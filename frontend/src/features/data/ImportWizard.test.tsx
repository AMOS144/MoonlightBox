import { act, cleanup, fireEvent, render as testingRender, screen } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterEach, expect, test, vi } from 'vitest'

import { ImportWizard } from './ImportWizard'
import { TestThemeProvider } from '../../test/TestThemeProvider'

function render(ui: ReactElement) {
  return testingRender(<TestThemeProvider>{ui}</TestThemeProvider>)
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

function jsonResponse(body: object, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function previewResponse() {
  return jsonResponse(
    {
      id: 'preview-1',
      message_count: 2,
      participants: ['甲', '乙'],
      time_range: ['2026-01-01T20:00:00', '2026-01-01T20:00:12'],
      kind_counts: { text: 2 },
      sample_messages: [],
      errors: [],
    },
    201,
  )
}

function confirmResponse() {
  return jsonResponse(
    {
      import_id: 'import-1',
      message_count: 2,
      created: true,
      analysis_job_id: 'job-1',
    },
    201,
  )
}

function jobResponse(
  status: string,
  progress: number,
  stage: string,
  options: { errorCode?: string; eventCount?: number } = {},
) {
  return jsonResponse({
    id: 'job-1',
    kind: 'event_analysis_v2',
    status,
    progress,
    checkpoint: { stage, event_count: options.eventCount },
    error_code: options.errorCode ?? null,
    error_message: '<secret>sk-sensitive</secret>',
  })
}

async function importFile() {
  const file = new File(['时间,发送者,类型,内容'], 'chat.csv', { type: 'text/csv' })
  fireEvent.change(screen.getByLabelText('WxEcho 导出目录'), {
    target: { files: [file] },
  })
  fireEvent.click(screen.getByRole('button', { name: '扫描并预览' }))
  await screen.findByText('共 2 条消息')
  fireEvent.change(screen.getByLabelText('我'), { target: { value: '乙' } })
  fireEvent.change(screen.getByLabelText('复刻对象'), { target: { value: '甲' } })
  fireEvent.click(screen.getByRole('button', { name: '确认导入' }))
  await screen.findByText('导入完成，已保存 2 条消息。')
}

test('按 V2 checkpoint 阶段显示真实进度并在成功后停止轮询', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(previewResponse())
    .mockResolvedValueOnce(confirmResponse())
    .mockResolvedValueOnce(jobResponse('running', 0.05, 'loading_messages'))
    .mockResolvedValueOnce(jobResponse('running', 0.25, 'extracting_candidates'))
    .mockResolvedValueOnce(jobResponse('running', 0.55, 'reviewing_persistence'))
    .mockResolvedValueOnce(jobResponse('running', 0.85, 'ranking'))
    .mockResolvedValueOnce(jobResponse('running', 0.95, 'publishing'))
    .mockResolvedValueOnce(jobResponse('succeeded', 1, 'completed', { eventCount: 1 }))
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  expect(screen.getByText('请选择 WxEcho 导出目录')).toBeInTheDocument()
  await importFile()
  expect(await screen.findByText(/整理消息.*5%/)).toBeInTheDocument()
  for (const expected of [
    /提取候选.*25%/,
    /复核持续性.*55%/,
    /合并排名.*85%/,
    /发布节点.*95%/,
  ]) {
    await act(() => vi.advanceTimersByTimeAsync(800))
    expect(await screen.findByText(expected)).toBeInTheDocument()
  }
  await act(() => vi.advanceTimersByTimeAsync(800))
  expect(await screen.findByText('分析完成，生成 1 个关键节点。')).toBeInTheDocument()
  const callsAtSuccess = fetch.mock.calls.length
  await act(() => vi.advanceTimersByTimeAsync(2400))
  expect(fetch).toHaveBeenCalledTimes(callsAtSuccess)
})

test.each([
  ['node_analysis_authentication', '请配置节点分析 API'],
  ['node_analysis_missing_api_key', '请配置节点分析 API'],
  ['node_analysis_rate_limit', '服务繁忙或受限，请重试分析'],
  ['node_analysis_invalid_response', '请检查模型配置与兼容性'],
])('失败代码 %s 显示安全提示且不泄漏错误详情', async (errorCode, copy) => {
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValueOnce(previewResponse())
      .mockResolvedValueOnce(confirmResponse())
      .mockResolvedValueOnce(jobResponse('failed', 0.4, 'reviewing_persistence', { errorCode })),
  )
  render(<ImportWizard projectId="project-1" />)
  await importFile()
  expect(await screen.findByText(new RegExp(copy))).toBeInTheDocument()
  expect(screen.queryByText(/sk-sensitive/)).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '重试分析' })).toBeInTheDocument()
})

test('中断后可重试且防重复点击，成功后恢复轮询', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  let resolveResume: ((response: Response) => void) | undefined
  let resolvePoll: ((response: Response) => void) | undefined
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(previewResponse())
    .mockResolvedValueOnce(confirmResponse())
    .mockResolvedValueOnce(jobResponse('interrupted', 0.45, 'reviewing_persistence'))
    .mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          resolveResume = resolve
        }),
    )
    .mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          resolvePoll = resolve
        }),
    )
    .mockResolvedValueOnce(jobResponse('succeeded', 1, 'completed', { eventCount: 2 }))
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  await importFile()
  const retry = await screen.findByRole('button', { name: '重试分析' })
  fireEvent.click(retry)
  fireEvent.click(retry)
  expect(retry).toBeDisabled()
  expect(
    fetch.mock.calls.filter(([path]) => path === '/api/jobs/job-1/resume'),
  ).toHaveLength(1)
  await act(async () => resolveResume?.(jobResponse('queued', 0.45, 'reviewing_persistence')))
  expect(await screen.findByText(/等待分析任务开始/)).toBeInTheDocument()
  await act(async () => resolvePoll?.(jobResponse('running', 0.5, 'reviewing_persistence')))
  expect(await screen.findByText(/复核持续性.*50%/)).toBeInTheDocument()
  await act(() => vi.advanceTimersByTimeAsync(800))
  expect(await screen.findByText('分析完成，生成 2 个关键节点。')).toBeInTheDocument()
})

test('取消状态不继续轮询且不显示重试按钮', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(previewResponse())
    .mockResolvedValueOnce(confirmResponse())
    .mockResolvedValueOnce(jobResponse('cancelled', 0.3, 'extracting_candidates'))
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  await importFile()
  expect(await screen.findByText('分析已取消。')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '重试分析' })).not.toBeInTheDocument()
  const callsAtCancel = fetch.mock.calls.length
  await act(() => vi.advanceTimersByTimeAsync(2400))
  expect(fetch).toHaveBeenCalledTimes(callsAtCancel)
})

test('扫描失败显示安全中文提示且扫描中禁止重复提交', async () => {
  let resolveScan: ((response: Response) => void) | undefined
  const fetch = vi.fn(
    () =>
      new Promise<Response>((resolve) => {
        resolveScan = resolve
      }),
  )
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  const file = new File(['内容'], 'chat.csv', { type: 'text/csv' })
  fireEvent.change(screen.getByLabelText('WxEcho 导出目录'), {
    target: { files: [file] },
  })
  const scan = screen.getByRole('button', { name: '扫描并预览' })
  fireEvent.click(scan)
  fireEvent.click(scan)
  expect(screen.getByRole('button', { name: '正在扫描……' })).toBeDisabled()
  expect(fetch).toHaveBeenCalledTimes(1)

  resolveScan?.(jsonResponse({ detail: '<secret>内部错误</secret>' }, 500))
  expect(await screen.findByRole('alert')).toHaveTextContent('扫描失败，请检查文件后重试')
  expect(screen.queryByText(/内部错误/)).not.toBeInTheDocument()
})

test('一次选择导出目录并自动优先识别 chat.csv 与媒体文件', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(previewResponse())
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  const chatText = new File(['文本'], 'chat.txt', { type: 'text/plain' })
  const chatCsv = new File(['表格'], 'chat.csv', { type: 'text/csv' })
  const sticker = new File(['图片'], 'sticker.gif', { type: 'image/gif' })
  const manifest = new File(['{}'], 'manifest.json', { type: 'application/json' })
  Object.defineProperty(chatText, 'webkitRelativePath', {
    value: '洪欣羽/chat.txt',
  })
  Object.defineProperty(chatCsv, 'webkitRelativePath', {
    value: '洪欣羽/chat.csv',
  })
  Object.defineProperty(sticker, 'webkitRelativePath', {
    value: '洪欣羽/media/sticker.gif',
  })
  Object.defineProperty(manifest, 'webkitRelativePath', {
    value: '洪欣羽/manifest.json',
  })

  fireEvent.change(screen.getByLabelText('WxEcho 导出目录'), {
    target: { files: [chatText, sticker, manifest, chatCsv] },
  })
  fireEvent.click(screen.getByRole('button', { name: '扫描并预览' }))
  await screen.findByText('共 2 条消息')

  const body = fetch.mock.calls[0]?.[1]?.body as FormData
  expect((body.get('file') as File).name).toBe('chat.csv')
  expect(body.getAll('media_files')).toEqual([sticker])
  expect(body.getAll('media_paths')).toEqual(['洪欣羽/media/sticker.gif'])
  expect(screen.queryByLabelText('聊天记录文件')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('微信媒体目录（可选）')).not.toBeInTheDocument()
})

test('目录中没有聊天文件时显示明确错误', () => {
  render(<ImportWizard projectId="project-1" />)
  const image = new File(['图片'], 'photo.jpg', { type: 'image/jpeg' })
  Object.defineProperty(image, 'webkitRelativePath', {
    value: '洪欣羽/media/photo.jpg',
  })

  fireEvent.change(screen.getByLabelText('WxEcho 导出目录'), {
    target: { files: [image] },
  })

  expect(screen.getByRole('alert')).toHaveTextContent(
    '没有找到 chat.csv、chat.json 或 chat.txt',
  )
  expect(screen.getByRole('button', { name: '扫描并预览' })).toBeDisabled()
})

test('换文件会取消旧扫描且旧响应不能覆盖新预览', async () => {
  let resolveOld: ((response: Response) => void) | undefined
  let oldSignal: AbortSignal | undefined
  const fetch = vi
    .fn()
    .mockImplementationOnce((_path: string, init?: RequestInit) => {
      oldSignal = init?.signal ?? undefined
      return new Promise<Response>((resolve) => {
        resolveOld = resolve
      })
    })
    .mockResolvedValueOnce(
      jsonResponse(
        {
          ...JSON.parse(await previewResponse().text()),
          message_count: 9,
          id: 'preview-new',
        },
        201,
      ),
    )
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  const input = screen.getByLabelText('WxEcho 导出目录')
  fireEvent.change(input, {
    target: { files: [new File(['旧'], 'chat.csv', { type: 'text/csv' })] },
  })
  fireEvent.click(screen.getByRole('button', { name: '扫描并预览' }))
  fireEvent.change(input, {
    target: { files: [new File(['新'], 'chat.csv', { type: 'text/csv' })] },
  })
  expect(oldSignal?.aborted).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: '扫描并预览' }))
  expect(await screen.findByText('共 9 条消息')).toBeInTheDocument()

  resolveOld?.(previewResponse())
  await act(async () => {})
  expect(screen.queryByText('共 2 条消息')).not.toBeInTheDocument()
  expect(screen.getByText('共 9 条消息')).toBeInTheDocument()
})

test('轮询遇到 4xx 立即停止并允许重新查询', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(previewResponse())
    .mockResolvedValueOnce(confirmResponse())
    .mockResolvedValueOnce(jsonResponse({ detail: '任务不可见' }, 403))
    .mockResolvedValueOnce(jobResponse('succeeded', 1, 'completed', { eventCount: 1 }))
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  await importFile()

  expect(await screen.findByRole('alert')).toHaveTextContent('无法查询分析进度')
  const callsAtFailure = fetch.mock.calls.length
  await act(() => vi.advanceTimersByTimeAsync(5000))
  expect(fetch).toHaveBeenCalledTimes(callsAtFailure)
  fireEvent.click(screen.getByRole('button', { name: '重新查询' }))
  expect(await screen.findByText('分析完成，生成 1 个关键节点。')).toBeInTheDocument()
})

test('轮询服务错误有限重试三次后停止', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(previewResponse())
    .mockResolvedValueOnce(confirmResponse())
    .mockResolvedValueOnce(jsonResponse({}, 500))
    .mockResolvedValueOnce(jsonResponse({}, 503))
    .mockRejectedValueOnce(new TypeError('network'))
    .mockResolvedValueOnce(jsonResponse({}, 500))
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  await importFile()
  for (let index = 0; index < 3; index += 1) {
    await act(() => vi.advanceTimersByTimeAsync(1500))
  }

  expect(await screen.findByRole('alert')).toHaveTextContent('多次查询失败')
  expect(fetch).toHaveBeenCalledTimes(6)
  await act(() => vi.advanceTimersByTimeAsync(5000))
  expect(fetch).toHaveBeenCalledTimes(6)
  expect(screen.getByRole('button', { name: '重新查询' })).toBeInTheDocument()
})

test('组件卸载时中止在途 Job 查询', async () => {
  let pollSignal: AbortSignal | undefined
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(previewResponse())
    .mockResolvedValueOnce(confirmResponse())
    .mockImplementationOnce((_path: string, init?: RequestInit) => {
      pollSignal = init?.signal ?? undefined
      return new Promise<Response>(() => {})
    })
  vi.stubGlobal('fetch', fetch)
  const view = render(<ImportWizard projectId="project-1" />)
  await importFile()
  expect(pollSignal?.aborted).toBe(false)
  view.unmount()
  expect(pollSignal?.aborted).toBe(true)
})

test('确认导入未完成时换文件会中止请求并忽略旧结果', async () => {
  let resolveConfirm: ((response: Response) => void) | undefined
  let confirmSignal: AbortSignal | undefined
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(previewResponse())
    .mockImplementationOnce((_path: string, init?: RequestInit) => {
      confirmSignal = init?.signal ?? undefined
      return new Promise<Response>((resolve) => {
        resolveConfirm = resolve
      })
    })
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  const input = screen.getByLabelText('WxEcho 导出目录')
  fireEvent.change(input, {
    target: { files: [new File(['旧'], 'chat.csv', { type: 'text/csv' })] },
  })
  fireEvent.click(screen.getByRole('button', { name: '扫描并预览' }))
  await screen.findByText('共 2 条消息')
  fireEvent.change(screen.getByLabelText('我'), { target: { value: '乙' } })
  fireEvent.change(screen.getByLabelText('复刻对象'), { target: { value: '甲' } })
  fireEvent.click(screen.getByRole('button', { name: '确认导入' }))
  expect(screen.getByRole('button', { name: '正在导入……' })).toBeDisabled()

  fireEvent.change(input, {
    target: { files: [new File(['新'], 'chat.csv', { type: 'text/csv' })] },
  })
  expect(confirmSignal?.aborted).toBe(true)
  resolveConfirm?.(confirmResponse())
  await act(async () => {})
  expect(screen.queryByText(/导入完成/)).not.toBeInTheDocument()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(fetch).toHaveBeenCalledTimes(2)
})

test('重试分析未完成时换文件会中止请求并忽略旧结果', async () => {
  let resolveRetry: ((response: Response) => void) | undefined
  let retrySignal: AbortSignal | undefined
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(previewResponse())
    .mockResolvedValueOnce(confirmResponse())
    .mockResolvedValueOnce(jobResponse('interrupted', 0.4, 'reviewing_persistence'))
    .mockImplementationOnce((_path: string, init?: RequestInit) => {
      retrySignal = init?.signal ?? undefined
      return new Promise<Response>((resolve) => {
        resolveRetry = resolve
      })
    })
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  await importFile()
  fireEvent.click(await screen.findByRole('button', { name: '重试分析' }))
  expect(await screen.findByRole('button', { name: '正在重试……' })).toBeDisabled()

  fireEvent.change(screen.getByLabelText('WxEcho 导出目录'), {
    target: { files: [new File(['新'], 'chat.csv', { type: 'text/csv' })] },
  })
  expect(retrySignal?.aborted).toBe(true)
  resolveRetry?.(jobResponse('queued', 0.4, 'reviewing_persistence'))
  await act(async () => {})
  expect(screen.queryByText(/等待分析任务开始/)).not.toBeInTheDocument()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(fetch).toHaveBeenCalledTimes(4)
})

test('换文件清空角色且新预览会清除不存在的旧参与者', async () => {
  const secondPreview = jsonResponse(
    {
      id: 'preview-2',
      message_count: 3,
      participants: ['丙', '丁'],
      time_range: null,
      kind_counts: { text: 3 },
      sample_messages: [],
      errors: [],
    },
    201,
  )
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(previewResponse())
    .mockResolvedValueOnce(secondPreview)
    .mockResolvedValueOnce(previewResponse())
  vi.stubGlobal('fetch', fetch)
  render(<ImportWizard projectId="project-1" />)
  const input = screen.getByLabelText('WxEcho 导出目录')
  fireEvent.change(input, {
    target: { files: [new File(['一'], 'chat.csv', { type: 'text/csv' })] },
  })
  fireEvent.click(screen.getByRole('button', { name: '扫描并预览' }))
  await screen.findByText('共 2 条消息')
  fireEvent.change(screen.getByLabelText('我'), { target: { value: '乙' } })
  fireEvent.change(screen.getByLabelText('复刻对象'), { target: { value: '甲' } })

  fireEvent.click(screen.getByRole('button', { name: '扫描并预览' }))
  await screen.findByText('共 3 条消息')
  expect(screen.getByLabelText('我')).toHaveValue('')
  expect(screen.getByLabelText('复刻对象')).toHaveValue('')
  expect(screen.getByRole('button', { name: '确认导入' })).toBeDisabled()

  fireEvent.change(input, {
    target: { files: [new File(['二'], 'chat.csv', { type: 'text/csv' })] },
  })
  fireEvent.click(screen.getByRole('button', { name: '扫描并预览' }))
  await screen.findByText('共 2 条消息')
  expect(screen.getByLabelText('我')).toHaveValue('')
  expect(screen.getByLabelText('复刻对象')).toHaveValue('')
  expect(screen.getByRole('button', { name: '确认导入' })).toBeDisabled()
})
