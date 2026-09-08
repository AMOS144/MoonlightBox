import { useEffect, useRef, useState } from 'react'
import { Alert, Button, Group, NativeSelect, Paper, Progress, Stack, Stepper, Text, ThemeIcon, Title } from '@mantine/core'
import { Link, useInRouterContext } from 'react-router-dom'

import { Icon } from '../../components/Icon'

type Preview = {
  id: string
  message_count: number
  participants: string[]
  time_range: [string, string] | null
  kind_counts: Record<string, number>
  sample_messages: Array<{
    timestamp: string
    sender: string
    kind: string
    content: string
  }>
  errors: Array<{ line: number; code: string; message: string }>
  media_file_count: number
  linked_sticker_count: number
  unlinked_sticker_count: number
  deduplicated_asset_count?: number
  avatar_status?: Record<string, boolean>
  failure_reasons?: Record<string, number>
}

type ImportWizardProps = {
  projectId: string
}

type ConfirmResult = {
  import_id: string
  message_count: number
  created: boolean
  analysis_job_id: string | null
  world_job_id: string | null
}

type AnalysisJob = {
  id: string
  kind: string
  status: 'queued' | 'running' | 'interrupted' | 'succeeded' | 'failed' | 'cancelled'
  progress: number
  checkpoint: {
    stage?: string
    event_count?: number
    analysis_job_id?: string
  } | null
  error_code: string | null
  error_message: string | null
}

const ANALYSIS_STAGE_LABELS: Record<string, string> = {
  bundles_ready: '整理会话窗口',
  indexing_world: '建立人物世界',
  compiling_profile: '编译人物背景',
  world_ready: '人物背景已建立',
  loading_messages: '整理消息',
  extracting_candidates: '提取候选',
  reviewing_persistence: '复核持续性',
  windows: '复核持续性',
  ranking: '合并排名',
  publishing: '发布节点',
  events_created: '发布节点',
  completed: '发布节点',
}

function analysisFailureMessage(errorCode: string | null): string {
  if (errorCode?.startsWith('lightrag_') || errorCode?.startsWith('world_')) {
    return '人物世界构建未完成，请检查 LightRAG Sidecar、抽取模型和嵌入模型配置后重试。'
  }
  const code = errorCode?.replace(/^node_analysis_/, '')
  if (code === 'authentication' || code === 'missing_api_key' || code === 'disabled') {
    return '节点分析服务尚未就绪，请配置节点分析 API 后重试。'
  }
  if (code === 'invalid_response') {
    return '节点分析响应无效，请检查模型配置与兼容性后重试。'
  }
  if (code && ['rate_limit', 'server', 'network', 'timeout'].includes(code)) {
    return '节点分析服务繁忙或受限，请重试分析。'
  }
  return '自动分析未完成，请重试；如持续失败，请检查节点分析配置。'
}

export function ImportWizard({ projectId }: ImportWizardProps) {
  const inRouter = useInRouterContext()
  const [file, setFile] = useState<File | null>(null)
  const [mediaFiles, setMediaFiles] = useState<File[]>([])
  const [directoryName, setDirectoryName] = useState('')
  const [preview, setPreview] = useState<Preview | null>(null)
  const [scanning, setScanning] = useState(false)
  const [scanError, setScanError] = useState('')
  const [selfParticipant, setSelfParticipant] = useState('')
  const [targetParticipant, setTargetParticipant] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [confirmed, setConfirmed] = useState<ConfirmResult | null>(null)
  const [confirmError, setConfirmError] = useState('')
  const [analysisJob, setAnalysisJob] = useState<AnalysisJob | null>(null)
  const [followupJobId, setFollowupJobId] = useState<string | null>(null)
  const [pollGeneration, setPollGeneration] = useState(0)
  const [pollError, setPollError] = useState('')
  const [pollStopped, setPollStopped] = useState(false)
  const [retrying, setRetrying] = useState(false)
  const [retryError, setRetryError] = useState('')
  const scanController = useRef<AbortController | null>(null)
  const confirmController = useRef<AbortController | null>(null)
  const retryController = useRef<AbortController | null>(null)
  const pollController = useRef<AbortController | null>(null)
  const scanRequestId = useRef(0)
  const flowGeneration = useRef(0)

  useEffect(
    () => () => {
      flowGeneration.current += 1
      scanController.current?.abort()
      confirmController.current?.abort()
      retryController.current?.abort()
      pollController.current?.abort()
    },
    [],
  )

  useEffect(() => {
    const jobId = followupJobId ?? confirmed?.world_job_id ?? confirmed?.analysis_job_id
    if (!jobId) return
    let cancelled = false
    let timer: number | undefined
    let activeController: AbortController | undefined
    let failureCount = 0
    const generation = flowGeneration.current
    setPollError('')
    setPollStopped(false)

    async function poll() {
      activeController = new AbortController()
      pollController.current = activeController
      try {
        const response = await fetch(`/api/jobs/${jobId}`, {
          signal: activeController.signal,
        })
        if (!response.ok && response.status < 500) {
          if (!cancelled && generation === flowGeneration.current) {
            setPollError('无法查询分析进度，请确认任务仍然存在后重新查询。')
            setPollStopped(true)
          }
          return
        }
        if (!response.ok) throw new Error('查询分析进度失败')
        const job = (await response.json()) as AnalysisJob
        if (cancelled || generation !== flowGeneration.current) return
        failureCount = 0
        setPollError('')
        setPollStopped(false)
        setAnalysisJob(job)
        if (job.status === 'queued' || job.status === 'running') {
          timer = window.setTimeout(poll, 800)
        } else if (
          job.status === 'succeeded' &&
          job.kind === 'lightrag_world_build_v1' &&
          typeof job.checkpoint?.analysis_job_id === 'string'
        ) {
          setFollowupJobId(job.checkpoint.analysis_job_id)
        }
      } catch (error) {
        if (
          cancelled ||
          generation !== flowGeneration.current ||
          (error instanceof DOMException && error.name === 'AbortError')
        ) {
          return
        }
        failureCount += 1
        if (failureCount > 3) {
          setPollError('多次查询失败，请检查网络或服务状态后重新查询。')
          setPollStopped(true)
        } else {
          setPollError('暂时无法获取分析进度，正在重试……')
          timer = window.setTimeout(poll, 1500)
        }
      }
    }

    void poll()
    return () => {
      cancelled = true
      activeController?.abort()
      if (pollController.current === activeController) {
        pollController.current = null
      }
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [confirmed?.analysis_job_id, confirmed?.world_job_id, followupJobId, pollGeneration])

  async function retryAnalysis() {
    const jobId = followupJobId ?? confirmed?.world_job_id ?? confirmed?.analysis_job_id
    if (!jobId || retrying) return
    retryController.current?.abort()
    const controller = new AbortController()
    retryController.current = controller
    const generation = flowGeneration.current
    setRetrying(true)
    setRetryError('')
    try {
      const response = await fetch(`/api/jobs/${jobId}/resume`, {
        method: 'POST',
        signal: controller.signal,
      })
      if (!response.ok) throw new Error('恢复分析失败')
      const job = (await response.json()) as AnalysisJob
      if (generation !== flowGeneration.current || controller.signal.aborted) return
      setAnalysisJob(job)
      setPollGeneration((generation) => generation + 1)
    } catch (error) {
      if (
        generation === flowGeneration.current &&
        !(error instanceof DOMException && error.name === 'AbortError')
      ) {
        setRetryError('重试请求失败，请稍后再试。')
      }
    } finally {
      if (generation === flowGeneration.current) {
        setRetrying(false)
      }
    }
  }

  async function scan() {
    if (!file || scanning) return
    scanController.current?.abort()
    const controller = new AbortController()
    scanController.current = controller
    const requestId = ++scanRequestId.current
    setScanning(true)
    setScanError('')
    const form = new FormData()
    form.append('file', file)
    mediaFiles.forEach((mediaFile) => {
      form.append('media_files', mediaFile)
      form.append(
        'media_paths',
        mediaFile.webkitRelativePath || mediaFile.name,
      )
    })
    try {
      const response = await fetch(`/api/projects/${projectId}/imports/preview`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      })
      if (!response.ok) throw new Error('扫描请求失败')
      const result = (await response.json()) as Preview
      if (requestId === scanRequestId.current && !controller.signal.aborted) {
        setPreview(result)
        setSelfParticipant((current) =>
          result.participants.includes(current) ? current : '',
        )
        setTargetParticipant((current) =>
          result.participants.includes(current) ? current : '',
        )
      }
    } catch (error) {
      if (
        requestId === scanRequestId.current &&
        !(error instanceof DOMException && error.name === 'AbortError')
      ) {
        setScanError('扫描失败，请检查文件后重试。')
      }
    } finally {
      if (requestId === scanRequestId.current) {
        setScanning(false)
      }
    }
  }

  async function confirm() {
    if (!preview || submitting) return
    confirmController.current?.abort()
    const controller = new AbortController()
    confirmController.current = controller
    const generation = flowGeneration.current
    setSubmitting(true)
    setConfirmError('')
    try {
      const response = await fetch(
        `/api/projects/${projectId}/imports/${preview.id}/confirm`,
        {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          self_participant: selfParticipant,
          target_participant: targetParticipant,
        }),
        },
      )
      if (!response.ok) {
        throw new Error('导入请求失败')
      }
      const result = (await response.json()) as ConfirmResult
      if (generation !== flowGeneration.current || controller.signal.aborted) return
      setConfirmed(result)
    } catch (error) {
      if (
        generation === flowGeneration.current &&
        !(error instanceof DOMException && error.name === 'AbortError')
      ) {
        setConfirmError('导入失败，请检查文件和角色选择后重试。')
      }
    } finally {
      if (generation === flowGeneration.current) {
        setSubmitting(false)
      }
    }
  }

  const canConfirm =
    preview !== null &&
    preview.participants.includes(selfParticipant) &&
    preview.participants.includes(targetParticipant) &&
    selfParticipant !== targetParticipant

  function selectExportDirectory(files: File[]) {
    flowGeneration.current += 1
    scanController.current?.abort()
    confirmController.current?.abort()
    retryController.current?.abort()
    pollController.current?.abort()
    scanRequestId.current += 1
    const chatFiles = files.filter((candidate) =>
      /^chat\.(csv|json|txt)$/i.test(candidate.name),
    )
    const priority = ['chat.csv', 'chat.json', 'chat.txt']
    const selectedChat =
      priority
        .map((name) =>
          chatFiles.find((candidate) => candidate.name.toLowerCase() === name),
        )
        .find((candidate) => candidate !== undefined) ?? null
    const firstPath = files[0]?.webkitRelativePath
    setDirectoryName(firstPath?.split('/')[0] || '')
    setFile(selectedChat)
    setMediaFiles(
      files.filter(
        (candidate) =>
          !chatFiles.includes(candidate) &&
          candidate.name.toLowerCase() !== 'manifest.json' &&
          !candidate.name.startsWith('.'),
      ),
    )
    setScanning(false)
    setSubmitting(false)
    setRetrying(false)
    setPreview(null)
    setScanError(
      files.length > 0 && selectedChat === null
        ? '没有找到 chat.csv、chat.json 或 chat.txt，请选择完整的 WxEcho 导出目录。'
        : '',
    )
    setConfirmed(null)
    setConfirmError('')
    setAnalysisJob(null)
    setFollowupJobId(null)
    setPollError('')
    setRetryError('')
    setSelfParticipant('')
    setTargetParticipant('')
  }

  return (
    <Stack gap="xl">
      <div><Text c="moon.4" fw={700} size="xs">聊天数据</Text><Title mt={5} order={1}>导入真实聊天</Title><Text c="dimmed" mt={7}>选择完整的 WxEcho 导出目录，原始消息不会被清洗过程覆盖。</Text></div>
      <Stepper active={!preview ? 0 : !confirmed ? 1 : analysisJob?.status !== 'succeeded' || analysisJob?.kind === 'lightrag_world_build_v1' ? 2 : 3} allowNextStepsSelect={false}>
        <Stepper.Step label="选择记录" />
        <Stepper.Step label="确认双方" />
        <Stepper.Step label="整理回忆" />
        <Stepper.Completed>导入与分析完成</Stepper.Completed>
      </Stepper>
      <Paper p="md" withBorder><Text fw={700}>请选择 WxEcho 导出目录</Text><Text c="dimmed" mt={4} size="sm">系统会自动识别聊天记录、图片、语音、表情和头像。聊天记录优先使用 chat.csv，其次是 chat.json 和 chat.txt。</Text></Paper>
      <Paper component="label" htmlFor="export-directory" p="lg" style={{ cursor: 'pointer' }} withBorder>
        <Group wrap="nowrap">
        <ThemeIcon color="moon" size={44} variant="light"><Icon name="folder" size={24} /></ThemeIcon>
        <div>
          <Text fw={700}>{directoryName || '选择 WxEcho 导出目录'}</Text>
          <Text c="dimmed" size="sm">
            {file
              ? `已识别 ${file.name} 和 ${mediaFiles.length} 个媒体文件`
              : '点击选择包含聊天记录和媒体的完整文件夹'}
          </Text>
        </div>
        <Button component="span" ml="auto" variant="default">选择目录</Button>
        </Group>
      </Paper>
      <input
        aria-label="WxEcho 导出目录"
        className="directory-picker__input"
        id="export-directory"
        multiple
        onChange={(event) => {
          selectExportDirectory(Array.from(event.target.files ?? []))
        }}
        ref={(element) => element?.setAttribute('webkitdirectory', '')}
        type="file"
      />
      <Button
        disabled={!file || scanning}
        loading={scanning}
        onClick={scan}
        type="button"
      >
        {scanning ? '正在扫描……' : '扫描并预览'}
      </Button>
      {scanError && <Alert color="red" role="alert">{scanError}</Alert>}

      {preview && (
        <Paper p="lg" withBorder>
          <Title order={3}>共 {preview.message_count} 条消息</Title>
          {preview.media_file_count > 0 ? (
            <p>
              已读取 {preview.media_file_count} 个媒体文件，关联{' '}
              {preview.linked_sticker_count} 个表情，
              {preview.unlinked_sticker_count} 个表情待关联；去重后{' '}
              {preview.deduplicated_asset_count ?? 0} 个资源。
            </p>
          ) : null}
          {Object.entries(preview.avatar_status ?? {}).map(([name, linked]) => (
            <p key={name}>
              {name} 的头像：{linked ? '已关联' : '未找到强关联资源'}
            </p>
          ))}
          {Object.entries(preview.failure_reasons ?? {})
            .filter(([, count]) => count > 0)
            .map(([reason, count]) => (
              <p className="media-warning" key={reason}>
                {reason}：{count}
              </p>
            ))}
          <NativeSelect
            aria-label="我"
            id="self-participant"
            label="我"
            mt="md"
            onChange={(event) => setSelfParticipant(event.target.value)}
            value={selfParticipant}
          >
            <option value="">请选择</option>
            {preview.participants.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </NativeSelect>
          <NativeSelect
            aria-label="复刻对象"
            id="target-participant"
            label="复刻对象"
            mt="md"
            onChange={(event) => setTargetParticipant(event.target.value)}
            value={targetParticipant}
          >
            <option value="">请选择</option>
            {preview.participants.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </NativeSelect>
          <Button
            disabled={!canConfirm || submitting || confirmed !== null}
            loading={submitting}
            mt="lg"
            onClick={confirm}
            type="button"
          >
            {confirmed ? '已完成导入' : submitting ? '正在导入……' : '确认导入'}
          </Button>
          {confirmed && (
                  <>
                    <Alert color="green" mt="md" role="status">
                      导入完成，已保存 {confirmed.message_count} 条消息。
                    </Alert>
                    {!analysisJob && (confirmed.world_job_id || confirmed.analysis_job_id) && <p>正在启动自动分析……</p>}
                    {analysisJob?.status === 'queued' && <p role="status">等待分析任务开始……</p>}
                    {analysisJob?.status === 'running' && (
                      <div role="status">
                        <Text mt="md">
                        {ANALYSIS_STAGE_LABELS[analysisJob.checkpoint?.stage ?? ''] ??
                          '分析关键节点'}
                        ……{Math.round(analysisJob.progress * 100)}%
                        </Text><Progress mt="xs" value={analysisJob.progress * 100} />
                      </div>
                    )}
                    {pollError && (
                      <p role={pollStopped ? 'alert' : 'status'}>{pollError}</p>
                    )}
                    {pollStopped && (
                      <button
                        onClick={() => setPollGeneration((generation) => generation + 1)}
                        type="button"
                      >
                        重新查询
                      </button>
                    )}
                    {analysisJob?.status === 'succeeded' && analysisJob.kind === 'lightrag_world_build_v1' && (
                      <p role="status">人物背景已经建立，正在启动关键节点分析……</p>
                    )}
                    {analysisJob?.status === 'succeeded' && analysisJob.kind !== 'lightrag_world_build_v1' && (
                      <div className="analysis-success">
                        <p role="status">
                          分析完成，生成 {analysisJob.checkpoint?.event_count ?? 0} 个关键节点。
                        </p>
                        {inRouter ? (
                          <Link className="primary-button" to={`/projects/${projectId}/events`}>
                            查看关键节点
                          </Link>
                        ) : (
                          <a className="primary-button" href={`/projects/${projectId}/events`}>
                            查看关键节点
                          </a>
                        )}
                      </div>
                    )}
                    {analysisJob?.status === 'cancelled' && <p role="status">分析已取消。</p>}
                    {(analysisJob?.status === 'failed' ||
                      analysisJob?.status === 'interrupted') && (
                      <div>
                        <p role="alert">{analysisFailureMessage(analysisJob.error_code)}</p>
                        <button
                          disabled={retrying}
                          onClick={retryAnalysis}
                          type="button"
                        >
                          {retrying ? '正在重试……' : '重试分析'}
                        </button>
                        {retryError && <p role="alert">{retryError}</p>}
                      </div>
                    )}
                  </>
          )}
          {confirmError && <Alert color="red" role="alert">{confirmError}</Alert>}
        </Paper>
      )}
    </Stack>
  )
}
