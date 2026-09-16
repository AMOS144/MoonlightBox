import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { notifications } from '@mantine/notifications'
import { TestThemeProvider } from './TestThemeProvider'
import { SettingsPage } from '../features/settings/SettingsPage'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

test('统一保存三类运行时模型配置，空 Key 保留且只提示已全部保存', async () => {
  const initial = {
    agent: { model: 'agent-old', endpoint: 'https://agent.test/v1/chat/completions', key_configured: true },
    lightrag: {
      llm_model: 'graph-old', llm_endpoint: 'https://graph.test/v1', llm_key_configured: true,
      embedding_model: 'embed-old', embedding_endpoint: 'https://embed.test/v1',
      embedding_dimension: 1024, embedding_key_configured: true,
    },
  }
  let payload: Record<string, unknown> | undefined
  const show = vi.spyOn(notifications, 'show').mockImplementation(() => 'notice-id')
  vi.stubGlobal('fetch', vi.fn(async (url, init) => {
    expect(url).toBe('/api/settings')
    if (init?.method === 'PUT') payload = JSON.parse(String(init.body))
    return new Response(JSON.stringify(initial), { status: 200 })
  }))
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(<TestThemeProvider><QueryClientProvider client={client}><SettingsPage /></QueryClientProvider></TestThemeProvider>)
  const modelInputs = await screen.findAllByLabelText('模型名称', { exact: false })
  expect(modelInputs).toHaveLength(3)
  expect(screen.getAllByLabelText('API Key', { exact: false })).toHaveLength(3)
  expect(screen.getAllByRole('button', { name: '测试连接' })).toHaveLength(3)
  fireEvent.change(modelInputs[0], { target: { value: 'agent-new' } })
  fireEvent.click(screen.getByRole('button', { name: '全部保存' }))
  await waitFor(() => expect(show).toHaveBeenCalledWith({ color: 'green', message: '已全部保存' }))
  expect(payload).toMatchObject({
    agent: { model: 'agent-new', api_key: null },
    lightrag: { llm_api_key: null, embedding_api_key: null },
  })
  expect(screen.queryByText(/重启|旧配置|已生效/)).not.toBeInTheDocument()
  client.clear()
})
