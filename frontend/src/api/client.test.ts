import { afterEach, expect, test, vi } from 'vitest'

import { request } from './client'

afterEach(() => vi.unstubAllGlobals())

test('空的 204 响应不会因为 JSON 解析再次失败', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(null, { status: 204 })))
  await expect(request<void>('/api/example', { method: 'DELETE' })).resolves.toBeUndefined()
})

test('非 JSON 服务错误转换为可操作的 ApiError', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('Bad Gateway', { status: 502 })))
  await expect(request('/api/example')).rejects.toMatchObject({
    status: 502,
    message: '暂时无法完成操作，请稍后重试。',
  })
})

test('200 HTML 也应作为协议错误，而不是成功数据', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>proxy</html>', { status: 200 })))
  await expect(request('/api/example')).rejects.toMatchObject({
    status: 200, body: { code: 'invalid_response' },
  })
})
