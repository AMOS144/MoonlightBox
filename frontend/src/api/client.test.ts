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
    message: '服务暂时不可用',
  })
})
