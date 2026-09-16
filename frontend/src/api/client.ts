import { userMessage } from '../components/feedback/messages'

export type ApiErrorBody = {
  code?: string
  message?: string
  detail?: string | { code?: string; message?: string }
  details?: unknown
}

export class ApiError extends Error {
  readonly status: number
  readonly body: ApiErrorBody

  constructor(status: number, body: ApiErrorBody) {
    const detail = typeof body.detail === 'object' && body.detail !== null ? body.detail : undefined
    super(userMessage(body.message ?? detail?.message ?? body.detail, status, body.code ?? detail?.code))
    this.status = status
    this.body = body
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })

  const rawBody = await response.text()
  let body: T | ApiErrorBody | undefined
  if (rawBody) {
    try {
      body = JSON.parse(rawBody) as T | ApiErrorBody
    } catch {
      if (response.ok) {
        throw new ApiError(response.status, {
          code: 'invalid_response', message: '服务返回了无法识别的数据，请重试',
        })
      }
      body = { message: '服务暂时不可用' }
    }
  }
  if (!response.ok) {
    throw new ApiError(response.status, (body ?? { message: '请求失败' }) as ApiErrorBody)
  }
  return body as T
}
