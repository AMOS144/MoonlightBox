export type ApiErrorBody = {
  code?: string
  message?: string
  detail?: string
  details?: unknown
}

export class ApiError extends Error {
  readonly status: number
  readonly body: ApiErrorBody

  constructor(status: number, body: ApiErrorBody) {
    super(body.message ?? body.detail ?? '请求失败')
    this.status = status
    this.body = body
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })

  const rawBody = await response.text()
  let body: T | ApiErrorBody | undefined
  if (rawBody) {
    try {
      body = JSON.parse(rawBody) as T | ApiErrorBody
    } catch {
      body = { message: response.ok ? '服务返回了无法识别的数据' : '服务暂时不可用' }
    }
  }
  if (!response.ok) {
    throw new ApiError(response.status, (body ?? { message: '请求失败' }) as ApiErrorBody)
  }
  return body as T
}
