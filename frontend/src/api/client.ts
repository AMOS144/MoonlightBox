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

  const body = (await response.json()) as T | ApiErrorBody
  if (!response.ok) {
    throw new ApiError(response.status, body as ApiErrorBody)
  }
  return body as T
}
