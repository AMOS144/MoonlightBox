import type { BranchMessage } from './types'

const TIME_SEPARATOR_GAP_MS = 5 * 60 * 1000

type ChatTime = {
  year: number
  month: number
  day: number
  hour: number
  minute: number
  epochMs: number
}

export function mergeBranchMessages(
  persisted: BranchMessage[],
  staged: BranchMessage[],
) {
  const merged = new Map<string, BranchMessage>()
  for (const message of [...persisted, ...staged]) {
    const key = message.client_message_id
      ? `client:${message.client_message_id}`
      : message.turn_id
        ? `${message.role}:${message.turn_id}:${message.bubble_index}`
        : message.id
    merged.set(key, message)
  }
  return [...merged.values()]
}

export function assistantRevealSchedule(messages: BranchMessage[]) {
  let elapsed = 0
  return messages.map((message) => {
    elapsed += Math.min(Math.max(message.delay_ms ?? 0, 0), 5000)
    return elapsed
  })
}

/**
 * 聊天记录使用后端保存的钟面时间，而不是浏览器转换后的时区时间。
 * 导入记录通常只有 created_at，Runtime 消息则优先使用 observed_at。
 */
export function messageChatTime(message: BranchMessage): string | null {
  return message.observed_at ?? message.created_at ?? null
}

/** 微信式时间条：首条、跨日或同日相隔五分钟以上时显示。 */
export function shouldShowChatTime(
  previous: BranchMessage | undefined,
  current: BranchMessage,
): boolean {
  const currentTime = parseChatTime(messageChatTime(current))
  if (!currentTime) return false
  if (!previous) return true
  const previousTime = parseChatTime(messageChatTime(previous))
  if (!previousTime) return true
  if (!isSameDate(previousTime, currentTime)) return true
  return currentTime.epochMs - previousTime.epochMs >= TIME_SEPARATOR_GAP_MS
}

/** 生成类似微信的紧凑中文时间标签；较早的导入记录保留完整年月日。 */
export function formatChatTime(value: string, reference = new Date()): string {
  const time = parseChatTime(value)
  if (!time) return ''
  const referenceTime: ChatTime = {
    year: reference.getFullYear(),
    month: reference.getMonth() + 1,
    day: reference.getDate(),
    hour: reference.getHours(),
    minute: reference.getMinutes(),
    epochMs: Date.UTC(
      reference.getFullYear(),
      reference.getMonth(),
      reference.getDate(),
      reference.getHours(),
      reference.getMinutes(),
    ),
  }
  const clock = `${pad(time.hour)}:${pad(time.minute)}`
  if (isSameDate(time, referenceTime)) return `今天 ${clock}`
  if (dayDistance(referenceTime, time) === 1) return `昨天 ${clock}`
  return `${time.year}年${time.month}月${time.day}日 ${clock}`
}

function parseChatTime(value: string | null): ChatTime | null {
  if (!value) return null
  // 保留 API 传回的年月日和钟面，不让用户浏览器所在地的时区改变聊天历史。
  const matched = /^(\d{4})-(\d{2})-(\d{2})[T\s](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?/.exec(value)
  if (!matched) return null
  const [year, month, day, hour, minute, second = '0', fraction = '0'] = matched
    .slice(1)
    .map((item) => item ?? '0')
  const values = [year, month, day, hour, minute, second].map(Number)
  if (values.some((item) => !Number.isFinite(item))) return null
  const [parsedYear, parsedMonth, parsedDay, parsedHour, parsedMinute, parsedSecond] = values
  const millisecond = Number(`0.${fraction}`) * 1000
  const epochMs = Date.UTC(
    parsedYear,
    parsedMonth - 1,
    parsedDay,
    parsedHour,
    parsedMinute,
    parsedSecond,
    millisecond,
  )
  if (!Number.isFinite(epochMs)) return null
  return {
    year: parsedYear,
    month: parsedMonth,
    day: parsedDay,
    hour: parsedHour,
    minute: parsedMinute,
    epochMs,
  }
}

function isSameDate(left: ChatTime, right: ChatTime): boolean {
  return left.year === right.year && left.month === right.month && left.day === right.day
}

function dayDistance(reference: ChatTime, value: ChatTime): number {
  return Math.round(
    (Date.UTC(reference.year, reference.month - 1, reference.day) -
      Date.UTC(value.year, value.month - 1, value.day)) /
      86_400_000,
  )
}

function pad(value: number): string {
  return String(value).padStart(2, '0')
}
