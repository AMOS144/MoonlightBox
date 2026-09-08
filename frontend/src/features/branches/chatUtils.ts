import type { BranchMessage } from './types'

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
