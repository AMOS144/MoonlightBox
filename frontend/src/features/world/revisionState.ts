import type { WorldRevision } from './types'

/** GET、SSE 和操作回执共用版本规则，迟到的旧响应不能恢复旧按钮状态。 */
export function newestRevision(previous: unknown, incoming: unknown): WorldRevision {
  const old = previous as WorldRevision | undefined
  const next = incoming as WorldRevision
  if (!old || old.id !== next.id) return next
  if (old.session_revision > next.session_revision) return old
  if (old.session_revision === next.session_revision
    && Date.parse(old.updated_at) > Date.parse(next.updated_at)) return old
  return next
}
