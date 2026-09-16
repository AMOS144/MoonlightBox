export type Branch = {
  id: string
  project_id: string
  origin_event_id: string
  title: string
  origin_time: string
  state_snapshot: Record<string, unknown>
  lifecycle_status: 'active' | 'archived' | 'upgraded' | 'preparing' | 'prepare_failed'
  replacement_branch_id: string | null
  generation_policy_version: string
  origin_import_id: string | null
  origin_boundary_message_id: string | null
  baseline_manifest_id: string | null
  baseline_job_id: string | null
  baseline_status: 'preparing' | 'ready' | 'failed'
  baseline_error_code: string | null
  baseline_error_message: string | null
  baseline_ready_at: string | null
  created_at: string
}

export type BranchPreparation = {
  branch_id: string
  title?: string
  origin_time?: string
  background_mode?: string | null
  background?: Record<string, unknown> | null
  status: 'preparing' | 'ready' | 'failed'
  progress: number | null
  stage: string
  message_count: number
  event_count: number
  error_code: string | null
  error_message: string | null
}

export type MessageType =
  | 'text'
  | 'sticker'
  | 'image'
  | 'audio'
  | 'video'
  | 'video_thumbnail'
  | 'call'
  | 'quote'
  | 'reaction'
  | 'system'
  | 'unknown'

export type BranchHistoryMessage = {
  id: string
  source_id: string
  role: 'self' | 'target'
  content: string
  type: MessageType
  media_asset_id: string | null
  timestamp: string
}

export type BranchHistoryPage = {
  items: BranchHistoryMessage[]
  next_cursor: string | null
  has_more: boolean
  manifest_id: string
}

export type BranchMessage = {
  id: string
  branch_id: string
  sequence: number
  role: 'user' | 'assistant'
  content: string
  type: MessageType
  media_asset_id: string | null
  turn_id: string
  bubble_index: number
  delay_ms: number
  generation_status: 'completed' | 'failed'
  generation_metadata: Record<string, unknown>
  client_message_id?: string | null
  observed_at?: string | null
  expression_plan_id?: string | null
  actor_intent?: string | null
  is_proactive?: boolean
  created_at: string
}

/** 聊天页展示用的分支当前时间；不包含 Runtime 的内部锚点。 */
export type BranchVirtualClock = {
  has_day_plan?: boolean
  day_plan_preparation?: {
    status: 'pending' | 'running' | 'ready' | 'retryable_failed' | 'blocked'
    retry_at?: string | null
    error_code?: string | null
  }
  virtual_now: string
  timezone: string
  status: 'running' | 'paused'
  time_scale: number
}
