export type RuntimeBranch = {
  id: string
  project_id: string
  origin_event_id: string
  model_version_id: string
  title: string
  origin_time: string
  lifecycle_status: string
  created_at: string
}

export type RuntimeBranchMessage = {
  id: string
  branch_id: string
  sequence: number
  role: 'user' | 'assistant' | 'system'
  content: string
  type: string
  media_asset_id: string | null
  turn_id: string
  bubble_index: number
  delay_ms: number
  generation_status: string
  generation_metadata: Record<string, unknown>
  client_message_id: string | null
  observed_at: string | null
  is_proactive: boolean
  created_at: string
}
