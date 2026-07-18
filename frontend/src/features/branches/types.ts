export type Branch = {
  id: string
  project_id: string
  origin_event_id: string
  model_version_id: string
  title: string
  origin_time: string
  state_snapshot: Record<string, unknown>
  created_at: string
}

export type BranchMessage = {
  id: string
  branch_id: string
  sequence: number
  role: 'user' | 'assistant'
  content: string
  generation_metadata: Record<string, unknown>
  created_at: string
}
