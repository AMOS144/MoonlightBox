export type EventNode = {
  id: string
  project_id: string
  type: string
  start_message_id: string
  end_message_id: string
  before_state: string
  after_state: string
  emotion_labels: string[]
  topic: string
  conflict_level: number
  importance: number
  reason: string
  evidence_ids: string[]
  status: string
  created_at: string
}
