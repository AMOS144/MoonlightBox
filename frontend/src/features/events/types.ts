export type V2EventScoreComponents = {
  state_change_strength: number
  persistence: number
  evidence_quality: number
  model_confidence: number
}

export type EventScoreComponents = {
  event_significance: number
  relationship_impact: number
  evidence_quality: number
  persistence: number
  type_support: number
  model_confidence: number
}

export type EventEvidenceSummary = {
  message_id: string
  sender: string
  timestamp: string
  content: string
}

export type EventNode = {
  id: string
  project_id: string
  type: string
  start_message_id: string
  end_message_id: string
  before_state: string | null
  after_state: string | null
  emotion_labels: string[]
  topic: string
  conflict_level: number
  importance: number
  reason: string
  evidence_ids: string[]
  lane: 'relationship' | 'shared_experience'
  event_status: 'occurred' | 'confirmed'
  title: string
  summary: string
  display_summary: string | null
  summary_status: string
  summary_model: string | null
  started_at: string | null
  ended_at: string | null
  source_lanes: Array<'relationship' | 'shared_experience'>
  status: string
  created_at: string
  score_components: EventScoreComponents | V2EventScoreComponents | null
  evidence_summaries: EventEvidenceSummary[]
  analysis_version: string | null
  prompt_version: string | null
  model: string | null
  revision_number: number | null
  analysis_run_id: string | null
}
