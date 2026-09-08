export type Branch = {
  id: string
  project_id: string
  origin_event_id: string
  model_version_id: string
  title: string
  origin_time: string
  state_snapshot: Record<string, unknown>
  lifecycle_status: 'active' | 'archived' | 'upgraded'
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
  status: 'preparing' | 'ready' | 'failed'
  progress: number
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

export type ConversationActorState = {
  status: 'idle' | 'observing' | 'thinking' | 'expressing' | 'waiting' | 'retrying'
  typing: boolean
  observed_message_sequence: number
  version: number
}

export type IdentityKernel = {
  id: string
  model_version_id: string
  schema_version: string
  content: Record<string, unknown>
  evidence_message_ids: string[]
  field_evidence: Record<string, string[]>
  field_confidence: Record<string, number>
  acceptance_report_id: string | null
  created_at: string
  locked_at: string
}

export type BranchMemoryItem = {
  id: string
  kind: string
  content: string
  subject: string
  predicate: string
  object: string
  confidence: number
  importance: number
  valid_from: string
  valid_to: string | null
  source_episode_ids: string[]
  source_item_ids: string[]
  review_status: string
  verification_status: string
  claim_key: string | null
  stance: string
  state_version_id: string | null
  root_episode_hashes: string[]
}

export type BranchMemoryEpisode = {
  id: string
  user_turn_id: string
  assistant_turn_id: string
  user_content: string
  assistant_bubbles: Array<Record<string, unknown>>
  importance: number
  processing_status: string
  started_at: string
  ended_at: string
}

export type BranchStateVersion = {
  id: string
  version: number
  previous_version_id: string | null
  persona_state: Record<string, unknown>
  relationship_state: Record<string, unknown>
  user_model: Record<string, unknown>
  emotional_tendency: Record<string, unknown>
  active_belief_ids: string[]
  contested_belief_ids: string[]
  current_goals: Record<string, unknown>
  current_concerns: Record<string, unknown>
  memory_cutoff_version: number | null
  rollback_of_version_id: string | null
  reason: string
  source_episode_ids: string[]
  is_current: boolean
  created_at: string
  rolled_back_at: string | null
}

export type LongitudinalGrowthHealth = {
  status: 'insufficient' | 'observing' | 'validated' | 'frozen'
  processed_episode_count: number
  observation_span_days: number
  state_version_count: number
  approved_memory_count: number
  approved_reflection_count: number
  successful_cognitive_cycle_count: number
  failed_cognitive_cycle_count: number
  evidence_coverage_rate: number
  duplicate_lineage_count: number
  unmet_requirements: string[]
}

export type BranchMemoryOverview = {
  identity_kernel: IdentityKernel
  current_state: BranchStateVersion
  active_beliefs: BranchMemoryItem[]
  competing_beliefs: BranchMemoryItem[]
  recent_reflections: BranchMemoryItem[]
  pending_jobs: number
  failed_jobs: number
  evolution_frozen: boolean
  growth_health: LongitudinalGrowthHealth
}

export type BranchMemoryJob = {
  id: string
  kind: string
  status: string
  checkpoint: Record<string, unknown> | null
  error_code: string | null
  error_message: string | null
}
