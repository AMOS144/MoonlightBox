export type SourceStatus = 'direct' | 'summarized' | 'inferred' | 'human_corrected' | 'superseded'

export type SourcedStatement = {
  claim_id?: string
  text: string
  source_status: SourceStatus
  source_document_ids: string[]
  assertion_kind?: string
  referent?: string
  temporal_status?: string
  source_message_ids?: string[]
  evidence_quotes?: string[]
}

export type ProfileStatementSelection = {
  key: string
  section: string
  sectionLabel: string
  statement: SourcedStatement
  entry_id?: string
  module_id?: string
  field_path?: string
}

export type StoredProfileStatementSelection = {
  entry_id?: string
  module_id?: string
  field_path?: string
  claim_id?: string
  section: string
  section_label: string
  text: string
  source_message_ids: string[]
}

export type WorldProfileFields = {
  identity: {
    names: SourcedStatement[]
    aliases: SourcedStatement[]
    self_descriptions: SourcedStatement[]
    roles: SourcedStatement[]
  }
  work_and_education: SourcedStatement[]
  places: SourcedStatement[]
  social_relationships: SourcedStatement[]
  preferences: SourcedStatement[]
  recurring_activities: SourcedStatement[]
  routine_summary: {
    workdays: SourcedStatement[]
    weekends: SourcedStatement[]
    other_patterns: SourcedStatement[]
  }
  life_phases: SourcedStatement[]
  relationship_with_user: {
    overview: SourcedStatement[]
    changes_over_time: SourcedStatement[]
  }
  important_events: SourcedStatement[]
  unresolved_candidates: SourcedStatement[]
}

export type WorldGraph = {
  id: string
  status: string
  message_count: number
  bundle_count: number
  lightrag_version: string | null
  embedding_model: string | null
  embedding_dimension: number | null
  extraction_model: string | null
  chunking_strategy: string | null
  entity_prompt_version: string | null
  parent_version_id: string | null
  revision: number
  correction_head_hash: string
  change_set_id: string | null
  published_at: string | null
  superseded_at: string | null
  error_code: string | null
  error_message: string | null
  completed_at: string | null
}

export type WorldProfile = WorldProfileFields & {
  id: string
  project_id: string
  subject_person_id: string
  source_message_ids: string[]
  compiler_version: string
  agent_run_id: string | null
  generation_summary: Record<string, unknown>
  profile_v2: Record<string, unknown>
  profile_v3?: Record<string, unknown>
  profile_schema_version: string
  investigation_report: InvestigationReport
  created_at: string
  graph: WorldGraph
}

export type WorldProfileDraftRead = {
  profile: WorldProfileFields & {
    id: string
    source_message_ids: string[]
    agent_run_id: string | null
    generation_summary: Record<string, unknown>
    profile_v2: Record<string, unknown>
    profile_v3?: Record<string, unknown>
    profile_schema_version: string
    investigation_report: InvestigationReport
  }
  graph: Pick<
    WorldGraph,
    'id' | 'status' | 'revision' | 'parent_version_id' | 'message_count' | 'bundle_count'
  >
  section_tasks: SectionTaskAudit[]
}

export type InvestigationStatus = {
  section: string
  state: 'completed' | 'completed_without_evidence' | 'needs_more_research' | 'cloud_error' | 'schema_error' | 'budget_exhausted' | 'not_started'
  accepted_fact_count: number
  unresolved_questions: string[]
  error_code: string | null
  trace_refs: string[]
}

export type InvestigationReport = {
  section_statuses: InvestigationStatus[]
  unresolved_questions: string[]
  cloud_and_schema_errors: Array<{ section: string, code: string }>
  trace_refs: string[]
}

export type SectionTaskAudit = {
  section: string
  status: string
  attempted_queries: Array<{ query_id?: string, question?: string, mode?: string }>
  research_round: number
  // 这是查询 Phoenix 根 Span 的关联键，不是业务数据库中的一份调用统计副本。
  phoenix_execution_id: string | null
  // 仅供改造前没有 execution id 的任务按稳定 owner id 从 Phoenix 回退读取。
  phoenix_owner_id: string | null
  unresolved_questions: string[]
  error_code: string | null
  retry_attempt: number
}

export type PhoenixAgentExecutionSummary = {
  execution_id: string
  phoenix_base_url: string
  trace_id: string | null
  terminal_reason: string | null
  tool_call_count: number
  tool_success_count: number
  tool_error_count: number
  tool_empty_count: number
  tool_cancelled_count: number
}

export type WorldGraphChangeSet = {
  id: string
  project_id: string
  revision_session_id: string
  base_graph_version_id: string
  revision: number
  status: string
  profile_patch: Array<Record<string, unknown>>
  graph_operations: Array<Record<string, unknown>>
  affected_entities: string[]
  affected_relations: Array<Record<string, string>>
  regression_queries: Array<Record<string, string>>
  payload_hash: string
  candidate_graph_version_id: string | null
  execution_result: Record<string, unknown> | null
}

export type WorldRevision = {
  id: string
  project_id: string
  base_graph_version_id: string
  base_profile_id: string | null
  status: string
  session_revision: number
  scope: Record<string, unknown>
  agent_stage: string | null
  agent_stage_progress: number | null
  context_snapshot_id: string | null
  pending_turn_id: string | null
  understanding_revision: number
  understanding: {
    wrong_interpretation: string
    corrected_interpretation: string
    affected_dimensions: string[]
    affected_profile_sections?: string[]
    graph_change_requested?: boolean
    source_message_ids: string[]
    open_question: string | null
    summary_for_user: string
  } | null
  understanding_payload_hash: string | null
  selected_statements: StoredProfileStatementSelection[]
  messages: Array<{
    id: string
    turn_id: string
    role: 'user' | 'assistant'
    kind: string
    in_reply_to_turn_id: string | null
    context_snapshot_id: string | null
    session_revision: number
    content: string
    payload: Record<string, unknown> | null
    created_at: string
  }>
  change_set: WorldGraphChangeSet | null
  created_at: string
  updated_at: string
}

export type WorldBuildProgress = {
  job_id: string
  status: string
  stage: string | null
  progress: number
  completed_questions: number | null
  question_count: number | null
  indexed_bundles: number | null
  bundle_count: number | null
}

export type WorldGraphStatus = {
  id: string
  status: string
  message_count: number
  bundle_count: number
  completed_at: string | null
  build_progress: WorldBuildProgress | null
}

export type EntityMergeProposal = {
  id: string
  project_id: string
  graph_version_id: string
  source_entities: string[]
  target_entity: string
  reason: string
  evidence: Array<{
    entity?: string
    basis?: string
    source_document_id?: string | null
    message_id?: string | null
    quote?: string | null
    [key: string]: unknown
  }>
  decision: 'pending' | 'approve' | 'reject' | 'defer'
  review_note: string | null
  merge_result: Record<string, unknown> | null
  created_at: string
  reviewed_at: string | null
}

export type WorldGraphNode = {
  id: string
  entity_type: string | null
  description: string | null
}

export type WorldGraphEdge = {
  source: string
  target: string
  keywords: string | null
}

export type WorldGraphSnapshot = {
  nodes: WorldGraphNode[]
  edges: WorldGraphEdge[]
  truncated: boolean
}
