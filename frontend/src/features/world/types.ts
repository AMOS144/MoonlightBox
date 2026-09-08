export type SourceStatus = 'direct' | 'summarized' | 'inferred' | 'superseded'

export type SourcedStatement = {
  text: string
  source_status: SourceStatus
  source_document_ids: string[]
  assertion_kind?: string
  referent?: string
  temporal_status?: string
  source_message_ids?: string[]
  evidence_quotes?: string[]
}

export type WorldProfile = {
  id: string
  project_id: string
  subject_person_id: string
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
  source_message_ids: string[]
  compiler_version: string
  created_at: string
  graph: {
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
    error_code: string | null
    error_message: string | null
    completed_at: string | null
  }
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
