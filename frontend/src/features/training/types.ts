export type TrainingCheckpoint = {
  stage?: string
  iteration?: number
  total_iterations?: number
  train_loss?: number
  validation_loss?: number
  model_version_id?: string
  example_count?: number
  dataset_coverage?: {
    target_message_count: number
    covered_target_message_count: number
    split_counts: Record<'train' | 'valid' | 'test', number>
  }
  candidate_runs?: Record<
    string,
    {
      status: string
      metrics?: {
        validation_loss?: number
        style_score?: number
      }
      error?: string
    }
  >
  best_candidate_id?: string
  acceptance?: {
    passed: boolean
    semantic_passed: boolean
    style_passed: boolean
    sticker_passed: boolean
    memorization_passed: boolean
    comparisons?: Record<string, number>
    failure_reasons: string[]
  }
}

export type TrainingJob = {
  id: string
  kind: string
  status: string
  progress: number
  checkpoint: TrainingCheckpoint | null
  error_code: string | null
  error_message: string | null
}
