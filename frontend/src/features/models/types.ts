export type ModelVersion = {
  id: string
  project_id: string
  base_model: string
  adapter_path: string
  dataset_hash: string
  metrics: Record<string, number>
  status: string
  recommended: boolean
  created_at: string
}
