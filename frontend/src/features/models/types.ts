export type ModelVersion = {
  id: string
  project_id: string
  base_model: string
  adapter_path: string
  dataset_hash: string
  metrics: Record<string, number>
  status: string
  recommended: boolean
  active: boolean
  created_at: string
}

export function modelDisplayName(model: ModelVersion): string {
  if ((model.adapter_path ?? '').includes('evidence-layered-v3-candidate')) {
    return '记忆增强 v3'
  }
  return `数字人 ${model.id.slice(0, 8)}`
}
