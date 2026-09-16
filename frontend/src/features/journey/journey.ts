import { useQuery } from '@tanstack/react-query'
import { request } from '../../api/client'

export type StageState = 'not_started' | 'processing' | 'needs_user' | 'confirmed' | 'completed' | 'blocked' | 'paused'
export type JourneyAction = 'import' | 'participants' | 'profile' | 'nodes' | 'node_profile' | 'selection' | 'branches' | 'branch' | 'revision'
export type JourneyStage = { key: string; label: string; state: StageState; action: JourneyAction; detail: string; object_id: string | null }
export type Journey = {
  world_build?: { id: string; status: string; progress: number; error_message: string | null;
    failure?: { code: string; document_id?: string | null; chunk_id?: string | null } | null;
    indexed_bundles: number; bundle_count: number;
    recovery: { status?: string; retries?: number; max_retries?: number; reason?: string; error_code?: string; retry_at?: string | null } } | null
  project_id: string; stages: JourneyStage[]; tasks: (JourneyStage & { id: string })[]; next_action: JourneyStage; processing: boolean
  participants: { id: string; name: string; role: string; avatar_asset_id: string | null }[]
  imports: { id: string; preview_id: string; message_count: number; confirmed_at: string }[]
  time_range: [string | null, string | null]
  publication: { id: string; profile_id: string; graph_version_id: string } | null
  graph: { id: string; status: string } | null
  historical_branch_creation: { available: boolean; reason: string }
  branches: { id: string; title: string; origin_time: string; lifecycle_status: string; latest_message?: { text: string; role: string; virtual_time: string | null } | null }[]
}

export const journeyKey = (projectId: string) => ['journey', projectId] as const
export function useJourney(projectId: string) {
  return useQuery({ queryKey: journeyKey(projectId), queryFn: async () => {
    const value = await request<Journey>(`/api/projects/${projectId}/journey`)
    if (!value || !Array.isArray(value.stages) || !value.next_action || !Array.isArray(value.tasks)) {
      throw new Error('流程接口返回不完整，请确认后端已更新并重试。')
    }
    return value
  },
    enabled: Boolean(projectId), refetchInterval: q => q.state.data?.processing ? 3000 : false,
  })
}

// 只映射本项目的已知操作，服务端不能返回任意跳转地址。
export function actionPath(projectId: string, action: JourneyAction, id?: string | null) {
  const root = `/projects/${encodeURIComponent(projectId)}`
  switch (action) {
    case 'import': return `${root}/setup/import`
    case 'participants': return `${root}/setup/participants`
    case 'profile': return `${root}/world`
    case 'node_profile': return `${root}/node-profiles/${encodeURIComponent(id ?? '')}`
    case 'nodes': return `${root}/nodes${id ? `?investigation=${encodeURIComponent(id)}` : ''}`
    case 'selection': return `${root}/branches/new${id ? `?investigation=${encodeURIComponent(id)}` : ''}`
    case 'branches': return `${root}/branches`
    case 'branch': return id ? `${root}/branches/${encodeURIComponent(id)}` : `${root}/branches`
    case 'revision': return `${root}/world${id ? `?revision=${encodeURIComponent(id)}` : ''}`
  }
}

