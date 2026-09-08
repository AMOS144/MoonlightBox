import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { ApiError, request } from '../../api/client'
import type {
  BranchMemoryEpisode,
  BranchMemoryJob,
  BranchMemoryItem,
  BranchMemoryOverview,
  BranchStateVersion,
} from './types'

type Props = {
  projectId: string
  branchId: string
  onClose: () => void
}

export function BranchMemoryPanel({ projectId, branchId, onClose }: Props) {
  const queryClient = useQueryClient()
  const basePath = `/api/projects/${projectId}/branches/${branchId}/memory`
  const overview = useQuery({
    queryKey: ['branch-memory', branchId],
    queryFn: () => request<BranchMemoryOverview>(basePath),
    retry: false,
  })
  const episodes = useQuery({
    queryKey: ['branch-memory-episodes', branchId],
    queryFn: () => request<BranchMemoryEpisode[]>(`${basePath}/episodes`),
  })
  const versions = useQuery({
    queryKey: ['branch-memory-versions', branchId],
    queryFn: () => request<BranchStateVersion[]>(`${basePath}/versions`),
  })
  const jobs = useQuery({
    queryKey: ['branch-memory-jobs', branchId],
    queryFn: () => request<BranchMemoryJob[]>(`${basePath}/jobs`),
  })
  const refresh = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['branch-memory', branchId] }),
      queryClient.invalidateQueries({
        queryKey: ['branch-memory-episodes', branchId],
      }),
      queryClient.invalidateQueries({
        queryKey: ['branch-memory-versions', branchId],
      }),
      queryClient.invalidateQueries({
        queryKey: ['branch-memory-jobs', branchId],
      }),
      queryClient.invalidateQueries({ queryKey: ['branches', projectId] }),
      queryClient.invalidateQueries({ queryKey: ['branch-messages', branchId] }),
    ])
  }
  const migrate = useMutation({
    mutationFn: () => request(`${basePath}/migrate`, { method: 'POST' }),
    onSuccess: refresh,
  })
  const rollback = useMutation({
    mutationFn: (versionId: string) =>
      request(`${basePath}/versions/${versionId}/rollback`, { method: 'POST' }),
    onSuccess: refresh,
  })
  const retryJob = useMutation({
    mutationFn: (jobId: string) =>
      request(`${basePath}/jobs/${jobId}/retry`, { method: 'POST' }),
    onSuccess: refresh,
  })
  const episodeMap = new Map(
    (episodes.data ?? []).map((episode) => [episode.id, episode]),
  )
  const notMigrated =
    overview.error instanceof ApiError && overview.error.status === 409

  return (
    <aside aria-label="人格记忆" className="branch-memory-panel">
      <header className="branch-memory-panel__header">
        <div>
          <h2>人格记忆</h2>
          <p>仅显示当前分支的主体成长与证据</p>
        </div>
        <button aria-label="关闭人格记忆" onClick={onClose} type="button">
          ×
        </button>
      </header>
      {notMigrated ? (
        <div className="branch-memory-panel__empty">
          <p>该分支尚未建立持续人格记忆。</p>
          <button
            disabled={migrate.isPending}
            onClick={() => migrate.mutate()}
            type="button"
          >
            {migrate.isPending ? '迁移中…' : '开始迁移'}
          </button>
        </div>
      ) : overview.isLoading ? (
        <p className="branch-memory-panel__loading">正在读取人格记忆…</p>
      ) : overview.data ? (
        <div className="branch-memory-panel__content">
          <section>
            <h3>稳定人格内核</h3>
            <p className="memory-lock">训练后锁定，不会被单轮对话直接覆盖</p>
            <pre>{JSON.stringify(overview.data.identity_kernel.content, null, 2)}</pre>
          </section>
          <section>
            <h3>当前主体状态</h3>
            <MemoryState state={overview.data.current_state} />
          </section>
          <section aria-label="长期成长验证">
            <h3>长期成长验证</h3>
            <p>
              {growthStatusLabel(overview.data.growth_health.status)} · 已观察{' '}
              {overview.data.growth_health.processed_episode_count} 个 episode /{' '}
              {overview.data.growth_health.observation_span_days} 天
            </p>
            <p>
              状态版本 {overview.data.growth_health.state_version_count} · 长期记忆{' '}
              {overview.data.growth_health.approved_memory_count} · 阶段反思{' '}
              {overview.data.growth_health.approved_reflection_count}
            </p>
            <p>
              主体认知成功{' '}
              {overview.data.growth_health.successful_cognitive_cycle_count} · 失败{' '}
              {overview.data.growth_health.failed_cognitive_cycle_count}
            </p>
            <p>
              证据覆盖率{' '}
              {Math.round(overview.data.growth_health.evidence_coverage_rate * 100)}%
              {' · '}重复 lineage{' '}
              {overview.data.growth_health.duplicate_lineage_count}
            </p>
            {overview.data.growth_health.unmet_requirements.length ? (
              <ul>
                {overview.data.growth_health.unmet_requirements.map((requirement) => (
                  <li key={requirement}>{requirement}</li>
                ))}
              </ul>
            ) : (
              <p>长期人格成长已满足证据门槛。</p>
            )}
          </section>
          <MemoryGroup
            episodes={episodeMap}
            items={overview.data.active_beliefs}
            title="当前信念"
          />
          <MemoryGroup
            episodes={episodeMap}
            items={overview.data.competing_beliefs}
            title="竞争中的信念"
          />
          <MemoryGroup
            episodes={episodeMap}
            items={overview.data.recent_reflections}
            title="阶段反思"
          />
          <section>
            <h3>演化任务</h3>
            <p>
              待处理 {overview.data.pending_jobs} · 失败{' '}
              {overview.data.failed_jobs}
            </p>
            {overview.data.evolution_frozen ? (
              <p className="memory-error">演化已冻结，失败任务修复前状态不会变化。</p>
            ) : null}
            {(jobs.data ?? [])
              .filter((job) => job.status === 'failed')
              .map((job) => (
                <div className="memory-job" key={job.id}>
                  <span>{job.error_message ?? '任务失败'}</span>
                  <button
                    disabled={retryJob.isPending}
                    onClick={() => retryJob.mutate(job.id)}
                    type="button"
                  >
                    重试
                  </button>
                </div>
              ))}
          </section>
          <section>
            <h3>状态版本</h3>
            <div className="memory-version-list">
              {(versions.data ?? []).map((version) => (
                <div className="memory-version" key={version.id}>
                  <div>
                    <strong>版本 {version.version}</strong>
                    <span>{version.reason}</span>
                  </div>
                  {version.is_current ? (
                    <em>当前</em>
                  ) : (
                    <button
                      disabled={rollback.isPending}
                      onClick={() => {
                        if (
                          window.confirm(
                            `确认回滚到版本 ${version.version}？历史记录会保留。`,
                          )
                        ) {
                          rollback.mutate(version.id)
                        }
                      }}
                      type="button"
                    >
                      回滚到此版本
                    </button>
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>
      ) : (
        <p className="memory-error">人格记忆读取失败</p>
      )}
    </aside>
  )
}

function growthStatusLabel(status: BranchMemoryOverview['growth_health']['status']) {
  return {
    insufficient: '证据不足',
    observing: '观察中',
    validated: '已验证',
    frozen: '已冻结',
  }[status]
}

function MemoryState({ state }: { state: BranchStateVersion }) {
  return (
    <dl className="memory-state">
      <div>
        <dt>关系</dt>
        <dd>{JSON.stringify(state.relationship_state)}</dd>
      </div>
      <div>
        <dt>情绪倾向</dt>
        <dd>{JSON.stringify(state.emotional_tendency)}</dd>
      </div>
      <div>
        <dt>用户理解</dt>
        <dd>{JSON.stringify(state.user_model)}</dd>
      </div>
    </dl>
  )
}

function MemoryGroup({
  title,
  items,
  episodes,
}: {
  title: string
  items: BranchMemoryItem[]
  episodes: Map<string, BranchMemoryEpisode>
}) {
  return (
    <section>
      <h3>{title}</h3>
      {items.length === 0 ? (
        <p className="memory-empty">暂无</p>
      ) : (
        items.map((item) => (
          <details className="memory-item" key={item.id}>
            <summary>{item.content}</summary>
            <p>
              置信度 {Math.round(item.confidence * 100)}% · 重要度{' '}
              {item.importance}
            </p>
            <div>
              {item.source_episode_ids.map((episodeId) => {
                const episode = episodes.get(episodeId)
                return (
                  <blockquote key={episodeId}>
                    {episode
                      ? `用户：${episode.user_content}`
                      : `来源 episode：${episodeId}`}
                  </blockquote>
                )
              })}
            </div>
          </details>
        ))
      )}
    </section>
  )
}
