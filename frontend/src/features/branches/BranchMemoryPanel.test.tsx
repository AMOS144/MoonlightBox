import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { BranchMemoryPanel } from './BranchMemoryPanel'

afterEach(() => {
  vi.unstubAllGlobals()
})

test('展示人格成长证据并确认回滚', async () => {
  const posts: string[] = []
  vi.stubGlobal('confirm', vi.fn(() => true))
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        posts.push(path)
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'state-3',
              version: 3,
              previous_version_id: 'state-2',
              persona_state: {},
              relationship_state: { trust: 50 },
              user_model: {},
              emotional_tendency: {},
              active_belief_ids: [],
              reason: '回滚',
              source_episode_ids: [],
              is_current: true,
              created_at: '2026-07-20T00:00:00Z',
              rolled_back_at: null,
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }
      let body: unknown
      if (path.endsWith('/episodes')) {
        body = [
          {
            id: 'episode-1',
            user_turn_id: 'u1',
            assistant_turn_id: 'a1',
            user_content: '我们慢慢来',
            assistant_bubbles: [{ type: 'text', content: '好' }],
            importance: 8,
            processing_status: 'processed',
            started_at: '2026-07-20T00:00:00Z',
            ended_at: '2026-07-20T00:00:01Z',
          },
        ]
      } else if (path.endsWith('/versions')) {
        body = [
          {
            id: 'state-2',
            version: 2,
            previous_version_id: 'state-1',
            persona_state: {},
            relationship_state: { trust: 55 },
            user_model: {},
            emotional_tendency: {},
            active_belief_ids: ['belief-1'],
            reason: '当前成长',
            source_episode_ids: ['episode-1'],
            is_current: true,
            created_at: '2026-07-20T00:00:02Z',
            rolled_back_at: null,
          },
          {
            id: 'state-1',
            version: 1,
            previous_version_id: null,
            persona_state: {},
            relationship_state: { trust: 50 },
            user_model: {},
            emotional_tendency: {},
            active_belief_ids: [],
            reason: '初始状态',
            source_episode_ids: [],
            is_current: false,
            created_at: '2026-07-20T00:00:00Z',
            rolled_back_at: null,
          },
        ]
      } else if (path.endsWith('/jobs')) {
        body = []
      } else {
        body = {
          identity_kernel: {
            id: 'kernel-1',
            model_version_id: 'model-1',
            schema_version: 'v1',
            content: { persona: '稳定、直接' },
            evidence_message_ids: ['m1'],
            created_at: '2026-07-20T00:00:00Z',
            locked_at: '2026-07-20T00:00:00Z',
          },
          current_state: {
            id: 'state-2',
            version: 2,
            previous_version_id: 'state-1',
            persona_state: {},
            relationship_state: { trust: 55 },
            user_model: {},
            emotional_tendency: {},
            active_belief_ids: ['belief-1'],
            reason: '当前成长',
            source_episode_ids: ['episode-1'],
            is_current: true,
            created_at: '2026-07-20T00:00:02Z',
            rolled_back_at: null,
          },
          active_beliefs: [
            {
              id: 'belief-1',
              kind: 'belief',
              content: '我愿意慢慢建立信任',
              subject: 'digital_human',
              predicate: '关系态度',
              object: '慢慢建立信任',
              confidence: 0.72,
              importance: 8,
              valid_from: '2026-07-20T00:00:01Z',
              valid_to: null,
              source_episode_ids: ['episode-1'],
              source_item_ids: [],
              review_status: 'approved',
            },
          ],
          competing_beliefs: [],
          recent_reflections: [],
          pending_jobs: 0,
          failed_jobs: 0,
          evolution_frozen: false,
          growth_health: {
            status: 'observing',
            processed_episode_count: 1,
            observation_span_days: 0,
            state_version_count: 2,
            approved_memory_count: 1,
            approved_reflection_count: 0,
            successful_cognitive_cycle_count: 0,
            failed_cognitive_cycle_count: 0,
            evidence_coverage_rate: 1,
            duplicate_lineage_count: 0,
            unmet_requirements: ['需要至少 20 个已处理互动 episode'],
          },
        }
      }
      return Promise.resolve(
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
    }),
  )
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })

  render(
    <QueryClientProvider client={queryClient}>
      <BranchMemoryPanel
        branchId="b1"
        onClose={() => undefined}
        projectId="p1"
      />
    </QueryClientProvider>,
  )

  expect(await screen.findByText('稳定人格内核')).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: '长期成长验证' })).toBeInTheDocument()
  expect(screen.getByText(/观察中 · 已观察 1 个 episode/)).toBeInTheDocument()
  expect(screen.getByText('需要至少 20 个已处理互动 episode')).toBeInTheDocument()
  expect(screen.getByText('我愿意慢慢建立信任')).toBeInTheDocument()
  fireEvent.click(screen.getByText('我愿意慢慢建立信任'))
  expect(await screen.findByText('用户：我们慢慢来')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '回滚到此版本' }))

  await waitFor(() => {
    expect(posts).toContain(
      '/api/projects/p1/branches/b1/memory/versions/state-1/rollback',
    )
  })
})
