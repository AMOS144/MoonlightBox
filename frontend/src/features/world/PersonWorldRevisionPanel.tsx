import { userMessage } from '../../components/feedback/messages'
import { Alert, Badge, Button, Divider, Drawer, Group, Paper, Pill, Progress, ScrollArea, Stack, Text, Textarea, Title, Tooltip } from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Icon } from '../../components/Icon'
import { useEffect, useMemo, useRef, useState } from 'react'

import { request } from '../../api/client'
import { AgentConversationPanel } from '../../components/agent/AgentConversationPanel'
import { useSessionState } from '../../hooks/useSessionState'
import { useSearchParams } from 'react-router-dom'
import { restoreProfileSelections } from './profileSelection'
import { newestRevision } from './revisionState'
import type { ProfileStatementSelection, WorldGraphChangeSet, WorldRevision } from './types'

type Props = {
  projectId: string
  baseProfileId?: string
  opened: boolean
  selections: ProfileStatementSelection[]
  onClose: () => void
  onLockChange: (locked: boolean) => void
  onRestoreSelections: (selections: ProfileStatementSelection[]) => void
  onStartNew: () => void
}

const terminalStatuses = new Set(['published', 'cancelled', 'failed', 'stale'])
const nonCancellableStatuses = new Set(['approved', 'executing', 'validating'])

export function PersonWorldRevisionPanel({
  projectId,
  baseProfileId,
  opened,
  selections,
  onClose,
  onLockChange,
  onRestoreSelections,
  onStartNew,
}: Props) {
  const narrow = useMediaQuery('(max-width: 1180px)')
  const queryClient = useQueryClient()
  const [routeParams] = useSearchParams()
  const requestedSession = routeParams.get('revision')
  const storageKey = `moonlightbox:person-world-revision:${projectId}:${baseProfileId ?? 'global'}`
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [message, setMessage] = useSessionState(`world-reply:${projectId}:${sessionId ?? 'new'}`, '')
  const [isRevising, setIsRevising] = useState(false)
  const [sseUnavailable, setSseUnavailable] = useState(false)
  const createRequestKey = useRef(newRequestKey())
  const replyRequestKey = useRef(newRequestKey())

  useEffect(() => {
    const stored = requestedSession || window.localStorage.getItem(storageKey)
    if (stored) setSessionId(stored)
  }, [storageKey, requestedSession])

  useEffect(() => {
    if (sessionId) window.localStorage.setItem(storageKey, sessionId)
    else window.localStorage.removeItem(storageKey)
  }, [sessionId, storageKey])

  const revision = useQuery({
    queryKey: ['person-world-revision', projectId, sessionId],
    queryFn: () => request<WorldRevision>(
      `/api/projects/${projectId}/world-agent/sessions/${sessionId}`,
    ),
    enabled: Boolean(sessionId),
    structuralSharing: newestRevision,
    refetchInterval: (query) => {
      const currentStatus = query.state.data?.status
      return sseUnavailable && currentStatus && !terminalStatuses.has(currentStatus) ? 2000 : false
    },
  })

  useEffect(() => {
    if (!sessionId || typeof EventSource === 'undefined') {
      setSseUnavailable(Boolean(sessionId))
      return undefined
    }
    let closed = false
    const source = new EventSource(
      `/api/projects/${projectId}/world-agent/sessions/${sessionId}/events`,
    )
    source.addEventListener('revision', (event) => {
      try {
        const next = JSON.parse((event as MessageEvent<string>).data) as WorldRevision
        if (closed) return
        if (next.id !== sessionId || !Number.isInteger(next.session_revision)) throw new Error('无效会话事件')
        queryClient.setQueryData(['person-world-revision', projectId, sessionId], next)
        setSseUnavailable(false)
      } catch {
        // 收到格式损坏的事件时不采用它；短轮询会作为安全回退读取完整状态。
        setSseUnavailable(true)
      }
    })
    source.onopen = () => setSseUnavailable(false)
    source.onerror = () => {
      if (!closed) setSseUnavailable(true)
    }
    return () => {
      closed = true
      source.close()
    }
  }, [projectId, queryClient, sessionId])

  const current = revision.data
  const activeSessions = useQuery({
    queryKey: ['person-world-revisions', projectId],
    queryFn: () => request<WorldRevision[]>(`/api/projects/${projectId}/world-agent/sessions`),
    enabled: opened && !sessionId,
    staleTime: 10_000,
  })
  const locked = Boolean(
    sessionId
    && (revision.isLoading || (current && !terminalStatuses.has(current.status))),
  )
  useEffect(() => onLockChange(locked), [locked, onLockChange])
  useEffect(() => {
    if (!selections.length && current?.selected_statements.length) {
      onRestoreSelections(restoreProfileSelections(current.selected_statements))
    }
  }, [current?.selected_statements, onRestoreSelections, selections.length])

  const sourceMessageIds = useMemo(
    () => Array.from(new Set(selections.flatMap(
      (item) => item.statement.source_message_ids ?? [],
    ))),
    [selections],
  )
  const selectedStatements = useMemo(() => selections.map((item) => ({
    entry_id: item.entry_id,
    module_id: item.module_id,
    field_path: item.field_path,
    claim_id: item.statement.claim_id,
    section: item.section,
    section_label: item.sectionLabel,
    text: item.statement.text,
    source_message_ids: item.statement.source_message_ids ?? [],
  })), [selections])

  const refresh = () => {
    if (sessionId) {
      return queryClient.invalidateQueries({
        queryKey: ['person-world-revision', projectId, sessionId],
      })
    }
  }

  const create = useMutation({
    mutationFn: () => request<WorldRevision>(`/api/projects/${projectId}/world-agent/sessions`, {
      method: 'POST',
      body: JSON.stringify({
        message,
        base_profile_id: baseProfileId,
        idempotency_key: createRequestKey.current,
        source_message_ids: sourceMessageIds,
        selected_statements: selectedStatements,
      }),
    }),
    onSuccess: (data) => {
      setSessionId(data.id)
      setMessage('')
      createRequestKey.current = newRequestKey()
      queryClient.setQueryData(['person-world-revision', projectId, data.id], data)
    },
  })

  const reply = useMutation({
    mutationFn: () => {
      if (!current) throw new Error('纠正会话尚未就绪')
      return request<WorldRevision>(
        `/api/projects/${projectId}/world-agent/sessions/${sessionId}/messages`,
        {
          method: 'POST',
          body: JSON.stringify({
            message,
            session_revision: current.session_revision,
            in_reply_to_turn_id: current.pending_turn_id,
            idempotency_key: replyRequestKey.current,
          }),
        },
      )
    },
    onSuccess: (data) => {
      setMessage('')
      setIsRevising(false)
      replyRequestKey.current = newRequestKey()
      queryClient.setQueryData(['person-world-revision', projectId, data.id], data)
    },
  })

  const resolveScopeProposal = useMutation({
    mutationFn: ({ action, itemIds }: { action: 'include' | 'exclude', itemIds: number[] }) => {
      if (!current) throw new Error('纠正会话尚未就绪')
      return request<WorldRevision>(
        `/api/projects/${projectId}/world-agent/sessions/${sessionId}/scope`,
        {
          method: 'POST',
          body: JSON.stringify({
            action,
            claim_ids: [],
            proposal_item_ids: itemIds,
            graph_object_refs: [],
            session_revision: current.session_revision,
            idempotency_key: newRequestKey(),
          }),
        },
      )
    },
    onSuccess: (data) => {
      queryClient.setQueryData(['person-world-revision', projectId, data.id], data)
    },
  })

  const runChangeAction = (suffix: string, changeSet: WorldGraphChangeSet) =>
    request<Record<string, unknown>>(
      `/api/projects/${projectId}/world-agent/sessions/${sessionId}/${suffix}`,
      {
        method: 'POST',
        body: JSON.stringify({
          change_set_id: changeSet.id,
          revision: changeSet.revision,
          payload_hash: changeSet.payload_hash,
          session_revision: current?.session_revision,
        }),
      },
    )

  const confirm = useMutation({
    mutationFn: () => request<WorldGraphChangeSet | WorldRevision>(
      `/api/projects/${projectId}/world-agent/sessions/${sessionId}/confirm-understanding`,
      {
        method: 'POST',
        body: JSON.stringify({
          understanding_revision: revision.data?.understanding_revision,
          understanding_payload_hash: revision.data?.understanding_payload_hash,
          session_revision: revision.data?.session_revision,
        }),
      },
    ),
    onSuccess: refresh,
  })
  const approveProfile = useMutation({
    mutationFn: (changeSet: WorldGraphChangeSet) => runChangeAction('approve-profile', changeSet),
    onSuccess: refresh,
  })
  const approveGraph = useMutation({
    mutationFn: (changeSet: WorldGraphChangeSet) => runChangeAction('approve-graph', changeSet),
    onSuccess: refresh,
  })
  const publish = useMutation({
    mutationFn: (changeSet: WorldGraphChangeSet) => request<Record<string, unknown>>(
      `/api/projects/${projectId}/world-agent/sessions/${sessionId}/publish`,
      {
        method: 'POST',
        body: JSON.stringify({
          change_set_id: changeSet.id,
          session_revision: current?.session_revision,
        }),
      },
    ),
    onSuccess: () => {
      refresh()
      void queryClient.invalidateQueries({ queryKey: ['world-profile', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['node-profile', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['world-profile-status', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['project-jobs', projectId] })
    },
  })
  const cancel = useMutation({
    mutationFn: () => {
      if (!current) throw new Error('纠正会话尚未就绪')
      return request<WorldRevision>(
        `/api/projects/${projectId}/world-agent/sessions/${sessionId}/cancel`,
        {
          method: 'POST',
          body: JSON.stringify({
            session_revision: current.session_revision,
            idempotency_key: newRequestKey(),
          }),
        },
      )
    },
    onSuccess: refresh,
  })
  const retryAgentTurn = useMutation({
    mutationFn: () => {
      if (!current) throw new Error('纠正会话尚未就绪')
      return request<WorldRevision>(
        `/api/projects/${projectId}/world-agent/sessions/${sessionId}/retry-agent-turn`,
        {
          method: 'POST',
          body: JSON.stringify({
            session_revision: current.session_revision,
            idempotency_key: newRequestKey(),
          }),
        },
      )
    },
    onSuccess: (data) => {
      queryClient.setQueryData(['person-world-revision', projectId, data.id], data)
    },
  })

  if (!opened) return null

  const changeSet = current?.change_set
  const busy = cancel.isPending || create.isPending || reply.isPending || resolveScopeProposal.isPending || confirm.isPending
    || approveProfile.isPending || approveGraph.isPending || publish.isPending || retryAgentTurn.isPending
  const error = create.error || reply.error || resolveScopeProposal.error || confirm.error || approveProfile.error
    || approveGraph.error || publish.error || cancel.error || retryAgentTurn.error || revision.error
  const showComposer = !sessionId
    || current?.status === 'waiting_for_user'
    || current?.status === 'exploring'
    || isRevising

  const startNew = () => {
    setSessionId(null)
    setMessage('')
    setIsRevising(false)
    onStartNew()
  }

  const panel = (
    <Paper className="person-world-agent-panel" component="aside" radius="md" withBorder>
      <AgentConversationPanel title="修改人物背景" subject={`本次核对 ${current?.selected_statements.length ?? selections.length} 项人物描述`}
        scope="影响项目人物背景；图谱变更需另外批准，不会自动替换已有分支。"
        status={statusLabel(current?.status ?? 'exploring')} onClose={onClose} />

      <ScrollArea className="person-world-agent-panel__body">
        <Stack gap="md" p="md">
          {current ? (
            <Stack gap="xs">
              <Group justify="space-between">
                <Text fw={700} size="sm">会话</Text>
                <Badge variant="light">{statusLabel(current.status)}</Badge>
              </Group>
              {current.messages.map((item) => (
                isQuestionTurn(item) ? (
                  <QuestionTurnCard
                    key={item.id}
                    question={item.payload.question}
                    onChoose={(answer) => { if (!busy) setMessage(answer) }}
                  />
                ) : isScopeProposalTurn(item) ? (
                  <ScopeProposalCard
                    key={item.id}
                    proposal={item.payload.scope_proposal}
                    loading={busy}
                    onResolve={(action, itemIds) => resolveScopeProposal.mutate({ action, itemIds })}
                  />
                ) : (
                  <Paper
                    className={`person-world-message person-world-message--${item.role}`}
                    key={item.id}
                    p="sm"
                    radius="md"
                    withBorder
                  >
                    <Text c="dimmed" size="xs">
                      {item.role === 'user' ? '你' : 'PersonWorldAgent'}
                    </Text>
                    <Text mt={3} size="sm">{item.content}</Text>
                  </Paper>
                )
              ))}
            </Stack>
          ) : (
            <Stack gap="sm">
              <Alert color="moon" title="先选择要核对的陈述">
                可在左侧选择一条或多条，也可以不选择，直接补充人物世界中缺失的事实。
              </Alert>
              {activeSessions.data?.length ? (
                <Paper p="sm" withBorder>
                  <Text fw={600} size="xs">恢复尚未完成的纠正</Text>
                  <Stack gap="xs" mt="xs">
                    {activeSessions.data.map((item) => (
                      <Button
                        key={item.id}
                        disabled={busy}
                        justify="flex-start"
                        size="xs"
                        variant="light"
                        onClick={() => setSessionId(item.id)}
                      >
                        {statusLabel(item.status)} · {item.updated_at.slice(0, 16).replace('T', ' ')}
                      </Button>
                    ))}
                  </Stack>
                </Paper>
              ) : null}
            </Stack>
          )}

          {current?.status === 'understanding_ready' && current.understanding ? (
            <Paper p="md" withBorder>
              {typeof current.scope.profile_preview_error === 'string' ? (
                <Alert color="red" mb="sm">{userMessage(current.scope.profile_preview_error)}</Alert>
              ) : null}
              <StepTitle index="1" title="确认 Agent 的理解" />
              <Text mt="sm" size="sm">{current.understanding.summary_for_user}</Text>
              <Text c="dimmed" mt="xs" size="xs">
                正确解释：{current.understanding.corrected_interpretation}
              </Text>
              {current.understanding.affected_profile_sections?.length ? (
                <Text mt="xs" size="sm">直接修改范围：{current.understanding.affected_profile_sections.map(profileSectionLabel).join('、')}</Text>
              ) : null}
              {current.understanding.graph_change_requested !== undefined ? (
                <Text mt="xs" size="xs" c="dimmed">{current.understanding.graph_change_requested
                  ? '图谱变更会另行展示并请求批准，不随本次理解确认直接执行。'
                  : '本次只修改人物画像，不修改图谱。'}</Text>
              ) : null}
              <Group mt="md">
                <Button
                  leftSection={<Icon name="check" size={15} />}
                  loading={confirm.isPending}
                  disabled={busy}
                  size="xs"
                  onClick={() => confirm.mutate()}
                >
                  理解正确
                </Button>
                <Button
                  size="xs"
                  variant="light"
                  onClick={() => beginRevision(setMessage, setIsRevising)}
                  disabled={busy}
                >
                  继续说明
                </Button>
              </Group>
            </Paper>
          ) : null}

          {changeSet ? (
            <ChangeSetReview
              changeSet={changeSet}
              revisionStatus={current?.status ?? ''}
              busy={busy}
              onApproveProfile={() => approveProfile.mutate(changeSet)}
              onApproveGraph={() => approveGraph.mutate(changeSet)}
              onPublish={() => publish.mutate(changeSet)}
              onRevise={() => beginRevision(setMessage, setIsRevising)}
              evidenceContext={evidenceContext(projectId, sessionId, current?.context_snapshot_id)}
            />
          ) : null}

          {current && ['approved', 'executing', 'validating'].includes(current.status) ? (
            <Alert color="blue" title="正在构建候选人物世界">
              正在重建隔离图谱、重放历史纠正、重新编译 Profile 并执行回归检查。
              当前 Runtime 不受影响。
            </Alert>
          ) : null}
          {current && ['agent_queued', 'agent_running'].includes(current.status) ? (
            <Alert color="blue" title="正在核对人物资料">
              {agentStageLabel(current.agent_stage)}；完成后会在此处给出一个可确认的问题或共同理解。
              {typeof current.agent_stage_progress === 'number' ? (
                <Progress mt="sm" value={Math.round(current.agent_stage_progress * 100)} />
              ) : null}
            </Alert>
          ) : null}
          {current?.status === 'profile_compiling' ? (
            <Alert color="blue" title="正在生成合并修改预览">
              正在修订模块并让相关栏目重新判断。完成后会统一展示 Diff，尚未修改已发布画像或图谱。
              {typeof current.agent_stage_progress === 'number' ? <Progress mt="sm" value={Math.round(current.agent_stage_progress * 100)} /> : null}
            </Alert>
          ) : null}
          {current?.status === 'published' ? (
            <Alert color="green" icon={<Icon name="check" size={16} />} title="纠正已发布">
              Graph 与 Profile 已作为同一版本切换。
            </Alert>
          ) : null}
          {current?.status === 'failed' ? (
            <Alert color="red" title="候选版本验证失败">
              当前已发布人物世界没有变化。请查看问题后发起新的纠正。
            </Alert>
          ) : null}
          {current?.status === 'stale' ? (
            <Alert color="yellow" title="人物资料已有更新">
              本轮核对期间发布了新版本。请基于最新资料重新确认修改，避免覆盖其他更新。
              <Button mt="sm" size="xs" variant="light" onClick={startNew} disabled={busy}>
                基于最新版本重新开始
              </Button>
            </Alert>
          ) : null}
          {current?.status === 'agent_failed' ? (
            <Alert color="red" title="这次核对尚未完成">
              未产生任何修改或图谱写入。已保留本轮用户输入和范围；检查模型服务后可安全重试。
              <Button
                loading={retryAgentTurn.isPending}
                disabled={busy}
                mt="sm"
                size="xs"
                variant="light"
                onClick={() => retryAgentTurn.mutate()}
              >
                重试本回合
              </Button>
            </Alert>
          ) : null}
          {error ? <Alert color="red">操作失败：{userMessage(error)}</Alert> : null}
        </Stack>
      </ScrollArea>

      <div className="person-world-agent-panel__footer">
        {showComposer ? (
          <Stack gap="xs">
            {selections.length ? (
              <Pill.Group aria-label="已选核对范围">
                {selections.map((item) => (
                  <Tooltip key={item.key} label={item.statement.text} multiline maw={320}>
                    <Pill
                      withRemoveButton={!locked}
                      onRemove={() => onRestoreSelections(selections.filter((entry) => entry.key !== item.key))}
                      removeButtonProps={{ 'aria-label': `移除${item.label ?? item.sectionLabel}` }}
                    >
                      <Group gap={4} wrap="nowrap">
                        <Icon name="quote" size={12} />
                        {item.label ?? item.sectionLabel}
                      </Group>
                    </Pill>
                  </Tooltip>
                ))}
              </Pill.Group>
            ) : null}
            <Textarea
              disabled={busy}
              autosize
              minRows={3}
              maxRows={7}
              placeholder={!sessionId
                ? '说明哪里不对，或补充缺失事实…'
                : '回答 Agent，或继续说明你的修改意图…'}
              value={message}
              onChange={(event) => setMessage(event.currentTarget.value)}
            />
            <Button
              disabled={busy || (!message.trim() && !selections.length)}
              loading={create.isPending || reply.isPending}
              onClick={() => sessionId ? reply.mutate() : create.mutate()}
            >
              {!sessionId ? '开始核对' : '发送'}
            </Button>
          </Stack>
        ) : null}
        <Divider my="sm" />
        <Group justify="space-between">
          {current && terminalStatuses.has(current.status) ? (
            <Button
              leftSection={<Icon name="plus" size={15} />}
              size="xs"
              variant="light"
              onClick={startNew}
              disabled={busy}
            >
              新建纠正
            </Button>
          ) : sessionId && revision.isError ? (
            <Button
              leftSection={<Icon name="plus" size={15} />}
              size="xs"
              variant="light"
              onClick={startNew}
              disabled={busy}
            >
              清除并新建
            </Button>
          ) : <span />}
          {current
            && !terminalStatuses.has(current.status)
            && !nonCancellableStatuses.has(current.status) ? (
            <Button
              color="red"
              loading={cancel.isPending}
              disabled={busy}
              size="xs"
              variant="subtle"
              onClick={() => cancel.mutate()}
            >
              取消本次纠正
            </Button>
          ) : null}
        </Group>
      </div>
    </Paper>
  )
  // 窄屏用标准抽屉管理焦点、Esc 和遮罩，关闭不取消服务端修订。
  return narrow ? <Drawer opened={opened} onClose={onClose} position="right" size="md" title="修改人物背景" className="profile-revision-drawer">{panel}</Drawer> : panel
}

type QuestionTurnPayload = {
  premise: string
  decision_key: string
  question: string
  options: Array<{ id: string; label: string; effect: string; recommended?: boolean }>
}

function isQuestionTurn(item: WorldRevision['messages'][number]): item is WorldRevision['messages'][number] & {
  payload: { question: QuestionTurnPayload }
} {
  const candidate = item.payload?.question as Partial<QuestionTurnPayload> | undefined
  return item.kind === 'question'
    && typeof candidate === 'object'
    && candidate !== null
    && typeof candidate.premise === 'string'
    && typeof candidate.question === 'string'
    && Array.isArray(candidate.options)
}

function QuestionTurnCard({
  question,
  onChoose,
}: {
  question: QuestionTurnPayload
  onChoose: (answer: string) => void
}) {
  return (
    <Paper p="md" withBorder>
      <StepTitle index="?" title="待确认项" />
      <Text c="dimmed" mt="sm" size="xs">已确认前提</Text>
      <Text size="sm">{question.premise}</Text>
      <Text fw={600} mt="sm" size="sm">{question.question}</Text>
      {question.options.length ? (
        <Stack gap="xs" mt="sm">
          {question.options.map((option) => (
            <Button
              key={option.id}
              color={option.recommended ? 'moon' : 'gray'}
              justify="flex-start"
              size="xs"
              variant={option.recommended ? 'light' : 'default'}
              onClick={() => onChoose(option.label)}
            >
              <Stack align="flex-start" gap={2}>
                <Text inherit>{option.label}</Text>
                <Text c="dimmed" inherit size="xs">{option.effect}</Text>
              </Stack>
            </Button>
          ))}
        </Stack>
      ) : null}
      <Text c="dimmed" mt="sm" size="xs">你也可以在下方用自己的话回答。</Text>
    </Paper>
  )
}

type ScopeProposalPayload = {
  summary: string
  items: Array<{
    candidate_item: number
    reason: string
    effect: string
    section?: string
    statement?: string
    structural_relations?: string[]
  }>
}

function isScopeProposalTurn(item: WorldRevision['messages'][number]): item is WorldRevision['messages'][number] & {
  payload: { scope_proposal: ScopeProposalPayload }
} {
  const candidate = item.payload?.scope_proposal as Partial<ScopeProposalPayload> | undefined
  return item.kind === 'scope_proposal'
    && typeof candidate === 'object'
    && candidate !== null
    && typeof candidate.summary === 'string'
    && Array.isArray(candidate.items)
}

function ScopeProposalCard({
  proposal,
  loading,
  onResolve,
}: {
  proposal: ScopeProposalPayload
  loading: boolean
  onResolve: (action: 'include' | 'exclude', itemIds: number[]) => void
}) {
  const itemIds = proposal.items.map((item) => item.candidate_item)
  return (
    <Paper p="md" withBorder>
      <StepTitle index="范围" title="关联核对项" />
      <Text mt="sm" size="sm">{proposal.summary}</Text>
      <Stack gap="xs" mt="sm">
        {proposal.items.map((item) => (
          <Paper bg="dark.7" key={item.candidate_item} p="sm" radius="sm">
            <Text fw={600} size="xs">候选 {item.candidate_item}</Text>
            {item.statement ? <Text mt={3} size="xs">{item.statement}</Text> : null}
            <Text c="dimmed" mt={3} size="xs">为什么相关：{item.reason}</Text>
            <Text c="dimmed" mt={3} size="xs">纳入会怎样：{item.effect}</Text>
          </Paper>
        ))}
      </Stack>
      <Group mt="md">
        <Button
          loading={loading}
          size="xs"
          onClick={() => onResolve('include', itemIds)}
        >
          纳入本轮核对
        </Button>
        <Button
          loading={loading}
          size="xs"
          variant="light"
          onClick={() => onResolve('exclude', itemIds)}
        >
          本轮排除
        </Button>
      </Group>
    </Paper>
  )
}

function ChangeSetReview({
  changeSet,
  revisionStatus,
  busy,
  onApproveProfile,
  onApproveGraph,
  onPublish,
  onRevise,
  evidenceContext,
}: {
  changeSet: WorldGraphChangeSet
  revisionStatus: string
  busy: boolean
  onApproveProfile: () => void
  onApproveGraph: () => void
  onPublish: () => void
  onRevise: () => void
  evidenceContext: EvidenceContext | null
}) {
  return (
    <Stack gap="sm">
      <Paper p="md" withBorder>
        <StepTitle index="2" title="审核 Profile 修改" />
        <Stack gap="xs" mt="sm">
          {changeSet.profile_patch.length ? changeSet.profile_patch.map((operation, index) => (
            <ChangeRow
              key={`${String(operation.operation)}-${index}`}
              action={profileOperationLabel(String(operation.operation ?? ''))}
              target={profileSectionLabel(String(operation.section ?? ''))}
              before={optionalText(operation.current_text)}
              after={optionalText(operation.replacement_text)}
              reason={optionalText(operation.reason)}
              sourceMessageIds={stringValues(operation.source_message_ids)}
              evidenceContext={evidenceContext}
            />
          )) : <Text c="dimmed" size="xs">本次不修改目标人物 Profile。</Text>}
        </Stack>
        {revisionStatus === 'profile_review' ? (
          <Group mt="md">
            <Button disabled={busy} size="xs" onClick={onApproveProfile}>批准 Profile 修改</Button>
            <Button disabled={busy} size="xs" variant="light" onClick={onRevise}>继续修改</Button>
          </Group>
        ) : null}
      </Paper>

      <Paper p="md" withBorder>
        <StepTitle index="3" title="审核图谱修改" />
        <Stack gap="xs" mt="sm">
          {changeSet.graph_operations.length ? changeSet.graph_operations.map((operation, index) => (
            <ChangeRow
              key={`${String(operation.operation_id)}-${index}`}
              action={graphOperationLabel(String(operation.operation_type ?? ''))}
              target={graphOperationTarget(operation)}
              before={optionalText(operation.before_description)}
              after={optionalText(operation.after_description) ?? optionalText(operation.relation_description)}
              reason={optionalText(operation.reason)}
              warning={operation.cascade === true ? '会级联删除关联关系' : null}
              sourceMessageIds={stringValues(operation.source_message_ids)}
              evidenceContext={evidenceContext}
            />
          )) : <Text c="dimmed" size="xs">本次不修改 LightRAG 图谱。</Text>}
        </Stack>
        {changeSet.affected_entities.length || changeSet.affected_relations.length ? (
          <Paper bg="dark.7" mt="sm" p="sm" radius="sm">
            {changeSet.affected_entities.length ? (
              <Text size="xs">受影响实体：{changeSet.affected_entities.join('、')}</Text>
            ) : null}
            {changeSet.affected_relations.length ? (
              <Text c="dimmed" mt={changeSet.affected_entities.length ? 4 : 0} size="xs">
                受影响关系：{changeSet.affected_relations.map(relationSummary).join('；')}
              </Text>
            ) : null}
          </Paper>
        ) : null}
        <Text c="dimmed" mt="sm" size="xs">
          变更版本 r{changeSet.revision} · 审核摘要 {changeSet.payload_hash.slice(0, 12)}…
        </Text>
        {revisionStatus === 'graph_review' ? (
          <Group mt="md">
            <Button disabled={busy} size="xs" onClick={onApproveGraph}>批准并构建候选图</Button>
            <Button disabled={busy} size="xs" variant="light" onClick={onRevise}>继续修改</Button>
          </Group>
        ) : null}
      </Paper>

      {['publish_ready', 'failed'].includes(revisionStatus) ? (
        <Paper p="md" withBorder>
          <StepTitle index="4" title={revisionStatus === 'publish_ready' ? '验证并发布' : '验证结果'} />
          <RegressionChecks value={changeSet.execution_result?.validation} />
          {revisionStatus === 'publish_ready' ? (
            <Stack gap="xs" mt="md">
              <Button color="green" disabled={busy} fullWidth onClick={onPublish}>
                发布 Graph + Profile
              </Button>
              <Button disabled={busy} fullWidth variant="light" onClick={onRevise}>
                验证结果仍不对，继续修改
              </Button>
            </Stack>
          ) : (
            <Button disabled={busy} fullWidth mt="md" variant="light" onClick={onRevise}>
              根据失败结果继续修改
            </Button>
          )}
        </Paper>
      ) : null}
    </Stack>
  )
}

function ChangeRow({
  action,
  target,
  before,
  after,
  reason,
  warning,
  sourceMessageIds,
  evidenceContext,
}: {
  action: string
  target: string
  before?: string | null
  after?: string | null
  reason?: string | null
  warning?: string | null
  sourceMessageIds?: string[]
  evidenceContext: EvidenceContext | null
}) {
  return (
    <Paper bg="dark.7" p="sm" radius="sm">
      <Group gap="xs">
        <Badge size="xs" variant="light">{action}</Badge>
        <Text fw={600} size="xs">{target}</Text>
      </Group>
      {before ? <Text c="dimmed" mt={5} size="xs">原内容：{before}</Text> : null}
      {after ? <Text mt={3} size="xs">新内容：{after}</Text> : null}
      {reason ? <Text c="dimmed" mt={3} size="xs">原因：{reason}</Text> : null}
      {sourceMessageIds?.length ? <EvidenceMessages context={evidenceContext} ids={sourceMessageIds} /> : null}
      {warning ? <Text c="red.4" mt={3} size="xs">{warning}</Text> : null}
    </Paper>
  )
}

type EvidenceContext = {
  projectId: string
  sessionId: string
  snapshotId: string
}

type EvidenceMessage = {
  message_id: string
  timestamp: string
  participant: string
  role: string
  kind: string
  content: string
  is_focus: boolean
}

function evidenceContext(
  projectId: string,
  sessionId: string | null,
  snapshotId: string | null | undefined,
): EvidenceContext | null {
  return sessionId && snapshotId ? { projectId, sessionId, snapshotId } : null
}

function EvidenceMessages({ context, ids }: { context: EvidenceContext | null, ids: string[] }) {
  const [expanded, setExpanded] = useState(false)
  const query = useQuery({
    queryKey: [
      'person-world-evidence',
      context?.projectId,
      context?.sessionId,
      context?.snapshotId,
      ids,
    ],
    queryFn: () => {
      if (!context) throw new Error('当前变更没有可读取的冻结证据快照')
      const parameters = new URLSearchParams()
      ids.forEach((id) => parameters.append('message_ids', id))
      return request<EvidenceMessage[]>(
        `/api/projects/${context.projectId}/world-agent/sessions/${context.sessionId}/context-snapshots/${context.snapshotId}/messages?${parameters.toString()}`,
      )
    },
    enabled: expanded && Boolean(context),
    staleTime: 30_000,
  })
  return (
    <Stack gap={4} mt={3}>
      <Text c="dimmed" size="xs">证据消息：{ids.join('、')}</Text>
      <Button
        disabled={!context}
        loading={query.isFetching}
        size="compact-xs"
        variant="subtle"
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? '收起证据' : '展开原消息'}
      </Button>
      {expanded && query.isError ? <Text c="red.4" size="xs">无法读取本轮冻结证据。</Text> : null}
      {expanded && query.data ? query.data.map((message) => (
        <Paper bg="dark.6" key={message.message_id} p="xs" radius="sm">
          <Text c="dimmed" size="xs">
            {message.participant} · {message.timestamp}{message.is_focus ? ' · 直接证据' : ' · 上下文'}
          </Text>
          <Text mt={2} size="xs">{message.content}</Text>
        </Paper>
      )) : null}
    </Stack>
  )
}

function RegressionChecks({ value }: { value: unknown }) {
  const checks = isRecord(value) && Array.isArray(value.checks)
    ? value.checks.filter(isRecord)
    : []
  if (!checks.length) {
    return (
      <Text c="dimmed" mt="sm" size="xs">
        候选图、重新编译的 Profile 和回归检查均已完成。发布后新 Runtime 分支才会读取它。
      </Text>
    )
  }
  return (
    <Stack gap="xs" mt="sm">
      <Text c="dimmed" size="xs">候选图回归核验</Text>
      {checks.map((check, index) => {
        const passed = check.passed === true
        return (
          <Paper bg="dark.7" key={`${String(check.query ?? '')}:${index}`} p="sm" radius="sm">
            <Group gap="xs" wrap="nowrap">
              <Badge color={passed ? 'green' : 'red'} size="xs" variant="light">
                {passed ? '通过' : '未通过'}
              </Badge>
              <Text size="xs">{String(check.query ?? '未命名查询')}</Text>
            </Group>
            {typeof check.explanation === 'string' ? (
              <Text c="dimmed" mt={3} size="xs">{check.explanation}</Text>
            ) : null}
            {typeof check.expected_change === 'string' && check.expected_change ? (
              <Text c="dimmed" mt={3} size="xs">预期：{check.expected_change}</Text>
            ) : null}
            {typeof check.actual_context_excerpt === 'string' && check.actual_context_excerpt ? (
              <Text mt={3} size="xs">实际检索摘要：{check.actual_context_excerpt}</Text>
            ) : null}
            {Array.isArray(check.source_references) && check.source_references.length ? (
              <Text c="dimmed" mt={3} size="xs">
                来源：{check.source_references.filter((item): item is string => typeof item === 'string').join('、')}
              </Text>
            ) : null}
          </Paper>
        )
      })}
      <Text c="dimmed" size="xs">发布后新 Runtime 分支才会读取这一版本。</Text>
    </Stack>
  )
}

function relationSummary(value: Record<string, string>): string {
  const source = value.source_entity ?? value.source ?? ''
  const target = value.target_entity ?? value.target ?? ''
  const relation = value.relation_description ?? value.relation ?? ''
  return [source, relation, target].filter(Boolean).join(' → ') || '关系变更'
}

function stringValues(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function StepTitle({ index, title }: { index: string; title: string }) {
  return (
    <Group gap="xs">
      <Badge circle color="moon" size="sm">{index}</Badge>
      <Title order={5}>{title}</Title>
    </Group>
  )
}

function beginRevision(
  setMessage: (value: string) => void,
  setIsRevising: (value: boolean) => void,
) {
  setMessage('需要调整的是：')
  setIsRevising(true)
}

function newRequestKey() {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `revision-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function profileSectionLabel(section: string) {
  const labels: Record<string, string> = {
    identity: '身份与性格', life_context: '现实生活', social_world: '社会关系',
    agency: '偏好与目标', practices: '实践规律', life_course: '经历轨迹',
    relationship_with_user: '与用户的关系', overview: '整体画像',
  }
  return labels[section] ?? section
}

function profileOperationLabel(operation: string) {
  const labels: Record<string, string> = {
    ADD_STATEMENT: '新增',
    REMOVE_STATEMENT: '删除',
    REPLACE_STATEMENT: '替换',
    REPLACE_V3_SECTION: '更新栏目（含关联调整）',
  }
  return labels[operation] ?? operation
}

function graphOperationLabel(operation: string) {
  const labels: Record<string, string> = {
    CREATE_ENTITY: '新增实体',
    UPDATE_ENTITY: '更新实体',
    DELETE_ENTITY: '删除实体',
    CREATE_RELATION: '新增关系',
    UPDATE_RELATION: '更新关系',
    DELETE_RELATION: '删除关系',
    MERGE_ENTITIES: '合并实体',
  }
  return labels[operation] ?? operation
}

function graphOperationTarget(operation: Record<string, unknown>) {
  const entity = optionalText(operation.entity_name)
  if (entity) return entity
  const source = optionalText(operation.source_entity)
  const target = optionalText(operation.target_entity)
  if (source || target) return `${source ?? '?'} → ${target ?? '?'}`
  const sources = Array.isArray(operation.source_entities)
    ? operation.source_entities.filter((item): item is string => typeof item === 'string')
    : []
  return `${sources.join('、')} → ${target ?? '?'}`
}

function optionalText(value: unknown) {
  return typeof value === 'string' && value ? value : null
}

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    exploring: '核对中',
    agent_queued: '等待 Agent 任务',
    agent_running: 'Agent 正在核对',
    profile_compiling: '正在生成合并修改预览',
    waiting_for_user: '等待你的回答',
    waiting_for_scope: '等待确认范围',
    understanding_ready: '等待确认理解',
    profile_review: '等待审核 Profile',
    graph_review: '等待审核 Graph',
    approved: '已批准',
    executing: '构建候选图',
    validating: '验证中',
    publish_ready: '等待发布',
    published: '已发布',
    cancelled: '已取消',
    failed: '失败',
    stale: '基线已过期',
    agent_failed: 'Agent 执行失败',
  }
  return labels[status] ?? status
}

function agentStageLabel(stage: string | null) {
  const labels: Record<string, string> = {
    context_assembling: '正在装配本轮可读取的证据上下文',
    agent_deliberating: '正在判断本轮最小核对动作',
    tool_running: '正在读取受限范围内的原始消息与图谱线索',
    turn_validating: '正在校验本轮问题或共同理解',
  }
  return stage && labels[stage] ? labels[stage] : '正在准备本轮核对'
}
