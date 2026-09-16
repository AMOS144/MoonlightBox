import { userMessage } from '../../components/feedback/messages'
import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Group, Loader, Modal, Stack, Text, TextInput, Title } from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import { useNavigate, useParams } from 'react-router-dom'
import { Icon } from '../../components/Icon'
import { request } from '../../api/client'
import { V3ProfileGrid } from '../world/V3ProfileGrid'
import { PersonWorldRevisionPanel } from '../world/PersonWorldRevisionPanel'
import type { ProfileStatementSelection } from '../world/types'
import { useSessionState } from '../../hooks/useSessionState'

type Review = { profile_id: string; profile_v3: Record<string, unknown>; failed_sections: string[];
  approval_hash: string; publication_id: string | null; agent_run_id: string; source_draft_id: string;
  node_scope: { cutoff_at: string; timezone: string; preview_hash: string; investigation_id: string } }
export function NodeProfilePage() {
  const { projectId = '', profileId = '' } = useParams()
  return <NodeProfile key={`${projectId}:${profileId}`} projectId={projectId} profileId={profileId} />
}

export function NodeProfile({ projectId, profileId, onRefresh }: { projectId: string; profileId: string; onRefresh?: (profileId: string) => void }) {
  const navigate = useNavigate(), client = useQueryClient()
  const active = useRef(true)
  useEffect(() => { active.current = true; return () => { active.current = false } }, [])
  const wide = useMediaQuery('(min-width: 1181px)')
  // 宽屏默认左右并排（编译器式）：左画像右对话；用户可手动收起，窄屏由抽屉接管。
  const [openedPreference, setOpenedPreference] = useState<boolean | null>(null)
  const opened = openedPreference ?? wide ?? false
  const [locked, setLocked] = useState(false)
  const [editing, setEditing] = useState(true)
  const [branchModal, setBranchModal] = useState(false)
  const [created, setCreated] = useState<string | null>(null)
  const [selections, setSelections] = useState<ProfileStatementSelection[]>([])
  const [title, setTitle] = useState('新的时间分支')
  const root = `/api/projects/${projectId}/world-agent`
  const key = ['node-profile', projectId, profileId]
  const profile = useQuery({ queryKey: key, queryFn: () => request<Review>(`${root}/node-profiles/${profileId}`),
    refetchInterval: opened ? 3000 : false })
  const data = profile.data
  const [retryTask, setRetryTask] = useSessionState<{ job_id: string; section: string } | null>(`node-section-retry:${projectId}:${profileId}`, null)
  const retryJob = useQuery({
    queryKey: ['node-section-retry', projectId, retryTask?.job_id],
    queryFn: () => request<{ status: string; error_message?: string }>(`/api/jobs/${retryTask?.job_id}`),
    enabled: Boolean(retryTask),
    refetchInterval: q => ['queued', 'running', 'cancelling'].includes(q.state.data?.status ?? '') ? 1500 : false,
  })
  const retryBusy = Boolean(retryTask && !['failed', 'interrupted', 'cancelled'].includes(retryJob.data?.status ?? ''))
  const approve = useMutation({ mutationFn: () => request(`${root}/node-profiles/${data?.profile_id}/approve`, {
    method: 'POST', body: JSON.stringify({ approval_hash: data?.approval_hash }),
  }), onSuccess: () => { void client.invalidateQueries({ queryKey: key }) } })
  const create = useMutation({ mutationFn: () => request<{ id: string }>(`/api/projects/${projectId}/branches`, {
    method: 'POST', body: JSON.stringify({ title, publication_id: data?.publication_id,
      investigation_id: data?.node_scope.investigation_id, preview_hash: data?.node_scope.preview_hash }),
  }), onSuccess: () => { if (active.current) { setCreated(title); setBranchModal(false) } } })
  const retry = useMutation({ mutationFn: (section: string) => request<{ job_id: string; section: string }>(`${root}/runs/${data?.agent_run_id}/sections/${section}/retry`, {
    method: 'POST', body: JSON.stringify({ idempotency_key: crypto.randomUUID() }),
  }), onSuccess: value => setRetryTask(value) })
  const refreshDraft = useMutation({ mutationFn: () => request<Review>(`${root}/node-drafts/${data?.source_draft_id}/review`, { method: 'POST' }),
    onSuccess: value => {
      setRetryTask(null)
      if (active.current) {
        setSelections([])
        if (onRefresh) onRefresh(value.profile_id)
        else navigate(`/projects/${projectId}/node-profiles/${value.profile_id}`)
      }
      void client.invalidateQueries({ queryKey: ['node-profile', projectId] })
    } })
  const reviewedJob = useRef<string | null>(null)
  const refreshLatest = refreshDraft.mutate
  useEffect(() => {
    // 任务完成才生成新的审核版本；失败时保留回执和手动重读入口，不循环提交。
    if (data && retryTask && retryJob.data?.status === 'succeeded' && reviewedJob.current !== retryTask.job_id) {
      reviewedJob.current = retryTask.job_id
      refreshLatest()
    }
  }, [data, retryTask, retryJob.data?.status, refreshLatest])
  const error = profile.error || approve.error || create.error || retry.error || refreshDraft.error
  const openCorrection = () => { setEditing(true); setOpenedPreference(true) }
  return (
    <div className={`person-world-workbench${opened ? ' person-world-workbench--agent-open' : ''}`}>
      <main className="person-world-workbench__profile">
        <Stack gap="xl">
          {error && <Alert color="red">{userMessage(error)}</Alert>}
          {profile.isLoading && <Group justify="center" py={80}><Loader aria-label="正在读取人物背景" /></Group>}
          {retryJob.error && <Alert color="red">重试任务状态读取失败，进度仍保留。<Button onClick={() => void retryJob.refetch()}>重新读取任务</Button></Alert>}
          {retryTask && <Text role="status" size="sm">{retryJob.data?.status === 'succeeded' ? '栏目已完成，正在更新审核草稿。' : retryBusy ? '正在恢复所选栏目，可以离开，进度会保留。' : `栏目尚未完成。${userMessage(retryJob.data?.error_message)}`}</Text>}
          {data && <>
            <Alert color="yellow" title="待审核背景">确认并发布后才会用于此起点分支；其他分支的背景不受影响。</Alert>
            <Group justify="space-between" wrap="wrap">
              <div>
                <Title order={1}>审核人物背景</Title>
                <Text c="dimmed" mt={6}>
                  {data.node_scope.cutoff_at} · {data.node_scope.timezone} · {editing ? '点击内容勾选，在侧栏告诉 Agent 怎么改' : '此起点时刻的人物理解与生活背景'}
                </Text>
              </div>
              <Group>
                {editing && <Button variant="subtle" color="gray" leftSection={<Icon name="back" size={16} />} onClick={() => { setEditing(false); setOpenedPreference(false) }}>返回阅读</Button>}
                {!opened && <Button variant="light" onClick={openCorrection}>{selections.length ? `继续修改（已选 ${selections.length} 项）` : '指出问题或补充事实'}</Button>}
                {!data.publication_id && <Button color="green" disabled={locked || data.failed_sections.length > 0} loading={approve.isPending} onClick={() => approve.mutate()}>确认并发布这个版本</Button>}
              </Group>
            </Group>
            {data.failed_sections.length > 0 && <Alert color="orange" title="部分栏目需要恢复">
              <Group>{data.failed_sections.map(section => <Button key={section} variant="light" size="xs" disabled={locked || retryBusy || refreshDraft.isPending} loading={retry.isPending} onClick={() => retry.mutate(section)}>重试 {section}</Button>)}</Group>
              <Button mt="sm" variant="subtle" disabled={retry.isPending || (retryBusy && retryJob.data?.status !== 'succeeded')} loading={refreshDraft.isPending} onClick={() => refreshDraft.mutate()}>读取最新草稿</Button>
            </Alert>}
            {data.publication_id && <Alert color="green" title="此节点背景已发布">
              <Text size="sm">可以基于这个起点创建新的时间分支。</Text>
              <Button mt="sm" onClick={() => setBranchModal(true)}>准备分支</Button>
            </Alert>}
            {created && <Alert color="teal" title="分支已创建">分支「{created}」已创建。</Alert>}
            <V3ProfileGrid projectId={projectId} profile={data.profile_v3} editing={editing} locked={locked} selectedKeys={new Set(selections.map(s => s.key))}
              onToggle={item => setSelections(old => old.some(s => s.key === item.key) ? old.filter(s => s.key !== item.key) : [...old, item])}
              onToggleMany={(items, select) => setSelections(old => select
                ? [...old.filter(s => !items.some(i => i.key === s.key)), ...items]
                : old.filter(s => !items.some(i => i.key === s.key)))} />
          </>}
        </Stack>
      </main>
      {data && <PersonWorldRevisionPanel key={data.profile_id} projectId={projectId} baseProfileId={data.profile_id} opened={opened} selections={selections}
        onClose={() => { setOpenedPreference(false); setEditing(false); void profile.refetch() }} onLockChange={setLocked}
        onRestoreSelections={setSelections} onStartNew={() => setSelections([])} />}
      <Modal opened={branchModal} onClose={() => setBranchModal(false)} title="准备分支">
        <Stack>
          <TextInput label="分支名称" value={title} onChange={e => setTitle(e.currentTarget.value)} autoFocus />
          <Group justify="flex-end">
            <Button variant="subtle" color="gray" onClick={() => setBranchModal(false)}>取消</Button>
            <Button disabled={!title.trim()} loading={create.isPending} onClick={() => create.mutate()}>创建并准备</Button>
          </Group>
        </Stack>
      </Modal>
    </div>
  )
}
