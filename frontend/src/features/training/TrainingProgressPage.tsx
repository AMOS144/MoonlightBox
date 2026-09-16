import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { userMessage } from '../../components/feedback/messages'
import {
  Alert,
  Badge,
  Button,
  Card,
  Container,
  Group,
  List,
  Paper,
  Progress,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import { Icon } from '../../components/Icon'
import { Link, useParams } from 'react-router-dom'

import { request } from '../../api/client'
import type { TrainingJob } from './types'

const STAGE_LABELS: Record<string, string> = {
  preparing_dataset: '正在准备训练数据',
  loading_model: '正在加载基础模型',
  candidate_search: '正在筛选训练候选',
  full_training: '正在完整训练优胜候选',
  model_acceptance: '正在执行双门槛验收',
  training: '正在训练数字人',
  saving_adapter: '正在保存适配器',
  activating_model: '正在启用新模型',
  completed: '训练完成',
}

export function TrainingProgressPage() {
  const { projectId, jobId } = useParams()
  const queryClient = useQueryClient()
  const job = useQuery({
    queryKey: ['training-job', jobId],
    queryFn: () => request<TrainingJob>(`/api/jobs/${jobId}`),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'queued' || status === 'running' || status === 'cancelling' ? 1500 : false
    },
  })
  const cancel = useMutation({
    mutationFn: () =>
      request<TrainingJob>(`/api/jobs/${jobId}/cancel`, { method: 'POST' }),
    onSuccess: (result) => {
      queryClient.setQueryData(['training-job', jobId], result)
    },
  })
  const data = job.data
  const checkpoint = data?.checkpoint
  const stage = checkpoint?.stage ?? (data?.status === 'queued' ? 'queued' : '')
  const coverage = checkpoint?.dataset_coverage
  const acceptance = checkpoint?.acceptance
  const candidates = Object.entries(checkpoint?.candidate_runs ?? {})

  return (
    <Container className="training-progress" fluid p={0}>
      <Stack gap="lg">
      <div>
        <Text c="dimmed" fw={700} size="xs" tt="uppercase">数字人训练</Text>
        <Title order={1}>{STAGE_LABELS[stage] ?? '等待训练任务开始'}</Title>
      </div>
      {job.isLoading && <Text c="dimmed">正在读取训练状态……</Text>}
      {job.isError && <Alert color="red" icon={<Icon name="warning" />} title="读取失败">训练状态读取失败，请稍后重试。</Alert>}
      {data && (
        <>
          <Paper p="lg" radius="md" withBorder>
            <Group justify="space-between" mb="xs"><Text fw={600}>整体进度</Text><Text fw={700}>{Math.round(data.progress * 100)}%</Text></Group>
            <Progress value={data.progress * 100} size="lg" radius="xl" />
          </Paper>
          {coverage && (
            <SimpleGrid cols={{ base: 1, sm: 2 }}>
              <Card withBorder><Text c="dimmed" size="sm">目标消息覆盖</Text><Text fw={700} size="xl">
                  {coverage.covered_target_message_count} / {coverage.target_message_count}
                </Text></Card>
              <Card withBorder><Text c="dimmed" size="sm">数据切分</Text><Text fw={700}>
                  train {coverage.split_counts.train} · valid {coverage.split_counts.valid} ·
                  test {coverage.split_counts.test}
                </Text></Card>
            </SimpleGrid>
          )}
          {candidates.length > 0 && (
            <Paper aria-label="候选训练进度" p="lg" withBorder>
              <Title order={2} size="h3" mb="sm">候选进度</Title>
              <List spacing="xs">
                {candidates.map(([candidateId, candidate]) => (
                  <List.Item key={candidateId}>
                    <Text component="span" fw={700}>
                      {candidateId}
                      {candidateId === checkpoint?.best_candidate_id ? '（优胜）' : ''}
                    </Text>
                    {' · '}
                    {candidate.status}
                    {candidate.metrics?.validation_loss != null
                      ? ` · valid ${candidate.metrics.validation_loss}`
                      : ''}
                    {candidate.metrics?.style_score != null
                      ? ` · style ${candidate.metrics.style_score}`
                      : ''}
                    {candidate.error ? ` · ${candidate.error}` : ''}
                  </List.Item>
                ))}
              </List>
            </Paper>
          )}
          {acceptance && (
            <Paper aria-label="模型验收结果" p="lg" withBorder>
              <Title order={2} size="h3">双门槛验收</Title>
              <Text my="sm">
                事实与安全：{acceptance.semantic_passed ? '通过' : '失败'} ·
                风格：{acceptance.style_passed ? '通过' : '失败'} · sticker：
                {acceptance.sticker_passed ? '通过' : '失败'} · 背诵：
                {acceptance.memorization_passed ? '通过' : '失败'}
              </Text>
              <Group gap="xs">
                {Object.entries(acceptance.comparisons ?? {}).map(([name, value]) => (
                  <Badge color="gray" key={name} variant="light">{name} {value}</Badge>
                ))}
              </Group>
              {acceptance.failure_reasons.map((reason) => (
                <Alert color="red" icon={<Icon name="warning" />} key={reason} mt="sm">{userMessage(reason)}</Alert>
              ))}
            </Paper>
          )}
          {data.status === 'running' && (
            <SimpleGrid cols={{ base: 1, sm: 3 }}>
              <Card withBorder><Text c="dimmed" size="sm">迭代</Text><Text fw={700}>
                  {checkpoint?.iteration ?? 0} / {checkpoint?.total_iterations ?? '—'}
                </Text></Card>
              <Card withBorder><Text c="dimmed" size="sm">训练 loss</Text><Text fw={700}>{checkpoint?.train_loss ?? '—'}</Text></Card>
              <Card withBorder><Text c="dimmed" size="sm">验证 loss</Text><Text fw={700}>{checkpoint?.validation_loss ?? '—'}</Text></Card>
            </SimpleGrid>
          )}
          {(data.status === 'queued' || data.status === 'running') && (
            <Button color="red" variant="light"
              disabled={cancel.isPending}
              onClick={() => cancel.mutate()}
              type="button"
            >
              取消训练
            </Button>
          )}
          {data.status === 'cancelled' && <Alert color="gray">训练已取消</Alert>}
          {data.status === 'cancelling' && <Alert color="gray">正在取消，等待执行退出</Alert>}
          {(data.status === 'failed' || data.status === 'interrupted') && (
            <Alert color="red" icon={<Icon name="warning" />} title="训练未完成">
              {acceptance?.failure_reasons.includes(data.error_message ?? '')
                ? '请查看上方检查结果，处理后再恢复任务。'
                : userMessage(data.error_message ?? '可以恢复任务后重试。')}
            </Alert>
          )}
          {data.status === 'succeeded' && (
            <Alert color="green" icon={<Icon name="check" />} title="训练完成">
              <Text mb="sm">
                {checkpoint?.model_version_id
                  ? `新模型已原子启用：${checkpoint.model_version_id}`
                  : '训练已完成，模型状态以验收报告为准。'}
              </Text>
              <Button component={Link} size="xs" to={`/projects/${projectId}/models`}>查看模型</Button>
            </Alert>
          )}
        </>
      )}
      </Stack>
    </Container>
  )
}
