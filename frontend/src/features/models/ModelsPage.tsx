import { useQuery } from '@tanstack/react-query'
import { Accordion, Alert, Badge, Card, Group, Progress, SimpleGrid, Skeleton, Stack, Text, Title } from '@mantine/core'
import { useParams } from 'react-router-dom'

import { request } from '../../api/client'
import { SectionNav } from '../../components/SectionNav'
import { personaNav } from '../../components/sectionNavItems'
import { modelDisplayName, type ModelVersion } from './types'

export function ModelsPage() {
  const { projectId } = useParams()
  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => request<ModelVersion[]>(`/api/projects/${projectId}/models`),
    enabled: Boolean(projectId),
    refetchInterval: (query) =>
      query.state.data?.length ? false : 2000,
  })

  return (
    <Stack gap="xl">
      <SectionNav items={personaNav} label="数字人" />
      <Title order={1}>模型版本</Title>
      {models.isLoading ? <SimpleGrid cols={{ base: 1, md: 2 }}><Skeleton h={220} /><Skeleton h={220} /></SimpleGrid> : null}
      {models.isError ? <Alert color="red" role="alert" title="读取失败">模型状态读取失败，请重试。</Alert> : null}
      {models.isSuccess && models.data.length === 0 ? (
        <Alert color="gray" role="status" title="还没有可选训练模型">LoRA 为可选训练资产，当前聊天不依赖此项。</Alert>
      ) : null}
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        {models.data?.map((model) => (
          <Card key={model.id} padding="lg" withBorder>
            <Group justify="space-between"><Badge color={model.active ? 'green' : 'gray'} variant="light">{model.active ? '正在使用' : model.status}</Badge>{model.recommended ? <Badge color="moon">推荐</Badge> : null}</Group>
            <Title mt="lg" order={3}>{modelDisplayName(model)}</Title>
            <Stack gap="md" mt="lg">
              <div><Group justify="space-between"><Text size="sm">表达风格</Text><Text fw={700}>{Math.round((model.metrics.style_score ?? 0) * 100)}%</Text></Group><Progress mt={6} value={(model.metrics.style_score ?? 0) * 100} /></div>
              <div><Group justify="space-between"><Text size="sm">盲测胜率</Text><Text fw={700}>{Math.round((model.metrics.blind_win_rate ?? 0) * 100)}%</Text></Group><Progress color="violet" mt={6} value={(model.metrics.blind_win_rate ?? 0) * 100} /></div>
            </Stack>
            <Accordion mt="lg" variant="contained"><Accordion.Item value="technical"><Accordion.Control>技术详情</Accordion.Control><Accordion.Panel><Text size="sm">基础模型：{model.base_model}</Text><Text c="dimmed" size="xs">数据集：{model.dataset_hash.slice(0, 12)}</Text></Accordion.Panel></Accordion.Item></Accordion>
          </Card>
        ))}
      </SimpleGrid>
    </Stack>
  )
}
