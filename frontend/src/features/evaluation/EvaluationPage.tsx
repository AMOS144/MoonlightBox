import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  Button,
  Card,
  Group,
  Progress,
  SimpleGrid,
  Skeleton,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'

import { request } from '../../api/client'
import { SectionNav } from '../../components/SectionNav'
import { personaNav } from '../../components/sectionNavItems'
import type { ModelVersion } from '../models/types'

type BlindStudy = {
  id: string
  project_id: string
  model_version_id: string
  status: string
  minimum_ratings: number
  valid_rating_count: number
  candidate_preference_rate: number
  report: Record<string, unknown>
}

type BlindCase = {
  id: string
  context: string[]
  options: { a: string; b: string }
}

function persistentRaterKey(): string {
  const storageKey = 'moonlightbox-human-blind-rater'
  const existing = window.localStorage.getItem(storageKey)
  if (existing) return existing
  const created = window.crypto.randomUUID()
  window.localStorage.setItem(storageKey, created)
  return created
}

export function EvaluationPage() {
  const { projectId } = useParams()
  const queryClient = useQueryClient()
  const [caseIndex, setCaseIndex] = useState(0)
  const [raterKey] = useState(persistentRaterKey)
  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => request<ModelVersion[]>(`/api/projects/${projectId}/models`),
    enabled: Boolean(projectId),
  })
  const studies = useQuery({
    queryKey: ['human-blind-studies', projectId],
    queryFn: () =>
      request<BlindStudy[]>(`/api/evaluation/projects/${projectId}/blind-studies`),
    enabled: Boolean(projectId),
  })
  const activeStudy = studies.data?.find((study) => study.status === 'open')
  const cases = useQuery({
    queryKey: ['human-blind-cases', activeStudy?.id],
    queryFn: () =>
      request<BlindCase[]>(`/api/evaluation/blind-studies/${activeStudy?.id}/cases`),
    enabled: Boolean(activeStudy),
  })
  useEffect(() => setCaseIndex(0), [activeStudy?.id])

  const finalize = useMutation({
    mutationFn: (study: BlindStudy) =>
      request<BlindStudy>(
        `/api/evaluation/projects/${projectId}/models/${study.model_version_id}/blind-studies/${study.id}/finalize`,
        { method: 'POST' },
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['human-blind-studies', projectId] })
      queryClient.invalidateQueries({ queryKey: ['models', projectId] })
    },
  })
  const rate = useMutation({
    mutationFn: ({ caseId, choice }: { caseId: string; choice: 'a' | 'b' | 'tie' }) =>
      request<{ accepted: boolean }>(
        `/api/evaluation/blind-studies/${activeStudy?.id}/ratings`,
        {
          method: 'POST',
          body: JSON.stringify({ case_id: caseId, rater_key: raterKey, choice }),
        },
      ),
    onSuccess: () => {
      const next = caseIndex + 1
      if (activeStudy && cases.data && next >= cases.data.length) {
        finalize.mutate(activeStudy)
      } else {
        setCaseIndex(next)
      }
    },
  })
  const currentCase = cases.data?.[caseIndex]
  const completedStudy = studies.data?.find((study) => study.status !== 'open')

  return (
    <Stack gap="xl">
      <SectionNav items={personaNav} label="数字人" />
      <div>
        <Text c="moon.4" fw={700} size="xs">真人盲测</Text>
        <Title mt={5} order={1}>哪一句更像真实的她</Title>
        <Text c="dimmed" mt={7}>选项顺序已随机隐藏。新版本只有完成至少 20 次真人判断后，才可能接管聊天。</Text>
      </div>
      {studies.isLoading || cases.isLoading ? <Skeleton h={260} /> : null}
      {studies.isError || cases.isError ? <Alert color="red">盲测读取失败，请刷新重试。</Alert> : null}
      {activeStudy && currentCase ? (
        <Card padding="xl" withBorder>
          <Group justify="space-between">
            <Text fw={700}>第 {caseIndex + 1} / {cases.data?.length ?? 0} 题</Text>
            <Text c="dimmed" size="sm">不要猜哪边是模型，只按自然程度选择</Text>
          </Group>
          <Progress mt="md" value={(caseIndex / (cases.data?.length ?? 1)) * 100} />
          <Stack gap="xs" my="xl">
            {currentCase.context.map((line, index) => <Text key={`${index}-${line}`} size="sm">{line}</Text>)}
          </Stack>
          <SimpleGrid cols={{ base: 1, md: 2 }}>
            {(['a', 'b'] as const).map((choice) => (
              <Button
                h="auto"
                key={choice}
                loading={rate.isPending}
                onClick={() => rate.mutate({ caseId: currentCase.id, choice })}
                p="lg"
                variant="light"
              >
                <Text style={{ whiteSpace: 'pre-wrap' }}>{currentCase.options[choice]}</Text>
              </Button>
            ))}
          </SimpleGrid>
          <Button
            color="gray"
            loading={rate.isPending}
            mt="md"
            onClick={() => rate.mutate({ caseId: currentCase.id, choice: 'tie' })}
            variant="subtle"
          >
            两句一样自然
          </Button>
        </Card>
      ) : null}
      {finalize.isPending ? <Alert>正在汇总盲测并决定是否启用新版本……</Alert> : null}
      {!activeStudy && completedStudy ? (
        <Alert color={completedStudy.status === 'passed' ? 'green' : 'orange'}>
          最近一次真人盲测：{completedStudy.status === 'passed' ? '已通过' : '未通过'}，候选偏好率 {Math.round(completedStudy.candidate_preference_rate * 100)}%。
        </Alert>
      ) : null}
      {!activeStudy && !completedStudy && studies.isSuccess ? (
        <Alert color="gray">当前没有待评审的新版本。训练完成后，盲测题会自动出现在这里。</Alert>
      ) : null}
      {rate.isError || finalize.isError ? <Alert color="red">提交失败，本题尚未计入，请重试。</Alert> : null}
      <Title order={2}>版本质量记录</Title>
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        {models.data?.map((model) => {
          const preference = model.metrics.human_blind_preference_rate ?? model.metrics.blind_win_rate ?? 0
          return (
            <Card key={model.id} padding="lg" withBorder>
              <Title order={4}>{model.base_model}</Title>
              <Text mt="lg" size="sm">真人偏好率</Text>
              <Progress mt={6} value={preference * 100} />
              <Text c="dimmed" mt="xs" size="xs">{model.status}</Text>
            </Card>
          )
        })}
      </SimpleGrid>
    </Stack>
  )
}
