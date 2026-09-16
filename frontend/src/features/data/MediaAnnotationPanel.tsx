import { userMessage } from '../../components/feedback/messages'
import {
  Alert,
  Badge,
  Button,
  Card,
  Group,
  Image,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { ApiError, request } from '../../api/client'

type MediaAnnotation = {
  id: string
  asset_id: string
  modality: 'image' | 'audio' | 'video' | 'sticker'
  status: 'pending' | 'succeeded' | 'failed' | 'needs_review'
  summary: string
  transcript: string
  ocr_text: string
  safety_tags: string[]
  source_model: string
  source_version: string
  confidence: number
  reusable: boolean
  reuse_decision: 'pending' | 'approved' | 'blocked'
  failure_code: string | null
}

type MediaAnnotationSummary = {
  eligible: number
  annotated: number
  succeeded: number
  failed: number
  needs_review: number
  approved: number
  blocked: number
}

const statusLabels: Record<MediaAnnotation['status'], string> = {
  pending: '待理解',
  succeeded: '已理解',
  failed: '处理失败',
  needs_review: '需复核',
}

export function MediaAnnotationPanel({ projectId }: { projectId: string }) {
  const [items, setItems] = useState<MediaAnnotation[]>([])
  const [summary, setSummary] = useState<MediaAnnotationSummary | null>(null)
  const [error, setError] = useState('')
  const [updating, setUpdating] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const [annotations, annotationSummary] = await Promise.all([
        request<MediaAnnotation[]>(
          `/api/projects/${projectId}/media/annotations/list`,
        ),
        request<MediaAnnotationSummary>(
          `/api/projects/${projectId}/media/annotations/summary`,
        ),
      ])
      setItems(annotations)
      setSummary(annotationSummary)
      setError('')
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : '媒体语义状态加载失败')
    }
  }, [projectId])

  useEffect(() => {
    void load()
  }, [load])

  const counts = useMemo(
    () => ({
      understood: items.filter((item) => item.status === 'succeeded').length,
      approved: items.filter((item) => item.reusable).length,
      review: items.filter((item) => item.reuse_decision === 'pending').length,
    }),
    [items],
  )

  async function decide(item: MediaAnnotation, decision: 'approved' | 'blocked') {
    setUpdating(item.id)
    try {
      const updated = await request<MediaAnnotation>(
        `/api/projects/${projectId}/media/${item.asset_id}/annotation`,
        {
          method: 'PUT',
          body: JSON.stringify({
            summary: item.summary,
            transcript: item.transcript,
            ocr_text: item.ocr_text,
            safety_tags: item.safety_tags,
            confidence: item.confidence,
            status: item.status,
            reuse_decision: decision,
            failure_code: item.failure_code,
          }),
        },
      )
      setItems((current) => current.map((value) => (value.id === item.id ? updated : value)))
      void load()
      setError('')
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : '媒体复用决定保存失败')
    } finally {
      setUpdating(null)
    }
  }

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <div>
          <Title order={3}>图片与语音语义审核</Title>
          <Text c="dimmed" size="sm">
            本地模型只负责理解内容；只有你明确批准、且聊天时语义匹配的历史媒体才可能发送。
          </Text>
        </div>
        <Group gap="xs">
          <Badge variant="light">已理解 {counts.understood}</Badge>
          <Badge color="gray" variant="light">
            总进度 {summary?.annotated ?? 0}/{summary?.eligible ?? 0}
          </Badge>
          <Badge color="yellow" variant="light">待决定 {counts.review}</Badge>
          <Badge color="green" variant="light">可复用 {counts.approved}</Badge>
        </Group>
      </Group>
      {error ? <Alert color="red">{userMessage(error)}</Alert> : null}
      {items.length === 0 ? (
        <Alert color="gray">尚无语义标注。先运行本地媒体理解任务，系统不会直接猜测图片或录音内容。</Alert>
      ) : (
        <SimpleGrid cols={{ base: 1, md: 2 }}>
          {items.map((item) => {
            const canApprove =
              item.status === 'succeeded' &&
              item.confidence >= 0.75 &&
              item.safety_tags.length === 0 &&
              Boolean(item.modality === 'audio' ? item.transcript : item.summary)
            return (
              <Card key={item.id} padding="md" withBorder>
                <Stack gap="sm">
                  <Group justify="space-between">
                    <Group gap="xs">
                      <Badge>{item.modality}</Badge>
                      <Badge color={item.status === 'succeeded' ? 'green' : 'yellow'}>
                        {statusLabels[item.status]}
                      </Badge>
                    </Group>
                    <Text c="dimmed" size="xs">
                      置信度 {Math.round(item.confidence * 100)}%
                    </Text>
                  </Group>
                  {item.modality === 'image' ? (
                    <Image
                      alt={item.summary || '待审核图片'}
                      fit="contain"
                      h={180}
                      radius="sm"
                      src={`/api/projects/${projectId}/media/${item.asset_id}`}
                    />
                  ) : null}
                  {item.modality === 'audio' ? (
                    <audio
                      controls
                      preload="none"
                      src={`/api/projects/${projectId}/media/${item.asset_id}`}
                    />
                  ) : null}
                  <Text size="sm">{item.transcript || item.summary || '没有可用语义内容'}</Text>
                  {item.ocr_text ? <Text c="dimmed" size="xs">可见文字：{item.ocr_text}</Text> : null}
                  {item.safety_tags.length > 0 ? (
                    <Alert color="red" p="xs">敏感标记：{item.safety_tags.join('、')}</Alert>
                  ) : null}
                  <Group justify="flex-end">
                    <Button
                      color="gray"
                      loading={updating === item.id}
                      onClick={() => void decide(item, 'blocked')}
                      size="xs"
                      variant="subtle"
                    >
                      禁止复用
                    </Button>
                    <Button
                      disabled={!canApprove}
                      loading={updating === item.id}
                      onClick={() => void decide(item, 'approved')}
                      size="xs"
                    >
                      批准复用
                    </Button>
                  </Group>
                </Stack>
              </Card>
            )
          })}
        </SimpleGrid>
      )}
    </Stack>
  )
}
