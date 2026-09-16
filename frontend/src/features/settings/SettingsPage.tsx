import { Alert, Badge, Button, Card, Divider, Group, PasswordInput, Stack, Text, TextInput, Title } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { request } from '../../api/client'
import { PageHeader } from '../../components/PageHeader'
import { userMessage } from '../../components/feedback/messages'

type AgentConfig = { model: string; endpoint: string; key_configured: boolean }
type LightRAGConfig = {
  llm_model: string
  llm_endpoint: string
  llm_key_configured: boolean
  embedding_model: string
  embedding_endpoint: string
  embedding_dimension: number
  embedding_key_configured: boolean
}
type SettingsConfig = { agent: AgentConfig; lightrag: LightRAGConfig }
type EditableConfig = SettingsConfig & {
  agent_api_key: string
  llm_api_key: string
  embedding_api_key: string
}
type ServiceName = 'agent' | 'lightrag' | 'embedding'

const settingsKey = ['settings'] as const

function ServiceHeader({ title, description, configured }: { title: string; description: string; configured: boolean }) {
  return <Group justify="space-between" align="flex-start" wrap="nowrap">
    <div><Title order={3} size="h5">{title}</Title><Text size="sm" c="dimmed" mt={4}>{description}</Text></div>
    <Badge variant="light" color={configured ? 'green' : 'gray'}>{configured ? '已配置' : '未配置'}</Badge>
  </Group>
}

function SettingsForm({ config }: { config: SettingsConfig }) {
  const client = useQueryClient()
  const [value, setValue] = useState<EditableConfig>({ ...config, agent_api_key: '', llm_api_key: '', embedding_api_key: '' })
  const test = useMutation({
    mutationFn: (service: ServiceName) => {
      const payload = service === 'agent'
        ? { service, model: value.agent.model, endpoint: value.agent.endpoint, api_key: value.agent_api_key || null }
        : service === 'lightrag'
          ? { service, model: value.lightrag.llm_model, endpoint: value.lightrag.llm_endpoint, api_key: value.llm_api_key || null }
          : { service, model: value.lightrag.embedding_model, endpoint: value.lightrag.embedding_endpoint, api_key: value.embedding_api_key || null, embedding_dimension: value.lightrag.embedding_dimension }
      return request<{ message: string }>('/api/settings/test', { method: 'POST', body: JSON.stringify(payload) })
    },
    onSuccess: result => notifications.show({ color: 'green', message: result.message }),
  })
  const save = useMutation({
    mutationFn: () => request<SettingsConfig>('/api/settings', {
      method: 'PUT',
      body: JSON.stringify({
        agent: { model: value.agent.model.trim(), endpoint: value.agent.endpoint.trim(), api_key: value.agent_api_key.trim() || null },
        lightrag: {
          llm_model: value.lightrag.llm_model.trim(), llm_endpoint: value.lightrag.llm_endpoint.trim(), llm_api_key: value.llm_api_key.trim() || null,
          embedding_model: value.lightrag.embedding_model.trim(), embedding_endpoint: value.lightrag.embedding_endpoint.trim(),
          embedding_dimension: value.lightrag.embedding_dimension, embedding_api_key: value.embedding_api_key.trim() || null,
        },
      }),
    }),
    onSuccess: result => {
      client.setQueryData(settingsKey, result)
      setValue({ ...result, agent_api_key: '', llm_api_key: '', embedding_api_key: '' })
      notifications.show({ color: 'green', message: '已全部保存' })
    },
  })
  const testButton = (service: ServiceName) => <Button type="button" variant="light" loading={test.isPending && test.variables === service} disabled={test.isPending || save.isPending} onClick={() => test.mutate(service)}>测试连接</Button>
  return <form onSubmit={event => { event.preventDefault(); if (!save.isPending) save.mutate() }}>
    <Card withBorder radius="lg" padding="xl"><Stack gap="xl">
      <Stack gap="md">
        <ServiceHeader title="Agent 模型" description="用于人物调查、日程规划和对话。" configured={value.agent.key_configured} />
        <TextInput label="模型名称" value={value.agent.model} onChange={event => setValue({ ...value, agent: { ...value.agent, model: event.currentTarget.value } })} required />
        <TextInput label="API 地址" description="OpenAI 兼容接口的完整 Chat Completions 地址" value={value.agent.endpoint} onChange={event => setValue({ ...value, agent: { ...value.agent, endpoint: event.currentTarget.value } })} required type="url" />
        <PasswordInput label="API Key" description={value.agent.key_configured ? '留空保留已保存的 Key' : '填写服务商提供的 Key'} value={value.agent_api_key} onChange={event => setValue({ ...value, agent_api_key: event.currentTarget.value })} autoComplete="new-password" />
        <Group justify="flex-end">{testButton('agent')}</Group>
      </Stack>
      <Divider />
      <Stack gap="md">
        <ServiceHeader title="LightRAG LLM" description="负责图谱中的实体与关系抽取。" configured={value.lightrag.llm_key_configured} />
        <TextInput label="模型名称" value={value.lightrag.llm_model} onChange={event => setValue({ ...value, lightrag: { ...value.lightrag, llm_model: event.currentTarget.value } })} required />
        <TextInput label="服务地址" description="OpenAI 兼容 API 的基础地址" value={value.lightrag.llm_endpoint} onChange={event => setValue({ ...value, lightrag: { ...value.lightrag, llm_endpoint: event.currentTarget.value } })} required type="url" />
        <PasswordInput label="API Key" description={value.lightrag.llm_key_configured ? '留空保留已保存的 Key' : '填写服务商提供的 Key'} value={value.llm_api_key} onChange={event => setValue({ ...value, llm_api_key: event.currentTarget.value })} autoComplete="new-password" />
        <Group justify="flex-end">{testButton('lightrag')}</Group>
      </Stack>
      <Divider />
      <Stack gap="md">
        <ServiceHeader title="Embedding" description="负责聊天文本的向量化与图谱检索。" configured={value.lightrag.embedding_key_configured} />
        <TextInput label="模型名称" value={value.lightrag.embedding_model} onChange={event => setValue({ ...value, lightrag: { ...value.lightrag, embedding_model: event.currentTarget.value } })} required />
        <TextInput label="服务地址" description="OpenAI 兼容 Embeddings API 的基础地址" value={value.lightrag.embedding_endpoint} onChange={event => setValue({ ...value, lightrag: { ...value.lightrag, embedding_endpoint: event.currentTarget.value } })} required type="url" />
        <TextInput label="向量维度" type="number" min={1} max={16384} value={value.lightrag.embedding_dimension} onChange={event => setValue({ ...value, lightrag: { ...value.lightrag, embedding_dimension: Number(event.currentTarget.value) } })} required />
        <PasswordInput label="API Key" description={value.lightrag.embedding_key_configured ? '留空保留已保存的 Key' : '填写服务商提供的 Key'} value={value.embedding_api_key} onChange={event => setValue({ ...value, embedding_api_key: event.currentTarget.value })} autoComplete="new-password" />
        <Group justify="flex-end">{testButton('embedding')}</Group>
      </Stack>
      {(test.error || save.error) && <Alert color="red">{userMessage(test.error || save.error)}</Alert>}
      <Divider />
      <Group justify="flex-end"><Button type="submit" loading={save.isPending} disabled={test.isPending}>全部保存</Button></Group>
    </Stack></Card>
  </form>
}

export function SettingsPage() {
  const query = useQuery({ queryKey: settingsKey, queryFn: () => request<SettingsConfig>('/api/settings'), staleTime: 0 })
  return <Stack gap="lg" maw={820} w="100%" mx="auto" px={{ base: 'md', sm: 'lg' }} py={{ base: 'md', sm: 'xl' }}>
    <PageHeader title="设置" />
    <Text size="sm" c="dimmed">全局模型服务配置，适用于所有项目。</Text>
    {query.isPending ? <Card withBorder padding="xl"><Text size="sm">正在读取配置…</Text></Card>
      : query.isError ? <Alert color="red">{userMessage(query.error)}<Button variant="subtle" size="xs" onClick={() => void query.refetch()}>重新读取</Button></Alert>
        : <SettingsForm config={query.data} />}
  </Stack>
}
