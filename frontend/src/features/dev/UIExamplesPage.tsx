import { Button, Group, Paper, Stack, Text, TextInput, Title } from '@mantine/core'
import { Icon } from '../../components/Icon'
import { StageStatus } from '../../components/feedback/StageStatus'
import { AsyncState } from '../../components/feedback/AsyncState'
import { StageHandoff } from '../../components/feedback/StageHandoff'
import { AgentConversationPanel } from '../../components/agent/AgentConversationPanel'
import type { StageState } from '../journey/journey'

/** 开发环境组件基线；不连接业务写接口，不混进正式用户主线。 */
export function UIExamplesPage() {
  const states: StageState[] = ['not_started', 'processing', 'needs_user', 'confirmed', 'completed', 'blocked', 'paused']
  return <Stack p="xl" maw={900} mx="auto"><Title order={1}>交互与主题基线</Title>
    <Paper p="md" withBorder><Stack gap="sm"><Title order={2}>洪欣羽 · 人物与对话</Title><Text>正文 14px：今天想聊些什么？中文与 English 123。</Text><Text size="xs" c="dimmed">辅助信息 12px · 起点 2026年5月8日</Text><Group>{(['home', 'people', 'timeline', 'conversation', 'settings', 'next'] as const).map(name => <Icon key={name} name={name} />)}</Group></Stack></Paper>
    <Group>{states.map(state => <StageStatus key={state} state={state} />)}</Group>
    <Paper p="lg" withBorder><Stack><TextInput label="分支名称" placeholder="一个新的开始" /><TextInput label="有错误的字段" error="请确认选择的起点" />
      <Group><Button>主操作</Button><Button variant="light">次操作</Button><Button disabled>暂不可用</Button><Button loading>正在提交</Button></Group></Stack></Paper>
    <AsyncState loading /><AsyncState error={new Error('网络暂时不可用，输入保留。')} retry={() => {}} />
    <StageHandoff title="人物背景已确认" detail="可开始起点调查。" label="返回项目列表" to="/" />
    <AgentConversationPanel title="起点调查" subject="一次共同出行" scope="仅补充起点调查，不修改已发布图谱。" status="等待你回答">
      <TextInput label="那几天是否一起见面？" /><Button>发送回答并继续</Button>
    </AgentConversationPanel>
  </Stack>
}
