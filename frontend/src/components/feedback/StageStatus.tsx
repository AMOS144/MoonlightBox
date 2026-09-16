import { Badge } from '@mantine/core'
import type { StageState } from '../../features/journey/journey'

const stagePresentation: Record<StageState, { label: string; color: string }> = {
  not_started: { label: '尚未开始', color: 'gray' }, processing: { label: '系统处理中', color: 'blue' },
  needs_user: { label: '需要你操作', color: 'orange' }, confirmed: { label: '已确认', color: 'green' },
  completed: { label: '已完成', color: 'green' }, blocked: { label: '暂时受阻', color: 'red' }, paused: { label: '已暂停', color: 'gray' },
}
export function StageStatus({ state }: { state: StageState }) {
  const value = stagePresentation[state]
  return <Badge color={value.color} variant="light">{value.label}</Badge>
}
