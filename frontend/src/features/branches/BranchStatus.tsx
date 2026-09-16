import { Badge } from '@mantine/core'

export function BranchStatus({ status }: { status: string }) {
  const value = status === 'active' ? ['green', '可以聊天'] : status === 'preparing' ? ['blue', '正在准备'] : status === 'prepare_failed' ? ['red', '准备受阻'] : ['gray', '只读记录']
  return <Badge color={value[0]} variant="light">{value[1]}</Badge>
}
