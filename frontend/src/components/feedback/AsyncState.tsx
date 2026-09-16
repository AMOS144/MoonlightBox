import { Alert, Button, Group, Skeleton } from '@mantine/core'
import { userMessage } from './messages'

/** 初次加载可用骨架；后台刷新绝不替换已有内容。 */
export function AsyncState({ loading, error, retry }: { loading?: boolean; error?: Error | null; retry?: () => void }) {
  if (error) return <Alert color="red" title="暂时无法读取" role="alert"><Group justify="space-between">
    <span>{userMessage(error)}</span>{retry && <Button variant="subtle" onClick={retry}>重新读取</Button>}
  </Group></Alert>
  return loading ? <Skeleton height={90} aria-label="正在读取" /> : null
}
