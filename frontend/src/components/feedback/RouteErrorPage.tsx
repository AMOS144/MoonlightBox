import { Button, Group, Stack, Text, Title } from '@mantine/core'
import { isRouteErrorResponse, useRouteError } from 'react-router-dom'

/** 页面异常不清理本地草稿，也不向普通用户展示堆栈。 */
export function RouteErrorPage() {
  const error = useRouteError()
  const missing = isRouteErrorResponse(error) && error.status === 404
  return <Stack gap="md" maw={560} mx="auto" px="lg" pt={80} role="alert">
    <Text c="dimmed" size="sm">月光宝盒</Text>
    <Title order={2}>{missing ? '这个页面不存在' : '页面暂时没能打开'}</Title>
    <Text c="dimmed" size="sm">{missing ? '地址可能不完整，请从项目列表重新进入。' : '请重新加载页面。此操作不会清除已上传的预览或本地草稿。'}</Text>
    <Group><Button onClick={() => window.location.reload()}>重新加载</Button><Button component="a" href="/" variant="subtle">所有项目</Button></Group>
  </Stack>
}
