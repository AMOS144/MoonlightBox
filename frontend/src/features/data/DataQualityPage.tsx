import { List, Paper, Stack, Title } from '@mantine/core'
import { useParams } from 'react-router-dom'

import { SectionNav } from '../../components/SectionNav'
import { memoryNav } from '../../components/sectionNavItems'
import { ImportWizard } from './ImportWizard'
import { MediaAnnotationPanel } from './MediaAnnotationPanel'

export function DataQualityPage() {
  const { projectId } = useParams()

  if (!projectId) {
    return <p role="alert">缺少项目标识。</p>
  }

  return (
    <Stack gap="xl">
      <SectionNav items={memoryNav} label="回忆" />
      <ImportWizard projectId={projectId} />
      <MediaAnnotationPanel projectId={projectId} />
      <Paper p="lg" withBorder>
        <Title mb="sm" order={4}>数据处理原则</Title>
        <List c="dimmed" spacing="xs">
          <List.Item>原始消息保持不变，清洗和脱敏结果单独保存。</List.Item>
          <List.Item>连续短回复先合并，不按单字规则直接删除。</List.Item>
          <List.Item>无法解析的行会单独报告，不影响其他有效消息。</List.Item>
        </List>
      </Paper>
    </Stack>
  )
}
