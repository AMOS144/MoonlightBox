import { Accordion, Alert, List, Stack } from '@mantine/core'
import { useParams } from 'react-router-dom'

import { useQueryClient } from '@tanstack/react-query'
import { journeyKey, useJourney } from '../journey/journey'
import { ImportWizard } from './ImportWizard'
import { MediaAnnotationPanel } from './MediaAnnotationPanel'

export function DataQualityPage() {
  const { projectId } = useParams()
  const client = useQueryClient()
  const journey = useJourney(projectId ?? '')

  if (!projectId) {
    return <Alert color="red">缺少项目标识。</Alert>
  }

  return (
    <Stack className="setup-page" gap="xl">
      <ImportWizard projectId={projectId} savedImports={journey.data?.imports} onSaved={() => void client.invalidateQueries({ queryKey: journeyKey(projectId) })} />
      <Accordion variant="separated">
        <Accordion.Item value="media">
          <Accordion.Control>可选：媒体整理</Accordion.Control>
          <Accordion.Panel><MediaAnnotationPanel projectId={projectId} /></Accordion.Panel>
        </Accordion.Item>
        <Accordion.Item value="notes">
          <Accordion.Control>数据处理说明</Accordion.Control>
          <Accordion.Panel>
            <List c="dimmed" spacing="xs">
              <List.Item>原始消息保持不变，清洗和脱敏结果单独保存。</List.Item>
              <List.Item>连续短回复先合并，不按单字规则直接删除。</List.Item>
              <List.Item>无法解析的行会单独报告，不影响其他有效消息。</List.Item>
            </List>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Stack>
  )
}
