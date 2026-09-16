import { Alert, SimpleGrid, Stack, Text } from '@mantine/core'
import { useParams } from 'react-router-dom'

import { useJourney } from '../journey/journey'
import { BranchCard } from './BranchCard'
import { PageHeader } from '../../components/PageHeader'
import { AsyncState } from '../../components/feedback/AsyncState'

export function BranchListPage() {
  const { projectId } = useParams()
  const branches = useJourney(projectId ?? '')

  return (
    <Stack gap="xl">
      <PageHeader title="分支" />
      <AsyncState loading={branches.isLoading} error={branches.error} retry={() => void branches.refetch()} />
      <SimpleGrid cols={1} spacing="xs">
        {branches.data?.branches.map((branch) => (
          <BranchCard key={branch.id} branch={branch} to={branch.id} />
        ))}
      </SimpleGrid>
      {branches.isSuccess && branches.data.branches.length === 0 ? (
        <Alert color="gray" title="暂无分支"><Text>还没有创建任何分支。</Text></Alert>
      ) : null}
    </Stack>
  )
}
