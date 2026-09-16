import { useInfiniteQuery } from '@tanstack/react-query'

import { request } from '../../api/client'
import type { BranchHistoryPage } from './types'

export function adaptiveHistoryLimit(
  viewportHeight: number,
  averageMessageHeight = 72,
) {
  return Math.min(
    80,
    Math.max(20, Math.ceil((viewportHeight / averageMessageHeight) * 1.5)),
  )
}

export function useAdaptiveBranchHistory(
  projectId: string | undefined,
  branchId: string | undefined,
) {
  const limit = adaptiveHistoryLimit(
    typeof window === 'undefined' ? 800 : window.innerHeight,
  )
  const query = useInfiniteQuery({
    queryKey: ['branch-history', branchId, limit],
    queryFn: ({ pageParam }) => {
      const search = new URLSearchParams({ limit: String(limit) })
      if (pageParam) search.set('before', pageParam)
      return request<BranchHistoryPage>(
        `/api/projects/${projectId}/branches/${branchId}/history?${search}`,
      )
    },
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    enabled: Boolean(projectId && branchId),
  })
  const items =
    query.data?.pages
      .slice()
      .reverse()
      .flatMap((page) => page.items) ?? []
  return { ...query, items, limit }
}
