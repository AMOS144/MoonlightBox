import { MutationCache, QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Outlet } from 'react-router-dom'

const queryClient = new QueryClient({
  // 所有业务变更都刷新只读流程投影，页面不再各自拼不完整的依赖清单。
  mutationCache: new MutationCache({ onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: ['journey'] })
  } }),
  defaultOptions: {
    queries: { retry: false },
  },
})

export function AppShell() {
  return (
    <QueryClientProvider client={queryClient}>
      <Outlet />
    </QueryClientProvider>
  )
}
