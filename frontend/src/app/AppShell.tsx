import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Outlet } from 'react-router-dom'

const queryClient = new QueryClient({
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
