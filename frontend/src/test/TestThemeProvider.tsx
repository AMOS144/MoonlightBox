import { MantineProvider } from '@mantine/core'
import type { ReactNode } from 'react'

import { theme } from '../app/theme'

export function TestThemeProvider({ children }: { children: ReactNode }) {
  return <MantineProvider theme={theme}>{children}</MantineProvider>
}
