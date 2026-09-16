import { MantineProvider } from '@mantine/core'
import type { ReactNode } from 'react'

import { theme, cssVariablesResolver } from '../app/theme'

export function TestThemeProvider({ children }: { children: ReactNode }) {
  return <MantineProvider defaultColorScheme="dark" theme={theme} cssVariablesResolver={cssVariablesResolver}>{children}</MantineProvider>
}
