import { MantineProvider } from '@mantine/core'
import { Notifications } from '@mantine/notifications'
import { RouterProvider } from 'react-router-dom'

import { router } from './app/router'
import { theme, cssVariablesResolver } from './app/theme'

function App() {
  return (
    <MantineProvider defaultColorScheme="dark" theme={theme} cssVariablesResolver={cssVariablesResolver}>
      <Notifications position="top-right" />
      <RouterProvider router={router} />
    </MantineProvider>
  )
}

export default App
