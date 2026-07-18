import { createBrowserRouter } from 'react-router-dom'

import { DataQualityPage } from '../features/data/DataQualityPage'
import { ProjectCreatePage } from '../features/projects/ProjectCreatePage'
import { ProjectLayout } from '../features/projects/ProjectLayout'
import { ProjectListPage } from '../features/projects/ProjectListPage'
import { AppShell } from './AppShell'

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: '/', element: <ProjectListPage /> },
      { path: '/projects/new', element: <ProjectCreatePage /> },
      {
        path: '/projects/:projectId',
        element: <ProjectLayout />,
        children: [
          {
            index: true,
            element: (
              <section>
                <p className="eyebrow">项目概览</p>
                <h1>准备进入时间线</h1>
              </section>
            ),
          },
          { path: 'data', element: <DataQualityPage /> },
        ],
      },
    ],
  },
])
