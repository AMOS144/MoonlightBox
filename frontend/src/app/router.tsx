import { createBrowserRouter } from 'react-router-dom'

import { BranchChatPage } from '../features/branches/BranchChatPage'
import { BranchCreatePage } from '../features/branches/BranchCreatePage'
import { BranchListPage } from '../features/branches/BranchListPage'
import { DataQualityPage } from '../features/data/DataQualityPage'
import { EvaluationPage } from '../features/evaluation/EvaluationPage'
import { NodeReviewPage } from '../features/events/NodeReviewPage'
import { ModelsPage } from '../features/models/ModelsPage'
import { ProjectCreatePage } from '../features/projects/ProjectCreatePage'
import { ProjectLayout } from '../features/projects/ProjectLayout'
import { ProjectListPage } from '../features/projects/ProjectListPage'
import { TimelinePage } from '../features/timeline/TimelinePage'
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
          { path: 'events', element: <NodeReviewPage /> },
          { path: 'models', element: <ModelsPage /> },
          { path: 'timeline', element: <TimelinePage /> },
          { path: 'branches', element: <BranchListPage /> },
          { path: 'branches/new', element: <BranchCreatePage /> },
          { path: 'branches/:branchId', element: <BranchChatPage /> },
          { path: 'evaluation', element: <EvaluationPage /> },
        ],
      },
    ],
  },
])
