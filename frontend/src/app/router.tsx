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
import { ProjectOverviewPage } from '../features/projects/ProjectOverviewPage'
import { SpatialHeatmapPage } from '../features/spatial/SpatialHeatmapPage'
import { TimelinePage } from '../features/timeline/TimelinePage'
import { TrainingProgressPage } from '../features/training/TrainingProgressPage'
import { WorldProfilePage } from '../features/world/WorldProfilePage'
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
          { index: true, element: <ProjectOverviewPage /> },
          { path: 'data', element: <DataQualityPage /> },
          { path: 'events', element: <NodeReviewPage /> },
          { path: 'models', element: <ModelsPage /> },
          { path: 'timeline', element: <TimelinePage /> },
          { path: 'world', element: <WorldProfilePage /> },
          { path: 'world/places', element: <SpatialHeatmapPage /> },
          { path: 'training/:jobId', element: <TrainingProgressPage /> },
          { path: 'branches', element: <BranchListPage /> },
          { path: 'branches/new', element: <BranchCreatePage /> },
          { path: 'branches/:branchId', element: <BranchChatPage /> },
          { path: 'evaluation', element: <EvaluationPage /> },
        ],
      },
    ],
  },
])
