import { createBrowserRouter, Navigate } from 'react-router-dom'

import { ProjectLayout } from '../features/projects/ProjectLayout'
import { AppShell } from './AppShell'
import { RouteErrorPage } from '../components/feedback/RouteErrorPage'

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    errorElement: <RouteErrorPage />,
    children: [
      { element: <ProjectLayout />, children: [
      { path: '/', lazy: async () => ({ Component: (await import('../features/projects/ProjectListPage')).ProjectListPage }) },
      { path: '/settings', lazy: async () => ({ Component: (await import('../features/settings/SettingsPage')).SettingsPage }) },
      ...(import.meta.env.DEV ? [{ path: '/dev/ui', lazy: async () => ({ Component: (await import('../features/dev/UIExamplesPage')).UIExamplesPage }) }] : []),
      { path: '/projects/new', lazy: async () => ({ Component: (await import('../features/projects/ProjectCreatePage')).ProjectCreatePage }) },
      {
        path: '/projects/:projectId',
        children: [
          { index: true, lazy: async () => ({ Component: (await import('../features/projects/ProjectOverviewPage')).ProjectOverviewPage }) },
          { path: 'data', element: <Navigate to="../setup/import" replace /> },
          { path: 'setup/import', lazy: async () => ({ Component: (await import('../features/data/DataQualityPage')).DataQualityPage }) },
          { path: 'setup/participants', lazy: async () => ({ Component: (await import('../features/data/ParticipantsPage')).ParticipantsPage }) },
          { path: 'setup/graph', lazy: async () => ({ Component: (await import('../features/world/WorldGraphPage')).WorldGraphPage }) },
          { path: 'events', element: <Navigate to="../nodes" replace /> },
          { path: 'nodes', lazy: async () => ({ Component: (await import('../features/nodes/NodeInvestigationPage')).NodeInvestigationPage }) },
          { path: 'background', lazy: async () => ({ Component: (await import('../features/nodes/NodeBackgroundPage')).NodeBackgroundPage }) },
          { path: 'node-profiles/:profileId', lazy: async () => ({ Component: (await import('../features/nodes/NodeProfilePage')).NodeProfilePage }) },
          { path: 'events/legacy', lazy: async () => ({ Component: (await import('../features/events/LegacyEventsPage')).LegacyEventsPage }) },
          { path: 'models', lazy: async () => ({ Component: (await import('../features/models/ModelsPage')).ModelsPage }) },
          { path: 'timeline', element: <Navigate to="../nodes" replace /> },
          { path: 'advanced', element: <Navigate to="/settings" replace /> },
          { path: 'world', lazy: async () => ({ Component: (await import('../features/world/WorldProfilePage')).WorldProfilePage }) },
          { path: 'world/places', lazy: async () => ({ Component: (await import('../features/spatial/SpatialHeatmapPage')).SpatialHeatmapPage }) },
          { path: 'training/:jobId', lazy: async () => ({ Component: (await import('../features/training/TrainingProgressPage')).TrainingProgressPage }) },
          { path: 'branches', lazy: async () => ({ Component: (await import('../features/branches/BranchListPage')).BranchListPage }) },
          { path: 'branches/new', lazy: async () => ({ Component: (await import('../features/branches/BranchCreatePage')).BranchCreatePage }) },
          { path: 'branches/:branchId', lazy: async () => ({ Component: (await import('../features/branches/BranchChatPage')).BranchChatPage }) },
          {
            path: 'branches/:branchId/preparing',
            lazy: async () => ({ Component: (await import('../features/branches/BranchPreparationPage')).BranchPreparationPage }),
          },
          { path: 'evaluation', lazy: async () => ({ Component: (await import('../features/evaluation/EvaluationPage')).EvaluationPage }) },
        ],
      },
      ] },
    ],
  },
])
