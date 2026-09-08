import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'

import { TestThemeProvider } from '../../test/TestThemeProvider'
import { SpatialHeatmapPage } from './SpatialHeatmapPage'

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

test('没有地图 key 时仍展示地点权重与定位覆盖', async () => {
  vi.stubEnv('VITE_AMAP_JS_KEY', '')
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
    type: 'FeatureCollection',
    properties: {
      person_id: 'person-1',
      person_name: '妈妈',
      snapshot_id: 'snapshot-1',
      metric: 'activity',
      located_mass: 0.7,
      unlocated_mass: 0.3,
      normalization: 'relative',
      reliability: 0.81,
      data_warning: null,
    },
    features: [{
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [121.45, 31.2] },
      properties: {
        place_id: 'work',
        display_name: '公司',
        primary_role: 'workplace',
        role_distribution: { workplace: 1 },
        heat_raw: 0.7,
        heat_display: 1,
        activity_weight: 0.7,
        effective_visit_count: 12.7,
        effective_dwell_hours: 30,
        dwell_coverage: 0.6,
        evidence_strength: 0.91,
        first_visit_at: '2026-01-01T00:00:00Z',
        last_visit_at: '2026-08-01T00:00:00Z',
        geo_resolution: 'poi',
        uncertainty_radius_m: 50,
        privacy_level: 'blurred',
        coordinate_crs: 'GCJ-02',
      },
    }],
    unlocated_places: [{
      place_id: 'home',
      display_name: '家',
      primary_role: 'home',
      role_distribution: { home: 1 },
      heat_raw: 0.3,
      heat_display: 0.65,
      activity_weight: 0.3,
      effective_visit_count: 4,
      effective_dwell_hours: 0,
      dwell_coverage: 0,
      evidence_strength: 0.7,
      first_visit_at: null,
      last_visit_at: null,
      geo_resolution: 'semantic_only',
      uncertainty_radius_m: null,
      privacy_level: 'blurred',
      coordinate_crs: null,
    }],
  }), { status: 200 })))
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <TestThemeProvider>
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/projects/p1/world']}>
          <Routes>
            <Route path="/projects/:projectId/world" element={<SpatialHeatmapPage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    </TestThemeProvider>,
  )

  expect(await screen.findByRole('heading', { name: '妈妈的活动地图' })).toBeInTheDocument()
  expect(screen.getAllByText('70%')).toHaveLength(2)
  expect(screen.getByText('公司')).toBeInTheDocument()
  expect(screen.getByText('家')).toBeInTheDocument()
  expect(screen.getByText(/配置 VITE_AMAP_JS_KEY/)).toBeInTheDocument()
})
