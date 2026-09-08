export type HeatmapMetric = 'activity' | 'visits' | 'dwell'
export type HeatmapPrivacy = 'exact' | 'blurred' | 'hidden'

export type PlaceHeatProperties = {
  place_id: string
  display_name: string
  primary_role: string | null
  role_distribution: Record<string, number>
  heat_raw: number
  heat_display: number
  activity_weight: number
  effective_visit_count: number
  effective_dwell_hours: number
  dwell_coverage: number
  evidence_strength: number
  first_visit_at: string | null
  last_visit_at: string | null
  geo_resolution: string
  uncertainty_radius_m: number | null
  privacy_level: HeatmapPrivacy
  coordinate_crs: string | null
}

export type PlaceHeatFeature = {
  type: 'Feature'
  geometry: { type: 'Point'; coordinates: [number, number] }
  properties: PlaceHeatProperties
}

export type PlaceHeatmap = {
  type: 'FeatureCollection'
  properties: {
    person_id: string
    person_name: string
    snapshot_id: string | null
    graph_version?: string
    created_at?: string
    metric: HeatmapMetric
    located_mass: number
    unlocated_mass: number
    normalization: 'relative' | 'fixed'
    reliability?: number
    warnings?: string[]
    data_warning: string | null
    filters?: {
      from: string | null
      to: string | null
      weekdays: number[] | null
      hour_start: number | null
      hour_end: number | null
    }
  }
  features: PlaceHeatFeature[]
  unlocated_places: PlaceHeatProperties[]
}
