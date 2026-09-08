import { useMutation, useQuery } from '@tanstack/react-query'
import {
  Alert,
  Badge,
  Button,
  Card,
  Drawer,
  Group,
  Loader,
  Paper,
  Progress,
  SegmentedControl,
  Stack,
  Switch,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useEffect, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'

import { request } from '../../api/client'
import { SectionNav } from '../../components/SectionNav'
import { worldNav } from '../../components/sectionNavItems'
import type {
  HeatmapMetric,
  HeatmapPrivacy,
  PlaceHeatmap,
  PlaceHeatProperties,
} from './types'

type AmapRuntime = {
  Map: new (node: HTMLElement, options: Record<string, unknown>) => AmapMap
  Marker: new (options: Record<string, unknown>) => AmapMarker
  HeatMap?: new (map: AmapMap, options: Record<string, unknown>) => AmapHeatMap
  plugin(names: string[], callback: () => void): void
}

type AmapMap = {
  add(value: unknown): void
  destroy(): void
  on(event: string, handler: () => void): void
  setFitView(values?: unknown[], immediately?: boolean, avoid?: number[]): void
}

type AmapMarker = {
  on(event: string, handler: () => void): void
}

type AmapHeatMap = {
  setDataSet(value: {
    data: Array<{ lng: number; lat: number; count: number }>
    max: number
  }): void
  setMap(map: AmapMap | null): void
}

declare global {
  interface Window {
    AMap?: AmapRuntime
    _AMapSecurityConfig?: { securityJsCode: string }
  }
}

const metricOptions = [
  { label: '活动范围', value: 'activity' },
  { label: '到访次数', value: 'visits' },
  { label: '已知停留', value: 'dwell' },
]

export function SpatialHeatmapPage() {
  const { projectId } = useParams()
  const [metric, setMetric] = useState<HeatmapMetric>('activity')
  const [privacy, setPrivacy] = useState<HeatmapPrivacy>('blurred')
  const [showMarkers, setShowMarkers] = useState(true)
  const [dayFilter, setDayFilter] = useState<'all' | 'workdays' | 'weekend'>('all')
  const [hourFilter, setHourFilter] = useState<'all' | 'daytime' | 'evening'>('all')
  const [fromDate, setFromDate] = useState('')
  const [toDate, setToDate] = useState('')
  const [selected, setSelected] = useState<PlaceHeatProperties | null>(null)
  const [analysisRequested, setAnalysisRequested] = useState(false)
  const heatmap = useQuery({
    queryKey: [
      'place-heatmap', projectId, metric, privacy, dayFilter, hourFilter, fromDate, toDate,
    ],
    queryFn: () => request<PlaceHeatmap>(heatmapUrl({
      projectId: projectId ?? '', metric, privacy, dayFilter, hourFilter, fromDate, toDate,
    })),
    enabled: Boolean(projectId),
    refetchInterval: analysisRequested ? 3000 : false,
  })
  const analyze = useMutation({
    mutationFn: () => request<{ job_ids: string[] }>(
      `/api/projects/${projectId}/spatial-analysis`,
      { method: 'POST' },
    ),
    onSuccess: () => {
      setAnalysisRequested(true)
      void heatmap.refetch()
    },
  })
  useEffect(() => {
    if (heatmap.data?.properties.snapshot_id) setAnalysisRequested(false)
  }, [heatmap.data?.properties.snapshot_id])

  if (heatmap.isLoading) {
    return <Group justify="center" py={80}><Loader aria-label="正在生成活动地图" /></Group>
  }
  if (heatmap.isError || !heatmap.data) {
    return <Alert color="red" title="活动地图读取失败">请稍后重试。</Alert>
  }
  const data = heatmap.data
  const totalMass = data.properties.located_mass + data.properties.unlocated_mass
  const locatedRatio = totalMass > 0 ? data.properties.located_mass / totalMass : 0

  return (
    <Stack className="spatial-page" gap="lg">
      <SectionNav items={worldNav} label="世界" />
      <Group align="flex-end" justify="space-between" wrap="wrap">
        <div>
          <Text c="moon.4" fw={700} size="xs">PERSONAL PLACE GRAPH</Text>
          <Title mt={6} order={1}>{data.properties.person_name}的活动地图</Title>
          <Text c="dimmed" mt={8}>由聊天证据聚合的长期活动范围，不代表实时位置。</Text>
        </div>
        <Badge color={data.properties.reliability && data.properties.reliability >= 0.7 ? 'green' : 'yellow'} size="lg" variant="light">
          图可靠度 {formatPercent(data.properties.reliability ?? 0)}
        </Badge>
      </Group>

      <Paper className="spatial-controls" p="md" withBorder>
        <Stack gap="sm">
        <Group justify="space-between" wrap="wrap">
          <SegmentedControl
            data={metricOptions}
            onChange={(value) => setMetric(value as HeatmapMetric)}
            value={metric}
          />
          <Group gap="lg">
            <Switch checked onChange={() => undefined} label="热力层" readOnly />
            <Switch checked={showMarkers} label="地点标记" onChange={(event) => setShowMarkers(event.currentTarget.checked)} />
            <Switch disabled label="转移边" />
            <SegmentedControl
              data={[
                { label: '模糊', value: 'blurred' },
                { label: '精确', value: 'exact' },
                { label: '隐藏', value: 'hidden' },
              ]}
              onChange={(value) => setPrivacy(value as HeatmapPrivacy)}
              size="xs"
              value={privacy}
            />
          </Group>
        </Group>
        <Group gap="sm" wrap="wrap">
          <SegmentedControl
            data={[
              { label: '全部星期', value: 'all' },
              { label: '工作日', value: 'workdays' },
              { label: '周末', value: 'weekend' },
            ]}
            onChange={(value) => setDayFilter(value as typeof dayFilter)}
            size="xs"
            value={dayFilter}
          />
          <SegmentedControl
            data={[
              { label: '全天', value: 'all' },
              { label: '白天 8–18', value: 'daytime' },
              { label: '晚间 18–24', value: 'evening' },
            ]}
            onChange={(value) => setHourFilter(value as typeof hourFilter)}
            size="xs"
            value={hourFilter}
          />
          <TextInput
            aria-label="开始日期"
            onChange={(event) => setFromDate(event.currentTarget.value)}
            size="xs"
            type="date"
            value={fromDate}
          />
          <TextInput
            aria-label="结束日期"
            onChange={(event) => setToDate(event.currentTarget.value)}
            size="xs"
            type="date"
            value={toDate}
          />
        </Group>
        </Stack>
      </Paper>

      {data.properties.data_warning ? (
        <Alert color="yellow" title="数据提示">
          <Group justify="space-between" wrap="wrap">
            <Text size="sm">{data.properties.data_warning}</Text>
            {data.properties.snapshot_id === null ? (
              <Button
                loading={analyze.isPending}
                onClick={() => analyze.mutate()}
                size="xs"
                variant="light"
              >
                {analysisRequested ? '正在分析…' : '生成活动地图'}
              </Button>
            ) : null}
          </Group>
          {analyze.isError ? (
            <Text c="red" mt="xs" size="xs">
              无法入队，请先在后端启用空间分析并确认项目已有聊天导入。
            </Text>
          ) : null}
        </Alert>
      ) : null}

      <div className="spatial-workspace">
        <Paper className="spatial-map-shell" withBorder>
          <AmapHeatmap
            data={data}
            onSelect={setSelected}
            showMarkers={showMarkers}
          />
        </Paper>
        <Stack className="spatial-sidebar" gap="md">
          <Paper p="lg" withBorder>
            <Text c="dimmed" size="xs">已定位活动质量</Text>
            <Title mt={4} order={2}>{formatPercent(locatedRatio)}</Title>
            <Progress mt="sm" value={locatedRatio * 100} />
            <Text c="dimmed" mt="sm" size="xs">
              未定位质量 {formatNumber(data.properties.unlocated_mass)}；地图没有把语义地点伪装成坐标。
            </Text>
          </Paper>
          <Paper p="lg" withBorder>
            <Group justify="space-between">
              <Title order={4}>地点</Title>
              <Badge variant="light">{data.features.length + data.unlocated_places.length}</Badge>
            </Group>
            <Stack gap="xs" mt="md">
              {[...data.features.map((item) => item.properties), ...data.unlocated_places]
                .sort((left, right) => right.heat_raw - left.heat_raw)
                .map((place) => (
                  <Card
                    className="spatial-place-card"
                    key={place.place_id}
                    onClick={() => setSelected(place)}
                    padding="sm"
                    withBorder
                  >
                    <Group justify="space-between" wrap="nowrap">
                      <div>
                        <Text fw={600} size="sm">{place.display_name}</Text>
                        <Text c="dimmed" size="xs">{roleLabel(place.primary_role)} · {place.geo_resolution}</Text>
                      </div>
                      <Text c="moon.3" fw={700} size="sm">{formatPercent(place.activity_weight)}</Text>
                    </Group>
                  </Card>
                ))}
              {data.features.length + data.unlocated_places.length === 0 ? (
                <Text c="dimmed" size="sm">导入后的空间分析完成时，这里会出现地点。</Text>
              ) : null}
            </Stack>
          </Paper>
        </Stack>
      </div>

      <PlaceDetailDrawer onClose={() => setSelected(null)} place={selected} />
    </Stack>
  )
}

function AmapHeatmap({
  data,
  showMarkers,
  onSelect,
}: {
  data: PlaceHeatmap
  showMarkers: boolean
  onSelect: (place: PlaceHeatProperties) => void
}) {
  const container = useRef<HTMLDivElement>(null)
  const key = import.meta.env.VITE_AMAP_JS_KEY as string | undefined
  const securityCode = import.meta.env.VITE_AMAP_SECURITY_CODE as string | undefined
  const [mapError, setMapError] = useState<string | null>(null)

  useEffect(() => {
    if (!container.current || !key || !data.features.length) return
    let map: AmapMap | null = null
    let heatmap: AmapHeatMap | null = null
    let cancelled = false
    if (securityCode) window._AMapSecurityConfig = { securityJsCode: securityCode }
    loadAmap(key)
      .then(async () => {
        if (cancelled || !container.current || !window.AMap) return
        const AMap = window.AMap
        map = new AMap.Map(container.current, {
          mapStyle: 'amap://styles/dark',
          viewMode: '2D',
          zoom: 10,
        })
        await waitForMapReady(map)
        if (cancelled) return
        await loadHeatmapPlugin(AMap)
        if (cancelled || !AMap.HeatMap) return
        heatmap = new AMap.HeatMap(map, {
          radius: 64,
          opacity: [0.12, 0.88],
          gradient: {
            0.2: '#3b82f6',
            0.45: '#22d3ee',
            0.65: '#facc15',
            0.82: '#f97316',
            1: '#ef4444',
          },
        })
        const heatPoints = data.features.map((feature) => ({
          lng: feature.geometry.coordinates[0],
          lat: feature.geometry.coordinates[1],
          count: Math.max(feature.properties.heat_display, 0.01),
        }))
        heatmap.setDataSet({
          data: heatPoints,
          max: Math.max(...heatPoints.map((point) => point.count), 1),
        })
        if (showMarkers) {
          const markers = data.features.map((feature) => {
            const marker = new AMap.Marker({
              position: feature.geometry.coordinates,
              title: feature.properties.display_name,
            })
            marker.on('click', () => onSelect(feature.properties))
            map?.add(marker)
            return marker
          })
          if (markers.length) map.setFitView(markers, false, [60, 60, 60, 60])
        } else {
          map.setFitView(undefined, false, [60, 60, 60, 60])
        }
      })
      .catch((error: unknown) => {
        const detail = error instanceof Error ? error.message : String(error)
        console.error('AMap heatmap initialization failed', error)
        setMapError(`高德热图初始化失败：${detail}`)
      })
    return () => {
      cancelled = true
      heatmap?.setMap(null)
      map?.destroy()
    }
  }, [data, key, onSelect, securityCode, showMarkers])

  if (!key) {
    return (
      <div className="spatial-map-empty">
        <Title order={3}>热图数据已经准备好</Title>
        <Text c="dimmed" maw={480} ta="center">
          配置 VITE_AMAP_JS_KEY 后显示高德底图和 Loca 热力层；地点权重仍可在右侧查看。
        </Text>
      </div>
    )
  }
  if (!data.features.length) {
    return <div className="spatial-map-empty"><Text c="dimmed">当前没有可下发的地理坐标。</Text></div>
  }
  return (
    <>
      <div aria-label="人物活动热图" className="spatial-map" ref={container} />
      {mapError ? <Alert className="spatial-map-error" color="red">{mapError}</Alert> : null}
    </>
  )
}

function waitForMapReady(map: AmapMap): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(
      () => reject(new Error('等待高德地图渲染上下文超时')),
      10_000,
    )
    map.on('complete', () => {
      window.clearTimeout(timeout)
      resolve()
    })
  })
}

function loadHeatmapPlugin(AMap: AmapRuntime): Promise<void> {
  if (AMap.HeatMap) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(
      () => reject(new Error('加载 AMap.HeatMap 插件超时')),
      10_000,
    )
    AMap.plugin(['AMap.HeatMap'], () => {
      window.clearTimeout(timeout)
      if (AMap.HeatMap) resolve()
      else reject(new Error('AMap.HeatMap 插件不可用'))
    })
  })
}

function PlaceDetailDrawer({ place, onClose }: { place: PlaceHeatProperties | null; onClose: () => void }) {
  return (
    <Drawer onClose={onClose} opened={Boolean(place)} position="right" title="地点详情">
      {place ? (
        <Stack>
          <div><Text c="dimmed" size="xs">地点</Text><Title order={2}>{place.display_name}</Title></div>
          <Metric label="活动权重" value={formatPercent(place.activity_weight)} />
          <Metric label="有效到访" value={formatNumber(place.effective_visit_count)} />
          <Metric label="已知停留" value={`${formatNumber(place.effective_dwell_hours)} 小时`} />
          <Metric label="停留覆盖" value={formatPercent(place.dwell_coverage)} />
          <Metric label="证据强度" value={formatPercent(place.evidence_strength)} />
          <Metric label="最近到访" value={formatDate(place.last_visit_at)} />
          <Metric label="地理精度" value={`${place.geo_resolution} · ${place.privacy_level}`} />
        </Stack>
      ) : null}
    </Drawer>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return <Group justify="space-between"><Text c="dimmed" size="sm">{label}</Text><Text fw={600} size="sm">{value}</Text></Group>
}

let amapPromise: Promise<void> | null = null

function loadAmap(key: string): Promise<void> {
  if (window.AMap) return Promise.resolve()
  if (amapPromise) return amapPromise
  amapPromise = loadScript(`https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(key)}`)
  return amapPromise
}

function loadScript(source: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${source}"]`)
    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true })
      existing.addEventListener('error', () => reject(new Error('script load failed')), { once: true })
      return
    }
    const script = document.createElement('script')
    script.async = true
    script.src = source
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('script load failed'))
    document.head.append(script)
  })
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`
}

function formatNumber(value: number) {
  return new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 1 }).format(value)
}

function formatDate(value: string | null) {
  return value ? new Date(value).toLocaleString('zh-CN') : '未知'
}

function roleLabel(value: string | null) {
  return ({ home: '家', workplace: '公司', school: '学校' } as Record<string, string>)[value ?? ''] ?? '未确认角色'
}

function heatmapUrl(options: {
  projectId: string
  metric: HeatmapMetric
  privacy: HeatmapPrivacy
  dayFilter: 'all' | 'workdays' | 'weekend'
  hourFilter: 'all' | 'daytime' | 'evening'
  fromDate: string
  toDate: string
}) {
  const parameters = new URLSearchParams({
    metric: options.metric,
    privacy_level: options.privacy,
  })
  if (options.dayFilter === 'workdays') parameters.set('weekdays', '0,1,2,3,4')
  if (options.dayFilter === 'weekend') parameters.set('weekdays', '5,6')
  if (options.hourFilter === 'daytime') {
    parameters.set('hour_start', '8')
    parameters.set('hour_end', '18')
  }
  if (options.hourFilter === 'evening') {
    parameters.set('hour_start', '18')
    parameters.set('hour_end', '24')
  }
  if (options.fromDate) parameters.set('from', `${options.fromDate}T00:00:00`)
  if (options.toDate) parameters.set('to', `${options.toDate}T23:59:59`)
  return `/api/projects/${options.projectId}/place-heatmap?${parameters}`
}
