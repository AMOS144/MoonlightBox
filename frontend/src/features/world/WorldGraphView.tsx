import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import type { ForceGraphMethods } from 'react-force-graph-2d'
import { useQuery } from '@tanstack/react-query'
import { Loader, Paper, Stack, Text } from '@mantine/core'
import { useResizeObserver } from '@mantine/hooks'
import { request } from '../../api/client'
import { mergeGraphData } from './graphData'
import type { ForceGraphData, ForceLink, ForceNode } from './graphData'
import type { WorldGraphSnapshot } from './types'

type Props = {
  projectId: string
  /** 构建进行中时轮询快照，新节点会持续"长"进图里。 */
  building: boolean
}

function cssVar(name: string, fallback: string) {
  if (typeof window === 'undefined') return fallback
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

const EMPTY: ForceGraphData = { nodes: [], links: [] }

export default function WorldGraphView({ projectId, building }: Props) {
  const snapshot = useQuery({
    queryKey: ['world-graph', projectId],
    queryFn: () => request<WorldGraphSnapshot>(`/api/projects/${projectId}/world-profile/graph`),
    // 旧版 sidecar 没有 /graph 时主后端返回 503，静默降级为空态而不是刷错误。
    retry: false,
    refetchInterval: building ? 4000 : false,
  })
  const [data, setData] = useState<ForceGraphData>(EMPTY)
  useEffect(() => {
    if (snapshot.data) setData(prev => mergeGraphData(prev, snapshot.data))
  }, [snapshot.data])

  const [containerRef, rect] = useResizeObserver()
  const graphRef = useRef<ForceGraphMethods | undefined>(undefined)
  const [hovered, setHovered] = useState<ForceNode | null>(null)
  const [selected, setSelected] = useState<ForceNode | null>(null)

  const highlight = useMemo(() => {
    const focus = hovered ?? selected
    if (!focus) return null
    const nodes = new Set<string>([focus.id])
    const links = new Set<(typeof data.links)[number]>()
    for (const link of data.links) {
      const source = typeof link.source === 'object' ? link.source.id : link.source
      const target = typeof link.target === 'object' ? link.target.id : link.target
      if (source === focus.id || target === focus.id) {
        links.add(link)
        nodes.add(source)
        nodes.add(target)
      }
    }
    return { nodes, links }
  }, [hovered, selected, data.links])

  // 新快照合并后重新加热模拟，让新节点自然归位。
  useEffect(() => {
    graphRef.current?.d3ReheatSimulation()
  }, [data])

  const colors = {
    accent: cssVar('--accent', '#c9ad65'),
    accentStrong: cssVar('--accent-strong', '#d8c38f'),
    line: cssVar('--line-strong', '#4a423a'),
    text: cssVar('--text', '#f2eee8'),
    muted: cssVar('--muted', '#aaa19a'),
    memory: cssVar('--memory', '#b197fc'),
  }

  const loading = snapshot.isLoading
  const empty = !loading && data.nodes.length === 0

  return (
    <div className="world-graph">
      <div className="world-graph__canvas" ref={containerRef}>
        {empty ? (
          <Stack align="center" justify="center" gap="xs" style={{ height: '100%' }}>
            {building || loading ? <Loader size="sm" color="moon" /> : null}
            <Text c="dimmed" size="sm">
              {building || loading
                ? '图谱正在生长，第一批节点出现后就会显示在这里。'
                : '暂无图谱数据。'}
            </Text>
          </Stack>
        ) : (
          <ForceGraph2D
            ref={graphRef}
            width={Math.max(0, Math.floor(rect.width))}
            height={440}
            graphData={data}
            nodeId="id"
            backgroundColor="rgba(0,0,0,0)"
            linkColor={link => highlight
              ? highlight.links.has(link as ForceLink) ? colors.accent : 'rgba(74, 66, 58, 0.25)'
              : colors.line}
            linkWidth={link => highlight?.links.has(link as ForceLink) ? 1.6 : 0.7}
            linkDirectionalParticles={building ? 1 : 0}
            linkDirectionalParticleWidth={1.6}
            linkDirectionalParticleColor={() => colors.accentStrong}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const item = node as ForceNode
              const radius = 3.5 + Math.min(6, item.degree * 0.8)
              const dimmed = highlight !== null && !highlight.nodes.has(item.id)
              const isFocus = hovered?.id === item.id || selected?.id === item.id
              ctx.beginPath()
              ctx.arc(node.x ?? 0, node.y ?? 0, radius, 0, 2 * Math.PI)
              ctx.fillStyle = dimmed
                ? 'rgba(126, 118, 111, 0.35)'
                : isFocus ? colors.memory : colors.accent
              ctx.fill()
              const showLabel = isFocus || globalScale > 1.15 || item.degree >= 4
              if (showLabel && !dimmed) {
                const size = Math.max(3.4, 11 / globalScale)
                ctx.font = `${size}px "PingFang SC", "Microsoft YaHei", sans-serif`
                ctx.textAlign = 'center'
                ctx.textBaseline = 'top'
                ctx.fillStyle = isFocus ? colors.text : colors.muted
                ctx.fillText(item.id, node.x ?? 0, (node.y ?? 0) + radius + 1.5)
              }
            }}
            nodePointerAreaPaint={(node, color, ctx) => {
              const item = node as ForceNode
              const radius = 6 + Math.min(6, item.degree * 0.8)
              ctx.beginPath()
              ctx.arc(node.x ?? 0, node.y ?? 0, radius, 0, 2 * Math.PI)
              ctx.fillStyle = color
              ctx.fill()
            }}
            onNodeHover={node => setHovered((node as ForceNode | null) ?? null)}
            onNodeClick={node => setSelected(node as ForceNode)}
            onBackgroundClick={() => setSelected(null)}
            cooldownTime={2200}
            enableNodeDrag
          />
        )}
        {snapshot.data?.truncated ? (
          <Text className="world-graph__truncated" size="xs" c="dimmed">图谱较大，仅展示前 {data.nodes.length} 个节点</Text>
        ) : null}
      </div>
      {selected ? (
        <Paper className="world-graph__detail" p="sm" withBorder>
          <Text fw={600} size="sm">{selected.id}</Text>
          {selected.entityType ? <Text size="xs" c="moon.4" mt={2}>{selected.entityType}</Text> : null}
          {selected.description ? <Text size="sm" c="dimmed" mt={6} lineClamp={4}>{selected.description}</Text> : null}
        </Paper>
      ) : null}
    </div>
  )
}
