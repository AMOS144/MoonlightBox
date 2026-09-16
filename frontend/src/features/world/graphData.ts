import type { WorldGraphSnapshot } from './types'

export type ForceNode = {
  id: string
  entityType: string | null
  description: string | null
  degree: number
  x?: number
  y?: number
  vx?: number
  vy?: number
}

export type ForceLink = {
  source: string | ForceNode
  target: string | ForceNode
  keywords: string | null
}

export type ForceGraphData = { nodes: ForceNode[]; links: ForceLink[] }

/**
 * 把新一轮快照合并进已有图数据：按 id 复用旧节点对象，力导向模拟保留节点位置，
 * 轮询到新节点时它们在原图边上"长出来"，而不是整图重排。
 */
export function mergeGraphData(prev: ForceGraphData, snapshot: WorldGraphSnapshot): ForceGraphData {
  const degree = new Map<string, number>()
  const links: ForceLink[] = []
  const seenLinks = new Set<string>()
  for (const edge of snapshot.edges) {
    const key = `${edge.source}→${edge.target}`
    if (seenLinks.has(key)) continue
    seenLinks.add(key)
    links.push({ source: edge.source, target: edge.target, keywords: edge.keywords })
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1)
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1)
  }
  const prevById = new Map(prev.nodes.map(node => [node.id, node]))
  const nodes = snapshot.nodes.map(node => {
    const existing = prevById.get(node.id)
    if (existing) {
      existing.entityType = node.entity_type
      existing.description = node.description
      existing.degree = degree.get(node.id) ?? 0
      return existing
    }
    return {
      id: node.id,
      entityType: node.entity_type,
      description: node.description,
      degree: degree.get(node.id) ?? 0,
    }
  })
  return { nodes, links }
}
