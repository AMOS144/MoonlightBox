import { expect, test } from 'vitest'
import { mergeGraphData } from './graphData'
import type { ForceGraphData } from './graphData'
import type { WorldGraphSnapshot } from './types'

function snapshot(nodes: string[], edges: Array<[string, string]>): WorldGraphSnapshot {
  return {
    nodes: nodes.map(id => ({ id, entity_type: 'person', description: `${id}的描述` })),
    edges: edges.map(([source, target]) => ({ source, target, keywords: null })),
    truncated: false,
  }
}

test('空图合并出全部节点与边，并统计度数', () => {
  const data = mergeGraphData({ nodes: [], links: [] }, snapshot(['甲', '乙', '丙'], [['甲', '乙'], ['甲', '丙']]))
  expect(data.nodes.map(node => node.id)).toEqual(['甲', '乙', '丙'])
  expect(data.links).toHaveLength(2)
  expect(data.nodes[0].degree).toBe(2)
  expect(data.nodes[1].degree).toBe(1)
})

test('已有节点对象被复用以保留力导向位置', () => {
  const prev: ForceGraphData = mergeGraphData({ nodes: [], links: [] }, snapshot(['甲', '乙'], [['甲', '乙']]))
  prev.nodes[0].x = 42
  const next = mergeGraphData(prev, snapshot(['甲', '乙', '丙'], [['甲', '乙'], ['乙', '丙']]))
  expect(next.nodes[0]).toBe(prev.nodes[0])
  expect(next.nodes[0].x).toBe(42)
  expect(next.nodes.map(node => node.id)).toEqual(['甲', '乙', '丙'])
})

test('消失的节点被移除，字段会更新为最新快照', () => {
  const prev = mergeGraphData({ nodes: [], links: [] }, snapshot(['甲', '乙'], [['甲', '乙']]))
  const next = mergeGraphData(prev, {
    nodes: [{ id: '甲', entity_type: 'place', description: '更新后' }],
    edges: [],
    truncated: false,
  })
  expect(next.nodes.map(node => node.id)).toEqual(['甲'])
  expect(next.links).toHaveLength(0)
  expect(next.nodes[0].entityType).toBe('place')
  expect(next.nodes[0].description).toBe('更新后')
  expect(next.nodes[0].degree).toBe(0)
})

test('重复边被去重', () => {
  const data = mergeGraphData({ nodes: [], links: [] }, {
    nodes: [{ id: '甲', entity_type: null, description: null }, { id: '乙', entity_type: null, description: null }],
    edges: [
      { source: '甲', target: '乙', keywords: 'a' },
      { source: '甲', target: '乙', keywords: 'b' },
    ],
    truncated: false,
  })
  expect(data.links).toHaveLength(1)
  expect(data.nodes[0].degree).toBe(1)
})
