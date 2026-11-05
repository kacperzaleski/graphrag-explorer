// Client-side graph state: what is currently drawn. The full graph never comes to the browser;
// it grows only through searches, expands, paths and answers (each a small subgraph from the API).

import { useCallback, useReducer } from 'react'
import type { GraphEdge, GraphNode, Subgraph } from '../api/types'

export const MAX_NODES = 150 // beyond this a force layout becomes unreadable (and slow)

export interface Highlight {
  nodes: Set<string>
  edges: Set<string>
}

interface State {
  nodes: Map<string, GraphNode>
  edges: Map<string, GraphEdge>
  expanded: Set<string> // nodes whose neighbours were already loaded
  highlight: Highlight
  warning: string | null
}

type Action =
  | { kind: 'merge'; subgraph: Subgraph; replace?: boolean; expandedKey?: string }
  | { kind: 'highlight'; highlight: Highlight }
  | { kind: 'clear' }

const EMPTY_HIGHLIGHT: Highlight = { nodes: new Set(), edges: new Set() }

const initial: State = {
  nodes: new Map(),
  edges: new Map(),
  expanded: new Set(),
  highlight: EMPTY_HIGHLIGHT,
  warning: null,
}

function reducer(state: State, action: Action): State {
  switch (action.kind) {
    case 'clear':
      return initial
    case 'highlight':
      return { ...state, highlight: action.highlight }
    case 'merge': {
      const base = action.replace ? initial : state
      const nodes = new Map(base.nodes)
      let skipped = 0
      for (const node of action.subgraph.nodes) {
        if (nodes.has(node.key)) continue
        if (nodes.size >= MAX_NODES) {
          skipped++
          continue
        }
        nodes.set(node.key, node)
      }
      const edges = new Map(base.edges)
      for (const edge of action.subgraph.edges) {
        // an edge is drawn only if both ends are on screen
        if (nodes.has(edge.source) && nodes.has(edge.target)) {
          // keep flags like `used` / `on_path` from the newest response
          edges.set(edge.id, { ...edges.get(edge.id), ...edge })
        }
      }
      const expanded = new Set(base.expanded)
      if (action.expandedKey) expanded.add(action.expandedKey)
      let warning: string | null = null
      if (skipped > 0) {
        warning = `Node limit (${MAX_NODES}) reached: ${skipped} nodes not shown. ` +
          'Clear the graph or collapse the view to explore further.'
      } else if (action.subgraph.truncated) {
        warning = 'Only the most relevant neighbours are shown (server-side limit).'
      }
      return { ...base, nodes, edges, expanded, warning, highlight: EMPTY_HIGHLIGHT }
    }
  }
}

export function useGraph() {
  const [state, dispatch] = useReducer(reducer, initial)
  return {
    ...state,
    merge: useCallback((subgraph: Subgraph, options: { replace?: boolean; expandedKey?: string } = {}) =>
      dispatch({ kind: 'merge', subgraph, ...options }), []),
    setHighlight: useCallback((highlight: Highlight) => dispatch({ kind: 'highlight', highlight }), []),
    clear: useCallback(() => dispatch({ kind: 'clear' }), []),
  }
}

/** Nodes and edges to emphasise for a subgraph: path edges and edges the answer used. */
export function highlightOf(subgraph: Subgraph): Highlight {
  const edges = subgraph.edges.filter(e => e.on_path || e.used)
  return {
    edges: new Set(edges.map(e => e.id)),
    nodes: new Set([...subgraph.seeds, ...edges.flatMap(e => [e.source, e.target])]),
  }
}

export function qualifier(edge: { start_year: number | null; end_year: number | null; stake_percent: number | null }): string {
  const years = edge.start_year || edge.end_year
    ? `${edge.start_year ?? '?'}–${edge.end_year ?? ''}`.replace(/–$/, '')
    : ''
  const stake = edge.stake_percent ? `${edge.stake_percent}%` : ''
  return [stake, years].filter(Boolean).join(', ')
}
