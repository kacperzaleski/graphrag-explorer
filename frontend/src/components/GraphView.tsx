// Cytoscape wrapper. React owns the data (props); Cytoscape owns rendering and layout.
// On every data change we diff: add new elements, remove gone ones, then lay out with
// randomize: false so existing nodes stay roughly where the user has learnt to find them.

import cytoscape from 'cytoscape'
import type { Core, ElementDefinition } from 'cytoscape'
import fcose from 'cytoscape-fcose'
import { useEffect, useRef, useState } from 'react'
import type { EntityType, GraphEdge, GraphNode } from '../api/types'
import { communityColor, TYPE_COLORS } from '../colors'
import type { ColorBy } from '../colors'
import { qualifier } from '../hooks/useGraph'
import type { Highlight } from '../hooks/useGraph'

cytoscape.use(fcose)

interface Props {
  nodes: Map<string, GraphNode>
  edges: Map<string, GraphEdge>
  highlight: Highlight
  hiddenTypes: Set<EntityType>
  colorBy: ColorBy
  selectedKey: string | null
  onNodeClick: (key: string) => void
}

interface Tooltip {
  x: number
  y: number
  text: string
}

const STYLE: cytoscape.StylesheetJson = [
  {
    selector: 'node',
    style: {
      'background-color': 'data(color)',
      label: 'data(name)',
      'font-size': 9,
      'min-zoomed-font-size': 6, // hide labels when zoomed far out instead of drawing a smear
      'text-valign': 'bottom',
      'text-margin-y': 3,
      color: '#333',
      'text-wrap': 'ellipsis',
      'text-max-width': '110px',
      // bigger = better connected in the whole graph, not just on screen
      width: 'mapData(degree, 0, 50, 14, 40)',
      height: 'mapData(degree, 0, 50, 14, 40)',
      'border-width': 0,
    },
  },
  {
    selector: 'edge',
    style: {
      width: 1.2,
      'line-color': '#B8B8B8',
      'target-arrow-color': '#B8B8B8',
      'target-arrow-shape': 'triangle',
      'arrow-scale': 0.7,
      'curve-style': 'bezier',
    },
  },
  { selector: 'node.selected', style: { 'border-width': 3, 'border-color': '#111' } },
  { selector: '.faded', style: { opacity: 0.18 } },
  {
    selector: 'edge.highlighted',
    style: {
      width: 3, 'line-color': '#D62728', 'target-arrow-color': '#D62728',
      label: 'data(label)', 'font-size': 9, 'text-rotation': 'autorotate',
      'text-background-color': '#fff', 'text-background-opacity': 0.85,
      'text-background-padding': '2px',
    },
  },
  { selector: 'node.highlighted', style: { 'border-width': 3, 'border-color': '#D62728' } },
  { selector: '.hidden', style: { display: 'none' } },
]

export function GraphView({ nodes, edges, highlight, hiddenTypes, colorBy, selectedKey, onNodeClick }: Props) {
  const container = useRef<HTMLDivElement>(null)
  const cy = useRef<Core | null>(null)
  const clickHandler = useRef(onNodeClick)
  const [tooltip, setTooltip] = useState<Tooltip | null>(null)
  // latest callback without re-binding cytoscape events
  useEffect(() => { clickHandler.current = onNodeClick }, [onNodeClick])

  // create once
  useEffect(() => {
    const instance = cytoscape({ container: container.current, style: STYLE, minZoom: 0.2, maxZoom: 3 })
    instance.on('tap', 'node', e => clickHandler.current(e.target.id()))
    instance.on('mouseover', 'node, edge', e => {
      const p = e.renderedPosition ?? e.target.renderedMidpoint()
      setTooltip({ x: p.x, y: p.y, text: e.target.data('tooltip') })
    })
    instance.on('mouseout', 'node, edge', () => setTooltip(null))
    instance.on('pan zoom', () => setTooltip(null))
    cy.current = instance
    return () => instance.destroy()
  }, [])

  // sync elements: diff instead of re-creating, so positions survive updates
  useEffect(() => {
    const instance = cy.current
    if (!instance) return
    const wanted = new Set([...nodes.keys(), ...edges.keys()])
    instance.elements().filter(el => !wanted.has(el.id())).remove()

    const added: ElementDefinition[] = []
    for (const node of nodes.values()) {
      if (instance.getElementById(node.key).nonempty()) continue
      added.push({
        group: 'nodes',
        data: {
          id: node.key, name: node.name, type: node.type, degree: node.degree ?? 1,
          community: node.community,
          tooltip: `${node.name}\n${node.type} · ${node.degree ?? 0} connections` +
            (node.community !== null ? ` · community ${node.community}` : ''),
        },
        position: spawnPosition(instance, node.key, edges),
      })
    }
    for (const edge of edges.values()) {
      const label = [edge.type, qualifier(edge)].filter(Boolean).join(' · ')
      const existing = instance.getElementById(edge.id)
      if (existing.nonempty()) {
        existing.data('label', label)
        continue
      }
      added.push({
        group: 'edges',
        data: {
          id: edge.id, source: edge.source, target: edge.target, label,
          tooltip: `${nodes.get(edge.source)?.name} → ${label} → ${nodes.get(edge.target)?.name}`,
        },
      })
    }
    if (added.length === 0) return
    instance.add(added)
    const layout = instance.layout({
      name: 'fcose', randomize: false, animate: true, animationDuration: 400,
      nodeRepulsion: 8000, idealEdgeLength: 110, fit: instance.nodes().length < 60, padding: 40,
    } as cytoscape.LayoutOptions)
    // fitting a handful of nodes would zoom in until labels are huge: cap the zoom afterwards
    layout.one('layoutstop', () => { if (instance.zoom() > 1.1) instance.zoom({ level: 1.1, renderedPosition: center(instance) }) })
    layout.run()
  }, [nodes, edges])

  // colours
  useEffect(() => {
    cy.current?.nodes().forEach(n => {
      n.data('color', colorBy === 'type'
        ? TYPE_COLORS[n.data('type') as EntityType] ?? '#888'
        : communityColor(n.data('community')))
    })
  }, [colorBy, nodes])

  // filters, selection, highlight
  useEffect(() => {
    const instance = cy.current
    if (!instance) return
    instance.batch(() => {
      instance.elements().removeClass('hidden faded highlighted selected')
      instance.nodes().filter(n => hiddenTypes.has(n.data('type'))).addClass('hidden')
      if (selectedKey) instance.getElementById(selectedKey).addClass('selected')
      if (highlight.edges.size + highlight.nodes.size > 0) {
        instance.elements().addClass('faded')
        instance.nodes().filter(n => highlight.nodes.has(n.id())).removeClass('faded').addClass('highlighted')
        instance.edges().filter(e => highlight.edges.has(e.id())).removeClass('faded').addClass('highlighted')
      }
    })
  }, [highlight, hiddenTypes, selectedKey, nodes, edges])

  return (
    <div className="graph-canvas">
      <div ref={container} className="cy" />
      {tooltip && (
        <div className="tooltip" style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}>
          {tooltip.text}
        </div>
      )}
    </div>
  )
}

function center(cy: Core) {
  return { x: cy.width() / 2, y: cy.height() / 2 }
}

/** New nodes appear next to an already placed neighbour (the node that was expanded),
 *  not at the origin: the layout then only has to nudge them. */
function spawnPosition(cy: Core, key: string, edges: Map<string, GraphEdge>) {
  for (const edge of edges.values()) {
    const other = edge.source === key ? edge.target : edge.target === key ? edge.source : null
    const anchor = other ? cy.getElementById(other) : null
    if (anchor && anchor.nonempty()) {
      const p = anchor.position()
      return { x: p.x + (Math.random() - 0.5) * 80, y: p.y + (Math.random() - 0.5) * 80 }
    }
  }
  const extent = cy.extent()
  return { x: (extent.x1 + extent.x2) / 2, y: (extent.y1 + extent.y2) / 2 }
}
