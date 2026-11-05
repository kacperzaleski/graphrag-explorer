import { useCallback, useMemo, useState } from 'react'
import { api } from './api/client'
import type { AskResponse, EntityType, Subgraph } from './api/types'
import type { ColorBy } from './colors'
import { AskPanel } from './components/AskPanel'
import { GraphView } from './components/GraphView'
import { InsightsPanel } from './components/InsightsPanel'
import { Legend } from './components/Legend'
import { NodePanel } from './components/NodePanel'
import { PathFinder } from './components/PathFinder'
import { SearchBox } from './components/SearchBox'
import { highlightOf, useGraph } from './hooks/useGraph'

type Tab = 'ask' | 'connect' | 'insights'

export default function App() {
  const graph = useGraph()
  const [tab, setTab] = useState<Tab>('insights')
  const [selected, setSelected] = useState<string | null>(null)
  const [hiddenTypes, setHiddenTypes] = useState<Set<EntityType>>(new Set())
  const [colorBy, setColorBy] = useState<ColorBy>('type')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  /** Load an entity's neighbours into the view (once per entity). */
  const expand = useCallback(async (key: string, replace = false) => {
    setSelected(key)
    if (!replace && graph.expanded.has(key)) return
    setLoading(true)
    setError(null)
    try {
      graph.merge(await api.neighbors(key), { replace, expandedKey: key })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [graph])

  /** Jump to an entity from a list: show it with its neighbourhood, adding to the view. */
  const open = useCallback((key: string) => expand(key, graph.nodes.size === 0), [expand, graph.nodes.size])

  const showSubgraph = useCallback((subgraph: Subgraph) => {
    graph.merge(subgraph, { replace: true })
    graph.setHighlight(highlightOf(subgraph))
    setSelected(null)
  }, [graph])

  const onAnswer = useCallback((response: AskResponse) => {
    if (response.subgraph.nodes.length > 0) showSubgraph(response.subgraph)
  }, [showSubgraph])

  const typeCounts = useMemo(() => {
    const counts = new Map<EntityType, number>()
    for (const n of graph.nodes.values()) counts.set(n.type, (counts.get(n.type) ?? 0) + 1)
    return counts
  }, [graph.nodes])

  const toggleType = (type: EntityType) => setHiddenTypes(prev => {
    const next = new Set(prev)
    if (next.has(type)) next.delete(type)
    else next.add(type)
    return next
  })

  return (
    <div className="app">
      <header>
        <h1>GraphRAG Explorer</h1>
        <span className="muted small">Austrian banks, owners &amp; regulators · from 22 Wikipedia articles</span>
        <div className="header-search">
          <SearchBox placeholder="Search an entity, e.g. BAWAG, Erste, Cerberus…" onPick={hit => expand(hit.key, true)} />
        </div>
        <button onClick={() => { graph.clear(); setSelected(null) }} disabled={graph.nodes.size === 0}>Clear</button>
      </header>

      <main>
        <aside className="panel left">
          <nav className="tabs">
            {(['insights', 'ask', 'connect'] as Tab[]).map(t => (
              <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>
            ))}
          </nav>
          {/* panels stay mounted so an answer survives switching tabs */}
          <div hidden={tab !== 'insights'}><InsightsPanel onOpen={open} /></div>
          <div hidden={tab !== 'ask'}><AskPanel onAnswer={onAnswer} onOpenNode={expand} /></div>
          <div hidden={tab !== 'connect'}><PathFinder onPath={showSubgraph} /></div>
        </aside>

        <section className="graph">
          {graph.nodes.size === 0 && (
            <div className="empty">
              <p><b>Start exploring:</b> search an entity above, pick one from the insights, ask a question, or connect two entities.</p>
              <p className="muted small">Click a node to expand its neighbours and see its evidence.</p>
            </div>
          )}
          <GraphView
            nodes={graph.nodes} edges={graph.edges} highlight={graph.highlight}
            hiddenTypes={hiddenTypes} colorBy={colorBy} selectedKey={selected} onNodeClick={expand}
          />
          {graph.nodes.size > 0 && (
            <Legend counts={typeCounts} hidden={hiddenTypes} onToggle={toggleType} colorBy={colorBy} onColorBy={setColorBy} />
          )}
          <div className="status">
            {loading && <span className="spinner">Loading…</span>}
            {graph.warning && <span className="warning">⚠ {graph.warning}</span>}
            {error && <span className="error">{error}</span>}
            {graph.highlight.edges.size > 0 && (
              <button className="small" onClick={() => graph.setHighlight({ nodes: new Set(), edges: new Set() })}>
                clear highlight
              </button>
            )}
            <span className="muted small">{graph.nodes.size} nodes · {graph.edges.size} edges</span>
          </div>
        </section>

        {selected && (
          <NodePanel key={selected} nodeKey={selected} onOpen={expand} onClose={() => setSelected(null)} />
        )}
      </main>
    </div>
  )
}
