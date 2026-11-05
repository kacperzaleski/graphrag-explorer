// "How is A connected to B?" without the LLM: shortest path(s) from the graph, instant and free.

import { useState } from 'react'
import { api } from '../api/client'
import type { EntityHit, Subgraph } from '../api/types'
import { SearchBox } from './SearchBox'

interface Props {
  onPath: (subgraph: Subgraph) => void
}

export function PathFinder({ onPath }: Props) {
  const [source, setSource] = useState<EntityHit | null>(null)
  const [target, setTarget] = useState<EntityHit | null>(null)
  const [maxHops, setMaxHops] = useState(4)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  const find = async () => {
    if (!source || !target) return
    setLoading(true)
    setMessage(null)
    try {
      const subgraph = await api.path(source.key, target.key, maxHops)
      const hops = subgraph.edges.filter(e => e.on_path).length
      setMessage(hops === 0
        ? `No connection within ${maxHops} hops (places don't count as connections).`
        : `Connected via ${hops} relation${hops > 1 ? 's' : ''}. Click the nodes for evidence.`)
      onPath(subgraph)
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <p className="muted small">Find how two entities are connected, straight from the graph.</p>
      <SearchBox placeholder="From… e.g. ÖGB" onPick={setSource} />
      <SearchBox placeholder="To… e.g. GoldenTree" onPick={setTarget} />
      <div className="row">
        <label className="small">
          max hops <input type="number" min={1} max={6} value={maxHops}
            onChange={e => setMaxHops(Number(e.target.value))} />
        </label>
        <button className="primary" disabled={!source || !target || loading} onClick={find}>
          {loading ? 'Searching…' : 'Connect'}
        </button>
      </div>
      {message && <p className="small">{message}</p>}
    </div>
  )
}
