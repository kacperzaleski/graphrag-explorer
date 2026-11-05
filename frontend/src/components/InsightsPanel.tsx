// Starting points for discovery: rankings from graph analytics (computed at ingest).

import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { Insights, RankedEntity } from '../api/types'
import { communityColor, TYPE_COLORS } from '../colors'

interface Props {
  onOpen: (key: string) => void
}

export function InsightsPanel({ onOpen }: Props) {
  const [insights, setInsights] = useState<Insights | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.insights().then(setInsights).catch(err => setError(err.message))
  }, [])

  if (error) return <p className="error">{error}</p>
  if (!insights) return <p className="muted">Loading…</p>
  return (
    <div className="insights">
      <Ranking
        title="Most influential owners"
        hint="PageRank over ownership: owning entities that themselves own a lot counts more."
        items={insights.top_owners}
        value={e => `${e.group_size} in group`}
        onOpen={onOpen}
      />
      <Ranking
        title="Bridges"
        hint="Betweenness centrality: entities on many shortest paths, linking otherwise separate groups."
        items={insights.bridges}
        value={e => (e.betweenness ?? 0).toFixed(2)}
        onOpen={onOpen}
      />
      <h4>Communities</h4>
      <p className="muted small">Louvain clusters: densely connected inside, sparsely between. Colour the graph by community to see them.</p>
      <ul className="communities">
        {insights.communities.map(c => (
          <li key={c.id}>
            <span className="dot" style={{ background: communityColor(c.id) }} />
            <b>{c.members[0]?.name}</b> <span className="muted small">+{c.size - 1}</span>
            <div className="small">
              {c.members.slice(1, 5).map((m, i) => (
                <span key={m.key}>{i > 0 && ', '}<a onClick={() => onOpen(m.key)}>{m.name}</a></span>
              ))}
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Ranking({ title, hint, items, value, onOpen }: {
  title: string
  hint: string
  items: RankedEntity[]
  value: (e: RankedEntity) => string
  onOpen: (key: string) => void
}) {
  return (
    <>
      <h4>{title}</h4>
      <p className="muted small">{hint}</p>
      <ol className="ranking">
        {items.map(e => (
          <li key={e.key}>
            <span className="dot" style={{ background: TYPE_COLORS[e.type] }} />
            <a onClick={() => onOpen(e.key)}>{e.name}</a>
            <span className="muted small"> {value(e)}</span>
          </li>
        ))}
      </ol>
    </>
  )
}
