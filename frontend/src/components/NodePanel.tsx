// Side panel for the selected entity: what we know, how it is connected (with evidence), and the
// source passages. Every claim in the graph can be traced back to text here.

import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { NodeDetails, Relation } from '../api/types'
import { TYPE_COLORS } from '../colors'
import { qualifier } from '../hooks/useGraph'

interface Props {
  nodeKey: string
  onOpen: (key: string) => void // jump to a related entity
  onClose: () => void
}

export function NodePanel({ nodeKey, onOpen, onClose }: Props) {
  const [details, setDetails] = useState<NodeDetails | null>(null)
  const [error, setError] = useState<string | null>(null)

  // the parent remounts this panel per entity (key={nodeKey}), so state starts empty
  useEffect(() => {
    let cancelled = false
    api.node(nodeKey)
      .then(d => { if (!cancelled) setDetails(d) })
      .catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [nodeKey])

  return (
    <aside className="panel right">
      <button className="close" onClick={onClose} aria-label="close">×</button>
      {error && <p className="error">{error}</p>}
      {!details && !error && <p className="muted">Loading…</p>}
      {details && (
        <>
          <h2>{details.name}</h2>
          <p>
            <span className="badge" style={{ background: TYPE_COLORS[details.type] }}>{details.type}</span>
            <span className="muted"> {details.degree ?? 0} connections · {details.mention_count ?? 0} passages</span>
          </p>
          {details.description && <p>{details.description}</p>}
          {details.aliases.length > 1 && (
            <p className="muted small">Also written as: {details.aliases.filter(a => a !== details.name).join(', ')}</p>
          )}

          <h3>Connections ({details.relations.length})</h3>
          <ul className="relations">
            {sortRelations(details.relations).map(r => (
              <li key={r.id}>
                <div>
                  {r.outgoing ? '→' : '←'} <b>{r.type}</b>{' '}
                  <a onClick={() => onOpen(r.other_key)}>{r.other_name}</a>
                  {qualifier(r) && <span className="qualifier"> {qualifier(r)}</span>}
                </div>
                <blockquote>“{r.evidence[0]}”<span className="muted small"> — {r.chunk_ids.join(', ')}</span></blockquote>
              </li>
            ))}
          </ul>

          <h3>Source passages ({details.chunks.length})</h3>
          {details.chunks.map(c => (
            <details key={c.id} className="passage">
              <summary>{c.title} › {c.section || 'Introduction'} <span className="muted small">{c.id}</span></summary>
              <p>{c.text.split('\n\n').slice(1).join('\n\n')}</p>
              {c.url && <a href={c.url} target="_blank" rel="noreferrer">Wikipedia ↗</a>}
            </details>
          ))}
        </>
      )}
    </aside>
  )
}

// ownership first, then by year: reads like a history of the entity
const ORDER = ['OWNS', 'ACQUIRED', 'MERGED_WITH', 'SUPERVISES', 'LEADS', 'FOUNDED']
function sortRelations(relations: Relation[]) {
  const rank = (r: Relation) => {
    const i = ORDER.indexOf(r.type)
    return i === -1 ? ORDER.length : i
  }
  return [...relations].sort((a, b) =>
    rank(a) - rank(b) || (a.start_year ?? 9999) - (b.start_year ?? 9999))
}
