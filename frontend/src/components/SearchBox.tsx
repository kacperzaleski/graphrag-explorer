// Type-ahead entity search. Debounced (250 ms) so typing "raiffeisen" sends ~1 request, not 10;
// stale responses are ignored so a slow "ra" result can't overwrite a newer "raif" result.

import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { EntityHit } from '../api/types'
import { TYPE_COLORS } from '../colors'

interface Props {
  placeholder: string
  onPick: (hit: EntityHit) => void
}

export function SearchBox({ placeholder, onPick }: Props) {
  const [text, setText] = useState('')
  const [hits, setHits] = useState<EntityHit[]>([])
  const [open, setOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const latest = useRef(0)

  const active = open && text.trim().length >= 2

  useEffect(() => {
    if (!active) return
    const id = ++latest.current
    const timer = setTimeout(() => {
      api.search(text.trim())
        .then(result => { if (id === latest.current) { setHits(result); setError(null) } })
        .catch(err => { if (id === latest.current) setError(err.message) })
    }, 250)
    return () => clearTimeout(timer)
  }, [text, active])

  const pick = (hit: EntityHit) => {
    setOpen(false)
    setText(hit.name)
    onPick(hit)
  }

  return (
    <div className="search">
      <input
        value={text}
        placeholder={placeholder}
        onChange={e => { setText(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={e => { if (e.key === 'Enter' && hits[0]) pick(hits[0]) }}
      />
      {active && (hits.length > 0 || error) && (
        <ul className="search-results">
          {error && <li className="error">{error}</li>}
          {hits.map(hit => (
            <li key={hit.key} onMouseDown={() => pick(hit)}>
              <span className="dot" style={{ background: TYPE_COLORS[hit.type] }} />
              <span className="name">{hit.name}</span>
              <span className="muted">{hit.type.toLowerCase()} · {hit.degree ?? 0}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
