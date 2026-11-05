// Ask a question (vector or graph mode); show the answer with citations and hand the subgraph
// that was used to the graph view for highlighting.

import { useState } from 'react'
import { api } from '../api/client'
import type { AskResponse, Mode } from '../api/types'

interface Props {
  onAnswer: (response: AskResponse) => void
  onOpenNode: (key: string) => void
}

const EXAMPLES = [
  'How is the Austrian Trade Union Federation connected to GoldenTree?',
  'Which bank held 46% of BAWAG before the trade union bought it back?',
  'Who owned Hypo Alpe-Adria-Bank before it was nationalised?',
]

export function AskPanel({ onAnswer, onOpenNode }: Props) {
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState<Mode>('graph')
  const [hops, setHops] = useState(2)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AskResponse | null>(null)

  const ask = async (q: string) => {
    if (q.trim().length < 3) return
    setLoading(true)
    setError(null)
    try {
      const response = await api.ask({ question: q.trim(), mode, hops })
      setResult(response)
      onAnswer(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <textarea
        value={question}
        maxLength={500}
        rows={3}
        placeholder="Ask about banks, owners, regulators…"
        onChange={e => setQuestion(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(question) } }}
      />
      <div className="row">
        <select value={mode} onChange={e => setMode(e.target.value as Mode)} title="retrieval mode">
          <option value="graph">graph (GraphRAG)</option>
          <option value="vector">vector (plain RAG)</option>
        </select>
        <label className="small" title="graph expansion depth">
          hops <input type="number" min={1} max={3} value={hops} disabled={mode === 'vector'}
            onChange={e => setHops(Number(e.target.value))} />
        </label>
        <button className="primary" disabled={loading} onClick={() => ask(question)}>
          {loading ? 'Thinking…' : 'Ask'}
        </button>
      </div>
      {!result && !loading && (
        <div className="examples">
          {EXAMPLES.map(q => <a key={q} onClick={() => { setQuestion(q); ask(q) }}>{q}</a>)}
        </div>
      )}
      {error && <p className="error">{error}</p>}
      {result && <Answer result={result} onOpenNode={onOpenNode} />}
    </div>
  )
}

function Answer({ result, onOpenNode }: { result: AskResponse; onOpenNode: (key: string) => void }) {
  const used = result.subgraph.edges.filter(e => e.used || e.on_path)
  const name = (key: string) => result.subgraph.nodes.find(n => n.key === key)?.name ?? key
  return (
    <div className="answer">
      <p className={result.found_in_context ? '' : 'muted'}>{result.answer}</p>
      {result.issues.map(i => <p key={i} className="warning small">⚠ {i}</p>)}
      {used.length > 0 && (
        <>
          <h4>Graph facts used (highlighted)</h4>
          <ul className="facts">
            {used.map(e => (
              <li key={e.id}>
                <a onClick={() => onOpenNode(e.source)}>{name(e.source)}</a> <b>{e.type}</b>{' '}
                <a onClick={() => onOpenNode(e.target)}>{name(e.target)}</a>
              </li>
            ))}
          </ul>
        </>
      )}
      <h4>Sources ({result.citations.length})</h4>
      {result.citations.map(c => (
        <details key={c.chunk_id} className="passage">
          <summary>[{c.chunk_id}] {c.title} › {c.section || 'Introduction'}</summary>
          <p>{c.text.split('\n\n').slice(1).join('\n\n')}</p>
          {c.url && <a href={c.url} target="_blank" rel="noreferrer">Wikipedia ↗</a>}
        </details>
      ))}
      <p className="muted small">
        {result.mode} mode · {(result.timings_ms.total / 1000).toFixed(1)} s · {result.usage.model}
        {result.usage.cached ? ' (cached)' : ` · ${result.usage.prompt_tokens + result.usage.completion_tokens} tokens`}
      </p>
    </div>
  )
}
