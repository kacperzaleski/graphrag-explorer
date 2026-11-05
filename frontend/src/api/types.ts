// Mirrors backend/schemas.py. Keep both in sync: these types are the frontend's view of the API
// contract, so a renamed field shows up as a compile error instead of a blank UI.

export type EntityType =
  | 'BANK' | 'COMPANY' | 'PERSON' | 'REGULATOR' | 'GOVERNMENT' | 'LOCATION' | 'EVENT' | 'OTHER'

export interface GraphNode {
  key: string
  name: string
  type: EntityType
  degree: number | null
  hop: number
  community: number | null
}

export interface GraphEdge {
  id: string
  source: string // entity keys
  target: string
  type: string
  chunk_ids: string[]
  on_path: boolean
  used: boolean
  start_year: number | null // enrichment pass, ownership-like edges only
  end_year: number | null
  stake_percent: number | null
}

export interface Subgraph {
  seeds: string[]
  truncated: boolean
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface Citation {
  chunk_id: string
  doc: string
  title: string
  section: string
  url: string | null
  text: string
}

export type Mode = 'vector' | 'graph'

export interface AskRequest {
  question: string
  mode: Mode
  hops: number
}

export interface AskResponse {
  question: string
  mode: Mode
  answer: string
  found_in_context: boolean
  citations: Citation[]
  passages: string[]
  issues: string[]
  subgraph: Subgraph
  timings_ms: Record<string, number>
  usage: { model: string; cached: boolean; prompt_tokens: number; completion_tokens: number }
}

export interface EntityHit {
  key: string
  name: string
  type: EntityType
  degree: number | null
  description: string | null
  score: number
}

export interface Relation {
  id: string
  type: string
  outgoing: boolean
  other_key: string
  other_name: string
  other_type: EntityType
  chunk_ids: string[]
  evidence: string[]
  start_year: number | null
  end_year: number | null
  stake_percent: number | null
}

export interface Passage {
  id: string
  doc: string
  title: string
  section: string
  url: string | null
  text: string
}

export interface NodeDetails {
  key: string
  name: string
  type: EntityType
  description: string | null
  aliases: string[]
  degree: number | null
  mention_count: number | null
  relations: Relation[]
  chunks: Passage[]
}

export interface RankedEntity {
  key: string
  name: string
  type: EntityType
  degree: number | null
  owner_rank: number | null
  group_size: number | null
  betweenness: number | null
  community: number | null
}

export interface Community {
  id: number
  size: number
  members: { key: string; name: string; type: EntityType }[]
}

export interface Insights {
  top_owners: RankedEntity[]
  largest_groups: RankedEntity[]
  bridges: RankedEntity[]
  communities: Community[]
}
