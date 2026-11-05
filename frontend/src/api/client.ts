// Typed API client: one function per endpoint, all errors normalised to ApiError.

import type {
  AskRequest, AskResponse, EntityHit, Insights, NodeDetails, Subgraph,
} from './types'

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  status: number
  requestId: string | null

  constructor(status: number, message: string, requestId: string | null) {
    super(message)
    this.status = status
    this.requestId = requestId // shown in the UI so a user can report it and we can grep logs
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new ApiError(0, 'API not reachable. Is the backend running?', null)
  }
  const requestId = response.headers.get('X-Request-ID')
  if (!response.ok) {
    throw new ApiError(response.status, await errorMessage(response), requestId)
  }
  return response.json() as Promise<T>
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) {
      // FastAPI 422: list of {loc, msg}
      return body.detail.map((d: { loc: string[]; msg: string }) =>
        `${d.loc.slice(1).join('.')}: ${d.msg}`).join('; ')
    }
  } catch {
    // not JSON
  }
  return `Request failed (${response.status})`
}

const query = (params: Record<string, string | number>) =>
  new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString()

export const api = {
  ask: (body: AskRequest) =>
    request<AskResponse>('/ask', { method: 'POST', body: JSON.stringify(body) }),
  search: (q: string, limit = 8) =>
    request<EntityHit[]>(`/graph/search?${query({ q, limit })}`),
  neighbors: (nodeId: string, hops = 1, limit = 25) =>
    request<Subgraph>(`/graph/neighbors?${query({ node_id: nodeId, hops, limit })}`),
  path: (source: string, target: string, maxHops = 4) =>
    request<Subgraph>(`/graph/path?${query({ source, target, max_hops: maxHops })}`),
  node: (nodeId: string) =>
    request<NodeDetails>(`/graph/node/${encodeURIComponent(nodeId)}`),
  insights: (limit = 8) => request<Insights>(`/graph/insights?${query({ limit })}`),
}
