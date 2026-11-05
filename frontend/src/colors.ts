import type { EntityType } from './api/types'

// One colour per entity type; the same palette as the Streamlit prototype.
export const TYPE_COLORS: Record<EntityType, string> = {
  BANK: '#4C78A8',
  COMPANY: '#F58518',
  PERSON: '#54A24B',
  REGULATOR: '#E45756',
  GOVERNMENT: '#B279A2',
  LOCATION: '#9D755D',
  EVENT: '#D4A017',
  OTHER: '#8C8C8C',
}

export const ENTITY_TYPES = Object.keys(TYPE_COLORS) as EntityType[]

// Colour by Louvain community (ids are ordered by community size, 0 = largest).
const COMMUNITY_PALETTE = [
  '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
  '#8C564B', '#E377C2', '#17BECF', '#BCBD22', '#7F7F7F',
]

export function communityColor(community: number | null): string {
  return community === null ? '#C7C7C7' : COMMUNITY_PALETTE[community % COMMUNITY_PALETTE.length]
}

export type ColorBy = 'type' | 'community'
