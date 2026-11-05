// Legend doubling as type filter (click a type to hide/show it) plus the colour-mode switch.

import type { EntityType } from '../api/types'
import { ENTITY_TYPES, TYPE_COLORS } from '../colors'
import type { ColorBy } from '../colors'

interface Props {
  counts: Map<EntityType, number> // nodes per type currently on screen
  hidden: Set<EntityType>
  onToggle: (type: EntityType) => void
  colorBy: ColorBy
  onColorBy: (mode: ColorBy) => void
}

export function Legend({ counts, hidden, onToggle, colorBy, onColorBy }: Props) {
  return (
    <div className="legend">
      <div className="legend-modes">
        colour by
        <button className={colorBy === 'type' ? 'active' : ''} onClick={() => onColorBy('type')}>type</button>
        <button className={colorBy === 'community' ? 'active' : ''} onClick={() => onColorBy('community')}>
          community
        </button>
      </div>
      {ENTITY_TYPES.filter(t => counts.has(t)).map(type => (
        <button
          key={type}
          className={`legend-item ${hidden.has(type) ? 'off' : ''}`}
          onClick={() => onToggle(type)}
          title={hidden.has(type) ? 'show' : 'hide'}
        >
          <span className="dot" style={{ background: colorBy === 'type' ? TYPE_COLORS[type] : '#999' }} />
          {type.toLowerCase()} ({counts.get(type)})
        </button>
      ))}
    </div>
  )
}
