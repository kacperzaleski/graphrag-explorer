# GraphRAG Explorer: frontend

React + TypeScript + Vite, graph rendering with Cytoscape.js (`fcose` layout).

```bash
npm install
npm run dev     # http://localhost:5173, expects the API on http://localhost:8000
npm run build   # type-check (tsc) + production bundle in dist/
npm run lint    # oxlint
```

The API base URL can be changed with `VITE_API_URL`. `src/api/types.ts` mirrors `backend/schemas.py`.
Design notes: see "Phase 4" in the repository's `NOTES.md`.
