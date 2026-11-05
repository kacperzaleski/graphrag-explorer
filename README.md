# GraphRAG Explorer

Documents → knowledge graph → hybrid graph + vector retrieval → FastAPI → React graph explorer.
An end-to-end demo of taking GraphRAG from scripts to an app, with tests, evaluation and CI/CD.

> Work in progress, built in phases. Status: **Phase 4 (React explorer) done.**

## Setup (so far)

```bash
cp .env.example .env            # add at least GEMINI_API_KEY (free at aistudio.google.com)
uv sync                         # Python 3.12 venv + dependencies
docker compose up -d neo4j      # Neo4j 5 LTS; browser at http://localhost:7474
uv run python -m backend.check  # verifies Neo4j, chat model and embedding model
uv run python -m backend.ingest --alias-candidates   # builds the graph (idempotent, cached)
uv run --group prototype streamlit run prototype/app.py   # throwaway prototype UI
uv run python -m backend.pipeline "How is ÖGB connected to GoldenTree?"   # CLI, --mode vector|graph
uv run uvicorn backend.api:app --no-access-log   # API, docs at http://localhost:8000/docs
cd frontend && npm install && npm run dev            # explorer at http://localhost:5173
uv run pytest
```

The corpus (22 Wikipedia articles) is committed in `data/raw/`; `uv run python scripts/fetch_data.py`
re-downloads it. Ingestion calls the LLM once per chunk (~160 calls); results are cached in
`.cache/`, so re-runs are free, and a run interrupted by quota limits resumes where it stopped.

## Graph schema

```
(:Chunk {id, doc, title, section, text, url, embedding})-[:MENTIONS]->(:Entity {key, name, type, ...})
(:Entity)-[:RELATES {type, chunk_ids, evidence}]->(:Entity)
```
Entity types: BANK, COMPANY, PERSON, REGULATOR, GOVERNMENT, LOCATION, EVENT, OTHER.
Relation types: OWNS, ACQUIRED, MERGED_WITH, SUPERVISES, LEADS, FOUNDED, HEADQUARTERED_IN,
INVOLVED_IN, RELATED_TO. Entity merges beyond simple normalization are curated in
`data/aliases.yaml`.

## LLM providers

Each provider has a preset in `backend/config.py`, and all of them use the OpenAI-compatible API.
To switch, change `LLM_PROVIDER` (`gemini` | `cerebras` | `mistral` | `groq`) and set that provider's key.
Embeddings are configured separately (`EMBED_PROVIDER`: `gemini` | `mistral`), because not every
chat provider offers embeddings.

## Layout

| Path | Contents |
|---|---|
| `backend/` | Python package: config, LLM client, chunking, extraction, resolution, Neo4j, ingest |
| `frontend/` | React + TypeScript graph explorer (Phase 4) |
| `prototype/` | Streamlit prototype (Phase 1) |
| `scripts/` | data download |
| `data/raw/` | corpus: 22 Wikipedia excerpts (committed, see attribution) |
| `eval/` | questions + evaluation runner (Phase 5) |
| `infra/` | Jenkins, Kubernetes, Airflow (Phase 6) |
| `data/aliases.yaml` | curated entity aliases (human-reviewed merges) |
| `NOTES.md` | running interview notes: concepts and design decisions |

## Data attribution

Documents in `data/raw/` are excerpts of English Wikipedia articles, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Source URLs and revision ids are
in `data/raw/sources.json`.
