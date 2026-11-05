# GraphRAG Explorer

Documents → knowledge graph → hybrid graph + vector retrieval → FastAPI → React graph explorer.
An end-to-end demo of taking GraphRAG from scripts to an app, with tests, evaluation and CI/CD.

> Work in progress, built in phases. Status: **Phase 0 (scaffolding) done.**

## Setup (so far)

```bash
cp .env.example .env            # add at least GEMINI_API_KEY (free at aistudio.google.com)
uv sync                         # Python 3.12 venv + dependencies
docker compose up -d neo4j      # Neo4j 5 LTS; browser at http://localhost:7474
uv run python -m backend.check  # verifies Neo4j, chat model and embedding model
uv run pytest
```

## LLM providers

Each provider has a preset in `backend/config.py`, and all of them use the OpenAI-compatible API.
To switch, change `LLM_PROVIDER` (`gemini` | `cerebras` | `mistral` | `groq`) and set that provider's key.
Embeddings are configured separately (`EMBED_PROVIDER`: `gemini` | `mistral`), because not every
chat provider offers embeddings.

## Layout

| Path | Contents |
|---|---|
| `backend/` | Python package: config, Neo4j access, (soon) ingestion, retrieval, FastAPI |
| `frontend/` | React + TypeScript graph explorer (Phase 4) |
| `prototype/` | Streamlit prototype (Phase 1) |
| `scripts/` | data download |
| `data/raw/` | downloaded documents (git-ignored) |
| `eval/` | questions + evaluation runner (Phase 5) |
| `infra/` | Jenkins, Kubernetes, Airflow (Phase 6) |
| `NOTES.md` | running interview notes: concepts and design decisions |
