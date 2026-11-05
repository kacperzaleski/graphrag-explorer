# GraphRAG Explorer

Documents → knowledge graph → hybrid graph + vector retrieval → FastAPI → React graph explorer.
An end-to-end demo of taking GraphRAG from scripts to an app, with tests, evaluation and CI/CD.

> Work in progress, built in phases. Status: **Phase 5 (evaluation) done.**

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

## Evaluation

`uv run python -m eval.run` (details and methodology: `NOTES.md`, Phase 5). Questions are built from the
**text, not the graph** (otherwise graph mode would be graded on its own data). The answers come from Gemini and the
judge from Groq `gpt-oss-120b` (another vendor, against self-preference bias), with a second judge (Qwen) for
reliability. All comparisons are paired (both modes answer the same questions).

**Main set: 28 questions** (13 multi-hop, 10 lookup, 5 corpus-wide), `eval/results.md`:

| | vector (baseline) | graph |
|---|---|---|
| judge score [95% CI] | 0.91 [0.84, 0.98] | 0.93 [0.86, 0.98] |
| correct | 23/28 | 24/28 |
| recall@8 | 0.90 | 0.89 |
| LLM latency p50 / tokens per question | 1.3 s / 2.9k | 1.3 s / 3.5k |

**Pre-registered hard set: 8 "How is A connected to C?" questions** with a hidden bridge, `eval/results_hard.md`:

| | vector | graph |
|---|---|---|
| judge score | 0.38 | 0.44 (diff +0.06, CI [−0.12, +0.25]) |
| correct | 3/8 | 2/8 |
| recall@8 (without reranker) | 0.56 (**0.81**) | 0.50 (0.56) |

**Honest reading.**
- On this small, clean corpus, GraphRAG gives **no measurable QA advantage**. Differences are within noise, and it costs
  ~25% more tokens plus an extraction step.
- The first round showed a **ceiling effect**: generated questions reuse the passages' wording.
- The hypothesis that the graph helps on hidden-bridge connection questions is **not supported** at n = 8.
- The largest measured effect is that **the reranker hurts** connection questions.
- The graph's value here is what QA accuracy doesn't measure: exploration, provenance (every edge has its source
  quote), time and stake on ownership, analytics, and visible errors.
- Limitations: small n, LLM-generated questions, and judge validity still pending 10 human labels
  (`eval/calibration_sheet.md`).

## Data attribution

Documents in `data/raw/` are excerpts of English Wikipedia articles, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Source URLs and revision ids are
in `data/raw/sources.json`.
