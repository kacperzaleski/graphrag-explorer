# Interview notes: GraphRAG Explorer

Running notes, updated every phase. For each topic: **what it is**, **why I chose it**, **trade-offs**,
and **likely follow-up questions**. Phase 7 condenses this into `docs/INTERVIEW_NOTES.md`.

---

## Phase 0: Scaffolding

### Why GraphRAG at all (the 30-second pitch)
Plain vector RAG embeds text chunks and returns the ones most similar to the question. That works for
"what does document X say about Y", but fails when the answer is **spread across documents and
connected by relationships**: "Who ultimately owns bank X?" needs *X owned by Y* (doc 1) plus *Y owned by Z*
(doc 2). Neither chunk on its own is similar to the question. GraphRAG extracts entities and relations
into a knowledge graph once, at ingestion time. At query time it finds entry points, **follows edges** (multi-hop),
and hands the LLM the chunks that support those edges. Cost: an LLM call per chunk at ingestion time,
extraction errors, and a more complex system. That is why Phase 5 measures whether it actually helps.

### Repo layout: one Python package `backend/`, run with `python -m backend.x`
- One package means one import root, so `from backend.config import ...` works the same in scripts,
  tests, FastAPI and Docker. `python -m` runs modules from the repo root without fiddling with `sys.path`.
- `pyproject.toml` + **uv**: uv creates the venv, resolves and locks dependencies (`uv.lock`) and is fast. The lock
  file makes CI and Docker builds reproducible (same versions everywhere).
- `package = false`: this is an application, not a library to publish, so nothing is built or installed.

### Configuration: pydantic-settings + provider presets
- `Settings(BaseSettings)` reads env vars and `.env` and **validates types** at startup. A typo like
  `NEO4J_URI=` fails immediately, not halfway through ingestion. This is the
  [12-factor](https://12factor.net/config) principle: config in the environment, code identical across environments.
- **`SecretStr`** for keys and passwords: `print(settings)` and logs show `**********`. You must call
  `.get_secret_value()` explicitly, which makes a leak a conscious act.
- **Provider presets**: Gemini, Cerebras, Mistral and Groq all expose an **OpenAI-compatible API**, so one
  `openai` client handles all of them. Only `base_url`, key and model name differ, and those sit in a small dict.
  - *Why not LangChain / LiteLLM?* Another abstraction layer to explain and debug. With the OpenAI
    wire format as the common standard, a dict of four entries is enough. At BAWAG, switching to Azure OpenAI
    would be one more entry.
- **Chat and embeddings are configured separately**: Cerebras and Groq have no embeddings API, and you
  may want a fast chat model from one vendor and embeddings from another.
- **One key per provider** (`GEMINI_API_KEY`, `MISTRAL_API_KEY`, ...): switching chat provider must not
  break embeddings, which may still be on Gemini.
- `@lru_cache get_settings()`: one Settings object per process. In FastAPI it becomes a dependency that tests
  can override.

**Follow-up:** *"What if you change the embedding model later?"* All stored vectors become
incomparable with new query vectors: different models produce different vector spaces, even at the same
dimension. You must re-embed everything and recreate the vector index (which has a fixed dimension). In
production you'd version the index (`chunk_embedding_v2`), backfill, then switch reads over (blue/green).

### Embeddings: `gemini-embedding-001` at 768 dimensions
- Free tier, called over an API: no torch in the Docker image (sentence-transformers would add ~2 GB).
- The model natively returns 3072 dims, but it is trained with **Matryoshka Representation Learning**: the
  first *k* dimensions are themselves a usable embedding. Asking for 768 makes storage and index 4x smaller for
  a small quality loss. For a corpus of a few hundred chunks that's fine.
- Mistral's `mistral-embed` has a fixed size (1024), so we only send `dimensions` when the API supports it.
- *Trade-off vs a local model:* an API means network latency, a rate limit, and data leaving the machine. For
  a bank, that last point matters (see below).

### Chat model: `gemini-3.5-flash-lite` (default, overridable)
- Tested on 2026-11-05: the 2.5 models are closed to new users, and `gemini-3.8-flash` returned **503 (overloaded)**
  on the free tier. The flash-lite model supports **JSON-schema structured output**, which Phase 1 extraction needs.
- Lesson for production: free tiers are unreliable, so the client needs **timeouts, retries with backoff, and
  a rate limiter**. One test call hung for 45 s without a timeout.

**Follow-up:** *"Would you send bank documents to Gemini's free tier?"* No. The free tier may use the
data for training. Here the corpus is public Wikipedia text, so that's acceptable. In production: a
contract with no training on your data (Azure OpenAI in an EU region, matching BAWAG's stack), or a
self-hosted model. Mistral (EU company) is another option worth mentioning for data residency.

### Neo4j in Docker Compose
- Image `neo4j:5.26-community`: 5.26 is the **LTS** of the 5.x line. Native **vector indexes** exist since 5.11,
  so we need no separate vector database. Graph, vectors and fulltext all sit in one store and can be combined
  in one Cypher query.
- **Named volume** `neo4j_data`: data survives `docker compose down`; `down -v` wipes it.
- **Healthcheck** runs `cypher-shell 'RETURN 1'`, i.e. "the DB answers queries", not just "the port is
  open". Later, `depends_on: condition: service_healthy` makes the backend wait for that.
- Heap and page cache are capped so it runs on a laptop. *Page cache* = memory for caching the graph files
  on disk. *Heap* = memory for query execution.
- Ports: 7474 = HTTP / Browser UI, 7687 = **Bolt**, Neo4j's binary protocol used by drivers.
- `$$` in the compose file escapes `$` so the variable is expanded *inside* the container, not by compose.

### Async Neo4j driver
- The **driver** holds a connection pool: expensive to create, thread-safe, so it's created **once per process**
  (FastAPI lifespan in Phase 3). **Sessions** are cheap and short-lived: one per unit of work.
- *Why async?* FastAPI is async. During a Neo4j query or an LLM call the worker would otherwise block. With
  `await` the event loop serves other requests in the meantime. The work is **I/O-bound** (waiting on the
  network), and that is exactly what async is for. CPU-bound work wouldn't benefit.
- `verify_connectivity()` fails fast with a clear error, better than the first real query failing.

### Secrets hygiene
- `.env` is git-ignored, and only `.env.example` (no values) is committed. In CI and Kubernetes, secrets come from
  the Jenkins credential store or a K8s `Secret`, never from the repo.
- *If a key was committed by accident:* rotate it first, then purge it from history. Removing it in a new commit is
  not enough because it stays in git history.

### `backend/check.py` smoke test
Checks Neo4j, chat and embeddings in one command and reports *every* failure, not just the first.
Cheap insurance before an ingestion run that makes hundreds of API calls.
