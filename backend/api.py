"""FastAPI app: `uvicorn backend.api:app --reload --no-access-log`, docs at localhost:8000/docs.
(--no-access-log: the middleware below already writes one structured line per request.)

Design:
- Lifespan handler: expensive, shared objects (Neo4j driver = connection pool, LLM client,
  reranker model) are created once at startup and closed at shutdown, never per request.
- Dependency injection: endpoints receive the pipeline / repository via `Depends(...)`. Tests swap
  them with fakes through `app.dependency_overrides`, without a database or LLM.
- Liveness vs readiness: /health is a cheap "process is up"; /ready checks Neo4j. Kubernetes
  restarts a pod that fails liveness, but only stops routing traffic to one that fails readiness.
- Middleware: a request id per request (from X-Request-ID or new), returned in the response
  header and attached to every log line; one access-log line with status and duration.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from neo4j import AsyncDriver
from neo4j.exceptions import ServiceUnavailable
from openai import APIError

from backend.config import get_settings
from backend.db import create_driver, server_info
from backend.graph_queries import GraphRepository
from backend.logging_setup import request_id, setup_logging
from backend.pipeline import Pipeline
from backend.schemas import (
    MAX_HOPS,
    AskRequest,
    AskResponse,
    Citation,
    EntityHit,
    Health,
    NodeDetails,
    Subgraph,
    Usage,
)

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    driver = create_driver(settings)
    app.state.driver = driver
    app.state.pipeline = Pipeline.from_settings(settings, driver)
    app.state.graph = GraphRepository(driver, settings.hub_degree)
    # bounds concurrent LLM-backed requests: protects the provider quota and our memory
    app.state.ask_slots = asyncio.Semaphore(settings.max_concurrent_asks)
    # load the reranker now (in a thread: ~7 s of CPU work), not on the first user's request
    await asyncio.to_thread(app.state.pipeline.reranker.warm_up)
    log.info("startup complete", extra={"fields": {"llm": app.state.pipeline.llm.endpoint.model}})
    yield
    await driver.close()


app = FastAPI(title="GraphRAG Explorer API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,  # the React dev server / frontend origin only
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = request_id.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    log.info("request", extra={"fields": {
        "request_id": rid, "method": request.method, "path": request.url.path,
        "status": response.status_code, "ms": round((time.perf_counter() - start) * 1000, 1)}})
    return response


# --- dependencies ------------------------------------------------------------------------------

def get_driver(request: Request) -> AsyncDriver:
    return request.app.state.driver


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def get_graph(request: Request) -> GraphRepository:
    return request.app.state.graph


def get_ask_slots(request: Request) -> asyncio.Semaphore:
    return request.app.state.ask_slots


PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]
GraphDep = Annotated[GraphRepository, Depends(get_graph)]
NodeKey = Annotated[str, Query(min_length=1, max_length=200, description="entity key")]


# --- endpoints ---------------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Liveness: the process is up and serving. Deliberately no dependency checks."""
    return {"status": "ok"}


@app.get("/ready", response_model=Health)
async def ready(driver: Annotated[AsyncDriver, Depends(get_driver)], pipeline: PipelineDep):
    """Readiness: can we serve real traffic (Neo4j reachable)?"""
    endpoint = pipeline.llm.endpoint
    try:
        neo4j = await server_info(driver)
        status = "ok"
    except Exception as exc:  # report, don't crash: the probe needs an answer
        neo4j, status = f"unavailable: {type(exc).__name__}", "degraded"
    body = Health(status=status, neo4j=neo4j, llm=f"{endpoint.provider}/{endpoint.model}")
    if status != "ok":
        raise HTTPException(503, detail=body.model_dump())
    return body


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, pipeline: PipelineDep, graph: GraphDep,
              slots: Annotated[asyncio.Semaphore, Depends(get_ask_slots)]):
    async with slots:
        try:
            result = await pipeline.ask(body.question, body.mode, body.hops)
        except ServiceUnavailable as exc:
            raise HTTPException(503, "graph database unavailable") from exc
        except APIError as exc:  # quota, outage, timeout after all retries and fallbacks
            log.warning("llm failure", extra={"fields": {"error": str(exc)[:200]}})
            raise HTTPException(503, "language model unavailable, try again later") from exc
    a = result.answer
    passages = {p["id"]: p for p in await graph.chunks(a.cited_chunks)}
    return AskResponse(
        question=result.question, mode=result.mode, answer=a.answer,
        found_in_context=a.found_in_context,
        citations=[Citation(chunk_id=c, **{k: v for k, v in passages[c].items() if k != "id"})
                   for c in a.cited_chunks if c in passages],
        passages=[p.id for p in result.passages], issues=a.issues,
        subgraph=Subgraph(**result.subgraph) if result.subgraph else Subgraph(),
        timings_ms=result.timings_ms,
        usage=Usage(model=a.model, cached=a.cached, prompt_tokens=a.prompt_tokens,
                    completion_tokens=a.completion_tokens),
    )


@app.get("/graph/search", response_model=list[EntityHit])
async def search(graph: GraphDep, q: Annotated[str, Query(min_length=1, max_length=100)],
                 limit: Annotated[int, Query(ge=1, le=50)] = 10):
    """Type-ahead entity search over names and aliases."""
    return await graph.search_entities(q, limit)


@app.get("/graph/neighbors", response_model=Subgraph)
async def neighbors(graph: GraphDep, node_id: NodeKey,
                    hops: Annotated[int, Query(ge=1, le=MAX_HOPS)] = 1,
                    limit: Annotated[int, Query(ge=1, le=200)] = 30):
    """Subgraph around one entity (the explorer's expand-on-click). `truncated` = limit hit."""
    result = await graph.neighbors(node_id, hops, limit)
    if result is None:
        raise HTTPException(404, f"entity '{node_id}' not found")
    return result


@app.get("/graph/path", response_model=Subgraph)
async def path(graph: GraphDep, source: NodeKey, target: NodeKey,
               max_hops: Annotated[int, Query(ge=1, le=6)] = 4):
    """Shortest connection(s) between two entities, without the LLM."""
    if source == target:
        raise HTTPException(422, "source and target must differ")
    result = await graph.path(source, target, max_hops)
    if result is None:
        raise HTTPException(404, "source or target entity not found")
    return result


@app.get("/graph/node/{node_id}", response_model=NodeDetails)
async def node(graph: GraphDep, node_id: str):
    """Entity card: description, aliases, relations with evidence, source passages."""
    result = await graph.node(node_id)
    if result is None:
        raise HTTPException(404, f"entity '{node_id}' not found")
    return result


@app.get("/graph/stats")
async def stats(graph: GraphDep) -> dict:
    """Counts per entity and relation type (legend and filters in the UI)."""
    return await graph.stats()
