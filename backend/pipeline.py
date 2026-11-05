"""Question answering behind one entry point: `Pipeline.ask(question, mode="vector" | "graph")`.

vector (baseline, my earlier RAG demo's best setup): vector top-N chunks -> cross-encoder rerank ->
    top-k -> answer.
graph: hybrid entry search (vector + fulltext, RRF) -> seed entities -> paths between seeds +
    N-hop expansion -> chunks behind the subgraph edges + hybrid chunks -> rerank (path edges keep
    a reserved slot) -> answer with passages and graph facts.

`python -m backend.pipeline "question" [--mode vector] [--hops 2]` runs one question from the CLI.
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from neo4j import AsyncDriver

from backend import graph_retrieval as gr
from backend.answer import AnswerResult, generate_answer
from backend.cache import DiskCache
from backend.config import Settings, get_settings
from backend.db import create_driver
from backend.embeddings import Embedder
from backend.llm import LLM
from backend.reranker import make_reranker
from backend.search import Hit, fulltext_chunks, rrf, vector_chunks

Mode = Literal["vector", "graph"]


@dataclass
class AskResult:
    question: str
    mode: str
    answer: AnswerResult
    passages: list[Hit]  # what the LLM saw, in rank order
    subgraph: dict = field(default_factory=dict)  # nodes/edges for the UI ({} in vector mode)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Retrieval:
    passages: list[Hit]
    facts: list  # graph facts (Edge) for the prompt; [] in vector mode
    graph: gr.Subgraph | None
    timer: "_Timer"


class Pipeline:
    def __init__(self, settings: Settings, driver: AsyncDriver, embedder: Embedder, llm: LLM,
                 reranker):
        self.settings = settings
        self.driver = driver
        self.embedder = embedder
        self.llm = llm
        self.reranker = reranker

    @classmethod
    def from_settings(cls, settings: Settings, driver: AsyncDriver) -> "Pipeline":
        embedder = Embedder(settings.embed_endpoint(), cache=DiskCache("embeddings"),
                            texts_per_minute=settings.embed_texts_per_minute)
        llm = LLM(settings.chat_endpoint("answer"), rpm=settings.llm_rpm,
                  timeout=settings.llm_timeout, cache=DiskCache("llm"))
        return cls(settings, driver, embedder, llm, make_reranker(settings.reranker))

    async def retrieve(self, question: str, mode: Mode = "graph", hops: int | None = None,
                       reranker=None) -> Retrieval:
        """Everything before the LLM answer. Used alone by the evaluation's retrieval metrics
        and ablations, which therefore cost no LLM calls."""
        timer = _Timer()
        s = self.settings
        reranker = reranker or self.reranker
        vector = await self.embedder.embed_one(question)
        timer.lap("embed")

        if mode == "vector":
            candidates = await vector_chunks(self.driver, vector, s.candidate_k)
            timer.lap("search")
            passages = await reranker.rerank(question, candidates, s.top_k)
            timer.lap("rerank")
            return Retrieval(passages, [], None, timer)

        dense, sparse = await asyncio.gather(
            vector_chunks(self.driver, vector, s.candidate_k),
            fulltext_chunks(self.driver, question, s.candidate_k))
        fused = rrf(dense, sparse)[: s.candidate_k]
        timer.lap("search")

        seeds = await gr.find_seeds(self.driver, question, fused)
        graph = await gr.expand(self.driver, seeds, {h.id: h.score for h in fused},
                                hops if hops is not None else s.hops, s.max_nodes, s.hub_degree)
        timer.lap("expand")

        pool = await self._candidate_pool(fused, graph)
        reranked = await reranker.rerank(question, pool, len(pool))
        passages = gr.select_chunks(reranked, graph, s.top_k)
        timer.lap("rerank")
        return Retrieval(passages, gr.facts_for_answer(graph, passages), graph, timer)

    async def ask(self, question: str, mode: Mode = "graph", hops: int | None = None) -> AskResult:
        r = await self.retrieve(question, mode, hops)
        answer = await generate_answer(self.llm, question, r.passages, r.facts)
        r.timer.lap("answer")
        subgraph = gr.subgraph_to_dict(r.graph, set(answer.used_edges)) if r.graph else {}
        return AskResult(question, mode, answer, r.passages, subgraph, r.timer.laps)

    async def _candidate_pool(self, fused: list[Hit], graph: gr.Subgraph) -> list[Hit]:
        """Hybrid-search chunks + the chunks behind the subgraph edges (path edges first)."""
        edge_order = [*graph.path_edges, *(e for e in graph.edges if e not in graph.path_edges)]
        edge_chunks = list(dict.fromkeys(c for e in edge_order for c in graph.edges[e].chunk_ids))
        have = {h.id for h in fused}
        missing = [c for c in edge_chunks if c not in have][: self.settings.candidate_k]
        records, _, _ = await self.driver.execute_query(
            "UNWIND $ids AS id MATCH (c:Chunk {id: id}) RETURN c.id AS id, c.text AS text",
            ids=missing)
        return fused + [Hit(r["id"], 0.0, r["text"], {"graph": True}) for r in records]


class _Timer:
    def __init__(self):
        self.start = self.last = time.perf_counter()
        self.laps: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = round((now - self.last) * 1000, 1)
        self.laps["total"] = round((now - self.start) * 1000, 1)
        self.last = now


async def _main(question: str, mode: Mode, hops: int | None, as_json: bool) -> None:
    settings = get_settings()
    driver = create_driver(settings)
    try:
        result = await Pipeline.from_settings(settings, driver).ask(question, mode, hops)
    finally:
        await driver.close()
    if as_json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))
        return
    a = result.answer
    print(f"\n{a.answer}\n")
    print(f"citations: {a.cited_chunks}")
    if a.issues:
        print(f"issues: {a.issues}")
    if result.subgraph:
        g = result.subgraph
        print(f"seeds: {g['seeds']} | subgraph: {len(g['nodes'])} nodes, {len(g['edges'])} edges"
              f"{' (truncated)' if g['truncated'] else ''}")
        name = {n["key"]: n["name"] for n in g["nodes"]}
        for e in g["edges"]:
            if e["on_path"] or e["used"]:
                flags = [f for f, on in (("path", e["on_path"]), ("used", e["used"])) if on]
                print(f"  {name[e['source']]} -[{e['type']}]-> {name[e['target']]}  "
                      f"({', '.join(flags)})")
    print(f"passages: {[p.id for p in result.passages]}")
    print(f"timings ms: {result.timings_ms} | model {a.model}{' (cached)' if a.cached else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["vector", "graph"], default="graph")
    parser.add_argument("--hops", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main(args.question, args.mode, args.hops, args.json))
