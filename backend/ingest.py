"""Build the knowledge graph: `python -m backend.ingest`.

documents -> chunks -> embeddings -> LLM extraction -> entity resolution -> Neo4j

Idempotent: chunk ids are deterministic, writes are MERGEs, and LLM/embedding results are cached
on disk, so a second run makes zero API calls and leaves the graph unchanged.

Options:
  --limit-docs N       only the first N documents (smoke tests / CI)
  --alias-candidates   print groups of similar entity names to review for data/aliases.yaml
"""

import argparse
import asyncio
import json
import logging
import time

from backend import analytics, graph_store
from backend.cache import DiskCache
from backend.chunking import Chunk, chunk_document
from backend.config import REPO_ROOT, Settings, get_settings
from backend.db import create_driver
from backend.embeddings import Embedder
from backend.enrichment import enrich
from backend.extraction import ChunkExtraction, extract
from backend.llm import LLM
from backend.resolution import Resolver, alias_candidates, resolve

RAW_DIR = REPO_ROOT / "data" / "raw"
CONCURRENCY = 4  # parallel extraction requests; the rate limiter still spaces their starts

log = logging.getLogger("ingest")


def load_chunks(settings: Settings, limit_docs: int | None = None) -> tuple[list[Chunk], dict]:
    paths = sorted(RAW_DIR.glob("*.md"))[:limit_docs]
    if not paths:
        raise SystemExit(f"No documents in {RAW_DIR}: run `python scripts/fetch_data.py` first")
    chunks = [
        c
        for p in paths
        for c in chunk_document(p.stem, p.read_text(encoding="utf-8"), settings.chunk_tokens,
                                settings.chunk_overlap_tokens)
    ]
    sources_file = RAW_DIR / "sources.json"
    sources = json.loads(sources_file.read_text()) if sources_file.exists() else []
    urls = {s["doc"]: s["url"] for s in sources}
    return chunks, urls


async def extract_all(llm: LLM, chunks: list[Chunk]) -> list[ChunkExtraction]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    done = 0

    async def one(chunk: Chunk) -> ChunkExtraction:
        nonlocal done
        async with semaphore:
            try:
                result = await extract(llm, chunk)
            except Exception as exc:  # keep going: every success is cached for the next run
                log.error("chunk %s: extraction failed (%s)", chunk.id, str(exc)[:150])
                result = ChunkExtraction(chunk.id, dropped={"request_failed": 1})
        done += 1
        if done % 10 == 0 or done == len(chunks):
            print(f"  extracted {done}/{len(chunks)} chunks "
                  f"({llm.usage.calls} API calls, {llm.usage.cache_hits} cached)", flush=True)
        return result

    return await asyncio.gather(*(one(c) for c in chunks))


async def main(limit_docs: int | None, show_alias_candidates: bool) -> None:
    settings = get_settings()
    start = time.perf_counter()

    chunks, urls = load_chunks(settings, limit_docs)
    print(f"{len({c.doc for c in chunks})} documents -> {len(chunks)} chunks")

    embedder = Embedder(settings.embed_endpoint(), cache=DiskCache("embeddings"),
                        texts_per_minute=settings.embed_texts_per_minute)
    embeddings = await embedder.embed([c.text for c in chunks])
    print(f"embedded {len(chunks)} chunks ({embedder.calls} API calls)")

    llm = LLM(settings.chat_endpoint("extract"), rpm=settings.llm_rpm,
              timeout=settings.llm_timeout, cache=DiskCache("llm"))
    print(f"extracting with {llm.endpoint.provider}/{llm.endpoint.model} "
          f"(fallbacks: {', '.join(llm.endpoint.fallback_models) or 'none'})")
    extractions = await extract_all(llm, chunks)
    failed = [ex.chunk_id for ex in extractions if "request_failed" in ex.dropped]
    if failed:
        # Writing now would store a partial graph, and pruning would delete good data. Successful
        # extractions are cached, so simply re-running resumes where this run stopped.
        raise SystemExit(f"{len(failed)} chunks failed (quota/outage?). Nothing written to Neo4j; "
                         f"re-run later to resume ({llm.usage.calls} new results are cached).")

    graph = resolve(extractions, Resolver.from_file(), {c.doc: c.title for c in chunks})
    enrichment = await enrich(llm, graph, {c.id: c for c in chunks})
    print(f"enrichment: {enrichment}")
    if show_alias_candidates:
        print_alias_candidates(graph.entities)

    driver = create_driver(settings)
    try:
        await graph_store.create_schema(driver, settings.embed_dim())
        await graph_store.write_chunks(driver, chunks, embeddings, urls)
        await graph_store.write_graph(driver, graph)
        deleted = await graph_store.prune(driver, chunks, graph)
        await graph_store.write_degrees(driver)
        await graph_store.write_metrics(driver, analytics.compute(graph))
        counts = await graph_store.stats(driver)
    finally:
        await driver.close()

    dropped: dict[str, int] = {}
    for ex in extractions:
        for reason, n in ex.dropped.items():
            dropped[reason] = dropped.get(reason, 0) + n
    u = llm.usage
    print("\n=== graph ===")
    print(f"chunks {counts['chunks']} | entities {counts['entities']} | "
          f"relations {counts['relations']} | mentions {counts['mentions']}")
    print(f"pruned (no longer produced): {deleted}")
    print("=== extraction ===")
    print(f"API calls {u.calls} | cache hits {u.cache_hits} | "
          f"tokens in {u.prompt_tokens:,} / out {u.completion_tokens:,}")
    print(f"dropped items: {dropped or 'none'}")
    print(f"total time {time.perf_counter() - start:.0f}s")


def print_alias_candidates(entities) -> None:
    print("\n=== alias candidates (review, then add true variants to data/aliases.yaml) ===")
    for group in alias_candidates(entities):
        print(" / ".join(f"{r.name} [{r.type}, {len(r.chunk_ids)}x]" for r in group))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--alias-candidates", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main(args.limit_docs, args.alias_candidates))
