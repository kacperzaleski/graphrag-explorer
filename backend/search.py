"""Search primitives over the Neo4j indexes, plus Reciprocal Rank Fusion.

Two complementary retrievers:
- vector (semantic): finds paraphrases ("who controls" ~ "majority shareholder"), weak on exact
  names, abbreviations and numbers ("PSK", "2006").
- fulltext (Lucene, BM25-like): exact tokens and rare names, blind to paraphrase.
Hybrid search runs both and fuses the rankings with RRF.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from neo4j import AsyncDriver

RRF_K = 60  # standard constant from the RRF paper; dampens the influence of the very top ranks


@dataclass
class Hit:
    id: str
    score: float
    text: str = ""
    details: dict = field(default_factory=dict)  # per-retriever rank/score, for explainability


async def vector_chunks(driver: AsyncDriver, vector: list[float], k: int) -> list[Hit]:
    records, _, _ = await driver.execute_query(
        "CALL db.index.vector.queryNodes('chunk_embedding', $k, $vector) YIELD node, score "
        "RETURN node.id AS id, node.text AS text, score",
        k=k, vector=vector)
    return [Hit(r["id"], r["score"], r["text"], {"vector": {"rank": i, "score": r["score"]}})
            for i, r in enumerate(records, 1)]


async def fulltext_chunks(driver: AsyncDriver, question: str, k: int) -> list[Hit]:
    records, _, _ = await driver.execute_query(
        "CALL db.index.fulltext.queryNodes('chunk_text', $query, {limit: $k}) YIELD node, score "
        "RETURN node.id AS id, node.text AS text, score",
        query=lucene_query(question), k=k)
    return [Hit(r["id"], r["score"], r["text"], {"fulltext": {"rank": i, "score": r["score"]}})
            for i, r in enumerate(records, 1)]


async def fulltext_entities(driver: AsyncDriver, query: str, k: int) -> list[Hit]:
    records, _, _ = await driver.execute_query(
        "CALL db.index.fulltext.queryNodes('entity_text', $query, {limit: $k}) YIELD node, score "
        "RETURN node.key AS id, node.name AS name, node.type AS type, "
        "[node.name] + node.aliases AS names, score",
        query=lucene_query(query), k=k)
    return [Hit(r["id"], r["score"], r["name"], {"type": r["type"], "names": r["names"]})
            for r in records]


def rrf(*rankings: list[Hit], k: int = RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank of d).

    Uses only ranks, never raw scores: cosine similarity (0..1) and BM25 (unbounded) are on
    different scales, so adding or averaging them would be meaningless. A document ranked well by
    both retrievers beats one ranked first by only one of them."""
    fused: dict[str, Hit] = {}
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            scores[hit.id] += 1.0 / (k + rank)
            if hit.id in fused:
                fused[hit.id].details.update(hit.details)
            else:
                fused[hit.id] = Hit(hit.id, 0.0, hit.text, dict(hit.details))
    for hit_id, hit in fused.items():
        hit.score = scores[hit_id]
    return sorted(fused.values(), key=lambda h: -h.score)


_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')  # what Lucene's escape() covers


def lucene_query(text: str) -> str:
    """User text -> safe Lucene query: escape operators so "S&P: AG?" can't break the query
    (or be used to inject query syntax). Terms are OR-ed, so ranking does the work."""
    return _LUCENE_SPECIAL.sub(r"\\\1", text).strip() or "*"
