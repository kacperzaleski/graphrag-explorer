"""Graph retrieval: find seed entities, expand the subgraph around them, gather supporting chunks.

1. Seeds: entities whose names/aliases are (mostly) contained in the question (fulltext search +
   name coverage check), else the entities most mentioned by the best hybrid-search chunks.
   No extra LLM call.
2. Paths: with 2+ seeds, shortest paths between each pair (answers "how is A connected to B").
   Max length 2 * hops: two seeds each reaching N hops can meet in the middle.
3. Expansion: breadth-first, N hops, max_nodes. Neighbours are added in order of relevance (how
   well the chunks behind the connecting edge match the question). Locations ("Austria", "New
   York City") and hubs (degree > hub_degree) may be included but are not expanded or pathed
   through: "both are headquartered in Vienna" is not a meaningful connection, and one hop via
   "Austria" would pull in half the graph. Measured on this corpus: the high-degree nodes are the
   main banks themselves, so the type rule does the real work and the degree limit is a safety net.
4. Supporting chunks: the chunks behind the subgraph edges + the hybrid-search chunks, reranked.
   Each path edge keeps a reserved slot for its best chunk: the chunk proving a middle hop is often
   not similar to the question itself, and a plain top-k would drop exactly the multi-hop evidence.
"""

import re
from dataclasses import dataclass, field

from neo4j import AsyncDriver

from backend.answer import Edge
from backend.resolution import normalize_text
from backend.search import Hit, fulltext_entities

SEED_LIMIT = 4
# share of an entity name's words that must occur in the question: "Cerberus Capital Management"
# matches, "Raiffeisen Capital Management" (2 of 3 words) does not. Better than a relative BM25
# score threshold, which rewards long names: "Austrian Trade Union Federation" (4 matching words)
# scored 6.9 and pushed "GoldenTree" (3.1, a perfect match) below a 50% cut-off.
SEED_NAME_COVERAGE = 0.75
MAX_FACTS = 30  # graph facts shown to the answering LLM


@dataclass
class Node:
    key: str
    name: str
    type: str
    degree: int = 0
    hop: int = 0  # distance from the nearest seed
    community: int | None = None


@dataclass
class Subgraph:
    seeds: list[str] = field(default_factory=list)
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)  # relation id -> edge (display names)
    edge_ends: dict[str, tuple[str, str]] = field(default_factory=dict)  # relation id -> keys
    path_edges: list[str] = field(default_factory=list)  # relation ids on seed-to-seed paths
    truncated: bool = False  # node limit reached


async def find_seeds(driver: AsyncDriver, question: str, fused_chunks: list[Hit]) -> list[Node]:
    hits = await fulltext_entities(driver, question, k=10)
    question_words = set(_words(question))
    keys = []
    if hits:
        keys = [h.id for h in hits
                if _name_coverage(h.details["names"], question_words) >= SEED_NAME_COVERAGE]
        keys = keys[:SEED_LIMIT] or [hits[0].id]  # nothing covered well: trust the best match
    if not keys:  # no name in the question: entities most mentioned by the best chunks
        records, _, _ = await driver.execute_query("""
            UNWIND $chunks AS ch
            MATCH (:Chunk {id: ch.id})-[:MENTIONS]->(e:Entity)
            RETURN e.key AS key, sum(ch.score) AS score ORDER BY score DESC LIMIT $n""",
            chunks=[{"id": h.id, "score": h.score} for h in fused_chunks[:5]], n=SEED_LIMIT)
        keys = [r["key"] for r in records]
    return await load_nodes(driver, keys)


async def expand(driver: AsyncDriver, seeds: list[Node], chunk_relevance: dict[str, float],
                 hops: int, max_nodes: int, hub_degree: int) -> Subgraph:
    graph = Subgraph(seeds=[s.key for s in seeds], nodes={s.key: s for s in seeds})
    if len(seeds) >= 2:
        await add_paths(driver, graph, max_len=2 * hops, hub_degree=hub_degree)

    frontier = [s for s in seeds]
    for hop in range(1, hops + 1):
        sources = [n.key for n in frontier
                   if n.key in graph.seeds or (n.degree <= hub_degree and n.type != "LOCATION")]
        if not sources:
            break
        records, _, _ = await driver.execute_query("""
            UNWIND $keys AS key
            MATCH (n:Entity {key: key})-[r:RELATES]-(m:Entity)
            RETURN r.id AS id, r.type AS type, r.chunk_ids AS chunk_ids,
                   startNode(r).key AS source, endNode(r).key AS target,
                   m.key AS key, m.name AS name, m.type AS node_type, m.degree AS degree,
                   m.community AS community""",
            keys=sources)

        def relevance(r) -> tuple[float, int]:
            # relevance to the question first; ties (e.g. no question, in the explorer) go to
            # better-connected neighbours, which are more likely to lead somewhere
            score = max((chunk_relevance.get(c, 0.0) for c in r["chunk_ids"]), default=0.0)
            return score, r["degree"] or 0

        next_frontier = []
        for r in sorted(records, key=relevance, reverse=True):
            if r["key"] not in graph.nodes:
                if len(graph.nodes) >= max_nodes:
                    graph.truncated = True
                    continue
                node = Node(r["key"], r["name"], r["node_type"], r["degree"], hop,
                            r["community"])
                graph.nodes[node.key] = node
                next_frontier.append(node)
            _add_edge(graph, r)
        frontier = next_frontier
    fill_names(graph)
    return graph


def select_chunks(reranked: list[Hit], graph: Subgraph, k: int) -> list[Hit]:
    """Top-k passages, with one reserved slot per path edge (its best-scoring supporting chunk)."""
    by_id = {h.id: h for h in reranked}
    chosen: dict[str, Hit] = {}
    for edge_id in graph.path_edges:
        supporting = [by_id[c] for c in graph.edges[edge_id].chunk_ids if c in by_id]
        if supporting and len(chosen) < k:
            best = max(supporting, key=lambda h: h.score)
            chosen.setdefault(best.id, best)
    for hit in reranked:
        if len(chosen) >= k:
            break
        chosen.setdefault(hit.id, hit)
    return sorted(chosen.values(), key=lambda h: -h.score)


def facts_for_answer(graph: Subgraph, passages: list[Hit]) -> list[Edge]:
    """Graph facts for the prompt: path edges first, then edges backed by a chosen passage."""
    passage_ids = {p.id for p in passages}
    ordered = list(dict.fromkeys([
        *graph.path_edges,
        *(eid for eid, e in graph.edges.items() if passage_ids & set(e.chunk_ids)),
        *graph.edges,
    ]))
    return [graph.edges[eid] for eid in ordered[:MAX_FACTS]]


async def add_paths(driver: AsyncDriver, graph: Subgraph, max_len: int, hub_degree: int) -> None:
    """Shortest paths between every pair of seeds; their edges are marked as path edges."""
    # a path length cannot be a query parameter; max_len is an int from our own settings
    records, _, _ = await driver.execute_query(f"""
        UNWIND $pairs AS pair
        MATCH (a:Entity {{key: pair[0]}}), (b:Entity {{key: pair[1]}})
        MATCH p = shortestPath((a)-[:RELATES*..{int(max_len)}]-(b))
        // hubs and places may be endpoints but not stepping stones: "both are headquartered in
        // New York" is not a meaningful connection
        WHERE all(n IN nodes(p)[1..-1] WHERE n.degree <= $hub_degree AND n.type <> 'LOCATION')
        UNWIND relationships(p) AS r
        RETURN DISTINCT r.id AS id, r.type AS type, r.chunk_ids AS chunk_ids,
               startNode(r).key AS source, endNode(r).key AS target""",
        pairs=[[a, b] for i, a in enumerate(graph.seeds) for b in graph.seeds[i + 1:]],
        hub_degree=hub_degree)
    for r in records:
        _add_edge(graph, r)
        graph.path_edges.append(r["id"])
    path_keys = {k for r in records for k in (r["source"], r["target"])} - set(graph.nodes)
    for node in await load_nodes(driver, list(path_keys)):
        graph.nodes[node.key] = node  # path nodes always count, even beyond max_nodes


def subgraph_to_dict(graph: Subgraph, used_edges: set[str] = frozenset()) -> dict:
    """JSON-ready subgraph for the API / UI: nodes and edges keyed by entity key."""
    return {
        "seeds": graph.seeds,
        "truncated": graph.truncated,
        "nodes": [{"key": n.key, "name": n.name, "type": n.type, "degree": n.degree, "hop": n.hop,
                   "community": n.community}
                  for n in graph.nodes.values()],
        "edges": [{"id": eid, "source": graph.edge_ends[eid][0], "target": graph.edge_ends[eid][1],
                   "type": e.type, "chunk_ids": e.chunk_ids, "on_path": eid in graph.path_edges,
                   "used": eid in used_edges}
                  for eid, e in graph.edges.items()],
    }


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_text(text))


def _name_coverage(names: list[str], question_words: set[str]) -> float:
    """Best share of words of any of the entity's names (or aliases) found in the question."""
    return max((sum(w in question_words for w in ws) / len(ws)
                for ws in (_words(n) for n in names) if ws), default=0.0)


def _add_edge(graph: Subgraph, r) -> None:
    if r["id"] in graph.edges:
        return
    graph.edges[r["id"]] = Edge(r["id"], r["source"], r["type"], r["target"], list(r["chunk_ids"]))
    graph.edge_ends[r["id"]] = (r["source"], r["target"])


def fill_names(graph: Subgraph) -> None:
    """Edges were built with keys; swap in display names, dropping edges to excluded nodes."""
    for eid, (source, target) in list(graph.edge_ends.items()):
        if source not in graph.nodes or target not in graph.nodes:
            del graph.edges[eid], graph.edge_ends[eid]
            continue
        edge = graph.edges[eid]
        edge.source, edge.target = graph.nodes[source].name, graph.nodes[target].name


async def load_nodes(driver: AsyncDriver, keys: list[str]) -> list[Node]:
    if not keys:
        return []
    records, _, _ = await driver.execute_query("""
        UNWIND $keys AS key MATCH (e:Entity {key: key})
        RETURN e.key AS key, e.name AS name, e.type AS type, e.degree AS degree,
               e.community AS community""", keys=keys)
    by_key = {r["key"]: Node(r["key"], r["name"], r["type"], r["degree"], 0, r["community"])
              for r in records}
    return [by_key[k] for k in keys if k in by_key]  # keep the caller's (relevance) order
