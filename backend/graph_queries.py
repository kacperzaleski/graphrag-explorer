"""Read queries for the graph explorer (no LLM involved).

Wrapped in one class so the API depends on an interface, not on Neo4j: tests inject a fake
repository and run without a database.
"""

import re

from neo4j import AsyncDriver

from backend import graph_retrieval as gr
from backend.search import lucene_query


class GraphRepository:
    def __init__(self, driver: AsyncDriver, hub_degree: int):
        self.driver = driver
        self.hub_degree = hub_degree

    async def search_entities(self, q: str, limit: int) -> list[dict]:
        """Type-ahead search over entity names and aliases: "bawa" finds "BAWAG Group".

        Lucene gives prefix queries ("bawa*") a constant score, so all hits tie: ties are broken
        by degree, i.e. better-connected (more central) entities first."""
        records, _, _ = await self.driver.execute_query("""
            CALL db.index.fulltext.queryNodes('entity_text', $query, {limit: $pool})
            YIELD node, score
            RETURN node.key AS key, node.name AS name, node.type AS type,
                   node.degree AS degree, node.description AS description, score
            ORDER BY score DESC, node.degree DESC LIMIT $limit""",
            query=_prefix_query(q), pool=max(limit * 5, 50), limit=limit)
        return [dict(r) for r in records]

    async def neighbors(self, key: str, hops: int, limit: int) -> dict | None:
        """Subgraph around one entity (expand-on-click). Same BFS + hub rules as /ask."""
        seeds = await gr.load_nodes(self.driver, [key])
        if not seeds:
            return None
        graph = await gr.expand(self.driver, seeds, {}, hops, limit, self.hub_degree)
        return gr.subgraph_to_dict(graph)

    async def path(self, source: str, target: str, max_hops: int) -> dict | None:
        """Shortest connection between two entities, for "how is A connected to B"."""
        seeds = await gr.load_nodes(self.driver, [source, target])
        if len(seeds) < 2:
            return None
        graph = gr.Subgraph(seeds=[s.key for s in seeds], nodes={s.key: s for s in seeds})
        await gr.add_paths(self.driver, graph, max_hops, self.hub_degree)
        gr.fill_names(graph)
        return gr.subgraph_to_dict(graph)

    async def node(self, key: str, chunk_limit: int = 20) -> dict | None:
        """Entity card: details, relations with their evidence, and the source passages."""
        records, _, _ = await self.driver.execute_query("""
            MATCH (e:Entity {key: $key})
            CALL (e) {
                MATCH (e)-[r:RELATES]-(m:Entity)
                RETURN collect({id: r.id, type: r.type, outgoing: startNode(r) = e,
                                other_key: m.key, other_name: m.name, other_type: m.type,
                                chunk_ids: r.chunk_ids, evidence: r.evidence}) AS relations
            }
            CALL (e) {
                MATCH (c:Chunk)-[:MENTIONS]->(e)
                WITH c ORDER BY c.id LIMIT $chunk_limit
                RETURN collect({id: c.id, doc: c.doc, title: c.title, section: c.section,
                                url: c.url, text: c.text}) AS chunks
            }
            RETURN e.key AS key, e.name AS name, e.type AS type, e.description AS description,
                   e.aliases AS aliases, e.degree AS degree, e.mention_count AS mention_count,
                   relations, chunks""", key=key, chunk_limit=chunk_limit)
        return dict(records[0]) if records else None

    async def chunks(self, ids: list[str]) -> list[dict]:
        records, _, _ = await self.driver.execute_query("""
            UNWIND $ids AS id MATCH (c:Chunk {id: id})
            RETURN c.id AS id, c.doc AS doc, c.title AS title, c.section AS section,
                   c.url AS url, c.text AS text""", ids=ids)
        return [dict(r) for r in records]

    async def stats(self) -> dict:
        records, _, _ = await self.driver.execute_query("""
            MATCH (e:Entity) WITH e.type AS type, count(*) AS n ORDER BY n DESC
            RETURN collect({type: type, count: n}) AS entity_types""")
        relation_types, _, _ = await self.driver.execute_query("""
            MATCH ()-[r:RELATES]->() WITH r.type AS type, count(*) AS n ORDER BY n DESC
            RETURN collect({type: type, count: n}) AS relation_types""")
        return {**dict(records[0]), **dict(relation_types[0])}


def _prefix_query(q: str) -> str:
    """Every typed word also matches as a prefix ("bawa" -> "bawa*") for type-ahead search."""
    words = re.findall(r"\w+", q)
    if not words:
        return lucene_query(q)
    return " ".join(f"{lucene_query(w)}*" for w in words)
