"""Everything that writes the knowledge graph to Neo4j.

Graph schema:
  (:Chunk {id, doc, title, section, text, url, embedding})
  (:Entity {key, name, type, description, aliases, search_text, mention_count})
  (:Chunk)-[:MENTIONS]->(:Entity)
  (:Entity)-[:RELATES {id, type, chunk_ids, evidence}]->(:Entity)

Relation types are a property of one Neo4j relationship type (RELATES) instead of separate types
(:OWNS, :SUPERVISES...): Cypher cannot parameterize relationship types, and one type keeps the
traversal queries simple ("expand N hops over RELATES"). The cost: filtering by type is a property
check instead of an index-backed relationship type.

Idempotency: every write is a MERGE on a unique key, so running ingestion twice changes nothing.
Sync: after writing, anything that is no longer produced by the current corpus + extraction +
alias file is deleted, so the database always mirrors the last run (e.g. after merging two
entities via aliases.yaml, the old separate node disappears).
"""

from neo4j import AsyncDriver

from backend.chunking import Chunk
from backend.resolution import ResolvedGraph

BATCH = 500  # rows per UNWIND: one round trip per batch instead of one per row

CONSTRAINTS = [
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
]
FULLTEXT_INDEXES = [
    # names + aliases of entities, for "search a node by name" and hybrid entry-point search
    "CREATE FULLTEXT INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON EACH [e.search_text]",
    "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
]


async def create_schema(driver: AsyncDriver, embed_dim: int) -> None:
    for statement in CONSTRAINTS + FULLTEXT_INDEXES:
        await driver.execute_query(statement)
    records, _, _ = await driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name, options WHERE name = 'chunk_embedding' RETURN options"
    )
    if records:
        existing = records[0]["options"]["indexConfig"]["vector.dimensions"]
        if existing != embed_dim:
            raise RuntimeError(
                f"Vector index has {existing} dims but the embedding model has {embed_dim}. "
                "Changing the embedding model needs a fresh index: DROP INDEX chunk_embedding "
                "(or `docker compose down -v`) and re-ingest."
            )
        return
    # index options cannot be query parameters; embed_dim is an int from our own config
    await driver.execute_query(
        "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON c.embedding "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(embed_dim)}, "
        "`vector.similarity_function`: 'cosine'}}"
    )


async def write_chunks(driver: AsyncDriver, chunks: list[Chunk], embeddings: list[list[float]],
                       urls: dict[str, str]) -> None:
    rows = [{"id": c.id, "doc": c.doc, "title": c.title, "section": c.section, "text": c.text,
             "url": urls.get(c.doc), "embedding": e}
            for c, e in zip(chunks, embeddings, strict=True)]
    await _unwind(driver, rows, """
        MERGE (c:Chunk {id: row.id})
        SET c.doc = row.doc, c.title = row.title, c.section = row.section, c.text = row.text,
            c.url = row.url
        WITH c, row
        // stores the vector as a compact float array, the type the vector index expects
        CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
    """)


async def write_graph(driver: AsyncDriver, graph: ResolvedGraph) -> None:
    entities = [{"key": e.key, "name": e.name, "type": e.type, "description": e.description,
                 "aliases": sorted(e.names), "mention_count": len(e.chunk_ids),
                 "search_text": " | ".join([e.name, *sorted(set(e.names) - {e.name})])}
                for e in graph.entities.values()]
    await _unwind(driver, entities, """
        MERGE (e:Entity {key: row.key})
        SET e.name = row.name, e.type = row.type, e.description = row.description,
            e.aliases = row.aliases, e.mention_count = row.mention_count,
            e.search_text = row.search_text
    """)

    mentions = [{"chunk_id": c, "key": k} for c, k in sorted(graph.mentions)]
    await _unwind(driver, mentions, """
        MATCH (c:Chunk {id: row.chunk_id}), (e:Entity {key: row.key})
        MERGE (c)-[:MENTIONS]->(e)
    """)

    relations = [{"id": r.id, "source": r.source, "target": r.target, "type": r.type,
                  "chunk_ids": sorted(r.evidence), "evidence": [r.evidence[c] for c in
                                                                sorted(r.evidence)]}
                 for r in graph.relations.values()]
    await _unwind(driver, relations, """
        MATCH (s:Entity {key: row.source}), (t:Entity {key: row.target})
        MERGE (s)-[r:RELATES {type: row.type}]->(t)
        SET r.id = row.id, r.chunk_ids = row.chunk_ids, r.evidence = row.evidence
    """)


async def write_degrees(driver: AsyncDriver) -> None:
    """Precompute each entity's number of RELATES edges: retrieval uses it to recognise hubs
    without counting edges at query time (after prune, so deleted edges don't count)."""
    await driver.execute_query(
        "MATCH (e:Entity) SET e.degree = COUNT { (e)-[:RELATES]-() }")


async def prune(driver: AsyncDriver, chunks: list[Chunk], graph: ResolvedGraph) -> dict[str, int]:
    """Delete what the current run no longer produces. Returns counts per kind."""
    params = {
        "chunk_ids": [c.id for c in chunks],
        "keys": list(graph.entities),
        "relation_ids": list(graph.relations),
        "mention_ids": [f"{c}|{k}" for c, k in graph.mentions],
    }
    queries = {
        "relations": "MATCH ()-[r:RELATES]->() WHERE NOT r.id IN $relation_ids "
                     "DELETE r RETURN count(*) AS n",
        "mentions": "MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity) "
                    "WHERE NOT (c.id + '|' + e.key) IN $mention_ids DELETE m RETURN count(*) AS n",
        "entities": "MATCH (e:Entity) WHERE NOT e.key IN $keys DETACH DELETE e "
                    "RETURN count(*) AS n",
        "chunks": "MATCH (c:Chunk) WHERE NOT c.id IN $chunk_ids DETACH DELETE c "
                  "RETURN count(*) AS n",
    }
    deleted = {}
    for kind, query in queries.items():
        records, _, _ = await driver.execute_query(query, params)
        deleted[kind] = records[0]["n"]
    return deleted


async def stats(driver: AsyncDriver) -> dict[str, int]:
    records, _, _ = await driver.execute_query("""
        CALL () { MATCH (c:Chunk) RETURN count(c) AS chunks }
        CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
        CALL () { MATCH ()-[r:RELATES]->() RETURN count(r) AS relations }
        CALL () { MATCH ()-[m:MENTIONS]->() RETURN count(m) AS mentions }
        RETURN chunks, entities, relations, mentions
    """)
    return dict(records[0])


async def _unwind(driver: AsyncDriver, rows: list[dict], body: str) -> None:
    for start in range(0, len(rows), BATCH):
        await driver.execute_query(f"UNWIND $rows AS row {body}",
                                   {"rows": rows[start:start + BATCH]})
