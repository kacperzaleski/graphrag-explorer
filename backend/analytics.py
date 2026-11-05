"""Graph analytics for the explorer's insights panel, computed at ingest time with networkx.

Precomputed (not per request): the graph changes only on ingest, the metrics cost O(V*E), and the
UI can then sort and colour by them for free. The graph is small (~700 nodes) so networkx in
memory is fine; at scale this would be the Neo4j Graph Data Science library, run inside the DB.

Metrics, each answering an exploration question:
- owner_rank   "Who controls the most, directly or indirectly?" PageRank over ownership edges
               reversed (owned -> owner): an owner scores high if it owns things that are
               themselves important or that own a lot. Pure out-degree would miss indirect control.
- group_size   "How big is this corporate group?" Number of entities reachable via OWNS chains.
- betweenness  "Who connects otherwise separate parts?" Share of shortest paths that pass through
               the entity: bridges, brokers, the entities that link the Raiffeisen world to the
               Hypo world, for example.
- community    "Which entities belong together?" Louvain modularity communities: groups with many
               edges inside and few between them. Fixed seed, so ingest stays deterministic.

LOCATION nodes are left out of betweenness and communities: "both are in Vienna" would make
Vienna the top bridge and glue unrelated banks into one community (same reason as in retrieval).
"""

import networkx as nx

from backend.resolution import ResolvedGraph

# Only OWNS, not ACQUIRED: deals often concern a *part* of a company, and the extractor then
# writes e.g. "SIA ACQUIRED UniCredit" for "UniCredit sold its card business to SIA". With
# ACQUIRED included, that single wrong edge made SIA a top-5 "owner" with a 40-entity group.
# Real takeovers also produce an OWNS edge, so little is lost.
OWNERSHIP_TYPES = {"OWNS"}
SEED = 42


def compute(graph: ResolvedGraph) -> dict[str, dict]:
    """entity key -> {owner_rank, group_size, betweenness, community}"""
    keys = list(graph.entities)
    metrics: dict[str, dict] = {k: {} for k in keys}

    # control flows from owned to owner, so PageRank mass accumulates at (ultimate) owners
    control = nx.DiGraph()
    control.add_nodes_from(keys)
    owns = nx.DiGraph()
    owns.add_nodes_from(keys)
    for r in graph.relations.values():
        if r.type in OWNERSHIP_TYPES:
            control.add_edge(r.target, r.source)
            owns.add_edge(r.source, r.target)
    owner_rank = nx.pagerank(control, alpha=0.85)
    for k in keys:
        metrics[k]["owner_rank"] = owner_rank[k]
        metrics[k]["group_size"] = len(nx.descendants(owns, k))

    places = {k for k, e in graph.entities.items() if e.type == "LOCATION"}
    links = nx.Graph()
    links.add_nodes_from(k for k in keys if k not in places)
    for r in graph.relations.values():
        if r.source not in places and r.target not in places:
            links.add_edge(r.source, r.target)
    betweenness = nx.betweenness_centrality(links, normalized=True)
    communities = nx.community.louvain_communities(links, seed=SEED)
    # number communities by size (0 = largest) so ids are stable and meaningful in the UI
    for i, members in enumerate(sorted(communities, key=lambda c: (-len(c), min(c)))):
        for k in members:
            metrics[k]["community"] = i
    for k in keys:
        metrics[k]["betweenness"] = betweenness.get(k, 0.0)
        metrics[k].setdefault("community", None)  # locations
    return metrics
