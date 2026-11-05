from backend.analytics import compute
from backend.resolution import EntityRecord, RelationRecord, ResolvedGraph


def graph(edges, places=()):
    keys = {k for s, _, t in edges for k in (s, t)}
    entities = {k: EntityRecord(k, types={"LOCATION" if k in places else "BANK": 1}) for k in keys}
    relations = {f"{s}|{t_}|{t}": RelationRecord(s, t_, t) for s, t_, t in edges}
    return ResolvedGraph(entities, relations, set())


def test_ultimate_owner_ranks_highest_and_group_size_counts_chains():
    g = graph([("holding", "OWNS", "bank"), ("bank", "OWNS", "sub1"), ("bank", "OWNS", "sub2")])
    m = compute(g)
    assert max(m, key=lambda k: m[k]["owner_rank"]) == "holding"  # indirect control counts
    assert m["holding"]["group_size"] == 3 and m["bank"]["group_size"] == 2
    assert m["sub1"]["group_size"] == 0


def test_acquired_edges_do_not_transfer_ownership_rank():
    g = graph([("bank", "OWNS", "sub1"), ("bank", "OWNS", "sub2"), ("buyer", "ACQUIRED", "bank")])
    m = compute(g)
    assert m["buyer"]["group_size"] == 0
    assert m["bank"]["owner_rank"] > m["buyer"]["owner_rank"]


def test_locations_are_not_bridges():
    g = graph([("a", "HEADQUARTERED_IN", "vienna"), ("b", "HEADQUARTERED_IN", "vienna"),
               ("a", "OWNS", "c")], places={"vienna"})
    m = compute(g)
    assert m["vienna"]["betweenness"] == 0.0 and m["vienna"]["community"] is None
