from backend.answer import Edge
from backend.graph_retrieval import Subgraph, facts_for_answer, select_chunks
from backend.search import Hit


def graph_with_path() -> Subgraph:
    g = Subgraph(seeds=["ogb", "goldentree"])
    g.edges = {
        "ogb|OWNS|bawag": Edge("ogb|OWNS|bawag", "ÖGB", "OWNS", "BAWAG", ["bawag#001"]),
        "cerberus|OWNS|bawag": Edge("cerberus|OWNS|bawag", "Cerberus", "OWNS", "BAWAG",
                                    ["bawag#005", "cerberus#002"]),
        "x|RELATED_TO|y": Edge("x|RELATED_TO|y", "X", "RELATED_TO", "Y", ["other#000"]),
    }
    g.path_edges = ["ogb|OWNS|bawag", "cerberus|OWNS|bawag"]
    return g


def test_path_edges_keep_a_reserved_slot():
    reranked = [Hit("a#000", 9.0), Hit("b#000", 8.0), Hit("c#000", 7.0),
                Hit("cerberus#002", 1.0), Hit("bawag#005", 0.5), Hit("bawag#001", -2.0)]
    chosen = [h.id for h in select_chunks(reranked, graph_with_path(), k=4)]
    # a plain top-4 would be a, b, c, cerberus#002 and lose the ÖGB evidence (bawag#001)
    assert "bawag#001" in chosen
    assert "cerberus#002" in chosen  # best-scoring chunk of the Cerberus edge, not bawag#005
    assert "bawag#005" not in chosen
    assert chosen == ["a#000", "b#000", "cerberus#002", "bawag#001"]  # sorted by score


def test_reserved_slots_never_exceed_k():
    reranked = [Hit("bawag#001", 1.0), Hit("cerberus#002", 0.9)]
    assert len(select_chunks(reranked, graph_with_path(), k=1)) == 1


def test_facts_put_path_edges_first():
    facts = facts_for_answer(graph_with_path(), [Hit("other#000", 1.0)])
    assert [f.id for f in facts] == ["ogb|OWNS|bawag", "cerberus|OWNS|bawag", "x|RELATED_TO|y"]
