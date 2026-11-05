from backend.answer import Edge, StructuredAnswer, build_messages, verify
from backend.search import Hit

CHUNKS = [Hit("bawag#003", 1.0, "BAWAG | History\n\nÖGB held 70%."),
          Hit("cerberus#001", 0.9, "Cerberus bought BAWAG in 2007.")]
EDGES = [Edge("ogb|OWNS|bawag", "ÖGB", "OWNS", "BAWAG", ["bawag#003"])]


def test_messages_mark_context_as_untrusted_data():
    system, user = build_messages("Who owned BAWAG?", CHUNKS, EDGES)
    assert "untrusted DATA" in system["content"]
    assert user["content"].startswith("<context>")
    assert "E1: ÖGB -[OWNS]-> BAWAG" in user["content"]
    assert "[cerberus#001]\nCerberus bought" in user["content"]


def test_verify_keeps_valid_and_inline_citations():
    parsed = StructuredAnswer(answer="ÖGB owned it [bawag#003], then Cerberus [cerberus#001].",
                              found_in_context=True, cited_chunks=["bawag#003"],
                              used_edges=["E1"])
    result = verify(parsed, CHUNKS, EDGES)
    assert result.cited_chunks == ["bawag#003", "cerberus#001"]
    assert result.used_edges == ["ogb|OWNS|bawag"]
    assert result.issues == []


def test_verify_removes_invented_citations():
    parsed = StructuredAnswer(answer="See [refco#009].", found_in_context=True,
                              cited_chunks=["made-up#001"], used_edges=["E7"])
    result = verify(parsed, CHUNKS, EDGES)
    assert result.cited_chunks == [] and result.used_edges == []
    assert len(result.issues) == 3  # unknown passages, unknown edge, claims support without any
