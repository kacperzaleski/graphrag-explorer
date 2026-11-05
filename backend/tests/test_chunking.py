from backend.chunking import approx_tokens, chunk_document

DOC = """# BAWAG

BAWAG is a bank in Austria. It was founded in 1922.

## History

### Merger
In 2005 BAWAG merged with PSK. The new bank was called BAWAG PSK.
Another paragraph about the merger.

## Ownership
Cerberus bought the bank in 2007.
"""


def test_sections_titles_and_ids():
    chunks = chunk_document("bawag", DOC)
    assert [c.id for c in chunks] == ["bawag#000", "bawag#001", "bawag#002"]
    assert [c.section for c in chunks] == ["", "History > Merger", "Ownership"]
    assert all(c.title == "BAWAG" for c in chunks)
    # the header makes "the bank" resolvable for the LLM and the embedding
    assert chunks[2].text.startswith("BAWAG | Ownership\n\nCerberus")


def test_paragraphs_kept_as_lines():
    merger = chunk_document("bawag", DOC)[1]
    assert merger.body == ("In 2005 BAWAG merged with PSK. The new bank was called BAWAG PSK.\n"
                           "Another paragraph about the merger.")


def test_budget_and_overlap():
    sentences = [f"Sentence number {i} says that bank {i} owns bank {i + 1}." for i in range(60)]
    doc = "# Big\n\n## Facts\n" + " ".join(sentences)
    chunks = chunk_document("big", doc, max_tokens=120, overlap_tokens=30)
    assert len(chunks) > 3
    # small slack: token counts are summed per sentence, not over the joined text
    assert all(approx_tokens(c.text) <= 130 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        last_sentence = prev.body.split(". ")[-1]
        assert last_sentence in nxt.body  # the border sentence is repeated


def test_deterministic():
    assert chunk_document("bawag", DOC) == chunk_document("bawag", DOC)
