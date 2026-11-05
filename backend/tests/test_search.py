from backend.search import Hit, lucene_query, rrf


def ids(hits):
    return [h.id for h in hits]


def test_rrf_rewards_agreement_over_one_top_rank():
    vector = [Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.7)]
    fulltext = [Hit("d", 12.0), Hit("b", 9.0), Hit("a", 1.0)]
    fused = rrf(vector, fulltext)
    # a: 1/61 + 1/63 = .03227, b: 2/62 = .03226, d: 1/61 = .01639, c: 1/63 = .01587
    assert ids(fused) == ["a", "b", "d", "c"]


def test_rrf_ignores_raw_score_scales():
    # BM25 scores in the thousands must not dominate cosine scores below 1
    fused = rrf([Hit("x", 0.5), Hit("y", 0.4)], [Hit("y", 5000.0), Hit("x", 4000.0)])
    assert fused[0].score == fused[1].score


def test_rrf_merges_details():
    fused = rrf([Hit("a", 0.9, details={"vector": {"rank": 1}})],
                [Hit("a", 3.0, details={"fulltext": {"rank": 1}})])
    assert fused[0].details == {"vector": {"rank": 1}, "fulltext": {"rank": 1}}


def test_lucene_query_escapes_operators():
    assert lucene_query('Who owns "BAWAG" (AG)?') == r'Who owns \"BAWAG\" \(AG\)\?'
    assert lucene_query("a && b || c") == r"a \&\& b \|\| c"
    assert lucene_query("   ") == "*"
