from backend.enrichment import Observation, Qualifier, _apply, _grounded
from backend.resolution import RelationRecord

TEXT = "The Bayerische Landesbank bought the shares in 1995 and increased its stake to 46%."


def test_values_must_occur_in_the_passage():
    q = Qualifier(label="R1", start_year=1995, end_year=2004, stake_percent=46)
    # 2004 is not in this passage: dropped (it may be true, but this passage doesn't say it)
    assert _grounded(q, TEXT) == Observation(1995, None, 46)


def test_decimal_stakes_are_matched():
    text = "bought 74.82% of PSK"
    assert _grounded(Qualifier(label="R1", stake_percent=74.82), text).stake_percent == 74.82
    assert _grounded(Qualifier(label="R1", stake_percent=74.8), text).stake_percent is None


def test_apply_merges_several_statements():
    rel = RelationRecord("bayernlb", "OWNS", "bawag group")
    _apply(rel, [Observation(1995, None, 46.0), Observation(None, 2004, None),
                 Observation(1998, None, 50.0)])
    assert (rel.start_year, rel.end_year, rel.stake_percent) == (1995, 2004, 50.0)
