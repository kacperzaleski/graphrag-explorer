import pytest

from eval import metrics as m


def test_recall_and_complete():
    gold = ["a#0", "b#0"]
    assert m.recall_at_k(gold, ["a#0", "x", "b#0"], k=2) == 0.5
    assert m.complete_at_k(gold, ["a#0", "x", "b#0"], k=2) == 0.0
    assert m.complete_at_k(gold, ["a#0", "x", "b#0"], k=3) == 1.0


def test_entity_recall_uses_aliases_and_word_boundaries():
    variants = {"bayerische landesbank": {"bayerische landesbank", "bayernlb"},
                "bayernlb": {"bayerische landesbank", "bayernlb"}}
    assert m.entity_recall(["Bayerische Landesbank", "BAWAG"], "BayernLB owned 46% of BAWAG.",
                           variants) == 1.0
    assert m.entity_recall(["Erste"], "Ernst & Young audited it.", {}) == 0.0
    assert m.entity_recall(["UniCredit"], "UniCreditBank", {}) == 0.0  # no partial-word match


def test_bootstrap_ci_contains_mean_and_shrinks_with_n():
    small = [1.0, 0.0] * 5
    large = [1.0, 0.0] * 500
    lo_s, hi_s = m.bootstrap_ci(small, n=2000)
    lo_l, hi_l = m.bootstrap_ci(large, n=2000)
    assert lo_s <= 0.5 <= hi_s and lo_l <= 0.5 <= hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s)  # more questions -> narrower interval


def test_mcnemar_counts_only_discordant_pairs():
    a = [True, True, False, False, True]
    b = [True, False, True, True, True]
    result = m.mcnemar(a, b)
    assert (result["only_a"], result["only_b"]) == (1, 2)
    assert result["p"] == pytest.approx(1.0)  # 1 vs 2 is no evidence at all
    strong = m.mcnemar([False] * 12, [True] * 12)
    assert strong["p"] < 0.001


def test_wilcoxon_identical_is_one():
    assert m.wilcoxon([0.5, 1.0], [0.5, 1.0]) == 1.0


def test_cohen_kappa():
    assert m.cohen_kappa(["c", "i", "c", "i"], ["c", "i", "c", "i"]) == 1.0
    # 50% agreement when chance agreement is also 50% -> kappa 0
    assert m.cohen_kappa(["c", "c", "i", "i"], ["c", "i", "c", "i"]) == pytest.approx(0.0)
