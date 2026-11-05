"""Evaluation metrics and statistics (pure functions, unit-tested).

Retrieval: recall@k (share of gold passages retrieved) and complete@k (ALL gold passages
retrieved; a multi-hop question needs every hop's evidence, so partial recall is not enough).

Answers: an LLM judge score (correct 1 / partial 0.5 / wrong 0) and, as a deterministic
cross-check, entity recall (share of expected entities the answer mentions, alias-aware).

Statistics for n = 28 paired questions (both modes answer the SAME questions):
- bootstrap 95% CIs: resample questions with replacement, recompute the mean; the 2.5/97.5
  percentiles bound the mean. No normality assumption, works for bounded/skewed scores.
- McNemar's exact test for correct/incorrect pairs: only discordant questions (one mode right,
  the other wrong) carry information; under H0 they split 50/50 -> binomial test.
- Wilcoxon signed-rank for paired continuous scores (recall): ranks of the per-question
  differences, no normality assumption.
- Cohen's kappa for judge calibration: agreement corrected for the agreement expected by chance.
"""

import random
import re
from collections import Counter
from collections.abc import Sequence

from backend.resolution import normalize


def recall_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    return len(set(gold) & set(retrieved[:k])) / len(gold) if gold else 0.0


def complete_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    return float(set(gold) <= set(retrieved[:k]))


def entity_recall(expected: Sequence[str], answer: str, variants: dict[str, set[str]]) -> float:
    """variants: normalized key -> normalized surface forms (from the alias file), so "BayernLB"
    counts for an expected "Bayerische Landesbank"."""
    text = f" {normalize(answer)} "
    found = 0
    for entity in expected:
        forms = variants.get(normalize(entity), set()) | {normalize(entity)}
        if any(re.search(rf"\b{re.escape(f)}\b", text) for f in forms if f):
            found += 1
    return found / len(expected) if expected else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(values: Sequence[float], n: int = 10_000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(values)
    means = sorted(mean([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def paired_diff_ci(a: Sequence[float], b: Sequence[float], **kwargs) -> tuple[float, float]:
    """CI of mean(b - a), resampling *questions* (keeps the pairing)."""
    return bootstrap_ci([y - x for x, y in zip(a, b, strict=True)], **kwargs)


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact McNemar test. only_a: questions only A got right; only_b: only B got right."""
    from scipy.stats import binomtest

    only_a = sum(x and not y for x, y in zip(a_correct, b_correct, strict=True))
    only_b = sum(y and not x for x, y in zip(a_correct, b_correct, strict=True))
    n = only_a + only_b
    p = binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": p}


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import wilcoxon as scipy_wilcoxon

    if all(x == y for x, y in zip(a, b, strict=True)):
        return 1.0  # no differences at all: nothing to test
    return float(scipy_wilcoxon(a, b, zero_method="wilcox").pvalue)


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """1 = perfect agreement, 0 = no better than chance (given each rater's label frequencies)."""
    n = len(labels_a)
    observed = sum(x == y for x, y in zip(labels_a, labels_b, strict=True)) / n
    freq_a, freq_b = Counter(labels_a), Counter(labels_b)
    expected = sum(freq_a[c] * freq_b[c] for c in set(freq_a) | set(freq_b)) / n**2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
