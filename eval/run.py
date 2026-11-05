"""Compare vector vs graph mode: `uv run python -m eval.run [--limit N] [--retrieval-only]`.

1. Retrieval metrics for the main configs and ablations (reranker off, hops 1/3). These need no
   LLM call (embeddings are cached), so ablations are nearly free.
2. Answers + LLM judge + entity recall for the two main configs (LLM calls are cached on disk:
   a re-run costs nothing and still reports the original latency and tokens).
3. Statistics (bootstrap CIs, McNemar, Wilcoxon), judge calibration (Cohen's kappa against
   eval/manual_labels.jsonl, if present), per-type breakdown -> eval/results.md.
Per-question records go to eval/runs/latest.jsonl for error analysis.
"""

import argparse
import asyncio
import json
import random
import statistics
from dataclasses import dataclass, field

import yaml

from backend.cache import DiskCache
from backend.config import REPO_ROOT, get_settings
from backend.db import create_driver
from backend.llm import LLM
from backend.pipeline import Pipeline
from backend.reranker import NoReranker
from backend.resolution import normalize
from eval import metrics as m
from eval.judge import SCORES, judge

EVAL_DIR = REPO_ROOT / "eval"
K = 8  # = top_k passages given to the LLM


@dataclass(frozen=True)
class Config:
    name: str
    mode: str
    hops: int | None = None
    rerank: bool = True
    answer: bool = False  # main configs: also answer + judge


CONFIGS = [
    Config("vector", "vector", answer=True),  # baseline: my earlier RAG demo's best setup
    Config("graph", "graph", hops=2, answer=True),
    Config("vector, no rerank", "vector", rerank=False),
    Config("graph, no rerank", "graph", hops=2, rerank=False),
    Config("graph, 1 hop", "graph", hops=1),
    Config("graph, 3 hops", "graph", hops=3),
]


@dataclass
class Record:
    qid: str
    type: str
    config: str
    recall: float
    complete: float
    retrieval_ms: float
    passages: list[str]
    answer: str | None = None
    verdict: str | None = None
    judge_reasoning: str | None = None
    entity_recall: float | None = None
    llm_seconds: float | None = None
    tokens: int | None = None
    model: str | None = None  # answering model (fallbacks can differ between modes)
    verdict2: str | None = None  # second, independent judge (reliability check)
    issues: list[str] = field(default_factory=list)


async def evaluate(pipeline: Pipeline, judges: list[LLM], q: dict, cfg: Config,
                   variants: dict[str, set[str]], do_answer: bool) -> Record:
    reranker = None if cfg.rerank else NoReranker()
    r = await pipeline.retrieve(q["question"], cfg.mode, cfg.hops, reranker=reranker)
    ids = [p.id for p in r.passages]
    record = Record(q["id"], q["type"], cfg.name, m.recall_at_k(q["gold_chunks"], ids, K),
                    m.complete_at_k(q["gold_chunks"], ids, K), r.timer.laps["total"], ids)
    if not (cfg.answer and do_answer):
        return record
    from backend.answer import generate_answer

    a = await generate_answer(pipeline.llm, q["question"], r.passages, r.facts)
    v = await judge(judges[0], q["question"], q["reference_answer"], a.answer)
    record.answer, record.verdict, record.judge_reasoning = a.answer, v.verdict, v.reasoning
    if len(judges) > 1:
        record.verdict2 = (await judge(judges[1], q["question"], q["reference_answer"],
                                       a.answer)).verdict
    record.model = a.model
    record.entity_recall = m.entity_recall(q["expected_entities"], a.answer, variants)
    record.llm_seconds, record.tokens = a.llm_seconds, a.prompt_tokens + a.completion_tokens
    record.issues = a.issues
    return record


def alias_variants() -> dict[str, set[str]]:
    """normalized name -> all normalized names of the same alias group (for entity recall)."""
    groups = yaml.safe_load((REPO_ROOT / "data" / "aliases.yaml").read_text()) or {}
    variants: dict[str, set[str]] = {}
    for canonical, names in groups.items():
        forms = {normalize(n) for n in [canonical, *(names or [])]}
        for f in forms:
            variants[f] = forms
    return variants


async def run(questions_file: str, limit: int | None, retrieval_only: bool,
              second_judge: str | None) -> tuple[list[Record], str]:
    settings = get_settings()
    questions = [json.loads(line) for line in (REPO_ROOT / questions_file).open()][:limit]
    driver = create_driver(settings)
    try:
        pipeline = Pipeline.from_settings(settings, driver)
        judge_llm = LLM(settings.chat_endpoint("judge"), rpm=settings.llm_rpm,
                        timeout=settings.llm_timeout, cache=DiskCache("llm"))
        judges = [judge_llm]
        if second_judge:  # "provider:model", e.g. groq:llama-3.3-70b-versatile
            provider, model = second_judge.split(":", 1)
            ep = settings.model_copy(update={"judge_provider": provider, "judge_model": model})
            judges.append(LLM(ep.chat_endpoint("judge"), rpm=settings.llm_rpm,
                              timeout=settings.llm_timeout, cache=DiskCache("llm")))
        models = (f"answers: {pipeline.llm.endpoint.provider}/{pipeline.llm.endpoint.model} "
                  f"(fallbacks: {', '.join(pipeline.llm.endpoint.fallback_models) or 'none'}), "
                  f"judge: {judge_llm.endpoint.provider}/{judge_llm.endpoint.model}"
                  + (f" (second judge: {second_judge})" if second_judge else "") + ", "
                  f"reranker: {settings.reranker}")
        print(f"{models} | {len(questions)} questions x {len(CONFIGS)} configs")
        variants = alias_variants()
        # sequential: parallel runs compete for CPU in the reranker and distort latency
        semaphore = asyncio.Semaphore(1)

        async def one(q, cfg):
            async with semaphore:
                return await evaluate(pipeline, judges, q, cfg, variants, not retrieval_only)

        records = await asyncio.gather(*(one(q, c) for q in questions for c in CONFIGS))
        a, j = pipeline.llm.usage, judge_llm.usage
        print(f"LLM calls: answers {a.calls} (cached {a.cache_hits}), "
              f"judge {j.calls} (cached {j.cache_hits})")
    finally:
        await driver.close()
    return records, models


# --- reporting ---------------------------------------------------------------------------------

def fmt_ci(values: list[float]) -> str:
    lo, hi = m.bootstrap_ci(values)
    return f"{m.mean(values):.2f} [{lo:.2f}, {hi:.2f}]"


def report(records: list[Record], questions: dict[str, dict], models: str) -> str:
    by = {}
    for r in records:
        by.setdefault(r.config, []).append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r.qid)
    lines = ["# Evaluation results", "",
             f"{len(questions)} questions ({_type_counts(questions)}), k = {K} passages. "
             "Values: mean [bootstrap 95% CI]. Generated by `python -m eval.run`.", "",
             f"Models: {models}.", ""]

    lines += ["## Retrieval (no LLM involved)", "",
              "| config | recall@8 | complete@8 | retrieval latency p50 |", "|---|---|---|---|"]
    for cfg in CONFIGS:
        rs = by[cfg.name]
        lines.append(f"| {cfg.name} | {fmt_ci([r.recall for r in rs])} | "
                     f"{fmt_ci([r.complete for r in rs])} | "
                     f"{statistics.median(r.retrieval_ms for r in rs):.0f} ms |")

    answered = [c for c in CONFIGS if c.answer and by[c.name][0].verdict]
    if answered:
        lines += ["", "## Answers", "",
                  "| config | judge score | correct | entity recall | LLM latency p50 "
                  "| tokens / question |",
                  "|---|---|---|---|---|---|"]
        for cfg in answered:
            rs = by[cfg.name]
            lines.append(
                f"| {cfg.name} | {fmt_ci([SCORES[r.verdict] for r in rs])} | "
                f"{sum(r.verdict == 'correct' for r in rs)}/{len(rs)} | "
                f"{fmt_ci([r.entity_recall for r in rs])} | "
                f"{statistics.median(r.llm_seconds for r in rs):.1f} s | "
                f"{statistics.mean(r.tokens for r in rs):,.0f} |")
        lines += (_by_type(by, questions) + _significance(by) + _reliability(by)
                  + _calibration(by))
    lines += _per_question(by)
    return "\n".join(lines) + "\n"


def _type_counts(questions) -> str:
    counts = {}
    for q in questions.values():
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    return ", ".join(f"{n} {t.replace('_', '-')}" for t, n in counts.items())


def _by_type(by, questions) -> list[str]:
    lines = ["", "## By question type", "",
             "| type | n | recall@8 vector | recall@8 graph | judge vector | judge graph |",
             "|---|---|---|---|---|---|"]
    for t in dict.fromkeys(q["type"] for q in questions.values()):
        v = [r for r in by["vector"] if r.type == t]
        g = [r for r in by["graph"] if r.type == t]
        lines.append(f"| {t} | {len(v)} | {m.mean([r.recall for r in v]):.2f} | "
                     f"{m.mean([r.recall for r in g]):.2f} | "
                     f"{m.mean([SCORES[r.verdict] for r in v]):.2f} | "
                     f"{m.mean([SCORES[r.verdict] for r in g]):.2f} |")
    return lines


def _significance(by) -> list[str]:
    v, g = by["vector"], by["graph"]
    sv, sg = [SCORES[r.verdict] for r in v], [SCORES[r.verdict] for r in g]
    mc = m.mcnemar([r.verdict == "correct" for r in v], [r.verdict == "correct" for r in g])
    lo, hi = m.paired_diff_ci(sv, sg)
    rlo, rhi = m.paired_diff_ci([r.recall for r in v], [r.recall for r in g])
    return ["", "## Graph vs vector: paired tests", "",
            f"- Judge score difference (graph − vector): mean {m.mean(sg) - m.mean(sv):+.2f}, "
            f"95% CI [{lo:+.2f}, {hi:+.2f}]",
            f"- McNemar (correct vs not): only vector right {mc['only_a']}, only graph right "
            f"{mc['only_b']}, exact p = {mc['p']:.3f}",
            f"- Recall@8 difference: 95% CI [{rlo:+.2f}, {rhi:+.2f}], Wilcoxon p = "
            f"{m.wilcoxon([r.recall for r in v], [r.recall for r in g]):.3f}"]


def _reliability(by) -> list[str]:
    """Without human labels: agreement of two independent judges, and of the judge with the
    deterministic entity check. This measures reliability, not validity: two LLMs can share an
    error, which only human labels would reveal."""
    rs = [r for c in ("vector", "graph") for r in by[c]]
    lines = ["", "## Judge reliability (automatic)", ""]
    if all(r.verdict2 for r in rs):
        agree = sum(r.verdict == r.verdict2 for r in rs)
        lines.append(f"- Second judge agrees on {agree}/{len(rs)} answers, Cohen's kappa = "
                     f"{m.cohen_kappa([r.verdict for r in rs], [r.verdict2 for r in rs]):.2f}")
    judge_ok = ["yes" if r.verdict == "correct" else "no" for r in rs]
    entities_ok = ["yes" if r.entity_recall == 1.0 else "no" for r in rs]
    lines.append(f"- Judge 'correct' vs 'all expected entities mentioned': kappa = "
                 f"{m.cohen_kappa(judge_ok, entities_ok):.2f}")
    pairs = list(zip(by["vector"], by["graph"], strict=True))
    mixed = sum(v.model != g.model for v, g in pairs)
    lines.append(f"- Answering model differed between the two modes (provider fallback) on "
                 f"{mixed}/{len(pairs)} questions")
    return lines


def _calibration(by) -> list[str]:
    path = EVAL_DIR / "manual_labels.jsonl"
    if not path.exists():
        return ["", "## Judge calibration", "",
                "Pending: label the answers in `eval/calibration_sheet.md`, save them to "
                "`eval/manual_labels.jsonl`, re-run."]
    labels = [json.loads(line) for line in path.open()]
    judged = {(r.qid, r.config): r.verdict for rs in by.values() for r in rs if r.verdict}
    pairs = [(lab["label"], judged[(lab["qid"], lab["config"])]) for lab in labels
             if (lab["qid"], lab["config"]) in judged]
    human, model = zip(*pairs, strict=True)
    agree = sum(h == j for h, j in pairs)
    return ["", "## Judge calibration", "",
            f"{len(pairs)} answers labelled by hand: judge agrees on {agree}/{len(pairs)}, "
            f"Cohen's kappa = {m.cohen_kappa(human, model):.2f}.", "",
            "| question | config | human | judge |", "|---|---|---|---|",
            *[f"| {lab['qid']} | {lab['config']} | {lab['label']} | "
              f"{judged.get((lab['qid'], lab['config']), '-')} |" for lab in labels]]


def _per_question(by) -> list[str]:
    lines = ["", "## Per question (error analysis)", "",
             "| id | type | recall vector | recall graph | judge vector | judge graph |",
             "|---|---|---|---|---|---|"]
    for v, g in zip(by["vector"], by["graph"], strict=True):
        lines.append(f"| {v.qid} | {v.type} | {v.recall:.2f} | {g.recall:.2f} | "
                     f"{v.verdict or '-'} | {g.verdict or '-'} |")
    return lines


def calibration_sheet(records: list[Record], questions: dict[str, dict], n: int = 10) -> str:
    """A stratified sample (both modes, all types) for hand labelling."""
    rng = random.Random(1)
    strata: dict[tuple[str, str], list[Record]] = {}
    for r in records:
        if r.verdict:
            strata.setdefault((r.config, r.type), []).append(r)
    for group in strata.values():
        rng.shuffle(group)
    sample: list[Record] = []
    while len(sample) < n and any(strata.values()):  # round-robin: every (mode, type) gets turns
        for group in strata.values():
            if group and len(sample) < n:
                sample.append(group.pop())
    lines = ["# Judge calibration sheet", "",
             "Label each answer as correct / partial / incorrect against the reference, "
             "WITHOUT looking at the judge. Then write one JSON line per answer to "
             "`eval/manual_labels.jsonl`, e.g.", "",
             '`{"qid": "q01", "config": "graph", "label": "correct"}`', ""]
    for r in sample:
        q = questions[r.qid]
        lines += [f"## {r.qid} ({r.config}, {r.type})", f"**Q:** {q['question']}", "",
                  f"**Reference:** {q['reference_answer']}", "", f"**Answer:** {r.answer}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="first N questions (smoke test)")
    parser.add_argument("--retrieval-only", action="store_true", help="no LLM calls")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--out", default="eval/results.md")
    parser.add_argument("--second-judge", default=None, help="provider:model")
    args = parser.parse_args()
    records, models = asyncio.run(run(args.questions, args.limit, args.retrieval_only,
                                      args.second_judge))
    questions = {q["id"]: q for q in map(json.loads, (REPO_ROOT / args.questions).open())
                 if any(r.qid == q["id"] for r in records)}
    (EVAL_DIR / "runs").mkdir(exist_ok=True)
    out = REPO_ROOT / args.out
    with (EVAL_DIR / "runs" / f"{out.stem}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    text = report(records, questions, models)
    out.write_text(text)
    if not args.retrieval_only and not (EVAL_DIR / "manual_labels.jsonl").exists():
        (EVAL_DIR / "calibration_sheet.md").write_text(calibration_sheet(records, questions))
    print(text)


if __name__ == "__main__":
    main()
