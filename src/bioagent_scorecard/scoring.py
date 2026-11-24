"""Outcomes, metrics and the failure taxonomy.

Metric definitions follow the LAB-Bench paper:
- accuracy  = correct / all tasks
- precision = correct / answered tasks (a refusal is not an answer)
- coverage  = answered / all tasks
Each rate has a Wilson 95% interval, so a reader sees how much the number can
move by chance on a small subset.
"""

from __future__ import annotations

import collections
import math

# Level-1 failure taxonomy. The harness assigns it with no judgement.
# A reviewer adds a level-2 label by hand later (not done in this project).
OUTCOMES = (
    "correct",
    "wrong",          # A parsable letter, but not the correct one.
    "refused",        # The refusal option.
    "unparsable",     # No valid letter in the answer.
    "safety_refusal", # The provider's safety classifier declined (e.g. Claude stop_reason
                      # "refusal", category "bio"). Counted apart from "refused".
    "agent_error",    # The agent crashed or the API returned an error.
    "timeout",        # The task went over the time limit.
    "over_budget",    # The task went over the token or cost limit.
)

LEVEL2_LABELS = (
    "knowledge_gap", "reasoning_error", "calculation_error", "tool_not_used",
    "tool_misuse", "tool_failure", "retrieval_miss", "format_error",
    "question_ambiguous", "answer_key_suspect",
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(k: int, n: int) -> dict:
    """A rate with its Wilson interval. With n = 0 the rate is undefined, so
    value and ci95 are None (JSON null), not 0."""
    if n == 0:
        return {"value": None, "k": k, "n": 0, "ci95": None}
    lo, hi = wilson(k, n)
    return {"value": k / n, "k": k, "n": n, "ci95": [round(lo, 4), round(hi, 4)]}


def summarize(records: list[dict]) -> dict:
    n = len(records)
    counts = collections.Counter(r["outcome"] for r in records)
    correct = counts["correct"]
    answered = correct + counts["wrong"]
    secs = [r["seconds"] for r in records]
    costs = [r["cost_usd"] for r in records]
    letters = collections.Counter(r["letter"] for r in records if r.get("letter"))
    out = {
        "tasks": n,
        "outcomes": {o: counts.get(o, 0) for o in OUTCOMES},
        "accuracy": _rate(correct, n),
        "precision": _rate(correct, answered),
        "coverage": _rate(answered, n),
        "cost_usd": {
            "total": round(sum(costs), 6),
            "per_task_mean": round(sum(costs) / n, 6) if n else 0.0,
        },
        "seconds": {
            "total": round(sum(secs), 3),
            "per_task_mean": round(sum(secs) / n, 4) if n else 0.0,
            "per_task_p95": round(_pct(secs, 95), 4),
        },
        "tokens": {
            "input": sum(r["input_tokens"] for r in records),
            "output": sum(r["output_tokens"] for r in records),
        },
        "letter_distribution": dict(sorted(letters.items())),
    }
    by_sub = collections.defaultdict(list)
    for r in records:
        by_sub[r["subtask"]].append(r)
    out["by_subtask"] = {
        s: {"tasks": len(rs),
            "accuracy": _rate(sum(r["outcome"] == "correct" for r in rs), len(rs))}
        for s, rs in sorted(by_sub.items())
    }
    return out


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))
    return s[i]
