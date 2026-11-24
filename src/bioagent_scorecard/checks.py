"""Integrity checks ("cheating checks").

Each check reads the per-task records and the agent traces. A check returns
a list of flags. A flag does not change the score. The scorecard shows the
flags beside the score, and a reviewer reads the flagged traces.
"""

from __future__ import annotations

import collections
import math

# URLs where an agent can find the answer key. An agent with web access must
# not fetch them. Extend this list for each new benchmark.
ANSWER_KEY_SOURCES = (
    "huggingface.co/datasets/futurehouse",
    "datasets-server.huggingface.co",
    "github.com/Future-House/LAB-Bench",
    "github.com/Future-House/BixBench",
    "raw.githubusercontent.com/Future-House",
)

CANARY_MARKERS = ("canary GUID", "BENCHMARK DATA SHOULD NEVER APPEAR")


def check_answer_key_access(records: list[dict]) -> list[dict]:
    flags = []
    for r in records:
        for url in r.get("tool_urls", []):
            if any(src.lower() in url.lower() for src in ANSWER_KEY_SOURCES):
                flags.append({"check": "answer_key_access", "task_id": r["task_id"], "url": url})
    return flags


def check_canary_in_output(records: list[dict]) -> list[dict]:
    """The canary string in the agent's own words means the agent saw the data
    file, through a tool or through training."""
    return [
        {"check": "canary_in_output", "task_id": r["task_id"]}
        for r in records
        if any(m in (r.get("answer_text") or "") for m in CANARY_MARKERS)
    ]


def check_position_bias(records: list[dict], alpha: float = 0.001) -> list[dict]:
    """Chi-square test on the chosen letter positions.

    The shuffle puts the correct answer in each position with equal chance.
    A strong skew in the chosen letters means the agent (or a bug in the
    harness) prefers a position. The test uses the most common option count.
    """
    letters = [r["letter"] for r in records if r.get("letter")]
    if len(letters) < 50:
        return []
    k = collections.Counter(r["n_options"] for r in records).most_common(1)[0][0]
    letters = [r["letter"] for r in records if r.get("letter") and r["n_options"] == k]
    counts = collections.Counter(letters)
    n = len(letters)
    exp = n / k
    chi2 = sum((counts.get(chr(65 + i), 0) - exp) ** 2 / exp for i in range(k))
    p = _chi2_sf(chi2, k - 1)
    if p < alpha:
        return [{"check": "position_bias", "chi2": round(chi2, 2), "df": k - 1,
                 "p": p, "counts": dict(sorted(counts.items()))}]
    return []


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-square (Wilson-Hilferty normal approximation)."""
    if x <= 0:
        return 1.0
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def check_duplicate_options(records: list[dict]) -> list[dict]:
    """Tasks where the benchmark lists the same option text twice (after
    whitespace normalisation). The harness shows each text once, so the score
    is fair, but the flag tells the reader that these tasks differ from the
    raw benchmark file. One flag per run, not one per task."""
    ids = [r["task_id"] for r in records if r.get("dropped_duplicates")]
    if not ids:
        return []
    return [{"check": "duplicate_options", "tasks": len(ids), "task_ids": ids}]


def check_missing_price(records: list[dict]) -> list[dict]:
    """Tasks whose model has no row in prices.json. Their cost counts as 0,
    so the run's $ figures are too low until the row is added."""
    bad = [r for r in records if r.get("cost_unknown")]
    if not bad:
        return []
    return [{"check": "missing_price", "tasks": len(bad),
             "models": sorted({r.get("model", "") for r in bad})}]


def run_all(records: list[dict]) -> list[dict]:
    return (check_answer_key_access(records)
            + check_canary_in_output(records)
            + check_position_bias(records)
            + check_duplicate_options(records)
            + check_missing_price(records))
