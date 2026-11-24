"""No-model shortcut baselines, computed exactly from the task file.

LAB-Bench issue #18 shows that "pick the longest option" scores 0.332 on
ProtocolQA against 0.232 for chance. A model score is only meaningful above
these floors, so the scorecard prints them beside every run.

Each rule gives an expected accuracy. When several options tie, the rule picks
one of them at random, so the task adds 1/(number tied) if the ideal answer is
among them ("ties split evenly", the same convention as issue #18).
The rules never pick the refusal option. Option text is compared after
whitespace normalisation and duplicate removal (see mcq.unique_options).
"""

from __future__ import annotations

import collections
from typing import Callable

from .mcq import unique_options
from .tasks import Task

Rule = Callable[[list[str]], list[int]]


def _argmax_all(values: list[int]) -> list[int]:
    top = max(values)
    return [i for i, v in enumerate(values) if v == top]


def _argmin_all(values: list[int]) -> list[int]:
    low = min(values)
    return [i for i, v in enumerate(values) if v == low]


RULES: dict[str, Rule] = {
    "longest_chars": lambda opts: _argmax_all([len(o) for o in opts]),
    "longest_words": lambda opts: _argmax_all([len(o.split()) for o in opts]),
    "shortest_chars": lambda opts: _argmin_all([len(o) for o in opts]),
    "chance_without_refusal": lambda opts: list(range(len(opts))),
}


def expected_hit(task: Task, rule: Rule) -> float:
    options, _ = unique_options(task)
    answerable = options[:-1]          # The last option is the refusal text.
    picks = rule(answerable)
    return (1.0 / len(picks)) if 0 in picks else 0.0   # Index 0 is the ideal answer.


def chance_with_refusal(task: Task) -> float:
    options, _ = unique_options(task)
    return 1.0 / len(options)


def shortcut_baselines(tasks: list[Task]) -> dict:
    """Expected accuracy of each rule, overall and per subtask."""
    def table(ts: list[Task]) -> dict:
        n = len(ts)
        row = {name: round(sum(expected_hit(t, r) for t in ts) / n, 4) for name, r in RULES.items()}
        row["chance_with_refusal"] = round(sum(chance_with_refusal(t) for t in ts) / n, 4)
        row["tasks"] = n
        return row

    by_sub = collections.defaultdict(list)
    for t in tasks:
        by_sub[t.subtask].append(t)
    return {"overall": table(tasks),
            "by_subtask": {s: table(ts) for s, ts in sorted(by_sub.items())}}


def format_markdown(result: dict) -> str:
    cols = ["tasks", "chance_with_refusal", "chance_without_refusal",
            "longest_chars", "longest_words", "shortest_chars"]
    lines = ["| subtask | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in [("ALL", result["overall"]), *result["by_subtask"].items()]:
        lines.append(f"| {name} | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)
