"""Floor and ceiling anchors (the Arc Virtual Cell Challenge style scale).

Raw accuracy hides two facts: a no-information shortcut already scores above
chance, and the best known performance is below 1. The anchor scale puts both
on one line:

    s = (accuracy - floor) / (ceiling - floor)

- floor   = the best no-model shortcut on the same tasks: "pick the longest
            option" (exact expected value, baselines.py). Chance with the
            refusal option is shown too.
- ceiling = human experts from the LAB-Bench paper, per subtask, averaged
            over the same tasks (anchors.json; a reported value).

s = 0 means "no better than the shortcut". s = 1 means "as good as the
human experts". s is not clipped: it can be below 0 or above 1.
s is NOT an accuracy. Always print the raw accuracy and its interval next to it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .baselines import shortcut_baselines
from .tasks import Task

ANCHORS_FILE = Path(__file__).with_name("anchors.json")

CAPTION = ("s = (accuracy - floor) / (ceiling - floor). Floor = the 'pick the longest option' "
           "shortcut on the same tasks; ceiling = LAB-Bench human experts (reported, per "
           "subtask, averaged over the same tasks). s = 0: no better than the shortcut; "
           "s = 1: human level. s is not an accuracy.")


def base_subtask(name: str) -> str:
    """'ORF-seq-AAid-v1-public' -> 'ORF-seq-AAid'."""
    for suffix in ("-v1-public", "-public", "-v1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_anchors(key: str = "lab-bench/SeqQA") -> dict:
    return json.loads(ANCHORS_FILE.read_text(encoding="utf-8")).get(key, {})


def anchors_for(tasks: list[Task], key: str = "lab-bench/SeqQA") -> dict:
    """Floors and the ceiling for exactly these tasks."""
    if not tasks:
        return {"tasks": 0, "floor_chance": None, "floor_longest": None,
                "ceiling": None, "ceiling_name": None, "ceiling_status": None}
    b = shortcut_baselines(tasks)["overall"]
    out = {"tasks": len(tasks),
           "floor_chance": b["chance_with_refusal"],
           "floor_longest": b["longest_chars"],
           "ceiling": None, "ceiling_name": None, "ceiling_status": None}
    ceil = load_anchors(key).get("ceiling") or {}
    by_sub = ceil.get("by_subtask") or {}
    vals = [by_sub.get(base_subtask(t.subtask)) for t in tasks]
    if tasks and all(v is not None for v in vals):
        out["ceiling"] = round(sum(vals) / len(vals), 4)
        out["ceiling_name"] = ceil.get("name")
        out["ceiling_status"] = "reported"
    return out


def anchor_score(acc: float | None, floor: float | None, ceiling: float | None) -> float | None:
    if acc is None or floor is None or ceiling is None or ceiling <= floor:
        return None
    return round((acc - floor) / (ceiling - floor), 3)
