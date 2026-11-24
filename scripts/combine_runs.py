"""Build one 600-task result per row from an old full run plus a re-run of the
tasks that changed.

The first full run used a stale SeqQA copy. 480 tasks are the same in the
pinned file; 120 tasks were replaced upstream. This script takes, per row:
- the old records whose task id is in the pinned task file (unchanged tasks),
- the re-run records for the other ids,
and checks that the result covers every task of the pinned file exactly once.
It never writes into the run folders.

Usage (from the repo root):
    PYTHONPATH=src python3 scripts/combine_runs.py --tasks data/lab-bench/SeqQA.jsonl \
        --old-tasks <old task file> --row B <old run dir> <rerun dir> --row A <old run dir> <rerun dir> \
        [--json out.json]
The first --row is the "treatment" row of the paired test (B wins / A wins).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from bioagent_scorecard import anchors, scoring
from bioagent_scorecard.runner import _load_records
from bioagent_scorecard.tasks import load_jsonl, load_meta, pin_status


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _task_sha(old_dir_or_new: Path) -> str | None:
    m = json.loads((Path(old_dir_or_new) / "manifest.json").read_text())
    return (m.get("tasks") or {}).get("sha256_now")


def combine(task_ids: set[str], old_dir: Path, new_dir: Path, *, same_content: set[str] | None = None,
            old_sha: str | None = None, new_sha: str | None = None) -> tuple[list[dict], dict]:
    """same_content: ids whose task content is identical in the old and the
    fresh task file; only these may keep an old record.
    old_sha / new_sha: the task-file sha256 each manifest must record."""
    for d, want in ((old_dir, old_sha), (new_dir, new_sha)):
        if want is not None and _task_sha(d) != want:
            raise SystemExit(f"{d}: manifest task sha256 {_task_sha(d)} is not {want}")
    old = _load_records(Path(old_dir) / "results.jsonl")
    new = _load_records(Path(new_dir) / "results.jsonl")
    new_ids = {r["task_id"] for r in new}
    if len(new_ids) != len(new):
        raise SystemExit(f"{new_dir}: duplicate task ids")
    allowed = task_ids if same_content is None else same_content
    kept_old = [r for r in old if r["task_id"] in allowed and r["task_id"] not in new_ids]
    dropped = [r for r in old if r["task_id"] not in task_ids]
    stray = [r["task_id"] for r in new if r["task_id"] not in task_ids]
    if stray:
        raise SystemExit(f"{new_dir}: {len(stray)} re-run ids are not in the task file")
    recs = kept_old + new
    ids = [r["task_id"] for r in recs]
    if len(ids) != len(set(ids)) or set(ids) != task_ids:
        missing = task_ids - set(ids)
        raise SystemExit(f"combined result does not cover the task file once: "
                         f"{len(missing)} missing, {len(ids) - len(set(ids))} duplicates")
    bad = [r["task_id"] for r in recs if r.get("outcome") not in ("correct", "wrong", "refused")]
    if bad:
        raise SystemExit(f"{len(bad)} records are not correct/wrong/refused (first {bad[:3]}); "
                         "re-run them before you combine")
    info = {"old_dir": str(old_dir), "new_dir": str(new_dir), "from_old": len(kept_old),
            "from_rerun": len(new), "old_dropped": len(dropped),
            "outcomes_not_scored_normally": bad,
            "old_manifest_git": json.loads((Path(old_dir) / "manifest.json").read_text())["git_commit"],
            "new_manifest_git": json.loads((Path(new_dir) / "manifest.json").read_text())["git_commit"]}
    return recs, info


def acc(recs: list[dict]) -> dict:
    k = sum(r["outcome"] == "correct" for r in recs)
    r = scoring._rate(k, len(recs))
    r["refused"] = sum(r_["outcome"] == "refused" for r_ in recs)
    r["wrong"] = sum(r_["outcome"] == "wrong" for r_ in recs)
    return r


def fmt(a: dict) -> str:
    if a["value"] is None:
        return "n/a"
    lo, hi = a["ci95"]
    return f"{a['value']:.3f} [{lo:.3f}, {hi:.3f}] ({a['k']}/{a['n']}, {a['refused']} refused, {a['wrong']} wrong)"


def paired(r1: list[dict], r2: list[dict]) -> dict:
    o1 = {r["task_id"]: r["outcome"] == "correct" for r in r1}
    o2 = {r["task_id"]: r["outcome"] == "correct" for r in r2}
    common = set(o1) & set(o2)
    b = sum(o1[i] and not o2[i] for i in common)
    c = sum(o2[i] and not o1[i] for i in common)
    return {"n": len(common), "first_only": b, "second_only": c,
            "diff_points": round(100 * (b - c) / len(common), 2) if common else None,
            "p_exact_mcnemar": round(mcnemar_exact(b, c), 4)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--row", nargs=3, action="append", metavar=("NAME", "OLD", "RERUN"),
                   required=True)
    p.add_argument("--old-tasks", required=True,
                   help="the task file the old runs used; only ids with identical content keep old records")
    p.add_argument("--json")
    a = p.parse_args(argv)
    tasks = load_jsonl(a.tasks)
    meta = load_meta(a.tasks)
    if pin_status(meta) != "match":
        raise SystemExit(f"{a.tasks} is not the pinned task file ({pin_status(meta)})")
    ids = {t.id for t in tasks}
    by_id = {t.id: t for t in tasks}
    old_tasks = {t.id: t for t in load_jsonl(a.old_tasks)}
    old_meta = load_meta(a.old_tasks)
    same = {i for i in ids if i in old_tasks and old_tasks[i] == by_id[i]}
    rows, out = {}, {"tasks": {"file": a.tasks, "sha256": meta["sha256_now"],
                               "hf_revision": meta.get("hf_revision"), "n": len(tasks)},
                     "rows": {}}
    for name, old, new in a.row:
        recs, info = combine(ids, Path(old), Path(new), same_content=same,
                             old_sha=old_meta["sha256_now"], new_sha=meta["sha256_now"])
        rows[name] = recs
        new_ids = {r["task_id"] for r in _load_records(Path(new) / "results.jsonl")}
        old_all = _load_records(Path(old) / "results.jsonl")
        an = anchors.anchors_for([by_id[r["task_id"]] for r in recs])
        a_all = acc(recs)
        out["rows"][name] = {
            "combine": info,
            "all": a_all,
            "anchor_s": anchors.anchor_score(a_all["value"], an["floor_longest"], an["ceiling"]),
            "unchanged": acc([r for r in recs if r["task_id"] not in new_ids]),
            "rerun_fresh": acc([r for r in recs if r["task_id"] in new_ids]),
            "old_stale_replaced": acc([r for r in old_all if r["task_id"] not in ids]),
            "old_full_run": acc(old_all),
            "by_subtask_rerun": {s: acc([r for r in recs if r["task_id"] in new_ids
                                         and r["subtask"] == s])
                                 for s in sorted({by_id[i].subtask for i in new_ids})},
        }
    out["anchors"] = anchors.anchors_for(tasks)
    out["old_tasks"] = {"file": a.old_tasks, "sha256": old_meta["sha256_now"],
                        "ids_with_same_content": len(same)}
    names = [r[0] for r in a.row]
    if len(names) == 2:
        r1, r2 = rows[names[0]], rows[names[1]]
        new_any = set()
        for _, _, new in a.row:
            new_any |= {r["task_id"] for r in _load_records(Path(new) / "results.jsonl")}
        out["paired"] = {
            "all": paired(r1, r2),
            "unchanged": paired([r for r in r1 if r["task_id"] not in new_any],
                                [r for r in r2 if r["task_id"] not in new_any]),
            "rerun_fresh": paired([r for r in r1 if r["task_id"] in new_any],
                                  [r for r in r2 if r["task_id"] in new_any]),
            "order": names,
        }
    print(f"Tasks: {a.tasks} sha256 {meta['sha256_now'][:12]} rev {meta.get('hf_revision')}")
    an = out["anchors"]
    print(f"Anchors (600): chance {an['floor_chance']:.3f}, longest {an['floor_longest']:.3f}, "
          f"human ceiling {an['ceiling']:.3f}")
    for name, r in out["rows"].items():
        print(f"\nrow {name}  ({r['combine']['from_old']} old + {r['combine']['from_rerun']} re-run)")
        for k in ("all", "unchanged", "rerun_fresh", "old_stale_replaced", "old_full_run"):
            print(f"  {k:20s} {fmt(r[k])}")
        print(f"  anchor s             {r['anchor_s']:.2f}")
        for s, v in r["by_subtask_rerun"].items():
            print(f"    {s:34s} {fmt(v)}")
    if "paired" in out:
        for k, v in out["paired"].items():
            if k != "order":
                print(f"paired {k}: {v}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
