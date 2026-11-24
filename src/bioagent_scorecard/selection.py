"""Stratified task selection (replaces "the first N rows").

SeqQA has 15 subtasks of 40 rows, and the file lists them in blocks. The
first 20 rows all come from one subtask (ORF-seq-AAid), so "--limit 20" used
to test one skill only. `stratified` spreads N tasks over the subtasks as
evenly as possible:

1. Each subtask gets floor(N / S) tasks. The rest go one each to the first
   subtasks in a seeded order.
2. Inside a subtask, the tasks are ordered by sha256("<seed>:<task id>") and
   the first k are taken.

The result depends only on the task ids, the subtask names, N and the seed.
It does not depend on the file order or the machine. A larger N with the same
seed keeps every task of a smaller N only when no subtask gets fewer tasks.
"""

from __future__ import annotations

import collections
import hashlib

from .tasks import Task


def _key(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def stratified(tasks: list[Task], n: int, seed: int = 0) -> list[Task]:
    """Return n tasks spread evenly over the subtasks, in a fixed order."""
    if n <= 0 or n >= len(tasks):
        return list(tasks)
    by_sub: dict[str, list[Task]] = collections.defaultdict(list)
    for t in tasks:
        by_sub[t.subtask].append(t)
    subs = sorted(by_sub, key=lambda s: _key(seed, "subtask:" + s))
    base, extra = divmod(n, len(subs))
    want = {s: base + (1 if i < extra else 0) for i, s in enumerate(subs)}
    # A subtask with fewer tasks than its share gives the rest to the others.
    short = 0
    for s in subs:
        if want[s] > len(by_sub[s]):
            short += want[s] - len(by_sub[s])
            want[s] = len(by_sub[s])
    while short:
        moved = False
        for s in subs:
            if short and want[s] < len(by_sub[s]):
                want[s] += 1
                short -= 1
                moved = True
        if not moved:
            break
    picked = []
    for s in sorted(by_sub):
        ranked = sorted(by_sub[s], key=lambda t: _key(seed, t.id))
        picked.extend(ranked[: want[s]])
    return picked


def run_order(tasks: list[Task], seed: int = 0) -> list[Task]:
    """The order in which a run calls the tasks: one task of each subtask in
    turn (round robin), so a run that stops early still covers all subtasks
    about evenly. The selection (and its task_ids) does not change; only the
    call order does. The order depends only on the ids, subtasks and seed."""
    by_sub: dict[str, list[Task]] = collections.defaultdict(list)
    for t in tasks:
        by_sub[t.subtask].append(t)
    subs = sorted(by_sub, key=lambda s: _key(seed, "subtask:" + s))
    queues = [sorted(by_sub[s], key=lambda t: _key(seed, "order:" + t.id)) for s in subs]
    out: list[Task] = []
    for i in range(max((len(q) for q in queues), default=0)):
        out.extend(q[i] for q in queues if i < len(q))
    return out


def describe(tasks: list[Task], picked: list[Task], n: int | None, seed: int) -> dict:
    """The selection record for the run manifest."""
    counts = collections.Counter(t.subtask for t in picked)
    ids = [t.id for t in picked]
    return {
        "method": "stratified_by_subtask" if n else "all",
        "requested": n,
        "selection_seed": seed,
        "n_selected": len(picked),
        "n_available": len(tasks),
        "per_subtask": dict(sorted(counts.items())),
        "task_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "task_ids": ids,
    }
