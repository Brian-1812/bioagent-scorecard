"""Command line: fetch, run, report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import DUMMY_AGENTS, get_agent
from .anchors import CAPTION
from .baselines import format_markdown, shortcut_baselines
from .runner import ChunkLimits, _load_records, run
from .tasks import LAB_BENCH_SUBSETS, fetch_lab_bench, load_jsonl, load_meta

#: Exit code of `run` when it stopped at a chunk or usage limit (not an error).
EXIT_STOPPED = 3


def _fmt_rate(r: dict) -> str:
    if r["value"] is None:
        return "n/a"
    lo, hi = r["ci95"]
    return f"{r['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_value(r: dict) -> str:
    return "n/a" if r["value"] is None else f"{r['value']:.3f}"


def _f(x, fmt="{:.3f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def five_hour_use(run_dir: Path) -> dict:
    """An estimate of the 5-hour usage window used per task in one run folder.

    Each result holds the account-wide rate limit after its call. An interval
    is two calls in a row, both with usage data, in the same window. The
    estimate is the sum of the rises over the intervals divided by the number
    of intervals. Other sessions on the account count too (the estimate is
    high); the values have a step of 0.01 (one interval is coarse). None when
    no interval has data, for example dummy or API agents."""
    rows = _load_records(run_dir / "results.jsonl")
    used, last, seen, intervals = 0.0, None, 0, 0
    for r in rows:
        fh = ((r.get("rate_limit") or {}).get("five_hour") or {})
        u = fh.get("utilization")
        if not isinstance(u, (int, float)):
            last = None                     # A gap: the next call starts a new segment.
            continue
        seen += 1
        if last is not None and fh.get("resets_at") == last[1]:
            used += max(0.0, float(u) - last[0])
            intervals += 1
        last = (float(u), fh.get("resets_at"))
    return {"tasks": len(rows), "tasks_with_usage": seen, "intervals": intervals,
            "five_hour_rise": round(used, 4),
            "per_task": round(used / intervals, 5) if intervals else None}

def report(run_dirs: list[Path]) -> str:
    rows = ["| agent | seed | tasks | accuracy [95% CI] | anchor s | refused | precision | "
            "coverage | errors | tool calls/task | $/task | s/task | 5h use/task | flags |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    anchor_lines, maxplan = set(), False
    for d in run_dirs:
        m = json.loads((d / "manifest.json").read_text())
        s = json.loads((d / "summary.json").read_text())
        n = s["tasks"] or 1
        a = s.get("anchors") or {}
        errors = sum(s["outcomes"].get(k, 0) for k in
                     ("agent_error", "unparsable", "timeout", "over_budget", "safety_refusal"))
        tasks = f"{s['tasks']}" + ("" if m.get("status", "complete") == "complete"
                                   else f" of {m.get('total')} (stopped)")
        rows.append(
            f"| {m['agent']['agent']} | {m['seed']} | {tasks} | {_fmt_rate(s['accuracy'])} | "
            f"{_f(s.get('anchor_score'), '{:.2f}')} | {s['outcomes']['refused']}/{s['tasks']} | "
            f"{_fmt_value(s['precision'])} | {_fmt_value(s['coverage'])} | {errors} | "
            f"{(s.get('tool_calls') or {}).get('total', 0) / n:.1f} | "
            f"{s['cost_usd']['per_task_mean']:.4f} | {s['seconds']['per_task_mean']:.1f} | "
            f"{_f(five_hour_use(d)['per_task'], '{:.4f}')} | "
            f"{', '.join(f['check'] for f in s['integrity_flags']) or '-'} |")
        if a:
            anchor_lines.add(f"{a['tasks']} tasks: floor chance {_f(a['floor_chance'])}, "
                             f"floor longest option {_f(a['floor_longest'])}, "
                             f"ceiling human experts {_f(a['ceiling'])} ({a.get('ceiling_status')})")
        maxplan |= "max-subscription" in str(m.get("billing", ""))
    if anchor_lines:
        rows += ["", "Anchors (same tasks as each row):"] + [f"- {x}" for x in sorted(anchor_lines)]
        rows += ["", CAPTION]
    if maxplan:
        rows += ["", "5h use/task: rise of the account-wide 5-hour usage window per task "
                 "(1.0 = one full window), an estimate. Other sessions on the account count "
                 "too, so it is high when the account is busy."]
    if maxplan:
        rows += ["", "Note: claude-cli and claude-sandbox rows ran on a Claude Max subscription "
                 "through Claude Code. Their $/task is the API list-price equivalent, not money spent."]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bioagent-scorecard")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download a LAB-Bench subset (network, no model calls)")
    f.add_argument("--subset", choices=LAB_BENCH_SUBSETS, default="SeqQA")
    f.add_argument("--data-dir", default="data")

    r = sub.add_parser("run", help="run one agent on one task file")
    r.add_argument("--tasks", required=True, help="path to a tasks JSONL file")
    r.add_argument("--agent", required=True)
    r.add_argument("--seed", type=int, default=0, help="option-order seed")
    r.add_argument("--limit", type=int,
                   help="run N tasks, spread evenly over the subtasks (stratified)")
    r.add_argument("--selection-seed", type=int, default=0,
                   help="seed for the stratified selection of --limit tasks")
    r.add_argument("--out", default="runs")
    r.add_argument("--resume", help="continue a stopped run in this run folder")
    r.add_argument("--dry-run", action="store_true",
                   help="dummy agents only; all network access is blocked")
    r.add_argument("--allow-model-calls", action="store_true",
                   help="required for agents that call a model (they use quota or money)")
    r.add_argument("--allow-unpinned", action="store_true",
                   help="run on a task file whose sha256 is not the pinned value")
    r.add_argument("--effort", help="claude agents: --effort level for claude -p")
    r.add_argument("--timeout", type=float, default=600.0,
                   help="claude agents: seconds per task before the process is stopped")
    r.add_argument("--max-tool-calls", type=int, help="claude-sandbox: tool calls per task")
    r.add_argument("--chunk-five-hour", type=float, default=0.18,
                   help="stop after this chunk used this share of the 5-hour window (0.18)")
    r.add_argument("--stop-seven-day", type=float, default=0.95,
                   help="stop when the 7-day window utilization reaches this value (0.95)")
    r.add_argument("--chunk-max-tasks", type=int, help="stop after this many tasks")

    b = sub.add_parser("baselines", help="print no-model shortcut baselines for a task file")
    b.add_argument("--tasks", required=True)
    b.add_argument("--json", action="store_true")

    rp = sub.add_parser("report", help="print a markdown table for run folders")
    rp.add_argument("runs", nargs="+")

    sub.add_parser("agents", help="list the agents")

    a = p.parse_args(argv)
    if a.cmd == "fetch":
        path = fetch_lab_bench(a.subset, a.data_dir)
        print(json.dumps(load_meta(path), indent=2))
        print(path)
    elif a.cmd == "run":
        tasks = load_jsonl(a.tasks)
        kw = {}
        if a.agent not in DUMMY_AGENTS:
            if not a.allow_model_calls:
                p.error(f"agent {a.agent!r} calls a model; add --allow-model-calls")
            kw = {"effort": a.effort, "timeout": a.timeout}
            if a.agent.startswith("claude-sandbox:") and a.max_tool_calls:
                kw["max_tool_calls"] = a.max_tool_calls
        chunk = ChunkLimits(max_five_hour=a.chunk_five_hour, stop_seven_day=a.stop_seven_day,
                            max_tasks=a.chunk_max_tasks)
        out = run(get_agent(a.agent, **kw), tasks, load_meta(a.tasks), a.out,
                  seed=a.seed, dry_run=a.dry_run, limit=a.limit,
                  selection_seed=a.selection_seed, chunk=chunk, resume_dir=a.resume,
                  allow_unpinned=a.allow_unpinned)
        s = json.loads((out / "summary.json").read_text())
        m = json.loads((out / "manifest.json").read_text())
        print(f"{out}: accuracy {_fmt_rate(s['accuracy'])}, outcomes {s['outcomes']}, "
              f"flags {[x['check'] for x in s['integrity_flags']]}")
        if m["status"] != "complete":
            c = m["chunks"][-1]
            rl = c.get("last_rate_limit") or {}
            fh = (rl.get("five_hour") or {}).get("utilization")
            sd = (rl.get("seven_day") or {}).get("utilization")
            print(f"STOPPED ({m['stop_reason']}) at {m['done']}/{m['total']} tasks; "
                  f"usage this chunk +{c['five_hour_used_this_chunk']:.2f} of the 5-hour window; "
                  f"now five_hour {fh}, seven_day {sd}. Continue with --resume {out}")
            return EXIT_STOPPED
    elif a.cmd == "baselines":
        res = shortcut_baselines(load_jsonl(a.tasks))
        print(json.dumps(res, indent=2) if a.json else format_markdown(res))
    elif a.cmd == "report":
        print(report([Path(x) for x in a.runs]))
    elif a.cmd == "agents":
        for name in sorted(DUMMY_AGENTS):
            print(name, "(dummy, no API calls)")
        print("claude-cli:<model>", "(row A, no tools, claude -p on a Claude subscription; needs --allow-model-calls)")
        print("claude-sandbox:<model>", "(row B, one sandboxed run_python tool; needs --allow-model-calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
