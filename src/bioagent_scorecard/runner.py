"""Run one agent on one task file and write the results.

Output folder (runs/<run_id>/):
- manifest.json  what was run: harness version, git commit, task file meta,
                 task selection, seed, agent config, prices used, host, chunks
- results.jsonl  one record per task (no question text, so the folder does not
                 copy benchmark data; answer_text has question text removed)
- summary.json   metrics, cost, time, outcome counts, integrity flags, anchors
- tool_logs/     row B only: the code and output of each tool call. It can
                 hold question text (sequences), so it is not for publishing.

Chunks: a run on a Claude subscription stops by itself when
- the plan reports a usage-limit error (the task is not scored and is retried
  in the next chunk), or
- the 5-hour window of the account reaches `stop_five_hour` (the shared
  ceiling: other sessions move it too),
- this chunk used `max_five_hour` of the 5-hour usage window (the sum of the
  rises of five_hour utilization that Claude Code reports per call), or
- the 7-day window reaches `stop_seven_day`, or
- this chunk ran `max_tasks` tasks.
`run(..., resume_dir=<run folder>)` continues a stopped run in the same folder.
"""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import fcntl
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, anchors, checks, scoring
from .agents import Agent
from .mcq import build_mcq, parse_letter
from .netguard import no_network
from .pricing import cost_usd, load_prices
from .selection import describe, run_order, stratified
from .tasks import Task, pin_status

ANSWER_TEXT_LIMIT = 2000


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, cwd=Path(__file__).parent, timeout=5)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
#: A nucleotide run: 16+ nucleotide letters (any case) where only non-letters
#: (spaces, commas, digits, "|", line breaks; at most 6 in a row) stand
#: between them. It catches codons, numbered bases and wrapped lines.
_SEQ = re.compile(r"[ACGTUNacgtun](?:[^A-Za-z]{0,6}[ACGTUNacgtun]){15,}")
COPY_MIN = 30          # Shortest copied run (characters) that is removed.


def _cut_common(text: str, src: str, min_len: int) -> str:
    """Replace each run of >= min_len characters that text shares with src
    (case-insensitive) by a marker."""
    if not src or len(text) < min_len:
        return text
    for _ in range(50):
        sm = difflib.SequenceMatcher(None, text.lower(), src.lower(), autojunk=False)
        m = sm.find_longest_match(0, len(text), 0, len(src))
        if m.size < min_len:
            break
        text = text[:m.a] + f"[task text: {m.size} chars]" + text[m.a + m.size:]
    return text


def strip_task_text(text: str, task: Task) -> str:
    """Remove benchmark text from free text that goes into results.jsonl.

    results.jsonl must not carry benchmark text (CC-BY-SA). Models quote the
    sequence, the question or an option, in any case or split into codons.
    1. Every nucleotide run (16+ letters, also split by spaces or hyphens)
       becomes "[sequence: N nt]".
    2. Every run of COPY_MIN+ characters shared with the question or an
       option (case-insensitive) becomes "[task text: N chars]".
    3. E-mail addresses become "[email]".
    Short quotes (an enzyme name, a 13-nt site) stay, so the answer stays
    readable. Limits: a protein sequence the model writes itself, or a
    paraphrase, stays.
    """
    if not text:
        return text

    def seq(m: re.Match) -> str:
        n = sum(c.isalpha() for c in m.group(0))
        return f"[sequence: {n} nt]" if n >= 16 else m.group(0)
    out = _SEQ.sub(seq, text)
    for src in (task.question, task.ideal, *task.distractors):
        out = _cut_common(out, src, COPY_MIN)
    return _EMAIL.sub("[email]", out)


def run_one(agent: Agent, task: Task, seed: int, prices: dict,
            max_seconds: float, max_cost_usd: float):
    """Run one task. Return (record, AgentResult or None)."""
    mcq = build_mcq(task, seed)
    t0 = time.monotonic()
    try:
        res = agent.answer(mcq, seed)
        err = res.error
    except Exception as e:                      # The harness never crashes on one task.
        res, err = None, f"{type(e).__name__}: {e}"
    seconds = time.monotonic() - t0
    rec = {
        "task_id": task.id,
        "subtask": task.subtask,
        "n_options": len(mcq.options),
        "dropped_duplicates": mcq.dropped_duplicates,
        "correct_letter": mcq.correct_letter,
        "refuse_letter": mcq.refuse_letter,
        "seconds": round(seconds, 4),
        "input_tokens": res.input_tokens if res else 0,
        "output_tokens": res.output_tokens if res else 0,
        "cost_usd": 0.0,
        "answer_text": strip_task_text((res.text or "")[:ANSWER_TEXT_LIMIT * 4], task)[:ANSWER_TEXT_LIMIT] if res else "",
        "tool_calls": len(res.tool_calls) if res else 0,
        "tool_errors": sum(not c.ok for c in res.tool_calls) if res else 0,
        "tool_urls": [c.url for c in res.tool_calls if c.url] if res else [],
        "tool_error_samples": ([strip_task_text(c.error, task)[:200] for c in res.tool_calls
                                if not c.ok][:3] if res else []),
        "letter": None,
        "error": strip_task_text(err, task) if err else err,
    }
    if res:
        rec["model"] = res.model
        rec["cached_input_tokens"] = res.cached_input_tokens
        rec["cache_write_tokens"] = res.cache_write_tokens
        rec["trace"] = res.trace[:20]           # Metadata only for the claude-cli adapters.
        rec["rate_limit"] = res.rate_limit
        try:
            rec["cost_usd"] = round(cost_usd(prices, res.model, res.input_tokens,
                                             res.output_tokens, res.cached_input_tokens,
                                             res.cache_write_tokens), 6)
        except KeyError as e:
            # No price row: the answer is still scored; the cost is unknown and
            # the run gets a missing_price flag.
            rec["cost_unknown"] = str(e)
    rec["safety_refusal"] = res.safety_refusal if res else ""
    if err:
        rec["outcome"] = "agent_error"
    elif rec["safety_refusal"]:
        rec["outcome"] = "safety_refusal"
    elif seconds > max_seconds:
        rec["outcome"] = "timeout"
    elif not rec.get("cost_unknown") and rec["cost_usd"] > max_cost_usd:
        rec["outcome"] = "over_budget"
    else:
        letter = parse_letter(res.text, mcq)
        rec["letter"] = letter
        if letter is None:
            rec["outcome"] = "unparsable"
        elif letter == mcq.refuse_letter:
            rec["outcome"] = "refused"
        elif letter == mcq.correct_letter:
            rec["outcome"] = "correct"
        else:
            rec["outcome"] = "wrong"
    return rec, res


@dataclasses.dataclass
class ChunkLimits:
    #: Stop when this chunk used this share of the 5-hour window (0.18 = 18 points).
    max_five_hour: float | None = 0.18
    #: Stop when the 7-day window utilization reaches this value.
    stop_seven_day: float | None = 0.95
    #: Stop when the 5-hour window utilization reaches this value. This is the
    #: shared ceiling of the whole account (other sessions move it too), not a
    #: share of this run.
    stop_five_hour: float | None = 0.90
    #: Stop after this many tasks in this chunk.
    max_tasks: int | None = None
    #: Stop after this many agent errors in a row (login expired, tool broken, ...).
    max_errors_in_a_row: int | None = 5
    #: Stop when a model agent reports no 5-hour usage data this many calls in a
    #: row: without it the 5-hour stop cannot work, so the run must not go on.
    max_calls_without_usage: int | None = 3


class UsageMeter:
    """Sums the rise of the 5-hour utilization over the calls of one chunk.

    Claude Code reports utilization after each call, so the first call of a
    chunk sets the start value and its own use is not counted (an under-count
    of one call). The value covers the whole account, so other sessions on it
    count too: the stop comes early, not late."""

    def __init__(self, limits: ChunkLimits):
        self.limits = limits
        self.used = 0.0
        self.last: tuple[float, object] | None = None
        self.last_rate: dict = {}
        self.tasks = 0
        self.errors_in_a_row = 0
        self.calls_without_usage = 0

    def update(self, rate: dict, error: bool = False, model_call: bool = True) -> str | None:
        """Count one finished task. Return a stop reason or None."""
        self.tasks += 1
        self.errors_in_a_row = self.errors_in_a_row + 1 if error else 0
        if rate:
            self.last_rate = rate
        fh = rate.get("five_hour") if rate else None
        if isinstance(fh, dict) and isinstance(fh.get("utilization"), (int, float)):
            self.calls_without_usage = 0
            u, resets = float(fh["utilization"]), fh.get("resets_at")
            if self.last is not None:
                lu, lr = self.last
                self.used += max(0.0, u - lu) if resets == lr else u   # New window: count from 0.
            self.last = (u, resets)
        elif model_call:
            self.calls_without_usage += 1
        return self.check()

    def check(self) -> str | None:
        """The stop reason for the current state, or None. The runner calls it
        before each task too, so a shared meter stops the next run at once."""
        lim = self.limits
        if lim.max_five_hour is not None and self.used >= lim.max_five_hour:
            return "chunk_limit_five_hour"
        fh = self.last_rate.get("five_hour") if self.last_rate else None
        if (lim.stop_five_hour is not None and isinstance(fh, dict)
                and isinstance(fh.get("utilization"), (int, float))
                and fh["utilization"] >= lim.stop_five_hour):
            return "five_hour_ceiling"
        sd = self.last_rate.get("seven_day") if self.last_rate else None
        if (lim.stop_seven_day is not None and isinstance(sd, dict)
                and isinstance(sd.get("utilization"), (int, float))
                and sd["utilization"] >= lim.stop_seven_day):
            return "seven_day_limit"
        if lim.max_tasks is not None and self.tasks >= lim.max_tasks:
            return "chunk_limit_tasks"
        if lim.max_errors_in_a_row is not None and self.errors_in_a_row >= lim.max_errors_in_a_row:
            return "errors_in_a_row"
        if (lim.max_calls_without_usage is not None
                and self.calls_without_usage >= lim.max_calls_without_usage):
            return "no_usage_data"
        return None

    def state(self) -> dict:
        return {"five_hour_used_this_chunk": round(self.used, 4),
                "last_rate_limit": self.last_rate, "tasks_this_chunk": self.tasks,
                "errors_in_a_row": self.errors_in_a_row}


#: Config fields that may change between chunks without changing the agent
#: (a Claude Code auto-update). The manifest records them per chunk.
VOLATILE_CONFIG = ("claude_code_version",)


def stable_config_hash(cfg: dict) -> str:
    return config_hash({k: v for k, v in cfg.items() if k not in VOLATILE_CONFIG})


def _load_records(path: Path) -> list[dict]:
    """Read results.jsonl. A broken last line (a crash mid-write) is dropped."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _write_records(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def run(agent: Agent, tasks: list[Task], task_meta: dict, out_root: str | Path,
        seed: int = 0, dry_run: bool = False, limit: int | None = None,
        max_seconds: float = 900.0, max_cost_usd: float = 2.0,
        selection_seed: int = 0, chunk: ChunkLimits | None = None,
        resume_dir: str | Path | None = None, allow_unpinned: bool = False,
        meter: "UsageMeter | None" = None, retry_errors: bool = True,
        task_filter: dict | None = None) -> Path:
    """Run `agent` on the selected tasks. Pass one `meter` to several runs to
    give them one shared chunk budget (scripts/pilot.py does this). On resume,
    tasks with outcome agent_error run again (retry_errors)."""
    if dry_run and agent.makes_api_calls:
        raise SystemExit(f"agent {agent.name!r} makes API calls; dry-run accepts only dummy agents")
    pin = pin_status(task_meta)
    if pin == "mismatch" and not allow_unpinned:
        raise SystemExit("task file sha256 is not the pinned value (pins.json); "
                         "use --allow-unpinned only after you check the change")
    if hasattr(agent, "preflight"):
        agent.preflight()                   # Row B: the sandbox must work here.
    selected = stratified(tasks, limit or 0, selection_seed)
    selection = describe(tasks, selected, limit, selection_seed)
    if task_filter:                         # The run uses a subset of the file.
        selection["task_filter"] = task_filter
    prices = load_prices()
    cfg = agent.config()
    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())   # noqa: E731

    if resume_dir:
        out = Path(resume_dir)
        old = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        old_hash = stable_config_hash(old.get("agent") or {})
        for field, theirs, mine in (("agent config", old_hash, stable_config_hash(cfg)),
                                    ("seed", old.get("seed"), seed)):
            if theirs != mine:
                raise SystemExit(f"resume: {field} differs ({theirs!r} != {mine!r})")
        if (old.get("tasks") or {}).get("sha256_now") != task_meta.get("sha256_now"):
            raise SystemExit("resume: the task file changed (sha256 differs)")
        for field, mine in (("max_seconds_per_task", max_seconds),
                            ("max_cost_usd_per_task", max_cost_usd)):
            if old.get(field) != mine:
                raise SystemExit(f"resume: {field} differs ({old.get(field)!r} != {mine!r})")
        if old.get("selection", {}).get("task_ids") != selection["task_ids"]:
            raise SystemExit("resume: the task selection differs (other --limit, seed or file)")
        run_id, started, chunks = old["run_id"], old["started_at"], old.get("chunks", [])
    else:
        stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
        base_id = f"{stamp}-{agent.name}-s{seed}-{stable_config_hash(cfg)}"
        run_id, n = base_id, 1
        while (Path(out_root) / run_id).exists():   # Same agent and seed in the same second.
            n += 1
            run_id = f"{base_id}-{n}"
        out = Path(out_root) / run_id
        out.mkdir(parents=True, exist_ok=False)
        started, chunks = now(), []
    if hasattr(agent, "tool_log_dir"):
        agent.tool_log_dir = str(out / "tool_logs")

    lock = open(out / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{out} is in use by another run")

    results = out / "results.jsonl"
    records = _load_records(results)
    if retry_errors and any(r.get("outcome") == "agent_error" for r in records):
        records = [r for r in records if r.get("outcome") != "agent_error"]
    _write_records(results, records)        # Also drops a broken last line.
    done = {r["task_id"] for r in records}
    # The full order, then the tasks not done: a resume keeps the order of a run
    # without a stop.
    todo = [t for t in run_order(selected, selection_seed) if t.id not in done]
    meter = meter or UsageMeter(chunk or ChunkLimits())
    tasks_before = meter.tasks
    state = {"stop_reason": None, "stopped_on": None}
    chunk_info = {"started_at": now(), "claude_code_version": cfg.get("claude_code_version"),
                  "git_commit": _git_commit(), "run_order": "round_robin_by_subtask"}
    t0 = time.monotonic()

    def write_manifest(status: str) -> None:
        wall = time.monotonic() - t0
        this = {**chunk_info, "finished_at": now(), "tasks_run": meter.tasks - tasks_before,
                "wall_seconds": round(wall, 3), "stop_reason": state["stop_reason"],
                "stopped_on": state["stopped_on"], "limits": dataclasses.asdict(meter.limits),
                **meter.state()}
        all_chunks = chunks + [this]
        summary = scoring.summarize(records)
        summary["wall_seconds"] = round(sum(c["wall_seconds"] for c in all_chunks), 3)
        summary["integrity_flags"] = checks.run_all(records)
        by_id = {t.id: t for t in selected}
        summary["anchors"] = anchors.anchors_for([by_id[r["task_id"]] for r in records
                                                  if r["task_id"] in by_id])
        a = summary["anchors"]
        summary["anchor_score"] = anchors.anchor_score(summary["accuracy"]["value"],
                                                       a["floor_longest"], a["ceiling"])
        summary["refusal_rate"] = scoring._rate(summary["outcomes"]["refused"], len(records))
        summary["tool_calls"] = {"total": sum(r.get("tool_calls", 0) for r in records),
                                 "errors": sum(r.get("tool_errors", 0) for r in records)}
        used_models = sorted(({cfg.get("model")} | {r.get("model") for r in records}) - {None, ""})
        manifest = {
            "harness": "bioagent-scorecard",
            "harness_version": __version__,
            "git_commit": chunk_info["git_commit"],
            "run_id": run_id,
            "status": status,
            "done": len(records),
            "total": len(selected),
            "stop_reason": state["stop_reason"],
            "dry_run": dry_run,
            "network_blocked": dry_run,   # Python-level guard only; see netguard.py.
            "billing": cfg.get("billing", "none" if not agent.makes_api_calls else "api"),
            "seed": seed,
            "limit": limit,
            "selection": selection,
            "max_seconds_per_task": max_seconds,
            "max_cost_usd_per_task": max_cost_usd,
            "agent": cfg,
            "agent_config_hash": config_hash(cfg),
            "agent_config_hash_stable": stable_config_hash(cfg),
            "tasks": {**task_meta, "pin": pin},
            "n_tasks": len(selected),
            "prices": {"as_of": prices.get("as_of"),
                       "models": {m: prices["models"].get(m) for m in used_models}},
            "host": {"python": sys.version.split()[0], "platform": platform.platform()},
            "chunks": all_chunks,
            "started_at": started,
            "finished_at": now(),
        }
        tmp = out / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        tmp.replace(out / "manifest.json")
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_manifest("running")               # A crash still leaves a manifest to resume from.
    guard = no_network() if dry_run else contextlib.nullcontext()
    final = "stopped"
    try:
        with guard, open(results, "a", encoding="utf-8") as f:
            for task in todo:
                reason = meter.check()
                if reason:
                    state["stop_reason"] = reason
                    break
                rec, res = run_one(agent, task, seed, prices, max_seconds, max_cost_usd)
                if res is not None and res.usage_limit_hit:
                    # Not scored: the plan refused the call. The next chunk retries it.
                    state["stop_reason"] = "usage_limit_error"
                    state["stopped_on"] = {"task_id": task.id, "error": rec["error"]}
                    meter.last_rate = res.rate_limit or meter.last_rate
                    break
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                f.flush()
                reason = meter.update(res.rate_limit if res else {},
                                      error=rec["outcome"] == "agent_error",
                                      model_call=bool(agent.makes_api_calls))
                if reason and len(records) < len(selected):
                    state["stop_reason"] = reason
                    break
        final = "complete" if len(records) >= len(selected) else "stopped"
    except BaseException:
        state["stop_reason"] = state["stop_reason"] or "crashed"
        raise
    finally:
        write_manifest(final)
        lock.close()
    return out
