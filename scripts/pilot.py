"""Run several agents on one stratified subset in chunks of Max-plan usage.

All runs of one chunk share one UsageMeter, so a chunk stops when it used
--chunk-five-hour of the 5-hour window, at a usage-limit error, when the
7-day window reaches --stop-seven-day, or after agent errors in a row. Run the
same command again to continue: a complete run is skipped, a stopped run is
resumed in its own folder (finished tasks never run again).

Modes:
- default: one chunk, then exit (0 = all complete, 3 = stopped at a limit).
- --sanity N: run the first N tasks of each agent (those not done yet), print
  them with a check of the outcomes, then exit 0 (or 4 if a check failed).
- --auto: chunk after chunk until all runs are complete:
  * chunk limit: go on at once while the 5-hour window is at most
    --continue-below; else pause until the window resets.
  * usage-limit error or 7-day limit: pause until that window resets. A
    usage-limit error while both windows still have room is treated as a
    transient API error and gets a backoff instead.
  * a usage limit with no usable reset time (missing or already past): wait 5
    then 15 min and count a strike, so the loop cannot spin.
  * agent errors in a row, missing usage data, a crash: wait 5 min, then 15
    min. The same stop 3 times with no task that finished without an error:
    exit 4 (blocked).
  * agent_error records stay in place during the chunks, so a few tasks that
    always fail cannot block the run. After all runs finish, --retry-passes
    passes (3) run those tasks again.
  * strikes, the retry passes and the pause time live in auto-state.json, so
    they survive the exit 5 of a long pause.
  A pause longer than --max-inline-wait seconds ends the process with exit 5
  and progress.json {"status": "paused", "pause_until": <epoch>}; run the same
  command again after that time. Shorter waits sleep in the process.
  --on-event CMD runs CMD with one status line as its last argument after each
  chunk, pause and end (errors of CMD are ignored).

progress.json and report.md in --out are rewritten after every chunk.

Example (from the repo root):
    PYTHONPATH=src python3 scripts/pilot.py --tasks data/lab-bench/SeqQA.jsonl \
        --limit 20 --out runs/pilot --allow-model-calls \
        --agents claude-cli:claude-sonnet-5-5 claude-sandbox:claude-sonnet-5-5

Exit codes: 0 = all complete, 3 = stopped at a limit (one-chunk mode),
4 = blocked or sanity check failed, 5 = paused (run again after pause_until).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from bioagent_scorecard.agents import get_agent
from bioagent_scorecard.mcq import build_mcq, parse_letter
from bioagent_scorecard.cli import EXIT_STOPPED, report
from bioagent_scorecard.runner import (ANSWER_TEXT_LIMIT, ChunkLimits, UsageMeter,
                                       _load_records, run, stable_config_hash)
from bioagent_scorecard.selection import describe, stratified
from bioagent_scorecard.tasks import load_jsonl, load_meta

EXIT_BLOCKED = 4
EXIT_PAUSED = 5
BACKOFF = (300, 900, 2700)      # Seconds per strike; the block comes at strike MAX_STRIKES.
RESET_MARGIN = 120              # Seconds after a window reset before the next call.
#: Longest pause per window, so a wrong unit (milliseconds) cannot sleep for ever.
MAX_PAUSE = {"five_hour": 6 * 3600, "seven_day": 8 * 86400}
MAX_STRIKES = 3                 # The same stop this often with no progress = blocked.
EXPECTED_OUTCOMES = {"correct", "wrong", "refused"}


def find_run(out: Path, cfg_hash: str, seed: int, ids: list[str]) -> Path | None:
    """The matching run folder: a complete one first, else the newest."""
    found = []
    for m in sorted(out.glob("*/manifest.json"), reverse=True):
        d = json.loads(m.read_text())
        # The stable hash leaves out the Claude Code version (auto-updates).
        if (stable_config_hash(d.get("agent") or {}) == cfg_hash and d.get("seed") == seed
                and d.get("selection", {}).get("task_ids") == ids):
            found.append((d.get("status") == "complete", m.parent))
    for complete, path in found:            # Newest first.
        if complete:
            return path
    return found[0][1] if found else None


def _manifest(d: Path) -> dict:
    return json.loads((d / "manifest.json").read_text())


def _errors(d: Path) -> int:
    return sum(r.get("outcome") == "agent_error" for r in _load_records(d / "results.jsonl"))


def _util(rate: dict, window: str):
    return ((rate or {}).get(window) or {}).get("utilization")


def _rejected(rate: dict) -> bool:
    """True when the plan refused a call: a window status, or the top status."""
    if (rate or {}).get("status") == "rejected" or (rate or {}).get("is_using_overage"):
        return True
    return any(isinstance(w, dict) and w.get("status") == "rejected"
               for w in (rate or {}).values())


def _resets(rate: dict, window: str):
    return ((rate or {}).get(window) or {}).get("resets_at")


class Pilot:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.tasks, self.meta = load_jsonl(a.tasks), load_meta(a.tasks)
        self.task_filter = None
        if getattr(a, "task_ids", None):
            self.tasks, self.task_filter = filter_tasks(self.tasks, a.task_ids)
            if 0 < a.limit < len(self.tasks):
                raise SystemExit(f"--limit {a.limit} would drop some of the {len(self.tasks)} "
                                 "--task-ids tasks; leave --limit out to run them all")
        self.ids = describe(self.tasks, stratified(self.tasks, a.limit, a.selection_seed),
                            a.limit, a.selection_seed)["task_ids"]
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.total = len(self.ids) * len(a.agents)
        self.last_rate: dict = {}
        self._agents: dict = {}
        self.state_file = self.out / "auto-state.json"

    def limits(self, max_tasks: int | None = None) -> ChunkLimits:
        return ChunkLimits(max_five_hour=self.a.chunk_five_hour,
                           stop_seven_day=self.a.stop_seven_day,
                           stop_five_hour=self.a.stop_five_hour, max_tasks=max_tasks)

    def agent(self, name: str):
        """One agent object per name: get_agent runs `claude --version`."""
        if name not in self._agents:
            self._agents[name] = get_agent(name, timeout=self.a.timeout)
        return self._agents[name]

    def find(self, name: str) -> Path | None:
        return find_run(self.out, stable_config_hash(self.agent(name).config()),
                        self.a.seed, self.ids)

    def dirs(self) -> list[Path]:
        """The run folder of each agent that has one, in --agents order."""
        return [d for d in (self.find(n) for n in self.a.agents) if d]

    def done(self) -> int:
        return sum(_manifest(d)["done"] for d in self.dirs())

    def errors(self) -> int:
        """Tasks with outcome agent_error over all runs."""
        return sum(_errors(d) for d in self.dirs())

    def chunk(self, retry_errors_of_complete: bool = False) -> tuple[str, str | None, UsageMeter]:
        """Run one chunk over all agents. Return (status, stop reason, meter)."""
        meter = UsageMeter(self.limits())
        for name in self.a.agents:
            agent = self.agent(name)
            prev = self.find(name)
            if prev and _manifest(prev).get("status") == "complete":
                if not (retry_errors_of_complete and _errors(prev)):
                    print(f"{name}: complete in {prev}, skipped", flush=True)
                    continue
            # Errors of another agent must not stop this one.
            meter.errors_in_a_row = 0
            print(f"{name}: {'resume ' + str(prev) if prev else 'new run'}", flush=True)
            d = run(agent, self.tasks, self.meta, self.out, seed=self.a.seed, limit=self.a.limit,
                    selection_seed=self.a.selection_seed, resume_dir=prev, meter=meter,
                    retry_errors=retry_errors_of_complete, task_filter=self.task_filter)
            self.last_rate = meter.last_rate or self.last_rate
            m = _manifest(d)
            s = json.loads((d / "summary.json").read_text())
            print(f"{name}: {m['status']} {m['done']}/{m['total']}, outcomes {s['outcomes']}",
                  flush=True)
            if m["status"] != "complete":
                return "stopped", m["stop_reason"], meter
        return "complete", None, meter

    def sanity(self, n: int) -> int:
        """Run the first n tasks of each agent and check the outcomes."""
        ok = True
        by_id = {t.id: t for t in self.tasks}
        for name in self.a.agents:
            agent = self.agent(name)
            prev = self.find(name)
            have = _manifest(prev)["done"] if prev else 0
            first = _load_records(prev / "results.jsonl")[:n] if prev else []
            if any(r.get("outcome") == "agent_error" for r in first):
                have = 0                    # Run those tasks again (retry_errors drops them).
            if have < n:
                meter = UsageMeter(self.limits(max_tasks=n - have))
                prev = run(agent, self.tasks, self.meta, self.out, seed=self.a.seed,
                           limit=self.a.limit, selection_seed=self.a.selection_seed,
                           resume_dir=prev, meter=meter, retry_errors=True)
                self.last_rate = meter.last_rate or self.last_rate
                m = _manifest(prev)
                if m["done"] < n and m["status"] != "complete":
                    print(f"SANITY {name}: stopped early ({m['stop_reason']})")
                    ok = False
            recs = _load_records(prev / "results.jsonl")[:n]
            for r in recs:
                print(f"SANITY {name}: {r['subtask']} letter={r.get('letter')!r} "
                      f"correct={r.get('correct_letter')} refuse={r.get('refuse_letter')} "
                      f"outcome={r['outcome']} s={r.get('seconds')} tools={r.get('tool_calls')} "
                      f"err={(r.get('error') or '')[:80]!r}")
            problems, skipped = [], 0
            for r in recs:
                if r["outcome"] not in EXPECTED_OUTCOMES:
                    problems.append(f"{r['task_id'][:8]} outcome {r['outcome']}")
                    continue
                # The letter must come from the answer text, and the outcome from
                # the letter. A broken parser or scorer fails here.
                mcq = build_mcq(by_id[r["task_id"]], self.a.seed)
                text = r.get("answer_text") or ""
                parsed = parse_letter(text, mcq)
                # A long answer is cut to ANSWER_TEXT_LIMIT, so the tag with the
                # letter can be missing from the saved text: skip that case.
                if len(text) >= ANSWER_TEXT_LIMIT:
                    skipped += 1        # The saved text is cut, so the tag can be gone.
                elif parsed != r.get("letter"):
                    problems.append(f"{r['task_id'][:8]} letter {r.get('letter')!r} "
                                    f"!= parse {parsed!r}")
                if mcq.correct_letter != r.get("correct_letter"):
                    problems.append(f"{r['task_id'][:8]} correct letter not reproducible")
                want = ("correct" if r.get("letter") == r.get("correct_letter") else
                        "refused" if r.get("letter") == r.get("refuse_letter") else "wrong")
                if r["outcome"] != want:
                    problems.append(f"{r['task_id'][:8]} outcome {r['outcome']} != {want}")
            if agent.makes_api_calls and not any(
                    isinstance(_util(r.get("rate_limit") or {}, "five_hour"), (int, float))
                    for r in recs):
                # Without usage data the ceiling cannot work: the run must not start.
                problems.append("no five_hour usage data in any result")
            if not any(r["outcome"] == "correct" for r in recs):
                problems.append("no correct answer in the first tasks (scoring suspect)")
            if skipped and skipped == len(recs):
                problems.append(f"every answer reached {ANSWER_TEXT_LIMIT} characters, "
                                "so the parser was never checked")
            if problems or not recs:
                ok = False
                print(f"SANITY {name}: FAIL -> " + "; ".join(problems[:6] or ["no results"]))
            else:
                print(f"SANITY {name}: OK ({len(recs)} tasks"
                      + (f", {skipped} answers too long to re-parse)" if skipped else ")"))
        self.write_progress("sanity_ok" if ok else "sanity_fail", None)
        return 0 if ok else EXIT_BLOCKED

    def write_progress(self, status: str, reason: str | None, pause_until: float | None = None,
                       note: str = "") -> str:
        dirs = self.dirs()
        rows = []
        for d in dirs:
            m, s = _manifest(d), json.loads((d / "summary.json").read_text())
            rows.append({"agent": m["agent"].get("agent"), "run": d.name, "status": m["status"],
                         "done": m["done"], "total": m["total"], "accuracy": s["accuracy"],
                         "outcomes": s["outcomes"]})
        fh, sd = _util(self.last_rate, "five_hour"), _util(self.last_rate, "seven_day")
        done = sum(r["done"] for r in rows)
        until = (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pause_until))
                 if pause_until else None)
        prog = {"status": status, "stop_reason": reason, "pause_until": pause_until,
                "pause_until_utc": until, "done": done, "total": self.total,
                "five_hour": fh, "seven_day": sd, "note": note, "rows": rows,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = self.out / "progress.json.tmp"
        tmp.write_text(json.dumps(prog, indent=2))
        tmp.replace(self.out / "progress.json")
        if dirs:
            rtmp = self.out / "report.md.tmp"
            rtmp.write_text(report(dirs) + "\n")
            rtmp.replace(self.out / "report.md")
        accs = ", ".join(f"{r['agent']} {r['done']}/{r['total']} acc "
                         f"{(r['accuracy'] or {}).get('value') or 0:.3f}" for r in rows)
        line = (f"{status}{' (' + reason + ')' if reason else ''}: {done}/{self.total} tasks; "
                f"5h {fh}, 7d {sd}; {accs}"
                + (f"; pause until {until}" if until else "")
                + (f"; {note}" if note else ""))
        print(line, flush=True)
        if self.a.on_event:
            try:
                subprocess.run(shlex.split(self.a.on_event) + [line], timeout=60, check=False)
            except Exception as e:      # noqa: BLE001 - a broken hook must not stop the run.
                print(f"on-event failed: {e!r}", flush=True)
        return line

    def write_progress_safe(self, *a, **kw) -> str:
        try:
            return self.write_progress(*a, **kw)
        except Exception as e:          # noqa: BLE001 - a report error must not stop the run.
            print(f"write_progress failed: {e!r}", flush=True)
            return ""

    def wait(self, until: float, reason: str) -> bool:
        """Sleep until `until`, or end the process (return False) for a long pause."""
        secs = until - time.time()
        if secs <= 0:
            return True
        if secs > self.a.max_inline_wait:
            self.write_progress_safe("paused", reason, pause_until=until)
            return False
        self.write_progress_safe("waiting", reason, pause_until=until)
        time.sleep(secs)
        return True

    @staticmethod
    def window_reset(window: str, rate: dict) -> float | None:
        """When `window` resets, or None when that time is missing or stale.

        A stale time (already past) means the data does not describe the
        current window, so the caller must use a backoff instead."""
        r = _resets(rate, window)
        if not isinstance(r, (int, float)) or float(r) <= time.time():
            return None
        # A wrong unit (milliseconds) would sleep for ever: cap the pause.
        cap = MAX_PAUSE.get(window, 6 * 3600)
        return min(float(r) + RESET_MARGIN, time.time() + cap)

    # ---- state that must survive exit 5 (the process ends in a long pause) ----

    def load_state(self) -> dict:
        try:
            d = json.loads(self.state_file.read_text())
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_state(self, **kw) -> None:
        d = self.load_state()
        d.update(kw)
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(self.state_file)

    def auto(self) -> int:
        st = self.load_state()
        strikes = int(st.get("strikes") or 0)
        last_reason = st.get("last_reason")
        retry_passes = int(st.get("retry_passes") or 0)
        # A retry pass dropped the agent_error records of its tasks, so those
        # tasks must run again before the run is complete. Until that is done,
        # any record counts as progress.
        recovering = bool(st.get("recovering"))
        retry_complete = bool(st.get("retry_pending"))
        # A start before the saved pause time: wait again, do not run early.
        until = st.get("pause_until")
        if isinstance(until, (int, float)) and until > time.time():
            if not self.wait(float(until), st.get("stop_reason") or "pause"):
                return EXIT_PAUSED
        while True:
            try:
                before = self.done() if recovering else self.done() - self.errors()
            except Exception as e:      # noqa: BLE001 - a broken folder must not stop the loop.
                print(f"progress count failed: {e!r}", flush=True)
                before = -1
            try:
                status, reason, meter = self.chunk(retry_errors_of_complete=retry_complete)
                rate = meter.last_rate or self.last_rate
                self.last_rate = rate
            except SystemExit:
                raise                   # A config error (selection, pin): not transient.
            except Exception as e:      # noqa: BLE001 - a crash is a stop, then a retry.
                status, reason, rate = "stopped", f"crashed: {e!r}"[:300], self.last_rate
            was_retry, retry_complete = retry_complete, False
            if was_retry:
                self.save_state(retry_pending=False)
            try:
                # Progress = a task that finished WITHOUT an agent error. Error
                # records stay in place, so counting them would hide a failure
                # that repeats. While the run gets
                # back the records a failed retry pass dropped, any new record
                # counts.
                progressed = ((self.done() if recovering else
                               self.done() - self.errors()) > before)
            except Exception:           # noqa: BLE001
                progressed = False

            if status == "complete":
                recovering = False      # Every task has a record again.
                errs = self.errors()
                if errs and retry_passes < self.a.retry_passes:
                    retry_passes += 1
                    until = time.time() + BACKOFF[0]
                    self.save_state(retry_passes=retry_passes, strikes=0, last_reason=None,
                                    pause_until=until, stop_reason="agent_error_retry",
                                    retry_pending=True, recovering=False)
                    self.write_progress_safe("retrying", "agent_error",
                                        note=f"{errs} agent_error tasks, pass {retry_passes} "
                                             f"of {self.a.retry_passes}")
                    if not self.wait(until, "agent_error_retry"):
                        return EXIT_PAUSED
                    retry_complete = True
                    continue
                self.save_state(strikes=0, last_reason=None, pause_until=None,
                                retry_passes=retry_passes, recovering=False,
                                retry_pending=False)
                self.write_progress_safe("complete", None,
                                    note=f"{errs} agent_error tasks left" if errs else "")
                return 0

            # A retry pass that fails again is not a reason to block: the normal
            # chunks keep the error records, so the run can still finish.
            if was_retry and reason in ("errors_in_a_row", "no_usage_data"):
                retry_passes, recovering = self.a.retry_passes, True
                self.save_state(retry_passes=retry_passes, recovering=True,
                                retry_pending=False, strikes=0, last_reason=None)
                strikes, last_reason = 0, None
                self.write_progress_safe("running", reason,
                                         note="retry pass failed; run the dropped tasks again")
                continue

            # A usage-limit error while both windows have room is a transient API
            # error (a 429 of the service), not the plan limit: back off instead.
            if reason == "usage_limit_error" and not _rejected(rate):
                fh_u, sd_u = _util(rate, "five_hour"), _util(rate, "seven_day")
                room = (isinstance(fh_u, (int, float)) and isinstance(sd_u, (int, float))
                        and fh_u < (self.a.stop_five_hour or 1.0) - 0.10
                        and sd_u < (self.a.stop_seven_day or 1.0) - 0.10)
                if room:
                    reason = "transient_api_error"

            if reason in ("chunk_limit_five_hour", "five_hour_ceiling", "usage_limit_error",
                          "seven_day_limit"):
                window = "five_hour"
                if (reason == "seven_day_limit"
                        or (rate.get("seven_day") or {}).get("status") == "rejected"):
                    window = "seven_day"
                # Below the ceiling and only the per-chunk cap: go on at once.
                fh = _util(rate, "five_hour")
                if (reason == "chunk_limit_five_hour" and isinstance(fh, (int, float))
                        and fh <= self.a.continue_below and (progressed
                                                             or last_reason != reason)):
                    strikes, last_reason = 0, reason
                    self.save_state(strikes=0, last_reason=reason, pause_until=None)
                    self.write_progress_safe("running", reason, note="next chunk")
                    continue
                until = self.window_reset(window, rate)
                if until is None:
                    # No usable reset time (missing or stale): wait a fixed time
                    # and count a strike, so this cannot busy loop. Progress does
                    # not clear it: one task per chunk would spin for ever.
                    strikes = strikes + 1 if last_reason == reason else 1
                    last_reason = reason
                    if strikes >= MAX_STRIKES:
                        self.save_state(strikes=strikes, last_reason=reason)
                        self.write_progress_safe("blocked", reason,
                                            note="a usage limit with no reset time, 3 times")
                        return EXIT_BLOCKED
                    until = time.time() + BACKOFF[min(strikes - 1, len(BACKOFF) - 1)]
                else:
                    strikes, last_reason = 0, reason
                self.save_state(strikes=strikes, last_reason=last_reason, pause_until=until,
                                stop_reason=reason)
                if not self.wait(until, reason):
                    return EXIT_PAUSED
                continue

            # Errors in a row, missing usage data, or a crash: back off, try again.
            # no_usage_data always counts: without the meter the ceiling cannot work.
            key = (reason or "").split(":")[0]
            if key == "no_usage_data" or (not progressed and key == last_reason):
                strikes += 1
            else:
                strikes = 1
            last_reason = key
            if strikes >= MAX_STRIKES:
                self.save_state(strikes=strikes, last_reason=key)
                self.write_progress_safe("blocked", reason,
                                         note=f"the same stop {MAX_STRIKES} times")
                return EXIT_BLOCKED
            until = time.time() + BACKOFF[min(strikes - 1, len(BACKOFF) - 1)]
            self.save_state(strikes=strikes, last_reason=key, pause_until=until,
                            stop_reason=reason)
            if not self.wait(until, reason or "stopped"):
                return EXIT_PAUSED

    def once(self) -> int:
        status, reason, meter = self.chunk()
        self.last_rate = meter.last_rate or self.last_rate
        print(f"{'ALL COMPLETE' if status == 'complete' else 'STOPPED (' + str(reason) + ')'}; "
              f"usage this chunk +{meter.used:.2f} of the 5-hour window; "
              f"last rate limit {json.dumps(meter.last_rate)}", flush=True)
        self.write_progress(status, reason)
        print(report(self.dirs()))
        return 0 if status == "complete" else EXIT_STOPPED


def filter_tasks(tasks: list, ids_file: str) -> tuple[list, dict]:
    """Keep only the tasks whose id is a line of ids_file (use: re-run only the
    tasks that changed). An id that is not in the task file is an error, so a
    typo or another data version cannot shrink the run without notice."""
    raw = Path(ids_file).read_bytes()
    want = [x.strip() for x in raw.decode("utf-8").splitlines() if x.strip()]
    have = {t.id for t in tasks}
    missing = [i for i in want if i not in have]
    if missing or not want or len(set(want)) != len(want):
        raise SystemExit(f"--task-ids {ids_file}: {len(missing)} ids are not in the task file "
                         f"(first {missing[:3]}), or the list is empty or has duplicates")
    keep = set(want)
    return [t for t in tasks if t.id in keep], {
        "file": str(ids_file), "sha256": hashlib.sha256(raw).hexdigest(), "n": len(keep)}


def _opt_float(x: str) -> float | None:
    """A float, or None for 'none' (no limit)."""
    return None if x.strip().lower() in ("none", "off", "") else float(x)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="stratified N tasks (default 20; with --task-ids default all listed)")
    p.add_argument("--task-ids", metavar="FILE",
                   help="run only the tasks whose id is a line of FILE (before --limit)")
    p.add_argument("--selection-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/pilot")
    p.add_argument("--allow-model-calls", action="store_true")
    p.add_argument("--chunk-five-hour", type=_opt_float, default=0.18,
                   help="stop a chunk after this rise of the 5-hour window; "
                        "'none' = no per-chunk cap, only the ceilings")
    p.add_argument("--stop-seven-day", type=_opt_float, default=0.95)
    p.add_argument("--stop-five-hour", type=_opt_float, default=0.90,
                   help="shared ceiling: stop when the account 5-hour window reaches this")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--sanity", type=int, default=0, metavar="N")
    p.add_argument("--auto", action="store_true")
    p.add_argument("--continue-below", type=float, default=0.70,
                   help="--auto: after a chunk limit, go on at once while the 5-hour "
                        "window is at most this value")
    p.add_argument("--max-inline-wait", type=float, default=float("inf"),
                   help="--auto: a longer pause ends the process with exit 5")
    p.add_argument("--retry-passes", type=int, default=3,
                   help="--auto: passes over the agent_error tasks after all runs finish")
    p.add_argument("--on-event", default="")
    a = p.parse_args(argv)
    if a.limit is None:
        a.limit = 0 if a.task_ids else 20      # 0 = all tasks.
    if not a.allow_model_calls:
        p.error("the pilot calls models; add --allow-model-calls")
    pilot = Pilot(a)
    if a.sanity:
        return pilot.sanity(a.sanity)
    return pilot.auto() if a.auto else pilot.once()


if __name__ == "__main__":
    sys.exit(main())
