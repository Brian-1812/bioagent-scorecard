"""Usage per task in the report; round-robin run order; the pilot script."""

import json
import tempfile
import unittest
from pathlib import Path

from bioagent_scorecard.cli import five_hour_use
from bioagent_scorecard.selection import run_order
from bioagent_scorecard.tasks import Task

SYNTHETIC = Path(__file__).parent / "fixtures" / "synthetic.jsonl"
EXPECTED_ORDER = ["a0", "b0", "c0", "a2", "c1", "a1"]


def _row(u, resets=100):
    return {"task_id": "x", "rate_limit": {"five_hour": {"utilization": u, "resets_at": resets}}}


class TestFiveHourUse(unittest.TestCase):
    def write(self, rows, tail=""):
        d = Path(tempfile.mkdtemp())
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + tail)
        return d

    def test_sums_rises_per_interval(self):
        d = self.write([_row(0.10), _row(0.12), _row(0.12), _row(0.15)])
        u = five_hour_use(d)
        self.assertAlmostEqual(u["five_hour_rise"], 0.05)
        self.assertEqual(u["intervals"], 3)
        self.assertAlmostEqual(u["per_task"], 0.05 / 3, places=4)

    def test_new_window_starts_again(self):
        # The window resets between call 2 and call 3: that pair is not an interval.
        d = self.write([_row(0.90), _row(0.95), _row(0.05, 200), _row(0.10, 200)])
        u = five_hour_use(d)
        self.assertAlmostEqual(u["five_hour_rise"], 0.10)
        self.assertEqual(u["intervals"], 2)
        self.assertAlmostEqual(u["per_task"], 0.05)

    def test_rows_without_usage_break_the_chain(self):
        d = self.write([_row(0.1), {"task_id": "a"}, _row(0.2), _row(0.25)])
        u = five_hour_use(d)
        self.assertEqual((u["tasks"], u["tasks_with_usage"], u["intervals"]), (4, 3, 1))
        self.assertAlmostEqual(u["per_task"], 0.05)

    def test_no_interval_is_none(self):
        for rows in ([], [_row(0.3)], [{"task_id": "a"}, {"task_id": "b", "rate_limit": {}}],
                     [_row(0.1, 1), _row(0.2, 2), _row(0.3, 3)]):
            self.assertIsNone(five_hour_use(self.write(rows))["per_task"], rows)
        self.assertIsNone(five_hour_use(Path(tempfile.mkdtemp()))["per_task"])

    def test_truncated_last_line(self):
        d = self.write([_row(0.1), _row(0.2)], tail='{"task_id": "x", "rate_')
        u = five_hour_use(d)
        self.assertEqual(u["tasks"], 2)
        self.assertAlmostEqual(u["per_task"], 0.1)


class TestRunOrder(unittest.TestCase):
    def tasks(self):
        return [Task(id=f"{sub}{i}", benchmark="b", subset="s", subtask=sub, question="q",
                     ideal="i", distractors=("d",))
                for sub, n in (("a", 3), ("b", 1), ("c", 2)) for i in range(n)]

    def test_same_set_round_robin(self):
        ts = self.tasks()
        order = run_order(ts, 0)
        self.assertEqual(sorted(t.id for t in order), sorted(t.id for t in ts))
        # The first 3 calls cover all 3 subtasks; the next 2 cover a and c.
        self.assertEqual({t.subtask for t in order[:3]}, {"a", "b", "c"})
        self.assertEqual({t.subtask for t in order[3:5]}, {"a", "c"})

    def test_fixed_order(self):
        # A change of the hash keys changes this order; then update it on purpose.
        self.assertEqual([t.id for t in run_order(self.tasks(), 0)], EXPECTED_ORDER)

    def test_does_not_depend_on_input_order(self):
        ts = self.tasks()
        self.assertEqual([t.id for t in run_order(ts, 1)],
                         [t.id for t in run_order(list(reversed(ts)), 1)])


class TestResumeOrder(unittest.TestCase):
    """A stopped run resumes in the same order as a run without a stop, also
    for a folder made before the round-robin order (same task_ids)."""

    def test_resume_keeps_order(self):
        from bioagent_scorecard.agents import get_agent
        from bioagent_scorecard.runner import ChunkLimits, run
        from bioagent_scorecard.tasks import load_jsonl, load_meta
        path = str(SYNTHETIC)
        tasks, meta = load_jsonl(path), load_meta(path)
        agent = get_agent("first")
        root = Path(tempfile.mkdtemp())
        full = run(agent, tasks, meta, root / "a", dry_run=True, allow_unpinned=True)
        want = [json.loads(x)["task_id"] for x in (full / "results.jsonl").read_text().splitlines()]
        # An old folder: the first 2 tasks in file (block) order, then a resume.
        part = run(agent, tasks, meta, root / "b", dry_run=True, allow_unpinned=True,
                   chunk=ChunkLimits(max_tasks=2))
        lines = (part / "results.jsonl").read_text().splitlines()
        old_first = [t.id for t in tasks[:2]]
        recs = [json.loads(x) for x in lines]
        for r, tid in zip(recs, old_first):
            r["task_id"] = tid
        (part / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        run(agent, tasks, meta, root / "b", dry_run=True, allow_unpinned=True, resume_dir=part)
        got = [json.loads(x)["task_id"] for x in (part / "results.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(got), sorted(want))
        self.assertEqual(len(got), len(set(got)))
        self.assertEqual(got[2:], [t for t in want if t not in old_first])
        m = json.loads((part / "manifest.json").read_text())
        self.assertEqual(m["status"], "complete")
        self.assertEqual(m["chunks"][-1]["run_order"], "round_robin_by_subtask")


class TestFiveHourCeiling(unittest.TestCase):
    """The shared account ceiling: 0.90 of each window."""

    def rate(self, fh, sd=0.1):
        return {"five_hour": {"utilization": fh, "resets_at": 1},
                "seven_day": {"utilization": sd, "resets_at": 2}}

    def meter(self, **kw):
        from bioagent_scorecard.runner import ChunkLimits, UsageMeter
        return UsageMeter(ChunkLimits(**kw))

    def test_stops_at_the_ceiling_even_without_a_chunk_cap(self):
        m = self.meter(max_five_hour=None, stop_five_hour=0.90)
        self.assertIsNone(m.update(self.rate(0.50)))
        self.assertIsNone(m.update(self.rate(0.89)))
        self.assertEqual(m.update(self.rate(0.90)), "five_hour_ceiling")
        # The stop stays while the window is high: the next run stops at once.
        self.assertEqual(m.check(), "five_hour_ceiling")

    def test_other_sessions_count(self):
        # This run adds 0.02; the account jumps to 0.95 from another session.
        m = self.meter(max_five_hour=None, stop_five_hour=0.90)
        m.update(self.rate(0.30))
        self.assertEqual(m.update(self.rate(0.95)), "five_hour_ceiling")

    def test_ceiling_off(self):
        m = self.meter(max_five_hour=None, stop_five_hour=None, stop_seven_day=None)
        self.assertIsNone(m.update(self.rate(0.99, sd=0.99)))

    def test_seven_day_ceiling(self):
        m = self.meter(max_five_hour=None, stop_five_hour=0.90, stop_seven_day=0.90)
        self.assertEqual(m.update(self.rate(0.10, sd=0.90)), "seven_day_limit")


def _load_pilot():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPilotAuto(unittest.TestCase):
    """The --auto loop decisions, with chunk() faked (no model calls)."""

    def make(self, steps, **kw):
        from unittest import mock
        pilot = _load_pilot()
        out = Path(tempfile.mkdtemp())
        args = ["--tasks", str(SYNTHETIC), "--agents", "first", "--out", str(out),
                "--allow-model-calls", "--auto", "--limit", "0"]
        for k, v in kw.items():
            args += [f"--{k.replace('_', '-')}", str(v)]
        p = pilot.Pilot.__new__(pilot.Pilot)
        ns = _parse(pilot, args)
        pilot.Pilot.__init__(p, ns)
        it = iter(steps)
        done = {"n": 0, "err": 0}

        def fake_chunk(retry_errors_of_complete=False):
            status, reason, rate, progressed = next(it)
            if progressed:
                done["n"] += 1
            elif progressed is None:        # A task finished, but with an error.
                done["n"] += 1
                done["err"] += 1
            m = pilot.UsageMeter(p.limits())
            m.last_rate = rate
            if isinstance(status, Exception):
                raise status
            return status, reason, m

        p.chunk = fake_chunk
        p.done = lambda: done["n"]
        p.errors = lambda: done["err"]
        p.dirs = lambda: []
        p.write_progress_safe = p.write_progress
        sleeps = []
        patch = mock.patch.object(pilot.time, "sleep", lambda s: sleeps.append(s))
        return pilot, p, out, sleeps, patch

    def rate(self, fh, sd=0.1, fh_reset=None, sd_reset=None, sd_status="allowed"):
        import time as _t
        now = _t.time()
        return {"five_hour": {"utilization": fh, "resets_at": fh_reset or now + 3600},
                "seven_day": {"utilization": sd, "resets_at": sd_reset or now + 86400,
                              "status": sd_status}}

    def test_chunk_limit_low_window_goes_on(self):
        pilot, p, out, sleeps, patch = self.make([
            ("stopped", "chunk_limit_five_hour", self.rate(0.5), True),
            ("complete", None, self.rate(0.6), True)])
        with patch:
            self.assertEqual(p.auto(), 0)
        self.assertEqual(sleeps, [])
        self.assertEqual(json.loads((out / "progress.json").read_text())["status"], "complete")

    def test_chunk_limit_high_window_pauses(self):
        import time as _t
        reset = _t.time() + 4000
        r = self.rate(0.85, fh_reset=reset)
        pilot, p, out, sleeps, patch = self.make(
            [("stopped", "chunk_limit_five_hour", r, True)], max_inline_wait=600)
        with patch:
            self.assertEqual(p.auto(), pilot.EXIT_PAUSED)
        prog = json.loads((out / "progress.json").read_text())
        self.assertEqual(prog["status"], "paused")
        self.assertAlmostEqual(prog["pause_until"], reset + pilot.RESET_MARGIN, places=3)

    def test_ceiling_always_pauses(self):
        # five_hour_ceiling never continues at once, also below --continue-below.
        import time as _t
        reset = _t.time() + 50
        pilot, p, out, sleeps, patch = self.make([
            ("stopped", "five_hour_ceiling", self.rate(0.91, fh_reset=reset), True),
            ("complete", None, self.rate(0.2), True)])
        with patch:
            self.assertEqual(p.auto(), 0)
        self.assertEqual(len(sleeps), 1)

    def test_opt_float(self):
        pilot = _load_pilot()
        self.assertIsNone(pilot._opt_float("none"))
        self.assertEqual(pilot._opt_float("0.9"), 0.9)

    def test_seven_day_rejected_waits_for_seven_day(self):
        import time as _t
        sd_reset = _t.time() + 100
        r = self.rate(0.2, sd=1.0, sd_reset=sd_reset, sd_status="rejected")
        pilot, p, out, sleeps, patch = self.make([
            ("stopped", "usage_limit_error", r, False),
            ("complete", None, self.rate(0.1), True)])
        with patch:
            self.assertEqual(p.auto(), 0)
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], sd_reset + pilot.RESET_MARGIN - _t.time(), delta=5)

    def test_same_error_three_times_blocks(self):
        pilot, p, out, sleeps, patch = self.make(
            [("stopped", "errors_in_a_row", self.rate(0.1), False)] * 3)
        with patch:
            self.assertEqual(p.auto(), pilot.EXIT_BLOCKED)
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(json.loads((out / "progress.json").read_text())["status"], "blocked")

    def test_progress_resets_the_strikes(self):
        pilot, p, out, sleeps, patch = self.make([
            ("stopped", "errors_in_a_row", self.rate(0.1), False),
            ("stopped", "errors_in_a_row", self.rate(0.1), False),
            ("stopped", "errors_in_a_row", self.rate(0.1), True),
            ("complete", None, self.rate(0.1), True)])
        with patch:
            self.assertEqual(p.auto(), 0)

    def test_usage_limit_without_reset_time_cannot_spin(self):
        # No rate data at all: the pause must still be a real wait, then block.
        pilot, p, out, sleeps, patch = self.make(
            [("stopped", "usage_limit_error", {}, False)] * 3)
        with patch:
            self.assertEqual(p.auto(), pilot.EXIT_BLOCKED)
        self.assertEqual(len(sleeps), 2)
        self.assertGreaterEqual(min(sleeps), pilot.BACKOFF[0] - 5)

    def test_stale_reset_time_cannot_spin(self):
        import time as _t
        past = self.rate(0.95, fh_reset=_t.time() - 500)
        pilot, p, out, sleeps, patch = self.make([
            ("stopped", "usage_limit_error", past, False),
            ("complete", None, self.rate(0.2), True)])
        with patch:
            self.assertEqual(p.auto(), 0)
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], pilot.BACKOFF[0] - 5)

    def test_transient_429_gets_a_backoff_not_a_window_pause(self):
        import time as _t
        far = self.rate(0.20, sd=0.05, fh_reset=_t.time() + 10_000)
        pilot, p, out, sleeps, patch = self.make([
            ("stopped", "usage_limit_error", far, True),
            ("complete", None, far, True)])
        with patch:
            self.assertEqual(p.auto(), 0)
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], pilot.BACKOFF[0], delta=5)

    def test_no_usage_data_blocks_even_with_progress(self):
        pilot, p, out, sleeps, patch = self.make(
            [("stopped", "no_usage_data", self.rate(0.3), True)] * 3)
        with patch:
            self.assertEqual(p.auto(), pilot.EXIT_BLOCKED)

    def test_strikes_survive_a_restart(self):
        rate = self.rate(0.3)
        for i in range(2):
            pilot, p, out, sleeps, patch = self.make(
                [("stopped", "errors_in_a_row", rate, False)], max_inline_wait=1)
            if i:
                p.out = self.shared
                p.state_file = self.shared / "auto-state.json"
                p.save_state(pause_until=None)      # The outer loop slept already.
            else:
                self.shared = p.out
            with patch:
                rc = p.auto()
            self.assertEqual(rc, pilot.EXIT_PAUSED)
        # Third start with the saved strikes: this one blocks.
        pilot, p, out, sleeps, patch = self.make(
            [("stopped", "errors_in_a_row", rate, False)], max_inline_wait=1)
        p.out, p.state_file = self.shared, self.shared / "auto-state.json"
        p.save_state(pause_until=None)
        with patch:
            self.assertEqual(p.auto(), pilot.EXIT_BLOCKED)

    def test_an_early_start_waits(self):
        import time as _t
        pilot, p, out, sleeps, patch = self.make([("complete", None, self.rate(0.2), True)])
        p.save_state(pause_until=_t.time() + 120, stop_reason="five_hour_ceiling")
        with patch:
            self.assertEqual(p.auto(), 0)
        self.assertEqual(len(sleeps), 1)
        self.assertGreater(sleeps[0], 60)

    def test_a_failing_retry_pass_does_not_block(self):
        pilot, p, out, sleeps, patch = self.make([
            ("complete", None, self.rate(0.2), True),       # done, with error records
            ("stopped", "errors_in_a_row", self.rate(0.2), False),   # retry pass fails
            ("complete", None, self.rate(0.2), False)], retry_passes=1)
        p.errors = lambda: 2
        with patch:
            self.assertEqual(p.auto(), 0)
        prog = json.loads((out / "progress.json").read_text())
        self.assertEqual(prog["status"], "complete")
        self.assertIn("agent_error tasks left", prog["note"])

    def test_error_records_are_not_progress(self):
        """Every task fails, but the records stay, so
        done() keeps rising. The run must still block, not call 1,200 times."""
        pilot, p, out, sleeps, patch = self.make(
            [("stopped", "errors_in_a_row", self.rate(0.3), None)] * 6)
        with patch:
            self.assertEqual(p.auto(), pilot.EXIT_BLOCKED)
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(json.loads((out / "progress.json").read_text())["status"], "blocked")

    def test_one_good_task_per_chunk_still_blocks_over_the_ceiling(self):
        """Over the ceiling with no reset time, one task per chunk
        must not clear the strike."""
        no_reset = {"five_hour": {"utilization": 0.93}, "seven_day": {"utilization": 0.5}}
        pilot, p, out, sleeps, patch = self.make(
            [("stopped", "five_hour_ceiling", no_reset, True)] * 4)
        with patch:
            self.assertEqual(p.auto(), pilot.EXIT_BLOCKED)

    def test_a_pause_is_capped(self):
        # resets_at in milliseconds must not sleep for ever.
        import time as _t
        bad = {"five_hour": {"utilization": 0.95, "resets_at": _t.time() * 1000},
               "seven_day": {"utilization": 0.5}}
        pilot, p, out, sleeps, patch = self.make([
            ("stopped", "five_hour_ceiling", bad, True),
            ("complete", None, self.rate(0.2), True)])
        with patch:
            self.assertEqual(p.auto(), 0)
        self.assertLessEqual(sleeps[0], pilot.MAX_PAUSE["five_hour"] + 5)

    def test_a_stale_reset_time_is_not_used(self):
        import time as _t
        pilot = _load_pilot()
        self.assertIsNone(pilot.Pilot.window_reset(
            "five_hour", {"five_hour": {"resets_at": _t.time() - 10}}))
        self.assertIsNotNone(pilot.Pilot.window_reset(
            "five_hour", {"five_hour": {"resets_at": _t.time() + 600}}))

    def test_a_failed_retry_pass_lets_the_dropped_tasks_come_back(self):
        """A retry pass drops the error records of its
        tasks. If it then fails, the normal chunks must be able to write those
        records again. Error records count as progress until the run is whole."""
        pilot, p, out, sleeps, patch = self.make([
            ("complete", None, self.rate(0.2), True),               # all tasks have records
            ("stopped", "errors_in_a_row", self.rate(0.2), False),  # retry pass fails
            ("stopped", "errors_in_a_row", self.rate(0.2), None),   # a record, with an error
            ("stopped", "errors_in_a_row", self.rate(0.2), None),
            ("complete", None, self.rate(0.2), None)], retry_passes=1)
        p.errors = lambda: 2
        with patch:
            self.assertEqual(p.auto(), 0)
        prog = json.loads((out / "progress.json").read_text())
        self.assertEqual(prog["status"], "complete")
        self.assertIn("agent_error tasks left", prog["note"])

    def test_a_retry_pass_survives_a_restart(self):
        """With a short --max-inline-wait the process exits
        before the retry wait ends. The next start must run the retry pass."""
        pilot, p, out, sleeps, patch = self.make(
            [("complete", None, self.rate(0.2), True)], max_inline_wait=1, retry_passes=1)
        p.errors = lambda: 1
        with patch:
            self.assertEqual(p.auto(), pilot.EXIT_PAUSED)
        self.assertTrue(json.loads((out / "auto-state.json").read_text())["retry_pending"])
        seen = {}
        pilot2, p2, out2, sleeps2, patch2 = self.make(
            [("complete", None, self.rate(0.2), True)], max_inline_wait=1, retry_passes=1)
        p2.out, p2.state_file = out, out / "auto-state.json"
        p2.errors = lambda: 1
        p2.save_state(pause_until=None)
        inner = p2.chunk

        def spy(retry_errors_of_complete=False):
            seen["retry"] = retry_errors_of_complete
            return inner(retry_errors_of_complete)

        p2.chunk = spy
        with patch2:
            p2.auto()
        self.assertTrue(seen.get("retry"), "the restart did not run the retry pass")

    def test_the_seven_day_pause_is_not_cut_to_six_hours(self):
        import time as _t
        pilot = _load_pilot()
        far = _t.time() + 3 * 86400
        got = pilot.Pilot.window_reset("seven_day", {"seven_day": {"resets_at": far}})
        self.assertAlmostEqual(got, far + pilot.RESET_MARGIN, places=3)

    def test_crash_is_retried(self):
        pilot, p, out, sleeps, patch = self.make([
            (RuntimeError("boom"), None, {}, False),
            ("complete", None, self.rate(0.1), True)])
        with patch:
            self.assertEqual(p.auto(), 0)
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], pilot.BACKOFF[0], delta=5)


class TestPilotSanity(unittest.TestCase):
    def run_sanity(self, agents, n=2):
        import contextlib
        import io
        pilot = _load_pilot()
        out = Path(tempfile.mkdtemp())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = pilot.main(["--tasks", str(SYNTHETIC), "--agents", *agents,
                             "--out", str(out), "--allow-model-calls", "--limit", "0",
                             "--sanity", str(n)])
        return rc, buf.getvalue(), out

    def test_sanity_ok(self):
        rc, text, out = self.run_sanity(["oracle"])
        self.assertEqual(rc, 0, text)
        self.assertIn("SANITY oracle: OK (2 tasks)", text)
        for d in out.glob("*/manifest.json"):
            self.assertEqual(json.loads(d.read_text())["done"], 2)

    def test_sanity_fails_when_nothing_is_correct(self):
        # The check is not circular: an agent that answers A every time fails.
        rc, text, _ = self.run_sanity(["first"])
        self.assertEqual(rc, 4)
        self.assertIn("no correct answer", text)

    def test_sanity_accepts_a_cut_answer_text(self):
        """A long answer is cut, so the tag can be missing from the
        saved text. That must not fail the check."""
        from bioagent_scorecard.runner import ANSWER_TEXT_LIMIT
        pilot = _load_pilot()
        rc, text, out = self.run_sanity(["oracle"])
        self.assertEqual(rc, 0)
        d = next(out.glob("*/results.jsonl")).parent
        recs = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]
        recs[0]["answer_text"] = "x" * ANSWER_TEXT_LIMIT      # The tag is gone.
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc2 = pilot.main(["--tasks", str(SYNTHETIC), "--agents", "oracle",
                              "--out", str(out), "--allow-model-calls", "--limit", "0",
                              "--sanity", "2"])
        self.assertEqual(rc2, 0, buf.getvalue())

    def test_sanity_catches_a_wrong_letter_field(self):
        pilot = _load_pilot()
        rc, text, out = self.run_sanity(["oracle"])
        self.assertEqual(rc, 0)
        d = next(out.glob("*/results.jsonl")).parent
        recs = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]
        recs[0]["letter"] = "B" if recs[0]["letter"] != "B" else "C"
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc2 = pilot.main(["--tasks", str(SYNTHETIC), "--agents", "oracle",
                              "--out", str(out), "--allow-model-calls", "--limit", "0",
                              "--sanity", "2"])
        self.assertEqual(rc2, 4, buf.getvalue())
        self.assertIn("!= parse", buf.getvalue())


def _parse(pilot, args):
    import argparse
    from unittest import mock
    captured = {}

    class Stop(Exception):
        pass

    def fake_init(self, a):
        captured["a"] = a
        raise Stop

    with mock.patch.object(pilot.Pilot, "__init__", fake_init):
        try:
            pilot.main(args)
        except Stop:
            pass
    return captured["a"]


if __name__ == "__main__":
    unittest.main()
