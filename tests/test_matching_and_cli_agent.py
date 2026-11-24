"""Offline tests for whitespace-normalised matching and
duplicate options (LAB-Bench #12), the longest-option baseline (LAB-Bench #18),
and adapter A through claude -p (recorded fake responses, no model calls).

Run: PYTHONPATH=src python3 -m unittest discover -s tests -v
"""

import json
import socket
import tempfile
import unittest
from pathlib import Path

from bioagent_scorecard import checks, scoring
from bioagent_scorecard.agents import get_agent
from bioagent_scorecard.agents.claude_cli import (API_KEY_VARS, ClaudeCLIAgent,
                                                  parse_cli_json)
from bioagent_scorecard.baselines import RULES, expected_hit, shortcut_baselines
from bioagent_scorecard.cli import main as cli_main
from bioagent_scorecard.mcq import REFUSE_TEXT, build_mcq, norm_ws, parse_letter, same_text
from bioagent_scorecard.netguard import NetworkBlocked, no_network
from bioagent_scorecard.runner import run
from bioagent_scorecard.tasks import Task, load_jsonl, load_meta

FIX = Path(__file__).parent / "fixtures"
SYNTH = FIX / "synthetic.jsonl"


def task(ideal, distractors, tid="t", question="Q?", subtask="s"):
    return Task(id=tid, benchmark="synthetic", subset="S", subtask=subtask,
                question=question, ideal=ideal, distractors=tuple(distractors))


class TestWhitespace(unittest.TestCase):
    def test_norm_ws(self):
        self.assertEqual(norm_ws("  GAATTC \n"), "GAATTC")
        self.assertEqual(norm_ws("Eco RI,\t\nHindIII "), "Eco RI,\nHindIII")   # Line break kept.
        self.assertEqual(norm_ws("a  b\n\n\nc \n"), "a b\nc")
        self.assertTrue(same_text("Eco RI,\nHindIII", "Eco RI, HindIII"))
        self.assertTrue(same_text("TAA\n", "TAA"))
        self.assertFalse(same_text("taa", "TAA"))      # Case is kept.

    def test_trailing_whitespace_ideal_still_maps(self):
        m = build_mcq(task("Met \n", ["Trp", "Leu  "]), 0)
        self.assertEqual(m.options[ord(m.correct_letter) - 65], "Met")
        self.assertIn("Leu", m.options)
        self.assertNotIn("Leu  ", m.options)

    def test_question_keeps_inner_newlines(self):
        m = build_mcq(task("A", ["B"], question="Seq:\nATGC\nGGCC\n\n"), 0)
        # Inner line breaks stay; the trailing blank lines go.
        self.assertIn("Question: Seq:\nATGC\nGGCC\n\nOptions:", m.prompt)
        self.assertNotIn("GGCC\n\n\n", m.prompt)

    def test_duplicate_of_ideal_is_dropped(self):
        # An ideal answer that appears again as a distractor, with a trailing space.
        t = task("EcoRI, BamHI", ["EcoRI, NotI", "EcoRI, BamHI ", "EcoRI, NotI"])
        m = build_mcq(t, 0)
        self.assertEqual(m.dropped_duplicates, 2)
        self.assertEqual(sorted(m.options), sorted(["EcoRI, BamHI", "EcoRI, NotI", REFUSE_TEXT]))
        self.assertEqual(m.options.count("EcoRI, BamHI"), 1)

    def test_shuffle_unchanged_without_duplicates(self):
        # Normalisation must not move letters for clean tasks (old runs stay comparable).
        for t in load_jsonl(SYNTH):
            m = build_mcq(t, 4)
            self.assertEqual(m.dropped_duplicates, 0)

    def test_duplicate_flag(self):
        recs = [{"task_id": "a", "dropped_duplicates": 2}, {"task_id": "b", "dropped_duplicates": 0}]
        f = checks.check_duplicate_options(recs)
        self.assertEqual(f[0]["tasks"], 1)
        self.assertEqual(f[0]["task_ids"], ["a"])
        self.assertEqual(checks.check_duplicate_options(recs[1:]), [])


class TestLongestBaseline(unittest.TestCase):
    def test_agent_picks_longest_not_refusal(self):
        t = task("the long correct answer", ["short", "mid size"])
        m = build_mcq(t, 0)
        out = get_agent("longest").answer(m, 0).text
        self.assertEqual(parse_letter(out, m), m.correct_letter)

    def test_agent_tie_break_is_seeded(self):
        t = task("aaaa", ["bbbb", "c"], tid="tie")
        m = build_mcq(t, 0)
        a = [get_agent("longest").answer(m, s).text for s in range(20)]
        b = [get_agent("longest").answer(m, s).text for s in range(20)]
        self.assertEqual(a, b)
        letters = {parse_letter(x, m) for x in a}
        tied = {chr(65 + m.options.index("aaaa")), chr(65 + m.options.index("bbbb"))}
        self.assertEqual(letters, tied)                 # Both tied options occur, nothing else.

    def test_expected_hit_splits_ties(self):
        self.assertEqual(expected_hit(task("long one", ["a", "b"]), RULES["longest_chars"]), 1.0)
        self.assertEqual(expected_hit(task("aaaa", ["bbbb", "c"]), RULES["longest_chars"]), 0.5)
        self.assertEqual(expected_hit(task("a", ["bbbb"]), RULES["longest_chars"]), 0.0)
        self.assertEqual(expected_hit(task("a", ["bbbb"]), RULES["shortest_chars"]), 1.0)
        self.assertAlmostEqual(expected_hit(task("a", ["b", "c"]), RULES["chance_without_refusal"]), 1 / 3)

    def test_baselines_table(self):
        res = shortcut_baselines(load_jsonl(SYNTH))
        self.assertEqual(res["overall"]["tasks"], 6)
        self.assertEqual(set(res["by_subtask"]), {t.subtask for t in load_jsonl(SYNTH)})
        for v in res["overall"].values():
            self.assertGreaterEqual(v, 0)

    def test_longest_runs_in_dry_run(self):
        out = run(get_agent("longest"), load_jsonl(SYNTH), load_meta(SYNTH),
                  tempfile.mkdtemp(), dry_run=True)
        s = json.loads((out / "summary.json").read_text())
        self.assertEqual(s["outcomes"]["unparsable"], 0)
        self.assertEqual(s["outcomes"]["refused"], 0)


class FakeRunner:
    def __init__(self, fixture, rc=0):
        self.stdout = (FIX / fixture).read_text()
        self.rc = rc
        self.calls = []

    def __call__(self, argv, stdin, env, cwd, timeout):
        self.calls.append({"argv": argv, "stdin": stdin, "env": env, "cwd": cwd})
        return self.rc, self.stdout, ""


class TestClaudeCLIAdapter(unittest.TestCase):
    def setUp(self):
        self.mcq = build_mcq(load_jsonl(SYNTH)[0], 0)

    def agent(self, fixture, rc=0):
        fake = FakeRunner(fixture, rc)
        return ClaudeCLIAgent("claude-haiku-5-5", runner=fake, version="test"), fake

    def test_success_fixture(self):
        a, fake = self.agent("claude_cli_success.json")
        r = a.answer(self.mcq, 0)
        self.assertEqual(r.text, "[ANSWER]B[/ANSWER]")
        self.assertEqual(r.error, "")
        self.assertEqual(r.model, "claude-haiku-5-5")
        self.assertEqual(r.input_tokens, 2 + 751)       # Fresh + cache write + cache read.
        self.assertEqual(r.cached_input_tokens, 0)
        self.assertEqual(r.cache_write_tokens, 751)
        self.assertEqual(r.output_tokens, 18)
        self.assertEqual(r.trace[0]["num_turns"], 1)

    def test_cost_matches_claude_code(self):
        # The harness $ must equal the total_cost_usd that Claude Code reported.
        from bioagent_scorecard.pricing import cost_usd, load_prices
        a, _ = self.agent("claude_cli_success.json")
        r = a.answer(self.mcq, 0)
        c = cost_usd(load_prices(), r.model, r.input_tokens, r.output_tokens,
                     r.cached_input_tokens, r.cache_write_tokens)
        self.assertAlmostEqual(c, r.trace[0]["cli_cost_usd"], places=9)

    def test_other_model_answer_is_an_error(self):
        d = json.loads((FIX / "claude_cli_success.json").read_text())
        d["modelUsage"] = {"claude-opus-5-5": d["modelUsage"]["claude-haiku-5-5"]}
        r = parse_cli_json(json.dumps(d), 0, "", "claude-haiku-5-5")
        self.assertIn("not only claude-haiku-5-5", r.error)

    def test_extra_model_in_usage_is_an_error(self):
        d = json.loads((FIX / "claude_cli_success.json").read_text())
        d["modelUsage"]["claude-opus-5-5"] = d["modelUsage"]["claude-haiku-5-5"]
        r = parse_cli_json(json.dumps(d), 0, "", "claude-haiku-5-5")
        self.assertIn("not only claude-haiku-5-5", r.error)

    def test_argv_has_no_tools_and_restricted(self):
        a, fake = self.agent("claude_cli_success.json")
        a.answer(self.mcq, 0)
        argv = fake.calls[0]["argv"]
        self.assertEqual(argv[:2], ["claude", "-p"])
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        for flag in ("--restricted", "--strict-mcp-config", "--no-session-persistence",
                     "--disable-slash-commands", "--system-prompt"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "")
        self.assertEqual(argv[argv.index("--permission-mode") + 1], "dontAsk")
        self.assertEqual(argv[argv.index("--output-format") + 1], "stream-json")
        self.assertNotIn("--mcp-config", argv)
        self.assertNotIn("--allowedTools", argv)
        self.assertEqual(fake.calls[0]["stdin"], self.mcq.prompt)   # Prompt via stdin, not argv.

    def test_api_key_is_removed_from_child_env(self):
        import os
        a, fake = self.agent("claude_cli_success.json")
        old = {k: os.environ.get(k) for k in API_KEY_VARS}
        try:
            for k in API_KEY_VARS:
                os.environ[k] = "sk-test-not-real"
            a.answer(self.mcq, 0)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for k in API_KEY_VARS:
            self.assertNotIn(k, fake.calls[0]["env"])

    def test_child_env_drops_routing_and_effort_vars(self):
        import os
        a, _ = self.agent("claude_cli_success.json")
        names = ["ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_EFFORT",
                 "MAX_THINKING_TOKENS", "CLAUDECODE", "CLAUDE_CODE_OAUTH_TOKEN"]
        old = {k: os.environ.get(k) for k in names}
        try:
            for k in names:
                os.environ[k] = "x"
            env = a.child_env()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for k in names[:-1]:
            self.assertNotIn(k, env)
        os.environ.setdefault("CLAUDE_CONFIG_DIR", "/x")
        try:
            self.assertIn("CLAUDE_CONFIG_DIR", a.child_env())
        finally:
            if os.environ.get("CLAUDE_CONFIG_DIR") == "/x":
                os.environ.pop("CLAUDE_CONFIG_DIR")
        self.assertIn("CLAUDE_CODE_OAUTH_TOKEN", env)   # The subscription login stays.
        self.assertIn("PATH", env)

    def test_refusal_fixture(self):
        a, _ = self.agent("claude_cli_refusal.json")
        r = a.answer(self.mcq, 0)
        self.assertTrue(r.safety_refusal)
        self.assertEqual(r.error, "")

    def test_error_fixture(self):
        a, _ = self.agent("claude_cli_error.json", rc=1)
        r = a.answer(self.mcq, 0)
        self.assertIn("429", r.error)

    def test_not_json(self):
        r = parse_cli_json("Credit balance too low", 1, "", "claude-haiku-5-5")
        self.assertIn("no JSON result", r.error)

    def test_multi_turn_is_an_error(self):
        d = json.loads((FIX / "claude_cli_success.json").read_text())
        d["num_turns"] = 3
        r = parse_cli_json(json.dumps(d), 0, "", "claude-haiku-5-5")
        self.assertIn("expected at most 1 turn", r.error)

    def test_config_says_max_plan(self):
        a, _ = self.agent("claude_cli_success.json")
        cfg = a.config()
        self.assertIn("max-subscription", cfg["billing"])
        self.assertEqual(cfg["model"], "claude-haiku-5-5")

    def test_run_with_fake_cli(self):
        a, _ = self.agent("claude_cli_success.json")
        # The json fixture has no rate_limit_event, so turn off the no-usage stop.
        from bioagent_scorecard.runner import ChunkLimits
        out = run(a, load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp(),
                  chunk=ChunkLimits(max_calls_without_usage=None))
        m = json.loads((out / "manifest.json").read_text())
        s = json.loads((out / "summary.json").read_text())
        self.assertIn("max-subscription", m["billing"])
        self.assertIn("claude-haiku-5-5", m["prices"]["models"])
        self.assertEqual(s["tasks"], 6)
        self.assertEqual(s["outcomes"]["agent_error"], 0)
        self.assertGreater(s["cost_usd"]["total"], 0)

    def test_dry_run_rejects_model_agent(self):
        a, _ = self.agent("claude_cli_success.json")
        with self.assertRaises(SystemExit):
            run(a, load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp(), dry_run=True)

    def test_cli_needs_allow_flag(self):
        import contextlib, io
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli_main(["run", "--agent", "claude-cli:claude-haiku-5-5", "--tasks", str(SYNTH),
                      "--out", tempfile.mkdtemp()])


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.mcq = build_mcq(load_jsonl(SYNTH)[0], 0)

    def test_disagreeing_tags_are_unparsable(self):
        self.assertIsNone(parse_letter("[ANSWER]A[/ANSWER] or [ANSWER]C[/ANSWER]", self.mcq))
        self.assertEqual(parse_letter("[ANSWER]C[/ANSWER] so [ANSWER]c[/ANSWER]", self.mcq), "C")

    def test_bare_lower_case_letter(self):
        self.assertEqual(parse_letter("c", self.mcq), "C")

    def test_undefined_rate_is_null(self):
        r = scoring._rate(0, 0)
        self.assertIsNone(r["value"])
        self.assertIsNone(r["ci95"])

    def test_missing_price_still_scores(self):
        class NoPrice:
            name, makes_api_calls = "noprice", False
            def config(self): return {"agent": "noprice"}
            def answer(self, mcq, seed):
                from bioagent_scorecard.agents.base import AgentResult
                return AgentResult(text="[ANSWER]A[/ANSWER]", model="model-without-price")
        out = run(NoPrice(), load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp())
        s = json.loads((out / "summary.json").read_text())
        self.assertEqual(s["outcomes"]["agent_error"], 0)
        self.assertEqual(s["tasks"], 6)
        self.assertIn("missing_price", [f["check"] for f in s["integrity_flags"]])

    def test_same_second_runs_do_not_clash(self):
        tmp = tempfile.mkdtemp()
        a = run(get_agent("first"), load_jsonl(SYNTH), load_meta(SYNTH), tmp, dry_run=True)
        b = run(get_agent("first"), load_jsonl(SYNTH), load_meta(SYNTH), tmp, dry_run=True)
        self.assertNotEqual(a, b)

    def test_netguard_blocks_dns_and_udp(self):
        with no_network():
            with self.assertRaises(NetworkBlocked):
                socket.gethostbyname("example.com")
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                with self.assertRaises(NetworkBlocked):
                    s.sendto(b"x", ("127.0.0.1", 9))
            finally:
                s.close()
        self.assertNotEqual(socket.gethostbyname.__name__, "deny")


if __name__ == "__main__":
    unittest.main()
