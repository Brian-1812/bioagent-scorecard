"""Offline tests. They use made-up questions, not benchmark data.

Run: PYTHONPATH=src python3 -m unittest discover -s tests -v
"""

import json
import socket
import tempfile
import unittest
from pathlib import Path

from bioagent_scorecard import checks, scoring
from bioagent_scorecard.agents import get_agent
from bioagent_scorecard.mcq import REFUSE_TEXT, build_mcq, parse_letter
from bioagent_scorecard.netguard import NetworkBlocked, no_network
from bioagent_scorecard.runner import run
from bioagent_scorecard.tasks import load_jsonl, load_meta

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.jsonl"


class TestMCQ(unittest.TestCase):
    def setUp(self):
        self.tasks = load_jsonl(FIXTURE)

    def test_shuffle_is_deterministic_per_seed(self):
        a = [build_mcq(t, 7).options for t in self.tasks]
        b = [build_mcq(t, 7).options for t in reversed(self.tasks)][::-1]
        self.assertEqual(a, b)          # Order of execution does not matter.
        c = [build_mcq(t, 8).options for t in self.tasks]
        self.assertNotEqual(a, c)

    def test_letters_point_to_ideal_and_refusal(self):
        for t in self.tasks:
            m = build_mcq(t, 0)
            self.assertEqual(m.options[ord(m.correct_letter) - 65], t.ideal)
            self.assertEqual(m.options[ord(m.refuse_letter) - 65], REFUSE_TEXT)
            self.assertIn(f"{m.correct_letter}) {t.ideal}", m.prompt)

    def test_parser(self):
        m = build_mcq(self.tasks[0], 0)
        self.assertEqual(parse_letter("thinking... [ANSWER] b [/ANSWER]", m), "B")
        # Tags that disagree are unparsable, not "last tag wins".
        self.assertIsNone(parse_letter("[ANSWER]A[/ANSWER] no, [ANSWER]C[/ANSWER]", m))
        self.assertEqual(parse_letter("C", m), "C")
        self.assertIsNone(parse_letter("I think C is right", m))   # No guessing.
        self.assertIsNone(parse_letter("[ANSWER]Z[/ANSWER]", m))   # Not an option.
        self.assertIsNone(parse_letter(None, m))


class TestScoring(unittest.TestCase):
    def test_wilson(self):
        lo, hi = scoring.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(scoring.wilson(0, 0), (0.0, 0.0))

    def test_position_bias_flag(self):
        recs = [{"letter": "A", "n_options": 5} for _ in range(100)]
        self.assertTrue(checks.check_position_bias(recs))
        recs = [{"letter": "ABCDE"[i % 5], "n_options": 5} for i in range(100)]
        self.assertFalse(checks.check_position_bias(recs))

    def test_answer_key_and_canary_flags(self):
        recs = [{"task_id": "t1", "tool_urls": ["https://huggingface.co/datasets/futurehouse/lab-bench"],
                 "answer_text": "canary GUID lab-bench:xyz"}]
        self.assertEqual(len(checks.check_answer_key_access(recs)), 1)
        self.assertEqual(len(checks.check_canary_in_output(recs)), 1)


class TestRun(unittest.TestCase):
    def _run(self, agent, **kw):
        tmp = tempfile.mkdtemp()
        out = run(get_agent(agent), load_jsonl(FIXTURE), load_meta(FIXTURE), tmp, dry_run=True, **kw)
        return json.loads((out / "summary.json").read_text()), out

    def test_oracle_scores_100(self):
        s, _ = self._run("oracle")
        self.assertEqual(s["accuracy"]["value"], 1.0)
        self.assertEqual(s["coverage"]["value"], 1.0)

    def test_refuse_has_zero_coverage(self):
        s, _ = self._run("refuse")
        self.assertEqual(s["outcomes"]["refused"], 6)
        self.assertEqual(s["coverage"]["value"], 0.0)
        self.assertIsNone(s["precision"]["value"])     # 0/0 is undefined.

    def test_manifest_and_no_question_text_in_results(self):
        _, out = self._run("random", seed=3)
        m = json.loads((out / "manifest.json").read_text())
        self.assertTrue(m["dry_run"])
        self.assertEqual(m["seed"], 3)
        self.assertIn("sha256_now", m["tasks"])
        text = (out / "results.jsonl").read_text()
        self.assertNotIn("reverse complement", text)

    def test_random_is_reproducible(self):
        a, _ = self._run("random", seed=5)
        b, _ = self._run("random", seed=5)
        self.assertEqual(a["outcomes"], b["outcomes"])
        self.assertEqual(a["letter_distribution"], b["letter_distribution"])


class TestNetGuard(unittest.TestCase):
    def test_blocks_connect(self):
        with no_network():
            with self.assertRaises(NetworkBlocked):
                socket.create_connection(("example.com", 443), timeout=1)
        # The guard restores the original functions.
        self.assertNotEqual(socket.create_connection.__name__, "deny")


if __name__ == "__main__":
    unittest.main()
