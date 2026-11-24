"""The fetch guard against stale data, and the --task-ids filter.

The first full run read SeqQA from the Hugging Face datasets-server cache. That cache
served an older commit, and the meta file recorded the newest one. These tests
check that `fetch` now takes only the pinned parquet file of the pinned
revision, and that it stops loudly when a SHA-256 does not match.
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from bioagent_scorecard import tasks as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

ROWS = [
    {"id": "t1", "question": "Q1?", "ideal": "A", "distractors": ["B", "C"],
     "canary": "c", "source": None, "subtask": "S1"},
    {"id": "t2", "question": "Q2?", "ideal": "D", "distractors": ["E"],
     "canary": "c", "source": None, "subtask": "S2"},
]
PARQUET = b"PAR1 fake parquet bytes PAR1"


def _jsonl_sha(rows) -> str:
    text = "".join(json.dumps(T.Task(id=r["id"], benchmark="lab-bench", subset="SeqQA",
                                     subtask=r["subtask"], question=r["question"],
                                     ideal=r["ideal"], distractors=tuple(r["distractors"]),
                                     canary=r["canary"]).to_dict(), ensure_ascii=False) + "\n"
                   for r in rows)
    return hashlib.sha256(text.encode()).hexdigest()


def _pins(**over):
    pin = {"hf_dataset": "futurehouse/lab-bench", "hf_revision": "a" * 40,
           "parquet_path": "SeqQA/train-00000-of-00001.parquet",
           "parquet_sha256": hashlib.sha256(PARQUET).hexdigest(),
           "rows": 2, "sha256": _jsonl_sha(ROWS)}
    pin.update(over)
    return {"lab-bench/SeqQA": pin}


class FakeDownload:
    def __init__(self, data=PARQUET):
        self.data, self.urls = data, []

    def __call__(self, url):
        self.urls.append(url)
        return self.data


class TestFetchGuard(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.out = self.dir / "lab-bench" / "SeqQA.jsonl"

    def fetch(self, pins, download=None, rows=ROWS):
        return T.fetch_lab_bench("SeqQA", self.dir, pins=pins, download=download or FakeDownload(),
                                 read_rows=lambda _p: [dict(r) for r in rows])

    def test_good_fetch_uses_the_pinned_revision_and_records_it(self):
        dl = FakeDownload()
        path = self.fetch(_pins(), dl)
        self.assertEqual(dl.urls, ["https://huggingface.co/datasets/futurehouse/lab-bench/resolve/"
                                   + "a" * 40 + "/SeqQA/train-00000-of-00001.parquet"])
        self.assertNotIn("datasets-server", dl.urls[0])
        meta = T.load_meta(path)
        self.assertEqual(meta["hf_revision"], "a" * 40)
        self.assertEqual(meta["parquet_sha256"], hashlib.sha256(PARQUET).hexdigest())
        self.assertEqual(meta["sha256_now"], _jsonl_sha(ROWS))
        self.assertEqual(len(T.load_jsonl(path)), 2)

    def test_parquet_sha_mismatch_stops_and_keeps_the_old_file(self):
        self.out.parent.mkdir(parents=True)
        self.out.write_text("old good file\n")
        with self.assertRaisesRegex(RuntimeError, "STALE OR CHANGED DATA"):
            self.fetch(_pins(), FakeDownload(b"other bytes"))
        self.assertEqual(self.out.read_text(), "old good file\n")
        self.assertTrue(self.out.with_suffix(".parquet.unpinned").exists())
        self.assertFalse(Path(str(self.out) + ".meta.json").exists())

    def test_jsonl_sha_mismatch_stops(self):
        changed = [dict(ROWS[0], ideal="B", distractors=["A", "C"]), ROWS[1]]   # A flipped key.
        with self.assertRaisesRegex(RuntimeError, "conversion changed"):
            self.fetch(_pins(), rows=changed)
        self.assertFalse(self.out.exists())
        self.assertTrue(self.out.with_suffix(".jsonl.unpinned").exists())

    def test_row_count_mismatch_stops(self):
        with self.assertRaises(RuntimeError):
            self.fetch(_pins(rows=3))

    def test_no_pin_or_incomplete_pin_stops_before_any_download(self):
        for pins in ({}, _pins(hf_revision=""), _pins(parquet_sha256=None)):
            dl = FakeDownload()
            with self.assertRaisesRegex(RuntimeError, "no complete pin"):
                self.fetch(pins, dl)
            self.assertEqual(dl.urls, [])

    def test_real_pin_is_the_fresh_revision_not_the_stale_cache(self):
        pin = json.loads(T.PINS_FILE.read_text())["lab-bench/SeqQA"]
        self.assertEqual(pin["hf_revision"], "5c77cec648430f30611808808861eb86f81d5eaa")
        self.assertEqual(pin["parquet_sha256"],
                         "5864776d3d85774f91d82432e30ac31e9222f1670621e8ea141414173c5e9b46")
        self.assertEqual(pin["sha256"],
                         "7fa41a863832e2eaed8e419ca0df7bda31ed24e82cc54de2f22520d434868054")
        stale = "7d2d113295d3fc78c35499690bdb6b376b3979f0534dfd1b5ba5c056beed9e78"
        self.assertEqual(T.pin_status({"source": "hf://datasets/futurehouse/lab-bench",
                                       "config": "SeqQA", "sha256_now": stale}), "mismatch")

    def test_no_code_reads_the_datasets_server(self):
        src = (ROOT / "src" / "bioagent_scorecard" / "tasks.py").read_text()
        self.assertNotIn("datasets-server.huggingface.co", src)


class TestTaskIdsLimit(unittest.TestCase):
    def test_limit_below_the_id_count_stops_and_default_takes_all(self):
        import pilot
        d = Path(tempfile.mkdtemp())
        tasks = d / "t.jsonl"
        tasks.write_text("".join(json.dumps(T.Task(id=f"t{i}", benchmark="b", subset="s", subtask="x",
                                                   question="q", ideal="a", distractors=("b",)).to_dict()) + "\n"
                                 for i in range(30)))
        ids = d / "ids.txt"
        ids.write_text("".join(f"t{i}\n" for i in range(25)))
        base = ["--tasks", str(tasks), "--task-ids", str(ids), "--out", str(d / "o"),
                "--agents", "x", "--allow-model-calls"]
        with self.assertRaises(SystemExit):
            pilot.main(base + ["--limit", "20"])
        import argparse
        seen = {}
        orig = pilot.Pilot.once
        pilot.Pilot.once = lambda self: seen.setdefault("n", len(self.ids)) and 0
        try:
            pilot.main(base)
        finally:
            pilot.Pilot.once = orig
        self.assertEqual(seen["n"], 25)


class TestTaskIdsFilter(unittest.TestCase):
    def setUp(self):
        import pilot
        self.pilot = pilot
        self.tasks = [T.Task(id=i, benchmark="b", subset="s", subtask="x", question="q",
                             ideal="a", distractors=("b",)) for i in ("t1", "t2", "t3")]

    def ids(self, text):
        p = Path(tempfile.mkdtemp()) / "ids.txt"
        p.write_text(text)
        return str(p)

    def test_keeps_only_listed_ids_and_records_the_file(self):
        kept, info = self.pilot.filter_tasks(self.tasks, self.ids("t3\nt1\n"))
        self.assertEqual([t.id for t in kept], ["t1", "t3"])
        self.assertEqual(info["n"], 2)
        self.assertEqual(len(info["sha256"]), 64)

    def test_unknown_empty_or_duplicate_ids_stop(self):
        for text in ("t1\nzz\n", "", "t1\nt1\n"):
            with self.assertRaises(SystemExit):
                self.pilot.filter_tasks(self.tasks, self.ids(text))


class TestCombine(unittest.TestCase):
    """scripts/combine_runs.py: old records for unchanged ids + re-run records."""

    def setUp(self):
        import combine_runs
        self.c = combine_runs

    def run_dir(self, recs):
        d = Path(tempfile.mkdtemp())
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (d / "manifest.json").write_text(json.dumps({"git_commit": "x"}))
        return d

    @staticmethod
    def rec(i, outcome):
        return {"task_id": i, "outcome": outcome, "subtask": "S"}

    def test_mcnemar_matches_the_first_run_value(self):
        self.assertAlmostEqual(self.c.mcnemar_exact(47, 28), 0.037, places=3)
        self.assertEqual(self.c.mcnemar_exact(0, 0), 1.0)

    def test_old_stale_ids_are_dropped_and_rerun_ids_added(self):
        old = self.run_dir([self.rec("keep", "correct"), self.rec("stale", "wrong")])
        new = self.run_dir([self.rec("fresh", "refused")])
        recs, info = self.c.combine({"keep", "fresh"}, old, new)
        self.assertEqual(sorted(r["task_id"] for r in recs), ["fresh", "keep"])
        self.assertEqual((info["from_old"], info["from_rerun"], info["old_dropped"]), (1, 1, 1))

    def test_same_id_with_changed_content_is_not_kept(self):
        old = self.run_dir([self.rec("keep", "correct"), self.rec("edited", "correct")])
        new = self.run_dir([self.rec("edited", "wrong")])
        recs, _ = self.c.combine({"keep", "edited"}, old, new, same_content={"keep"})
        self.assertEqual({r["task_id"]: r["outcome"] for r in recs}, {"keep": "correct", "edited": "wrong"})
        with self.assertRaises(SystemExit):        # Edited id, no re-run record: stop.
            self.c.combine({"keep", "edited"}, old, self.run_dir([]), same_content={"keep"})

    def test_manifest_task_sha_and_bad_outcomes_stop(self):
        old = self.run_dir([self.rec("keep", "correct")])
        with self.assertRaises(SystemExit):
            self.c.combine({"keep"}, old, self.run_dir([]), old_sha="f" * 64)
        err = self.run_dir([self.rec("keep", "agent_error")])
        with self.assertRaises(SystemExit):
            self.c.combine({"keep"}, err, self.run_dir([]))

    def test_missing_or_stray_ids_stop(self):
        old = self.run_dir([self.rec("keep", "correct")])
        with self.assertRaises(SystemExit):        # "fresh" has no record.
            self.c.combine({"keep", "fresh"}, old, self.run_dir([]))
        with self.assertRaises(SystemExit):        # A re-run id not in the task file.
            self.c.combine({"keep"}, old, self.run_dir([self.rec("zz", "correct")]))


if __name__ == "__main__":
    unittest.main()
