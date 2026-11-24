# bioagent-scorecard

> **Status: PAUSED (2026-11-24). No further work is planned.**
> Other groups published the same comparison before us (see [Prior work](#prior-work)),
> so the plan for a public leaderboard is dropped. The code and the numbers stay here
> as a record. Issues and pull requests are not monitored.

## What this is

A small test harness for one question: **does a Python tool help an AI model answer
biology questions?**

It asks one model the 600 multiple-choice questions of
[LAB-Bench](https://github.com/Future-House/LAB-Bench) SeqQA (DNA and protein sequence
tasks) two times:

- without tools;
- with one tool, a sandboxed Python interpreter.

Then it compares the two scores on the same questions.

## Result

Model: Claude Sonnet 5.5 through headless Claude Code 2.1.330. One run per question,
seed 0, 21 to 23 November 2026.

| setup | correct, of 600 | accuracy [95% interval] |
|---|---|---|
| with the Python tool | 559 | 93.2% [90.9, 94.9] |
| without tools | 545 | 90.8% [88.3, 92.9] |

In plain words: the model answers about **93 of 100 questions right with the tool and
91 of 100 without it. The difference is too small to call real.** The tool setup wins
on 33 questions and loses on 19 (exact McNemar test, p = 0.070). With 600 questions,
chance alone can make a gap of this size.

For scale: random guessing gets 19.8%, "always pick the longest option" gets 26.1%,
and the human-expert scores in the LAB-Bench paper average to 78.9% on these tasks
(per-subtask values, read from the paper by a script and not checked by hand).

Limits of this result:

- One model, one seed, one run per question.
- The two setups also have different system prompts, not only a different tool list.
- 480 answers come from the first run and 120 from a re-run one day later (see the
  next section).
- The runs used a Claude Max subscription, not an API key. Cost figures in this
  repository are API list-price equivalents, not money paid.
- The per-question result files are not in this repository. The summary is in
  [`docs/results/headline.json`](docs/results/headline.json); the full
  account is in [`docs/results.md`](docs/results.md), section 1.

## The stale-data trap

Our first full run gave different numbers: 84.8% with the tool, 81.7% without,
p = 0.037. We wrote "the tool gives a small, real gain". That was wrong. The cause
was the data, not the model.

1. The harness downloaded SeqQA through the Hugging Face datasets-server `/rows` API
   (the API behind the Dataset Viewer).
2. That API served an **older copy** of the file. Our record said revision `5c77cec6`,
   but the content came from before commit `e2978291`.
3. 120 of the 600 questions were different: all 40 questions of each of 3 primer
   subtasks. The other 480 were the same in every field.
4. On the 120 old questions the model scored about 50%, with many refusals. On the
   120 current questions it scores 94%.

Nothing failed and nothing warned. One stale download changed the conclusion from
"the tool helps" to "no clear difference".

**The guard** (`src/bioagent_scorecard/tasks.py`, `pins.json`):

- `fetch` downloads the parquet file from the git tree of one pinned commit, not from
  the datasets-server.
- `fetch` checks the SHA-256 of the parquet file and of the JSONL file that it writes.
  If one value is not the pinned value, it stops with `STALE OR CHANGED DATA` and
  leaves the existing task file unchanged.
- `run` refuses a fetched task file whose hash is not the pinned hash, unless you give
  `--allow-unpinned`.
- Known gap: a task file at another path, with no meta file, runs with no check.

The lesson for any evaluation: pin the dataset revision, download the file of that
revision, and check its hash before you trust a score.

Credit: another person noted the stale Dataset Viewer copy in
[LAB-Bench issue #20](https://github.com/Future-House/LAB-Bench/issues/20) before we
posted our correction there. Our part is the exact mechanism (the `/rows` API) and
this guard.

## Reproduce

No model calls, no API key, no cost (this is what CI runs):

```bash
git clone https://github.com/Brian-1812/bioagent-scorecard.git
cd bioagent-scorecard
python3 -m venv .venv && . .venv/bin/activate   # Python 3.10 or newer
pip install pyarrow                             # only `fetch` needs it
scripts/dry_run.sh SeqQA 0
```

This runs the unit tests, downloads the pinned SeqQA file (with the hash checks),
runs five dummy agents with the network blocked, and prints the score table with the
no-model baselines.

The model runs need more: Claude Code with a logged-in account (the runs use that
account's usage limits), and Linux with root and systemd for the sandbox.

```bash
PYTHONPATH=src python3 scripts/pilot.py --tasks data/lab-bench/SeqQA.jsonl \
    --limit 600 --out runs/full --allow-model-calls --auto \
    --agents claude-cli:claude-sonnet-5-5 claude-sandbox:claude-sonnet-5-5
```

A new run will not give the same answers question by question: the model is not
deterministic. The sandbox and the usage-limit stops are described in
[`docs/harness.md`](docs/harness.md); the flags of `pilot.py` are in its docstring.

## Prior work

This comparison is not new. Published before this repository:

- Meta, *Code World Model Preparedness Report*
  ([arXiv 2605.00932](https://arxiv.org/abs/2605.00932)): LAB-Bench SeqQA with and
  without a Python tool, for 4 models.
- Edison Scientific, [LABBench2 leaderboard](https://advances.edisonscientific.com/benchmarks/labbench2/):
  "Base" against "Tools" setups on SeqQA2, the successor of SeqQA.
- [LAB-Bench issue #20](https://github.com/Future-House/LAB-Bench/issues/20): defects
  in the SeqQA data and the stale Dataset Viewer copy.

What may still be useful here: the fetch guard, and a worked example of how a stale
download changed a headline.

## Files

- [`docs/harness.md`](docs/harness.md): how the harness works (commands, sandbox,
  usage-limit stops, anchors, layout).
- [`docs/results.md`](docs/results.md): every result, newest first, with the limits of
  each one.
- [`docs/dataset-findings.md`](docs/dataset-findings.md): notes on the SeqQA data.
- [`docs/seqqa-replaced-ids.json`](docs/seqqa-replaced-ids.json): the ids of the 120
  replaced questions (ids only).

## Licences

- Code: Apache-2.0 ([`LICENSE`](LICENSE)).
- LAB-Bench data: CC-BY-SA-4.0, by FutureHouse. This repository does not contain it.
  `fetch` downloads it to `data/`, which git ignores.
- Results derived from the benchmark: CC-BY-SA-4.0.

LAB-Bench is by FutureHouse / Edison Scientific. This project is not affiliated with
them or with Anthropic.
