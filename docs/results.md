# Results log

> **PAUSED (2026-11-24).** No further work is planned. Other groups published the same
> comparison first (see the README, "Prior work"). This page stays as a record.

This page holds every result of the project, newest first.

- Section 1 is the final result.
- Sections 2 to 6 are the record of the earlier runs. Sections 2 to 6 used a stale
  copy of the data (see section 1). Their numbers are superseded.

All model runs used a Claude Max subscription through headless Claude Code, not an API
key. Every $ figure is the API list-price equivalent, not money paid.

Rows: **A** = the model with no tools. **B** = the model with one sandboxed Python tool
(`run_python`). The anchor `s` is `(accuracy - longest-option shortcut) / (human - longest-option shortcut)`.

Tags: **[checked]** = we checked it at the source. **[measured]** = from our run files.
**[reported]** = a third party reports it; we did not reproduce it. **[inference]** = our
reasoning, not a measurement.

## 1. Final result on the current data (2026-11-23)

**Why the first result changed.** The first full run (section 2) read SeqQA from the Hugging Face
datasets-server cache (`refs/convert/parquet`). That cache held the SeqQA parquet of
the commits before `e2978291` (for example `2050d471`), but our meta file
recorded `5c77cec6`. The fresh SeqQA parquet first appears at `e2978291`,
5 seconds before `5c77cec6`. The parquet file
of `5c77cec6` (sha256 `5864776d...`) is the same as GitHub `998a8e0` on all 600
tasks and all fields. Against the 600 tasks we ran:
- 480 tasks are the same in every field (id, question, ideal, distractors, canary, subtask).
- 120 tasks were replaced: all 40 of `PCR-gene-enzprimers`, `PCR-seq-enzprimers`
  and `PCR-geneprimers-enz`. They have new ids, genes and enzymes. Only 9 old
  question texts appear again (with new ids). Ids: `docs/seqqa-replaced-ids.json`.
So the change is 120 tasks in 3 subtasks, not the 80 enzprimers tasks only.

**What ran.** Both rows on the 120 new tasks only (`pilot.py --task-ids`), with
the same agent config (stable hashes `09c8a7b540de` B and `8b0e5866f4ea` A, the
same as the first full run), the same model (Sonnet 5.5), Claude Code 2.1.330, seed 0, on a
Claude Max subscription. 240 of 240 tasks, 0 agent errors. The first-run results for the 480
unchanged tasks stay as they are. `scripts/combine_runs.py` joins them and
stops if any task of the pinned file is missing or doubled. The old run
folders are unchanged.

| row (600 tasks, pinned `5c77cec6`) | accuracy [95% CI] | correct | refused | wrong | anchor s |
|---|---|---|---|---|---|
| **B: sandboxed `run_python` (main setup)** | **0.932 [0.909, 0.949]** | 559 | 6 | 35 | 1.27 |
| A: no tools (baseline) | 0.908 [0.883, 0.929] | 545 | 3 | 52 | 1.23 |

Anchors on the 600 fresh tasks: chance 0.198, longest-option shortcut **0.261**
(0.269 on the stale copy), human experts 0.789 **[reported, per subtask, not
checked by a person]**. Both intervals are now above the human figure.

**Paired test (same 600 tasks).** Row B is correct on 33 tasks that row A
misses; row A is correct on 19 that row B misses. That is +2.3 points, exact
McNemar **p = 0.070**. **The first-run claim "the tool gives a small, real gain" (p =
0.037) does not hold on the fresh data.** The point estimate still favours
row B, but 600 tasks do not show the gain at the 0.05 level. On the 480
unchanged tasks: 32 against 18, p = 0.065. On the 120 new tasks: 1 against 1.

**The 120 new tasks.** Both rows: 113 correct, 5 wrong, 2 refused (0.942
[0.884, 0.972]). They differ on 2 tasks only. Row B made 0 tool calls there.
In the first full run, row B also made 0 tool calls in these 3 subtasks (0 of 120), so this
is the same behaviour, not a broken tool. The stale 120 had scored 0.525 (B)
and 0.483 (A), with 55 and 59 refusals.

| subtask (40 new tasks each) | B | A | human (reported) |
|---|---|---|---|
| PCR-gene-enzprimers | 0.975 | 0.975 | 0.83 |
| PCR-seq-enzprimers | 0.950 (2 refused) | 0.950 (2 refused) | 0.81 |
| PCR-geneprimers-enz | 0.900 | 0.900 | 0.94 |

**The 0.848 / 0.931 choice is gone.** It came from 80 tasks that are not in the
current data. The single public headline is now row B, **0.932 [0.909, 0.949]
on the 600 tasks of `5c77cec6` / `998a8e0`**, with row A 0.908 as the baseline.
Note: 0.931 (old, 520 tasks) and 0.932 (new, 600 tasks) are close by chance.
They are different task sets.

**What this does not prove.**
- The 480 old results ran on 2026-11-21/22, the 120 new ones on 2026-11-22/23.
  The config, the CLI version and the model name are the same. A change on the
  model server between those days is possible; we cannot see it **[not checked]**.
- One seed, one run per task, one model. The two rows also differ in the system
  prompt, not only in the tool.
- The human ceiling and the 0.83 / 0.81 expert scores are for the paper's
  version of these subtasks; we do not know which version the experts saw.
- Our question 2 on LAB-Bench issue #20 is not settled. In the fresh data the keyed
  reverse primer body is still the coding strand 3' end, not its reverse
  complement (40 of 40 `PCR-seq-enzprimers` tasks, checked). Both rows still
  pick that key in 38 of 40 there. If the authors call it a key error, the
  number to fix is these tasks, not the whole headline **[inference: we did not
  check the `PCR-gene-enzprimers` primers against the E. coli genes]**.

How the two runs were joined:
`PYTHONPATH=src python3 scripts/combine_runs.py --tasks data/lab-bench/SeqQA.jsonl --old-tasks <first-run task file> --row B <first-run B dir> <re-run B dir> --row A <first-run A dir> <re-run A dir> --json docs/results/headline.json`.
The run folders are not in this repository. The output of this command is `docs/results/headline.json` (run-folder names and commit ids in it were made neutral).

## 2. First full run (2026-11-22): superseded

> **Superseded by section 1.** This run used a stale copy of SeqQA (HF datasets-server cache, content of commit `2050d471`). 120 of its 600 tasks are not in the current data. Keep this section as the record of what ran.

Complete: 1,200 of 1,200 tasks, **0 agent errors, 0 timeouts, 0 unparsable
answers, 0 safety refusals**. (Row B had 27 failed tool calls inside otherwise
good tasks.) One model (Sonnet 5.5), two setups, the same 600 tasks, seed 0,
one run per task, through `claude -p` on a Claude Max subscription.

| row | accuracy [95% CI] | anchor s | refused | precision | coverage | tool calls/task | s/task | $/task (list-price eq.) |
|---|---|---|---|---|---|---|---|---|
| **B: sandboxed `run_python`** | **0.848 [0.817, 0.875]** | 1.11 | 59/600 | 0.941 | 0.902 | 0.62 | 11.1 | 0.0237 |
| **A: no tools** | **0.817 [0.784, 0.846]** | 1.05 | 60/600 | 0.907 | 0.900 | 0.0 | 14.5 | 0.0266 |

Anchors on these 600 tasks: chance 0.199, longest-option shortcut 0.269, human
experts 0.789 **[reported, LAB-Bench paper, per subtask, not checked by a
person; the paper also let its experts answer "unsure"]**. Both **point
estimates** are above that ceiling, but row A's interval [0.784, 0.846]
contains 0.789, so only row B is clearly above it.

**The tool seemed to give a small, real gain (section 1 retracts this).** The two intervals overlap. The right
test is the paired one, because both rows answered the same 600 tasks: row B is
correct on **47** tasks row A misses, row A is correct on **28** tasks row B
misses (exact McNemar, binomial, two-sided **p = 0.037**). The gain is **+3.2
points** (509 against 490 correct). It holds on the 520 tasks that are not in
question (33 against 18, p = 0.049) and not on the 80 that are (14 against 10,
p = 0.54). It is concentrated in three subtasks: ORF-seq-numlen,
Prop-seq-gcpercent and ORF-seq-AAid.

**What this does not prove.** One seed, one run per task, one model. The two
rows also use different system prompts (row A says there are no tools), so the
sandbox tool is not the only difference between them. Row B was also faster
(11.1 s against 14.5 s per task) and used fewer output tokens (0.75 M against
1.12 M); the natural reading is that the model runs a short script instead of a
long reasoning chain, but that is **an inference from the token counts, not a
measured cause**.

**Where the score really sits [measured].** 80 of the 600 tasks are the two
subtasks under question in `docs/dataset-findings.md` F-B. Both rows behave the
same way on them:

| group | row B (sandbox) | row A (no tools) |
|---|---|---|
| the 80 tasks in question | 0.312 [0.222, 0.421], 55 refused | 0.263 [0.179, 0.368], 59 refused |
| the other 520 tasks | **0.931 [0.906, 0.950]**, 4 refused | **0.902 [0.873, 0.925]**, 1 refused |
| all 600 tasks | 0.848 | 0.817 |

Three facts about those 80 tasks:
1. Almost every refusal of the whole run comes from them: 55 of row B's 59 and
   59 of row A's 60.
2. **Neither row gives a wrong answer there.** Every one of the 80 tasks ends
   either correct or refused, in both rows (B: 25 correct, 55 refused; A: 21
   correct, 59 refused). The model does not guess there; it says it cannot
   answer.
3. The two rows refuse the same task in 45 of the 80.

The rows share the harness, the task prompt and the refusal option, so this is
not two independent systems agreeing. What it does show is that the failure
sits in the model's reading of those 80 questions, and that the model knows it
is not sure.

**So the headline of this scorecard depends on an open dataset question:**
0.848 / 0.817 over all 600 tasks, or 0.931 / 0.902 over the 520 not in
question. The scorecard reports both, and the question goes to the LAB-Bench
authors before anything is published. Note the counter-evidence in
`dataset-findings.md`: the paper's own experts scored 0.83 and 0.81 on these
two subtasks, so a human can answer them. That favours a model failure over a
broken answer key.

**Cost of the run [measured].** Per task the 5-hour usage window rose
**0.0006** for row A (which ran almost alone on the account) and **0.0029** for
row B (part of which shared the account with a second heavy session); both come
from the per-call meter readings in the results. While a second heavy session
ran, the measured cost was 0.012 per task. **Do not use the
`five_hour_used_this_chunk` field of the manifest for this:** it adds the whole
utilization again when a chunk crosses a window reset, so it over-counts.

Tokens: 4.34 M input and 1.87 M output, which is **$30.15** at the API list
prices in the manifest ($2 / $10 per M, as of 2026-11-17). The subscription paid
usage, not money.

Time: **4.25 h of running** and **11.7 h of pauses** at the 0.90 ceiling, so
15.9 h from the first full-run task (2026-11-21 18:29 UTC) to the last
(2026-11-22 10:23 UTC), over 4 usage windows. (A 5-task probe on 2026-11-21
13:37 makes the manifest span longer.) The run paused and resumed by itself 3
times and lost no task. The harness changed 4 times during the run (one file,
`runner.py`, 11 lines, from bug fixes); the task set, the prompts and
the scoring did not change.

## 3. Pilot on 20 tasks, 4 cells (2026-11-21)

**Pilot, 20 stratified tasks (seed 0), all four cells.** Sonnet 5.5 is now the
main scan model; the Haiku rows stay as the baseline.

| row (20 stratified tasks, seed 0) | accuracy [95% CI] | anchor s | refused | s/task | tokens in/out per task | list-price eq. $/task | 5h window per task |
|---|---|---|---|---|---|---|---|
| Haiku A, no tools | 14/20 = 0.70 [0.48, 0.86] | 0.84 | 3/20 | 14.6 | 1,976 / 2,765 | 0.0018 | 0.0063 |
| Haiku B, sandbox (14 of 20, stopped; an earlier sandbox version) | 13/14 = 0.93 [0.69, 0.99] | 1.31 | 1/14 | 11.0 | 4,254 / 1,936 | 0.0015 | 0.0039 |
| **Sonnet A, no tools** | 16/20 = 0.80 [0.58, 0.92] | 1.01 | 3/20 | 14.6 | 1,955 / 1,858 | 0.0264 | 0.0068 |
| **Sonnet B, sandbox** | 18/20 = 0.90 [0.70, 0.97] | 1.19 | 2/20 | 11.0 | 4,786 / 1,192 | 0.0227 | 0.0074 |

Anchors for these 20 tasks: chance 0.212, longest-option shortcut 0.229, human
ceiling 0.792 (reported). Anchors for all 600: chance 0.199, longest option
0.269. s = (accuracy - longest) / (ceiling - longest), so s > 1 is above the
reported human average.

**What the pilot shows [measured, 20 tasks per cell, wide intervals]:** the
sandboxed Python tool adds about 10 points in both models (Haiku 0.70 -> 0.93,
Sonnet 0.80 -> 0.90), and it also lowers the refusals (3 -> 2 for Sonnet). The
intervals of the two Sonnet rows overlap, so 20 tasks cannot prove the gap; the
600-task run can (+-4 points). Row B calls the tool on about half the tasks
(0.6 calls per task), so the gain comes from a minority of tasks.
**Not proven:** that the tool, and not the longer prompt, causes the gain.

**Usage [measured].** 13 s per task on average. The 5-hour window of the
account rose about 0.007 per task while two other sessions used the same account, and about 0.0025
per task in a quiet chunk. 1,200 tasks (both rows, 600 each) therefore need
about 4.3 h of wall time and about 3 full 5-hour windows.

**Run controls.** For this run the stop was the shared account meter: 0.90 of the 5-hour
window and 0.90 of the 7-day window, read from the `rate_limit_event` of each
call. `scripts/pilot.py --auto` pauses at a ceiling or a usage-limit error
until that window resets, then resumes by itself; finished tasks never run
again. Transient errors wait 5 min, then 15 min; the same stop three times
with no task that finished without an error ends the run as blocked (exit 4).
A task whose error repeats keeps its record, so a few such tasks cannot block
the others.

**Sanity check before the full run [measured]:** the first 5 tasks of each row
(5 different subtasks) parse to a letter, every outcome follows from its letter,
0 errors, 5-13 s per task. Both rows refused the same PCR primer task; see
`docs/dataset-findings.md` F-B.

## 4. Row B (sandboxed Python tool) and the first pilot (2026-11-21)

- **Row B built** (see `harness.md`). `--safe-mode` also turns off `--mcp-config` servers (live check), so rows A and B now both use `--tools "" --restricted --setting-sources "" --strict-mcp-config --permission-mode dontAsk --output-format stream-json`. Claude Code's built-in plugins and a short account block still load in both rows.
- **Row A prompt fixed:** it says there are no tools and asks for one letter from reasoning; the refusal option stays.
- **Chunk stops:** the stream has a `rate_limit_event` after each call with five_hour and seven_day utilization of the whole account. A run stops at a usage-limit error, at +0.18 of the 5-hour window per chunk, at seven_day 0.95, after 5 agent errors in a row, or when usage data is missing. `--resume` and `scripts/pilot.py` continue.
- **Stratified `--limit`**, SeqQA sha256 pinned (`pins.json`), copied task text stripped from results.
- **Anchors** (Arc VCC style): `s = (acc - floor) / (ceiling - floor)`, floor = longest-option shortcut on the same tasks (0.269 on all 600), ceiling = LAB-Bench human accuracy per subtask, averaged over the same tasks (0.789 on all 600) **[reported: LAB-Bench arXiv 2407.10362v3 Appendix B Table 2, read by a model extraction; the mean matches the 78.8% Biomni cites; a person has not checked the table]**.
- **Pilot (stopped by the chunk rule, 2026-11-21):** 34 of 80 planned tasks. The 5-hour window rose 0.57 -> 0.75 in 8 minutes, mostly from other sessions on the same account **[inference: 34 Haiku calls are $0.06 list-price equivalent]**. Seven_day 0.93.

| row (Haiku 5.5, 20 stratified tasks, seed 0) | accuracy [95% CI] | anchor s | refused | s/task | tokens in/out per task | list-price eq. $/task |
|---|---|---|---|---|---|---|
| A no tools | 14/20 = 0.70 [0.48, 0.85] | 0.84 | 3/20 | 15 | 1,976 / 2,765 | 0.0018 |
| B sandbox (stopped at 14 of 20; ran before later sandbox fixes) | 13/14 = 0.93 [0.69, 0.99] | 1.31 | 1/14 | 11 | 4,254 / 1,936 | 0.0015 |

Anchors for these 20 tasks: chance 0.212, longest option 0.229, human ceiling 0.792. Sonnet rows did not run. Refusals before the prompt fix (first smoke runs, old prompt, 5 first-row tasks, 2 runs): 9 of 10. After (row A, 20 stratified tasks): 3 of 20; one of the three still says it cannot check without code, two say no option matches. Both rows refused the same PCR-gene-enzprimers task (possible answer-key issue; not checked).

## 5. First checks (2026-11-20)

- **Benchmark bug found [checked]:** 6 of 600 SeqQA tasks (all `PCR-geneprimers-enz-v1-public`, for example `6ce2c743`) list the correct answer text twice. The old code graded only one copy as correct. The harness now shows each option text once (after whitespace normalisation, LAB-Bench #12) and flags the run `duplicate_options`.
- **Length shortcut on SeqQA [checked, exact expected values]:** "pick the longest option" scores 0.269 overall. Chance is 0.199 with the refusal option and 0.249 without it. Per subtask it reaches 0.379 (PCR-seq-enzprimers) and 0.363 (PCR-gene-enzprimers), against 0.25. In ORF-seq-AAseq the correct answer is never the longest. The `longest` agent (seed 0) scored 0.277.
- Dry run with 5 dummy agents: random 0.213, first 0.227 (position_bias), refuse 0, longest 0.277, oracle 1.0. Random and first moved by 0.001 because the 6 duplicate tasks now have fewer options.
- **Smoke runs (Claude Max subscription, Haiku 5.5, 5 tasks, `claude -p`, 2 runs):** run 1: 1 correct, 4 refused ("I can't run code here"); run 2: 5 refused. All 5 tasks are the first rows, which all come from one subtask (ORF-seq-AAid). This only shows that the plumbing works. It is not a score. List-price equivalent about $0.0009/task (the harness $ now equals Claude Code's own total_cost_usd), 9–13 s/task.

## 6. First dry run (2026-11-17)

- 11 unit tests passed.
- The fetch downloaded SeqQA, 600 rows, HF revision `5c77cec648430f30611808808861eb86f81d5eaa`, sha256 `7d2d1132…`. (This was the stale copy with a false revision record; see section 1.)
- The four dummy agents ran with the network blocked:

| agent | accuracy [95% CI] | precision | coverage | flags |
|---|---|---|---|---|
| random | 0.212 [0.181, 0.246] | 0.272 | 0.778 | – |
| first (always A) | 0.228 [0.197, 0.264] | 0.280 | 0.815 | position_bias |
| refuse | 0.000 [0.000, 0.006] | 0.000 | 0.000 | – |
| oracle (self-test) | 1.000 [0.994, 1.000] | 1.000 | 1.000 | – |

Random matches chance: 560 questions have 5 options and 40 have 6, so chance is about 0.20. The position-bias check flags only the "always A" agent.
