# SeqQA dataset findings (LAB-Bench v1, 600 tasks)

Findings about the benchmark data itself, from our runs. Task ids only, no task
text (CC-BY-SA, canary strings). We posted the open points on
[LAB-Bench issue #20](https://github.com/Future-House/LAB-Bench/issues/20).

Read F-C first: F-A and F-B describe a stale copy of the data. "Rows" are the two
setups: A = the model with no tools, B = the model with a sandboxed Python tool.

| id | subtasks | tasks | status |
|---|---|---|---|
| F-A duplicate options | PCR-geneprimers-enz-v1-public | 6 in the stale copy, 1 in `5c77cec6` | checked |
| F-B answer key or model reading, enzyme site orientation | PCR-seq-enzprimers-v1-public, PCR-gene-enzprimers-v1-public | 80 of 80 (stale copy only) | RESOLVED by F-C: these tasks are not in the current data; on the new 80 both rows score 0.95-0.975 |
| F-C our copy was stale (datasets-server cache) | the 3 PCR enzyme subtasks | 120 | checked; fetch now pins the parquet file and two SHA-256 values |

## F-A: the correct option text appears twice

6 of 600 tasks list the correct answer text two times. Old code graded only one
copy as correct. The harness now shows each option text once (after whitespace
normalisation) and flags the run `duplicate_options`. Example task
`6ce2c743`. Found 2026-11-20.

## F-B: the answer key and the orientation of the named enzyme site

Found 2026-11-21 by the sanity check of the full run. Both rows
(Sonnet 5.5 with no tools, and with the sandboxed Python tool) answered
"insufficient information to answer" on task `e1fa5428` (subtask
`PCR-seq-enzprimers-v1-public`). Both model answers give the same reason: in
the option the key marks as correct, the recognition site of the second named
enzyme is not in the orientation they expect, and they read it as the site of a
different enzyme.

A script over the whole subset shows the same pattern in **all 80 tasks** of the
two subtasks that name two enzymes (`PCR-seq-enzprimers-v1-public` and
`PCR-gene-enzprimers-v1-public`): the key's second primer always begins with the
reverse of the named site, and the first primer always begins with the site as
written. The third subtask (`PCR-geneprimers-enz-v1-public`, 40 tasks) names its
enzymes in another sentence form, so the script did not read it.

Two readings are possible, and **we did not settle which one is right**:
1. The key is consistent with a convention for writing the reverse primer, and
   the models misread it. Then the tasks are correct and this is a
   model failure (an expensive one: 80 of 600 tasks, 13%).
2. The key has a systematic error in those 80 tasks.

Both readings need a molecular biologist to confirm; neither is a claim we make
yet. What is checked: the pattern holds in 80 of 80 tasks, and it is
systematic, not a one-off typing error. The run keeps these tasks and scores
them as usual; the refusals stay visible in the `refused` column.

## F-B: what the full run shows [measured 2026-11-22, both rows, 600 tasks each]

The full run of both rows (Sonnet 5.5, with and without the sandboxed Python
tool, 0 agent errors) splits the subset in two:

| group | row B (sandbox) | row A (no tools) |
|---|---|---|
| the 80 tasks in question | 0.312, 55 refused | 0.263, 59 refused |
| the other 520 tasks | 0.931, 4 refused | 0.902, 1 refused |
| all 600 | 0.848, 59 refused | 0.817, 60 refused |

- Almost every refusal of the whole run comes from those 80 tasks: 55 of 59
  (row B) and 59 of 60 (row A).
- **Neither row ever answers one of the 80 wrongly.** Every task there ends
  correct or refused (B: 25 correct, 55 refused; A: 21 correct, 59 refused).
- The third primer subtask, `PCR-geneprimers-enz-v1-public` (40 tasks, the one
  that names its enzymes in another sentence form), scores 0.95 (B) and 0.925
  (A) with **no** refusals.
- The two rows share the harness, the prompt and the refusal option, so they
  are not two independent systems: 45 of the 80 are refused by both.

### Counter-evidence that a reader will raise first [checked in anchors.json]

The LAB-Bench paper reports **human expert accuracy of 0.83 on
`PCR-gene-enzprimers` and 0.81 on `PCR-seq-enzprimers`** (0.94 on the third
subtask). So experts answer these questions. A key that was simply broken
should also have stopped the experts.

That pushes the two readings apart:
1. **The key is right, and the model misreads it.** Supported by the human
   scores above. Then the 80 tasks measure a real gap: the model does not
   accept the convention used for the second primer, and refuses instead of
   answering. This is the more likely reading **[our inference, from the human
   scores; we did not check the chemistry]**.
2. **The key has a systematic error in those 80 tasks.** Supported by the
   pattern holding in 80 of 80 tasks and by the model giving the same specific
   objection every time.

**We do not say which is right, and we do not claim the key is wrong.** A
molecular biologist has to settle it. Either way the number has to be reported
twice, because a reader who believes reading 1 and a reader who believes
reading 2 want different totals: 0.848 / 0.817 over all 600 tasks, or
0.931 / 0.902 over the 520 not in question.

Outcome: F-C below. These 80 tasks are not in the current data, so the choice
between the two readings does not change the final result.


## F-C: our copy was a stale cache, not the current data [checked 2026-11-23]

The issue author on Future-House/LAB-Bench#20 said the HF parquet file of
`5c77cec6` already matches GitHub `998a8e0`. We checked it:

1. The parquet file of `5c77cec6` has sha256
   `5864776d3d85774f91d82432e30ac31e9222f1670621e8ea141414173c5e9b46`, the
   same as the LFS id that the HF API reports.
2. Converted to our JSONL it is the same as GitHub `998a8e0`
   (`SeqQA/*-public.jsonl`): 600 of 600 ids, and the same question, ideal,
   distractors, canary and subtask for every id.
3. Our old JSONL (sha256 `7d2d1132...`) is the conversion of the SeqQA
   parquet file (LFS `b8bb6a94`) of the HF commits before `e2978291`, for
   example `2050d471`. The fresh parquet first appears at `e2978291`
   (2025-09-27 00:42:29Z), 5 seconds before `5c77cec6`. The datasets-server `refs/convert/parquet` file gives the same
   JSONL. Our old fetch read `/rows` from that cache and wrote the newest
   commit id into the meta file, so the meta was false.
4. Against our old copy, 480 tasks are the same in every field. 120 tasks (all
   40 of `PCR-gene-enzprimers`, `PCR-seq-enzprimers` and
   `PCR-geneprimers-enz`) have new ids and new questions.

So F-B described the old 80 enzprimers tasks. They are gone from the current
data, and the third subtask changed too. On the 120 new tasks both rows score
0.942 (113 correct, 5 wrong, 2 refused); on the new enzprimers tasks 77 of 80.
F-A drops from 6 tasks to 1 (checked: one task with a repeated option text in the fresh copy). The final result is in `results.md`, section 1.

Still open (question 2 on issue #20): the keyed reverse primer body is the
coding strand 3' end, not its reverse complement. Checked again on 2026-11-23
in the fresh copy: 40 of 40 `PCR-seq-enzprimers` tasks (the same in the stale
copy). Both rows still pick that key in 38 of 40 of these tasks.
