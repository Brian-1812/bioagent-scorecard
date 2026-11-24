# The harness in detail

This page is the technical description of the harness. The short summary, the result and the project status are in
[`../README.md`](../README.md). The project is paused: no further work is planned.

## Quick start (no API keys, no cost)

```bash
# Python 3.10+. The harness uses the standard library only.
# `fetch` needs pyarrow to read the pinned parquet file.
python3 -m pip install pyarrow
scripts/dry_run.sh SeqQA 0
```

The dry run does four things:

1. It runs the offline unit tests on made-up questions.
2. It downloads the LAB-Bench SeqQA subset (600 questions) from Hugging Face.
3. It runs five dummy agents with all network access blocked.
4. It prints the scorecard table.

| dummy agent | what it shows |
|---|---|
| `random` | the chance floor (about 1 / number of options) |
| `first` | always option A; the position-bias check must flag it |
| `refuse` | always "insufficient information"; coverage must be 0 |
| `longest` | the longest option (not the refusal); the length shortcut from LAB-Bench #18 |
| `oracle` | reads the answer key; a harness self-test, never published |

## Commands

```bash
export PYTHONPATH=src
python3 -m pip install pyarrow   # fetch reads the pinned parquet file; runs do not need it
python3 -m bioagent_scorecard fetch --subset SeqQA
python3 -m bioagent_scorecard run --dry-run --agent random --seed 0 \
    --tasks data/lab-bench/SeqQA.jsonl --out runs/dry-run
python3 -m bioagent_scorecard report runs/dry-run/*
```

`baselines --tasks <file>` prints the exact expected accuracy of the no-model
shortcuts (chance with and without the refusal option, longest and shortest
option) per subtask. A model score is only meaningful above these floors.

## Benchmark text fixes

- Option text is compared after whitespace normalisation (LAB-Bench #12).
- Options that are the same text after normalisation are shown once. One
  SeqQA task in the pinned data (six in the older copy) lists the correct answer
  twice. The run flags such tasks as `duplicate_options`.

## Rows A and B: Claude through headless Claude Code

| row | agent name | tools |
|---|---|---|
| A (baseline) | `claude-cli:<model>` | none; the system prompt says so and asks for one letter from reasoning |
| B (main) | `claude-sandbox:<model>` | one MCP tool, `run_python` (`sandbox_mcp.py`) |

Both rows run one `claude -p` process per task with the same flags:
`--tools "" --restricted --setting-sources "" --strict-mcp-config
--permission-mode dontAsk --output-format stream-json`. Row B adds only
`--mcp-config` and `--allowedTools mcp__pysandbox__run_python`. See
`agents/claude_cli.py` for why each flag is there.

The `run_python` sandbox (`sandbox.py`) runs each program as user `nobody`
in new network, mount, PID, IPC and UTS namespaces, inside its own cgroup
(a transient systemd scope: memory, task and CPU caps for all its processes
together). The root file system is an allowlist: a read-only `/usr`, a
64 MB `/tmp`, a new `/proc`, four `/dev` nodes, a two-file `/etc` and the
read-only work folder. `pivot_root` detaches the host root. stdout and stderr
are capped at 16 MB on disk and 20,000 characters for the model. The
sandbox dies with its parent. Only the Python standard library is
available. It needs root and systemd; without them it does not run the code
(it fails closed), and a row-B run refuses to start. It is not a VM and has
no seccomp filter: the code shares the host kernel.

```bash
python3 -m bioagent_scorecard run --allow-model-calls \
    --agent claude-sandbox:claude-haiku-5-5 --limit 20 \
    --tasks data/lab-bench/SeqQA.jsonl --out runs/pilot
```

`--limit N` picks N tasks spread evenly over the subtasks (`selection.py`),
not the first N rows.

The child process gets no `ANTHROPIC_API_KEY`, so it uses the logged-in Claude
subscription. **Runs on a subscription cost usage-limit quota, not dollars.
The `$/task` column is then the API list-price equivalent. Any published
result from such a run must say so.**

### Chunks and usage limits

A run stops by itself, writes what it has, and exits with code 3 when (the
manifest is written at the start and after the run, also after a crash):

- the plan reports a usage-limit error (that task is not scored);
- this chunk used `--chunk-five-hour` (default 0.18) of the 5-hour window,
  from the `rate_limit_event` that Claude Code reports after each call;
- the 7-day window reaches `--stop-seven-day` (default 0.95);
- this chunk ran `--chunk-max-tasks` tasks;
- 5 tasks in a row end in `agent_error` (for example an expired login);
- a model call reports no 5-hour usage data 3 times in a row.

`--resume` runs `agent_error` tasks again and accepts a newer Claude Code
version (each chunk records its version).

`run --resume <run folder>` continues it. `scripts/pilot.py` runs several
agents as one chunk with one shared budget. The meter reads the account's
total use, so other sessions on the same account count too (the stop comes
early, not late).

### Anchors

`report` shows an anchor score next to the raw accuracy, in the style of the
Arc Virtual Cell Challenge: `s = (accuracy - floor) / (ceiling - floor)`. The
floor is the "pick the longest option" shortcut on the same tasks; the ceiling
is the LAB-Bench human-expert accuracy per subtask (a reported value,
`anchors.json`), averaged over the same tasks. `s` is not an accuracy.

The task file must match the SHA-256 in `pins.json` (`--allow-unpinned`
overrides it). `answer_text` in `results.jsonl` has copied question text
removed; `tool_logs/` (row B) keeps the full code and output and is not for
publishing.

## Layout

```
src/bioagent_scorecard/
  tasks.py      task records, fetch of the pinned HF parquet file, SHA-256 checks
  mcq.py        seeded option shuffle, prompt, strict answer parser
  agents/       the Agent interface; dummy agents; claude_cli.py (rows A and B)
  sandbox.py    the run_python sandbox (namespaces, uid nobody, rlimits)
  sandbox_mcp.py  a one-tool MCP server (stdio) around sandbox.py
  selection.py  stratified task selection for --limit
  anchors.py    floor and ceiling anchors (+ anchors.json); pins.json
  baselines.py  no-model shortcut baselines (chance, longest, shortest option)
  runner.py     runs one agent on one task file; writes manifest, results, summary
  scoring.py    accuracy / precision / coverage with Wilson intervals; outcome taxonomy
  checks.py     integrity checks: answer-key access, canary leak, position bias
  netguard.py   blocks all sockets in dry-run mode
  pricing.py    cost from token counts and prices.json
tests/          offline tests on a synthetic fixture
scripts/        dry_run.sh, pilot.py (several agents, one chunk)
.github/        CI: unit tests (Python 3.10, 3.12) and the dry run with no network
```

## Data and licenses

- Harness code: Apache-2.0 (`LICENSE`).
- LAB-Bench data: CC-BY-SA-4.0, by FutureHouse. The repo does not copy it.
  `fetch` downloads it to `data/`, which git ignores. Result files hold task
  ids only, not question text.
- Published results: CC-BY-SA-4.0, because they derive from the benchmark.

## Credits

LAB-Bench and BixBench are by FutureHouse / Edison Scientific. This project is
not affiliated with them.
