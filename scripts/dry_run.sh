#!/usr/bin/env bash
# Dry run: fetch one LAB-Bench subset, then score the dummy agents with all
# network access blocked. It makes no model API calls and costs nothing.
# Usage: scripts/dry_run.sh [subset] [seed]
set -euo pipefail
cd "$(dirname "$0")/.."
SUBSET=${1:-SeqQA}
SEED=${2:-0}
export PYTHONPATH=src

echo "== unit tests (offline, synthetic fixture)"
python3 -m unittest discover -s tests

echo "== fetch $SUBSET (Hugging Face, no model calls)"
if [ ! -f "data/lab-bench/$SUBSET.jsonl" ]; then
  python3 -m bioagent_scorecard fetch --subset "$SUBSET"
fi

echo "== dry run, network blocked"
OUT=runs/dry-run
mkdir -p "$OUT"
for agent in random first refuse longest oracle; do
  python3 -m bioagent_scorecard run --dry-run --agent "$agent" --seed "$SEED" \
    --tasks "data/lab-bench/$SUBSET.jsonl" --out "$OUT"
done

echo "== shortcut baselines (expected accuracy, no model)"
python3 -m bioagent_scorecard baselines --tasks "data/lab-bench/$SUBSET.jsonl"

echo "== report"
python3 -m bioagent_scorecard report "$OUT"/*
