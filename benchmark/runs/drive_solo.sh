#!/usr/bin/env bash
# SOLO：模型自己讀 README、自己規劃＋實作，沒有外部 mastermind 計畫。
# 用 headless Claude Code 計 token/turns/time/cost。
# 用法：./drive_solo.sh <model>     產出 runs/<model>-solo 與 metrics/<model>-solo.json
set -uo pipefail
BENCH=/Users/david/course/vibe-backtester/benchmark
model="${1:?usage: drive_solo.sh <model>}"
name="${model}-solo"

"$BENCH/new_run.sh" "$name" >/dev/null

read -r -d '' PROMPT <<'EOF'
Implement all three levels in this folder from scratch. Read level1/README.md,
level2/README.md, level3/README.md for the EXACT specs (formulas, tolerances,
look-ahead rules) and plan your approach yourself.
Edit ONLY these files:
- level1/indicators.py  (implement the 5 functions)
- level2/backtester.py  (implement run_backtest)
- level3/backtester/*.py (fix the bugs)
Do NOT modify any file under any tests/ directory. Do not read anything outside this folder.
Verify by running each level's public tests:
  /Users/david/course/vibe-backtester/benchmark/.venv/bin/python -m pytest level1/tests -q
  /Users/david/course/vibe-backtester/benchmark/.venv/bin/python -m pytest level2/tests -q
  /Users/david/course/vibe-backtester/benchmark/.venv/bin/python -m pytest level3/tests -q
Iterate until each level's public tests pass. Keep going until done.
EOF

cd "$BENCH/runs/$name"
claude -p "$PROMPT" --model "$model" --output-format json --dangerously-skip-permissions \
  > "$BENCH/runs/metrics/${name}.json" 2> "$BENCH/runs/metrics/${name}.err"
echo "SOLO_DONE model=$model exit=$?"
