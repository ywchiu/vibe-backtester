#!/usr/bin/env bash
# 用 headless Claude Code 讓某個模型「照 PLAN.md 實作」，並記錄 token/turns/time/cost。
# 用法：./drive_impl.sh <model>     例如 ./drive_impl.sh sonnet
set -uo pipefail
BENCH=/Users/david/course/vibe-backtester/benchmark
model="${1:?usage: drive_impl.sh <model>}"

# 乾淨資料夾（受測者視角）+ 放入 mastermind 的計畫
"$BENCH/new_run.sh" "$model" >/dev/null
cp "$BENCH/runs/PLAN.md" "$BENCH/runs/$model/PLAN.md"

read -r -d '' PROMPT <<'EOF'
Follow the implementation plan in PLAN.md to complete all three levels in this folder.
Edit ONLY these files:
- level1/indicators.py  (implement the 5 functions)
- level2/backtester.py  (implement run_backtest)
- level3/backtester/*.py (fix the bugs)
Do NOT modify any file under any tests/ directory. Do not read anything outside this folder.
After editing, verify by running each level's public tests:
  /Users/david/course/vibe-backtester/benchmark/.venv/bin/python -m pytest level1/tests -q
  /Users/david/course/vibe-backtester/benchmark/.venv/bin/python -m pytest level2/tests -q
  /Users/david/course/vibe-backtester/benchmark/.venv/bin/python -m pytest level3/tests -q
Iterate until each level's public tests pass. Keep going until done.
EOF

cd "$BENCH/runs/$model"
claude -p "$PROMPT" --model "$model" --output-format json --dangerously-skip-permissions \
  > "$BENCH/runs/metrics/$model.json" 2> "$BENCH/runs/metrics/$model.err"
echo "IMPL_DONE model=$model exit=$?"
