#!/usr/bin/env bash
# 用 Claude Code 指向 OpenRouter 的 DeepSeek 當實作者（Course 的方案 F 作法）。
# 隔離同其他 cell：workspace 在 repo 外、--setting-sources "" 裸模型、fresh process。
# 需要環境變數 OPENROUTER_API_KEY（若有 scratch/.or_key 會自動 source）。
#
# 用法：./drive_deepseek.sh <name> <slug> <cond>
#   slug = deepseek/deepseek-v4-flash | deepseek/deepseek-v4-pro
#   cond = solo | fplan
set -uo pipefail
BENCH=/Users/david/course/vibe-backtester/benchmark
SCRATCH=/private/tmp/claude-501/-Users-david-course-vibe-backtester/b43dd03c-bdbd-4f00-8a63-3c0d2b9a17bb/scratchpad/bench
PY="$SCRATCH/.venv/bin/python"
name="${1:?name}"; slug="${2:?slug}"; cond="${3:?cond}"
ws="$SCRATCH/$name"

[ -f "$SCRATCH/.or_key" ] && source "$SCRATCH/.or_key"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then echo "ERROR: OPENROUTER_API_KEY 未設定"; exit 2; fi

rm -rf "$ws"; mkdir -p "$ws"
rsync -a --exclude _grader --exclude __pycache__ --exclude test_hidden.py --exclude BUGS.md \
  --exclude PLAN.md \
  "$BENCH/level1" "$BENCH/level2" "$BENCH/level3" "$BENCH/data" "$ws/"
if [ "$cond" = "fplan" ]; then
  cp "$BENCH/runs/PLAN.md" "$ws/PLAN.md"
  TASK="Follow the implementation plan in PLAN.md to complete all three levels in this folder."
else
  TASK="Read level1/README.md, level2/README.md, level3/README.md and plan your own approach."
fi
PROMPT="$TASK
Work ONLY within this directory ($ws). Do NOT read or search any path outside it.
Edit ONLY level1/indicators.py, level2/backtester.py, level3/backtester/*.py.
Do NOT modify any file under any tests/ directory.
Verify by running each level's public tests:
  $PY -m pytest level1/tests -q
  $PY -m pytest level2/tests -q
  $PY -m pytest level3/tests -q
Iterate until each level's public tests pass. Keep going until done."

cd "$ws"
env \
  ANTHROPIC_BASE_URL="https://openrouter.ai/api" \
  ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY" \
  ANTHROPIC_API_KEY="" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="$slug" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="$slug" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="$slug" \
  CLAUDE_CODE_SUBAGENT_MODEL="$slug" \
  claude -p "$PROMPT" --model sonnet --output-format json \
    --setting-sources "" --dangerously-skip-permissions \
    > "$BENCH/runs/metrics/${name}.json" 2> "$BENCH/runs/metrics/${name}.err"
echo "DEEPSEEK_DONE name=$name slug=$slug cond=$cond exit=$?"
