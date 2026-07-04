#!/usr/bin/env bash
# 隔離實驗：workspace 建在 repo 外的 scratchpad，只含受測者可見檔案，
# 連 repo 路徑都不透露給 agent（python 用 scratch 內的 venv），杜絕偷看參考解答。
# 每次都是全新 workspace（rm -rf）+ 全新 process（headless），無 context 污染。
#
# 用法：./clean_run.sh <name> <agent> <cond>
#   agent = opus|sonnet|haiku|codex
#   cond  = solo | fplan   (fplan = 用 Opus 的 PLAN.md)
set -uo pipefail
BENCH=/Users/david/course/vibe-backtester/benchmark
SCRATCH=/private/tmp/claude-501/-Users-david-course-vibe-backtester/b43dd03c-bdbd-4f00-8a63-3c0d2b9a17bb/scratchpad/bench
PY="$SCRATCH/.venv/bin/python"
name="${1:?name}"; agent="${2:?agent}"; cond="${3:?cond}"
ws="$SCRATCH/$name"

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
Work ONLY within this directory ($ws). Do NOT read, list, or search any path outside it.
There is no reference solution available anywhere — implement purely from the READMEs/plan.
Edit ONLY level1/indicators.py, level2/backtester.py, level3/backtester/*.py.
Do NOT modify any file under any tests/ directory.
Verify by running each level's public tests:
  $PY -m pytest level1/tests -q
  $PY -m pytest level2/tests -q
  $PY -m pytest level3/tests -q
Iterate until each level's public tests pass. Keep going until done."

cd "$ws"
start=$SECONDS
if [ "$agent" = "codex" ]; then
  codex exec -C "$ws" --sandbox workspace-write --skip-git-repo-check -m gpt-5.5 "$PROMPT" \
    > "$BENCH/runs/metrics/${name}.codexlog" 2>&1
  dur=$((SECONDS-start))
  tok=$(grep -Eio 'tokens used[^0-9]*[0-9,]+' "$BENCH/runs/metrics/${name}.codexlog" | tail -1)
  echo "{\"agent\":\"codex\",\"duration_ms\":$((dur*1000)),\"codex_tokens\":\"${tok:-NA}\"}" \
    > "$BENCH/runs/metrics/${name}.json"
else
  # --setting-sources ""：停用 CLAUDE.md / skills / hooks / plugins / MCP
  # （例如 superpowers 的 SessionStart 注入），但保留 auth 與 model。
  # 讓每個模型都是乾淨裸模型，不被 harness 的 skill 污染行為。
  claude -p "$PROMPT" --model "$agent" --output-format json \
    --setting-sources "" --dangerously-skip-permissions \
    > "$BENCH/runs/metrics/${name}.json" 2> "$BENCH/runs/metrics/${name}.err"
fi
echo "CLEAN_DONE name=$name agent=$agent cond=$cond ws=$ws"
