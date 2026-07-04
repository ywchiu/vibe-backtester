#!/usr/bin/env bash
# 用 codex exec (gpt-5.5) 實作 benchmark，記錄 token 與時間。
# 用法：
#   ./drive_codex.sh <name>            # solo：codex 自己規劃+實作
#   ./drive_codex.sh <name> PLAN       # 照 runs/PLAN.md（Opus 的計畫）實作
set -uo pipefail
BENCH=/Users/david/course/vibe-backtester/benchmark
name="${1:?usage: drive_codex.sh <name> [PLAN]}"
useplan="${2:-}"

"$BENCH/new_run.sh" "$name" >/dev/null
if [ "$useplan" = "PLAN" ]; then
  cp "$BENCH/runs/PLAN.md" "$BENCH/runs/$name/PLAN.md"
  TASK='Follow the implementation plan in PLAN.md to complete all three levels in this folder.'
else
  TASK='Read level1/README.md, level2/README.md, level3/README.md and plan your own approach.'
fi

PROMPT="$TASK
Edit ONLY level1/indicators.py, level2/backtester.py, level3/backtester/*.py.
Do NOT modify any file under any tests/ directory. Do not read outside this folder.
Verify by running each level's public tests with:
  /Users/david/course/vibe-backtester/benchmark/.venv/bin/python -m pytest level1/tests -q
  (and level2/tests, level3/tests). Iterate until each level's public tests pass."

cd "$BENCH/runs/$name"
start=$SECONDS
codex exec -C "$BENCH/runs/$name" --sandbox workspace-write --skip-git-repo-check \
  -m gpt-5.5 "$PROMPT" > "$BENCH/runs/metrics/${name}.codexlog" 2>&1
dur=$((SECONDS - start))

# codex 自報 token（"tokens used N"）
tok=$(grep -Eio 'tokens used[^0-9]*[0-9,]+' "$BENCH/runs/metrics/${name}.codexlog" | tail -1)
echo "CODEX_DONE name=$name time=${dur}s  ${tok:-tokens=?}"