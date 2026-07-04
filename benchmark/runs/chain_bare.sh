#!/usr/bin/env bash
# 依序（不平行，時間才乾淨）重跑 6 個 claude cell，全部 --bare（無 skill 污染）。
set -uo pipefail
R=/Users/david/course/vibe-backtester/benchmark/runs
for cond in solo fplan; do
  for m in opus sonnet haiku; do
    echo ">>> $m/$cond"
    "$R/clean_run.sh" "c-$m-$cond" "$m" "$cond"
  done
done
echo "CHAIN_DONE"
