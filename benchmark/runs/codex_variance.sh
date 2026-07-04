#!/usr/bin/env bash
# Codex 變異測試：solo / fplan 各跑 N 次，記錄每次 token/time/score，看波動。
set -uo pipefail
BENCH=/Users/david/course/vibe-backtester/benchmark
SCRATCH=/private/tmp/claude-501/-Users-david-course-vibe-backtester/b43dd03c-bdbd-4f00-8a63-3c0d2b9a17bb/scratchpad/bench
PY="$SCRATCH/.venv/bin/python"
N="${1:-3}"
OUT="$BENCH/runs/metrics/codex_variance.tsv"
echo -e "cond\trep\ttime_s\ttotal_tok\tinput_tok\tcached_tok\toutput_tok\tscore" > "$OUT"

newest_session() {
  find ~/.codex/sessions -name '*.jsonl' -exec stat -f '%m %N' {} + 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}

for cond in solo fplan; do
  for rep in $(seq 1 "$N"); do
    name="c-cvar-${cond}-${rep}"
    ws="$SCRATCH/$name"
    rm -rf "$ws"; mkdir -p "$ws"
    rsync -a --exclude _grader --exclude __pycache__ --exclude test_hidden.py --exclude BUGS.md --exclude PLAN.md \
      "$BENCH/level1" "$BENCH/level2" "$BENCH/level3" "$BENCH/data" "$ws/"
    if [ "$cond" = fplan ]; then cp "$BENCH/runs/PLAN.md" "$ws/PLAN.md"
      TASK="Follow the implementation plan in PLAN.md to complete all three levels in this folder."
    else TASK="Read level1/README.md, level2/README.md, level3/README.md and plan your own approach."; fi
    PROMPT="$TASK
Work ONLY within this directory ($ws). Edit ONLY level1/indicators.py, level2/backtester.py, level3/backtester/*.py. Do NOT modify any tests/. Verify with: $PY -m pytest level1/tests level2/tests level3/tests -q (run per level). Iterate until public tests pass."
    start=$SECONDS
    codex exec -C "$ws" --sandbox workspace-write --skip-git-repo-check -m gpt-5.5 "$PROMPT" \
      > "$BENCH/runs/metrics/${name}.codexlog" 2>&1
    dur=$((SECONDS-start))
    sess=$(newest_session)
    read tot inp cac out < <("$PY" -c "
import json,sys
u={}
for line in open('$sess'):
    if 'total_token_usage' in line:
        import re;m=re.search(r'\"total_token_usage\":(\{[^}]*\})',line)
        if m: u=json.loads(m.group(1))
print(u.get('total_tokens',0),u.get('input_tokens',0),u.get('cached_input_tokens',0),u.get('output_tokens',0))
")
    score=$(bash "$BENCH/grade.sh" "$name" "$ws" 2>&1 | grep -oE '[0-9]+ passed' | grep -oE '^[0-9]+' | paste -sd+ - | bc)
    echo -e "${cond}\t${rep}\t${dur}\t${tot}\t${inp}\t${cac}\t${out}\t${score}/74" | tee -a "$OUT"
  done
done
echo "VARIANCE_DONE → $OUT"
