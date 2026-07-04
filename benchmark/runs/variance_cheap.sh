#!/usr/bin/env bash
# 便宜模型變異測試：Haiku 與 DeepSeek Flash 各 solo 跑 N 次。
set -uo pipefail
BENCH=/Users/david/course/vibe-backtester/benchmark
SCRATCH=/private/tmp/claude-501/-Users-david-course-vibe-backtester/b43dd03c-bdbd-4f00-8a63-3c0d2b9a17bb/scratchpad/bench
R="$BENCH/runs"
N="${1:-3}"
OUT="$R/metrics/variance_cheap.tsv"
echo -e "model\trep\ttime_s\ttotal_tok\tout_tok\tcost\tscore" > "$OUT"

emit() {  # name label costmode(native|flash)
  local name=$1 label=$2 mode=$3 f="$R/metrics/$1.json" ws="$SCRATCH/$1"
  read t tok out cost < <(python3 -c "
import json
d=json.load(open('$f'));u=d['usage']
tok=u['input_tokens']+u.get('cache_creation_input_tokens',0)+u.get('cache_read_input_tokens',0)+u['output_tokens']
if '$mode'=='native': cost=d['total_cost_usd']
else:
  inp=u['input_tokens'];cr=u.get('cache_read_input_tokens',0);cc=u.get('cache_creation_input_tokens',0);o=u['output_tokens']
  cost=inp*0.09e-6+(cr+cc)*0.018e-6+o*0.18e-6
print(round(d['duration_ms']/1000,1),tok,u['output_tokens'],round(cost,4))
")
  local score
  score=$(bash "$BENCH/grade.sh" "$name" "$ws" 2>&1 | grep -oE '[0-9]+ passed' | grep -oE '^[0-9]+' | paste -sd+ - | bc)
  echo -e "${label}\t${rep}\t${t}\t${tok}\t${out}\t${cost}\t${score}/74" | tee -a "$OUT"
}

for rep in $(seq 1 "$N"); do
  n="v-haiku-solo-$rep";   "$R/clean_run.sh" "$n" haiku solo >/dev/null 2>&1; emit "$n" Haiku native
  n="v-dsflash-solo-$rep"; "$R/drive_deepseek.sh" "$n" deepseek/deepseek-v4-flash solo >/dev/null 2>&1; emit "$n" DeepSeekFlash flash
done
echo "VARIANCE_CHEAP_DONE → $OUT"
