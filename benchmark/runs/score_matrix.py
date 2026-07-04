import json, re, subprocess, os
BENCH="/Users/david/course/vibe-backtester/benchmark"
SCRATCH="/private/tmp/claude-501/-Users-david-course-vibe-backtester/b43dd03c-bdbd-4f00-8a63-3c0d2b9a17bb/scratchpad/bench"
MAX={"L1":22,"L2":27,"L3p":11,"L3h":14}

def passed(line):
    m=re.search(r'(\d+) passed', line); return int(m.group(1)) if m else 0

def grade(name):
    ws=f"{SCRATCH}/{name}"
    out=subprocess.run(["bash",f"{BENCH}/grade.sh",name,ws],capture_output=True,text=True).stdout
    lines=[l for l in out.splitlines() if 'passed' in l or 'failed' in l or 'error' in l]
    # 4 result lines in order: L1, L2, L3 public, L3 hidden
    res=[l for l in out.splitlines() if re.search(r'\d+ (passed|failed|error)',l)]
    vals=[passed(l) for l in res[-4:]]
    while len(vals)<4: vals.insert(0,0)
    L1,L2,L3p,L3h=vals
    return dict(L1=L1,L2=L2,L3p=L3p,L3h=L3h,total=L1+L2+L3p+L3h)

def metrics(name):
    d=json.load(open(f"{BENCH}/runs/metrics/{name}.json"))
    if d.get("agent")=="codex":
        return dict(turns="~", sec=d["duration_ms"]/1000, cost=None,
                    out=d["output_tokens"]+d.get("reasoning_output_tokens",0),
                    total_tok=d["total_tokens"])
    u=d["usage"]
    return dict(turns=d["num_turns"], sec=d["duration_ms"]/1000, cost=d["total_cost_usd"],
                out=u["output_tokens"],
                total_tok=u["input_tokens"]+u["cache_creation_input_tokens"]+u["cache_read_input_tokens"]+u["output_tokens"])

models=["opus","sonnet","haiku","codex"]
print(f"{'cell':<16}{'score/74':>9}  {'turns':>5} {'time(s)':>8} {'cost':>9} {'out_tok':>8} {'total_tok':>10}")
print("-"*72)
rows={}
for cond in ["solo","fplan"]:
    for m in models:
        name=f"c-{m}-{cond}"
        g=grade(name); mt=metrics(name); rows[name]=(g,mt)
        cost = f"${mt['cost']:.4f}" if mt['cost'] is not None else "(codex)"
        print(f"{m+'/'+cond:<16}{g['total']:>4}/74   {str(mt['turns']):>5} {mt['sec']:>8.1f} {cost:>9} {mt['out']:>8,} {mt['total_tok']:>10,}")
    print()

# plan cost to add for fplan total-cost
mm=json.load(open(f"{BENCH}/runs/metrics/mastermind.json"))
plan_cost=mm["total_cost_usd"]; plan_sec=mm["duration_ms"]/1000; plan_turns=mm["num_turns"]
print(f"[mastermind plan (Opus): {plan_turns} turns, {plan_sec:.1f}s, ${plan_cost:.4f}]  → 加到 fplan 方案總成本")
PY_END=0
