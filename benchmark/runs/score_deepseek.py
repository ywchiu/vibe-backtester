import json, re, subprocess, os
BENCH="/Users/david/course/vibe-backtester/benchmark"
SCRATCH="/private/tmp/claude-501/-Users-david-course-vibe-backtester/b43dd03c-bdbd-4f00-8a63-3c0d2b9a17bb/scratchpad/bench"
# 真實 OpenRouter 定價（$/token）
PRICE={
 "flash":dict(inp=0.09e-6, out=0.18e-6, cache=0.018e-6),
 "pro":  dict(inp=0.435e-6, out=0.87e-6, cache=0.003625e-6),
}
def passed(l):
    m=re.search(r'(\d+) passed', l); return int(m.group(1)) if m else 0
def grade(name):
    ws=f"{SCRATCH}/{name}"
    if not os.path.isdir(ws): return None
    out=subprocess.run(["bash",f"{BENCH}/grade.sh",name,ws],capture_output=True,text=True).stdout
    res=[l for l in out.splitlines() if re.search(r'\d+ (passed|failed|error)',l)]
    vals=[passed(l) for l in res[-4:]]
    while len(vals)<4: vals.insert(0,0)
    return sum(vals), vals
def cost(name, kind):
    f=f"{BENCH}/runs/metrics/{name}.json"
    if not os.path.exists(f): return None
    d=json.load(open(f)); u=d.get("usage",{})
    inp=u.get("input_tokens",0); cr=u.get("cache_read_input_tokens",0)
    cc=u.get("cache_creation_input_tokens",0); out=u.get("output_tokens",0)
    p=PRICE[kind]
    # Anthropic 語意：input_tokens 已是未快取，不可再減 cache。
    c=inp*p["inp"] + (cr+cc)*p["cache"] + out*p["out"]
    return d.get("num_turns"), d.get("duration_ms",0)/1000, c, inp, out

cells=[("c-dsflash-solo","flash","solo"),("c-dsflash-fplan","flash","fplan"),
       ("c-dspro-solo","pro","solo"),("c-dspro-fplan","pro","fplan")]
print(f"{'cell':<20}{'score':>7}  {'turns':>5} {'time':>7} {'cost':>10}  {'in_tok':>9} {'out':>6}")
for name,kind,cond in cells:
    g=grade(name); m=cost(name,kind)
    if g is None or m is None: print(f"{name:<20}  (pending)"); continue
    total,vals=g; turns,sec,c,inp,out=m
    print(f"{name:<20}{total:>4}/74  {str(turns):>5} {sec:>6.1f}s ${c:>8.4f}  {inp:>9,} {out:>6,}")
