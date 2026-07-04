import json, os
BENCH="/Users/david/course/vibe-backtester/benchmark"
M=f"{BENCH}/runs/metrics"
def j(n): return json.load(open(f"{M}/{n}.json"))

# 定價 $/token
GPT=dict(inp=5e-6,out=30e-6,cache=0.5e-6)
FLASH=dict(inp=0.09e-6,out=0.18e-6,cache=0.018e-6)
PRO=dict(inp=0.435e-6,out=0.87e-6,cache=0.003625e-6)

def claude_row(n):
    d=j(n); u=d["usage"]
    tin=u["input_tokens"]+u["cache_creation_input_tokens"]+u["cache_read_input_tokens"]
    return d["num_turns"], d["duration_ms"]/1000, tin, u["output_tokens"], tin+u["output_tokens"], d["total_cost_usd"]

def codex_row(n):
    d=j(n); inp=d["input_tokens"]; cr=d["cached_input_tokens"]; out=d["output_tokens"]
    cost=(inp-cr)*GPT["inp"]+cr*GPT["cache"]+out*GPT["out"]
    return "~", d["duration_ms"]/1000, inp, out, d["total_tokens"], cost

def ds_row(n,P):
    d=j(n); u=d["usage"]
    inp=u["input_tokens"]; cr=u.get("cache_read_input_tokens",0); cc=u.get("cache_creation_input_tokens",0); out=u["output_tokens"]
    cost=(inp-cr-cc)*P["inp"]+(cr+cc)*P["cache"]+out*P["out"]
    return d["num_turns"], d["duration_ms"]/1000, inp+cc+cr, out, inp+cc+cr+out, cost

rows=[
 ("DeepSeek Flash","solo",ds_row("c-dsflash-solo",FLASH)),
 ("DeepSeek Flash","fplan",ds_row("c-dsflash-fplan",FLASH)),
 ("DeepSeek Pro","solo",ds_row("c-dspro-solo",PRO)),
 ("DeepSeek Pro","fplan",ds_row("c-dspro-fplan",PRO)),
 ("Haiku 4.5","solo",claude_row("c-haiku-solo")),
 ("Haiku 4.5","fplan",claude_row("c-haiku-fplan")),
 ("Sonnet 5","solo",claude_row("c-sonnet-solo")),
 ("Sonnet 5","fplan",claude_row("c-sonnet-fplan")),
 ("Opus 4.8","solo",claude_row("c-opus-solo")),
 ("Opus 4.8","fplan",claude_row("c-opus-fplan")),
 ("Codex gpt-5.5","solo",codex_row("c-codex-solo")),
 ("Codex gpt-5.5","fplan",codex_row("c-codex-fplan")),
]
plan=claude_row("mastermind")
print(f"{'model':<16}{'cond':<7}{'turns':>6}{'time(s)':>9}{'total_tok':>11}{'out_tok':>9}{'cost':>10}")
print("-"*70)
for name,cond,(t,s,tin,out,tot,c) in rows:
    print(f"{name:<16}{cond:<7}{str(t):>6}{s:>9.1f}{tot:>11,}{out:>9,}${c:>8.4f}")
print("-"*70)
print(f"{'MASTERMIND(Opus)':<16}{'plan':<7}{plan[0]:>6}{plan[1]:>9.1f}{plan[4]:>11,}{plan[3]:>9,}${plan[5]:>8.4f}")
