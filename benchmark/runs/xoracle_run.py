"""Cross-oracle check: compare my reference vs non-Claude contestant solutions
on UNSEEN random data. If all agree <1e-6, the reference isn't a Claude artifact."""
import os, json, subprocess, tempfile, shutil, math
import numpy as np, pandas as pd

BENCH = "/Users/david/course/vibe-backtester/benchmark"
SCRATCH = "/private/tmp/claude-501/-Users-david-course-vibe-backtester/b43dd03c-bdbd-4f00-8a63-3c0d2b9a17bb/scratchpad/bench"
PY = f"{SCRATCH}/.venv/bin/python"
WORKER = "/private/tmp/claude-501/-Users-david-course-vibe-backtester/b43dd03c-bdbd-4f00-8a63-3c0d2b9a17bb/scratchpad/xo_worker.py"

# --- 1) 產生 benchmark 沒看過的隨機資料（固定 seed 便於重現）---
rng = np.random.default_rng(999983)
n = 60
close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
dd = tempfile.mkdtemp()
pd.DataFrame({"Close": close}).to_csv(f"{dd}/close.csv", index=False)
prices = pd.DataFrame({"Open": close, "High": close*1.01, "Low": close*0.99,
                       "Close": close, "Volume": 1_000_000})
prices.to_csv(f"{dd}/prices.csv", index=False)
pd.DataFrame({"Signal": rng.integers(0, 2, n)}).to_csv(f"{dd}/signals2.csv", index=False)
pd.DataFrame({"Signal": rng.integers(-1, 2, n)}).to_csv(f"{dd}/signals3.csv", index=False)

# --- 2) 把「我的參考解答」組成一個標準 solution 目錄 ---
ref = tempfile.mkdtemp()
os.makedirs(f"{ref}/level1"); os.makedirs(f"{ref}/level2"); os.makedirs(f"{ref}/level3/backtester")
shutil.copy(f"{BENCH}/level1/_grader/reference_solution.py", f"{ref}/level1/indicators.py")
shutil.copy(f"{BENCH}/level2/_grader/reference_solution.py", f"{ref}/level2/backtester.py")
# level3 package: engine + metrics with proper relative import
met = open(f"{BENCH}/level3/_grader/reference_metrics.py").read()
eng = open(f"{BENCH}/level3/_grader/reference_engine.py").read().replace(
    "from reference_metrics import compute_metrics", "from .metrics import compute_metrics")
open(f"{ref}/level3/backtester/metrics.py", "w").write(met)
open(f"{ref}/level3/backtester/engine.py", "w").write(eng)
open(f"{ref}/level3/backtester/__init__.py", "w").write("from .engine import run_backtest\n")

solutions = {
    "MY-REF(Opus)": ref,
    "Codex(gpt5.5)": f"{SCRATCH}/c-codex-solo",
    "DeepSeekPro":   f"{SCRATCH}/c-dspro-solo",
    "DeepSeekFlash": f"{SCRATCH}/c-dsflash-solo",
}

def run(root, level):
    cwd = f"{root}/level{level}"
    r = subprocess.run([PY, WORKER, level, dd], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr[-300:]
    return json.loads(r.stdout), None

def flat(o):
    """flatten outputs to a numeric vector (NaN→marker)."""
    v = []
    def add(x):
        if isinstance(x, list):
            for e in x: add(e)
        elif isinstance(x, dict):
            for k in sorted(x): add(x[k])
        elif x is None: v.append(math.nan)
        else: v.append(float(x))
    add(o)
    return np.array(v, float)

for level in ["1", "2", "3"]:
    print(f"\n===== Level {level}: 與 MY-REF 的最大絕對差 (unseen 隨機資料) =====")
    base, err = run(ref, level)
    if err: print("  REF ERROR:", err); continue
    bv = flat(base)
    for name, root in solutions.items():
        if name == "MY-REF(Opus)": continue
        o, err = run(root, level)
        if err: print(f"  {name:<16} ERROR: {err}"); continue
        ov = flat(o)
        if ov.shape != bv.shape:
            print(f"  {name:<16} shape mismatch {ov.shape} vs {bv.shape}"); continue
        both_nan = np.isnan(ov) & np.isnan(bv)
        diff = np.where(both_nan, 0.0, np.abs(ov - bv))
        nan_mismatch = int((np.isnan(ov) ^ np.isnan(bv)).sum())
        print(f"  {name:<16} max|Δ|={np.nanmax(diff):.2e}   NaN位置不一致={nan_mismatch}   {'✅ 一致' if np.nanmax(diff)<1e-6 and nan_mismatch==0 else '⚠ 不一致'}")

shutil.rmtree(dd); shutil.rmtree(ref)
