"""Run one solution's outputs on shared UNSEEN random data. cwd = <soln>/levelN."""
import sys, os, json, math
import pandas as pd
sys.path.insert(0, os.getcwd())  # 讓 import 指向這個 solution 的 levelN 目錄
level, datadir = sys.argv[1], sys.argv[2]

def ser(s):
    return [None if pd.isna(x) else float(x) for x in s]

out = {}
if level == "1":
    import indicators as m
    close = pd.read_csv(f"{datadir}/close.csv")["Close"]
    out["sma"] = ser(m.sma(close, 5))
    out["ema"] = ser(m.ema(close, 5))
    out["daily_return"] = ser(m.daily_return(close))
    out["cumulative_return"] = ser(m.cumulative_return(close))
    out["max_drawdown"] = float(m.max_drawdown(close))
elif level == "2":
    import backtester as m
    prices = pd.read_csv(f"{datadir}/prices.csv")
    sig = pd.read_csv(f"{datadir}/signals2.csv")["Signal"]
    r = m.run_backtest(prices, sig)
    out["equity"] = ser(r["equity_curve"])
    out["metrics"] = {k: float(v) for k, v in r["metrics"].items()}
elif level == "3":
    import backtester as m
    prices = pd.read_csv(f"{datadir}/prices.csv")
    sig = pd.read_csv(f"{datadir}/signals3.csv")["Signal"]
    r = m.run_backtest(prices, sig)
    out["equity"] = ser(r["equity_curve"])
    out["metrics"] = {k: float(v) for k, v in r["metrics"].items()}
print(json.dumps(out))
