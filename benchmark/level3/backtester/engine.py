"""
回測引擎。支援多空訊號 {-1, 0, 1}。規格見 ../README.md。

核心規則：
  asset_ret[t] = close[t]/close[t-1] - 1（第 0 天視為 0）
  position[t]  = signal[t-1]（用昨天訊號，避免 look-ahead）
  turnover[t]  = |position[t] - position[t-1]|
  strat_ret[t] = position[t]*asset_ret[t] - (cost_bps/10000)*turnover[t]
  equity[t]    = initial_capital * cumprod(1 + strat_ret)[t]
"""
from __future__ import annotations

import pandas as pd

from .metrics import compute_metrics

INITIAL_CAPITAL = 100_000.0
COST_BPS = 10.0


def run_backtest(prices, signals, initial_capital=INITIAL_CAPITAL, cost_bps=COST_BPS) -> dict:
    close = prices["Close"].astype(float).reset_index(drop=True)
    sig = pd.Series(signals).astype(float).reset_index(drop=True)

    asset_ret = close.pct_change().fillna(0.0)
    position = sig.clip(lower=0)
    turnover = (position - position.shift(1).fillna(0.0)).abs()
    strat_ret = position * asset_ret
    equity = initial_capital * (1.0 + strat_ret).cumprod()

    metrics = compute_metrics(equity, strat_ret, position, initial_capital)
    equity.index = prices.index
    return {"equity_curve": equity, "metrics": metrics}
