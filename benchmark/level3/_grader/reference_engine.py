"""Level 3 參考 engine（GRADER ONLY / 正確版，支援多空 {-1,0,1}）。"""
from __future__ import annotations

import pandas as pd

from reference_metrics import compute_metrics

INITIAL_CAPITAL = 100_000.0
COST_BPS = 10.0


def run_backtest(prices, signals, initial_capital=INITIAL_CAPITAL, cost_bps=COST_BPS) -> dict:
    close = prices["Close"].astype(float).reset_index(drop=True)
    sig = pd.Series(signals).astype(float).reset_index(drop=True)

    asset_ret = close.pct_change().fillna(0.0)
    position = sig.shift(1).fillna(0.0)
    turnover = (position - position.shift(1).fillna(0.0)).abs()
    cost = cost_bps / 10_000.0
    strat_ret = position * asset_ret - cost * turnover
    equity = initial_capital * (1.0 + strat_ret).cumprod()

    metrics = compute_metrics(equity, strat_ret, position, initial_capital)
    equity.index = prices.index
    return {"equity_curve": equity, "metrics": metrics}
