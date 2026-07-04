"""
Level 2 — 參考解答（GRADER ONLY / 標準答案）。

不要連同任務交給受測模型。實作必須與 ../backtester.py docstring 的定義逐字一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CADENCE = 252
INITIAL_CAPITAL = 100_000.0
COST_BPS = 10.0


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = INITIAL_CAPITAL,
    cost_bps: float = COST_BPS,
) -> dict:
    close = prices["Close"].astype(float).reset_index(drop=True)
    sig = pd.Series(signals).astype(float).reset_index(drop=True)

    asset_ret = close.pct_change().fillna(0.0)
    position = sig.shift(1).fillna(0.0)
    prev_pos = position.shift(1).fillna(0.0)
    turnover = (position - prev_pos).abs()
    cost = cost_bps / 10_000.0
    strat_ret = position * asset_ret - cost * turnover
    equity = initial_capital * (1.0 + strat_ret).cumprod()

    metrics = _metrics(equity, strat_ret, position, initial_capital)
    equity.index = prices.index
    return {"equity_curve": equity, "metrics": metrics}


def _metrics(equity, strat_ret, position, initial_capital) -> dict:
    r = strat_ret.iloc[1:]
    n = len(r)
    std = r.std(ddof=1)

    total_return = float(equity.iloc[-1] / initial_capital - 1)
    ann_return = float((equity.iloc[-1] / initial_capital) ** (CADENCE / n) - 1) if n > 0 else 0.0
    volatility = float(std * np.sqrt(CADENCE)) if n > 1 else 0.0
    sharpe = float(np.sqrt(CADENCE) * r.mean() / std) if std and std > 0 else 0.0

    peak = equity.expanding(min_periods=1).max()
    max_dd = float(((equity - peak) / peak).min())

    prev_pos = position.shift(1).fillna(0.0)
    turnover = (position - prev_pos).abs()
    num_trades = int((turnover > 0).sum())

    invested = position > 0
    inv_days = int(invested.sum())
    win_days = int(((strat_ret > 0) & invested).sum())
    win_rate = float(win_days / inv_days) if inv_days > 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "num_trades": num_trades,
        "win_rate": win_rate,
    }
