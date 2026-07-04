"""Level 3 參考 metrics（GRADER ONLY / 正確版）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

CADENCE = 252


def compute_metrics(equity, strat_ret, position, initial_capital) -> dict:
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

    invested = position != 0
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
