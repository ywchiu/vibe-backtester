"""
績效指標計算。

規格（正確定義）見 ../README.md。令 r = strat_ret[1:]、n = len(r)：
  total_return      = equity[-1]/initial_capital - 1
  annualized_return = (equity[-1]/initial_capital)**(252/n) - 1
  volatility        = r.std(ddof=1)*sqrt(252)
  sharpe_ratio      = sqrt(252)*r.mean()/r.std(ddof=1)（rf=0；std=0 回傳 0）
  max_drawdown      = min((equity - expanding_max) / expanding_max)   # 是「比率」，介於 -1~0
  num_trades        = turnover>0 的天數
  win_rate          = (strat_ret>0 且 position!=0 天數)/(position!=0 天數)；分母 0 回傳 0
"""
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
    max_dd = float((equity - peak).min())

    prev_pos = position.shift(1).fillna(0.0)
    turnover = (position - prev_pos).abs()
    num_trades = int((turnover > 0).sum())

    invested = position != 0
    inv_days = int(invested.sum())
    win_days = int(((strat_ret > 0) & invested).sum())
    win_rate = float(win_days / inv_days)

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "num_trades": num_trades,
        "win_rate": win_rate,
    }
