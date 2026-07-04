"""
Level 2 — 簡單回測器（STARTER）

實作 run_backtest()。規格見同目錄 README.md，精確定義如下（reference 與
grader 都以這份定義為準，請完全照做，任何一步錯位都會被 hidden 測試抓到）。

只允許 pandas / numpy。純函式：不得修改輸入。誤差容忍 1e-6。

======================================================================
輸入
  prices : pd.DataFrame，至少含 'Close' 欄，index 為日期（可為 RangeIndex）
  signals: pd.Series，與 prices 等長，值 ∈ {0, 1}（0=空手，1=持有）
           signal[t] 是「在第 t 天收盤時、用截至第 t 天的資訊」決定的目標部位。
  initial_capital: 起始資金，預設 100_000.0
  cost_bps       : 每單位換手的交易成本（基點），預設 10.0，即 cost = cost_bps/10000

核心規則（避免 look-ahead bias）
  asset_ret[t] = close[t]/close[t-1] - 1        # asset_ret[0] = 0
  position[t]  = signal[t-1]                     # 用「昨天」的訊號，position[0] = 0
                 （等價 signals.shift(1).fillna(0)）
  turnover[t]  = |position[t] - position[t-1]|   # position[-1] 視為 0
  strat_ret[t] = position[t]*asset_ret[t] - (cost_bps/10000)*turnover[t]
  equity[t]    = initial_capital * cumprod(1 + strat_ret)[t]   # equity[0] = initial_capital

指標（全部由 equity / strat_ret / position 計算；令 r = strat_ret[1:]，n = len(r)）
  total_return      = equity[-1]/initial_capital - 1
  annualized_return = (equity[-1]/initial_capital)**(252/n) - 1
  volatility        = r.std(ddof=1) * sqrt(252)
  sharpe_ratio      = sqrt(252) * r.mean()/r.std(ddof=1)   # 無風險利率=0；std=0 時回傳 0.0
  max_drawdown      = min((equity - equity.expanding().max()) / equity.expanding().max())
  num_trades        = turnover>0 的天數（每次 0→1 與 1→0 各算一筆）
  win_rate          = (strat_ret>0 且 position>0 的天數) / (position>0 的天數)；分母為 0 回傳 0.0
======================================================================
"""
from __future__ import annotations

import pandas as pd

CADENCE = 252  # 每年交易日數
INITIAL_CAPITAL = 100_000.0
COST_BPS = 10.0


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = INITIAL_CAPITAL,
    cost_bps: float = COST_BPS,
) -> dict:
    """
    回傳 dict：
        {
          "equity_curve": pd.Series,   # 長度與 prices 相同，equity[0]=initial_capital
          "metrics": {
              "total_return": float,
              "annualized_return": float,
              "volatility": float,
              "sharpe_ratio": float,
              "max_drawdown": float,
              "num_trades": int,
              "win_rate": float,
          },
        }
    """
    raise NotImplementedError
