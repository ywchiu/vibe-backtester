"""
Level 1 — 參考解答（GRADER ONLY / 標準答案）。

不要連同任務一起交給受測模型。這支檔案是評分用的 oracle，
也用來產生 expected/expected_public.json（見 gen_expected.py）。
函式簽章與精確定義必須與 ../indicators.py 的 docstring 完全一致。
"""
from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def daily_return(close: pd.Series) -> pd.Series:
    return close / close.shift(1) - 1


def cumulative_return(close: pd.Series) -> pd.Series:
    return close / close.iloc[0] - 1


def max_drawdown(close: pd.Series) -> float:
    peak = close.expanding(min_periods=1).max()
    drawdown = (close - peak) / peak
    return float(drawdown.min())
