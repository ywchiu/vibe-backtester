"""
Level 1 — 技術指標計算（STARTER）

請實作以下 5 個純函式。規格與精確定義見同目錄 README.md。
只允許使用 pandas / numpy，不得讀取外部資料或網路。
所有函式必須是純函式：相同輸入永遠產生相同輸出，且不修改輸入。

誤差容忍：所有數值以絕對誤差 1e-6 比對。
"""
from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """簡單移動平均。前 window-1 個值為 NaN。回傳與 close 同 index 的 Series。"""
    raise NotImplementedError


def ema(close: pd.Series, window: int) -> pd.Series:
    """
    指數移動平均。

    定義：span=window、adjust=False，即 alpha = 2 / (window + 1)，
    以第一個 close 值作為種子（第一個輸出等於 close.iloc[0]，不是 NaN）。
    等價於 close.ewm(span=window, adjust=False).mean()。
    """
    raise NotImplementedError


def daily_return(close: pd.Series) -> pd.Series:
    """日報酬率 = close / close.shift(1) - 1。第一個值為 NaN。"""
    raise NotImplementedError


def cumulative_return(close: pd.Series) -> pd.Series:
    """累積報酬率 = close / close.iloc[0] - 1。第一個值為 0.0。"""
    raise NotImplementedError


def max_drawdown(close: pd.Series) -> float:
    """
    最大回撤（回傳單一 float，通常為負值或 0）。

    定義：peak = close 的 running maximum（expanding max）；
    drawdown = (close - peak) / peak；回傳 drawdown 的最小值。
    價格單調不跌時回傳 0.0。
    """
    raise NotImplementedError
