"""
用參考解答產生 public 測試的期望值（GRADER ONLY）。

輸出：../expected/expected_public.json
NaN 以 null 表示（載入時再轉回 np.nan）。

用法：
    python benchmark/level1/_grader/gen_expected.py
"""
import json
from pathlib import Path

import pandas as pd

import reference_solution as ref  # 同目錄

LEVEL1 = Path(__file__).resolve().parents[1]
DATA = LEVEL1.parent / "data"
EXPECTED = LEVEL1 / "expected"

# 與 test_public.py 共用的參數（若更動，兩邊要一致）
SMA_WINDOW = 5
EMA_WINDOW = 5


def ser(s: pd.Series):
    return [None if pd.isna(x) else float(x) for x in s]


def main():
    close = pd.read_csv(DATA / "sample_ohlcv_a.csv")["Close"]
    expected = {
        "sma_window": SMA_WINDOW,
        "ema_window": EMA_WINDOW,
        "sma": ser(ref.sma(close, SMA_WINDOW)),
        "ema": ser(ref.ema(close, EMA_WINDOW)),
        "daily_return": ser(ref.daily_return(close)),
        "cumulative_return": ser(ref.cumulative_return(close)),
        "max_drawdown": float(ref.max_drawdown(close)),
    }
    EXPECTED.mkdir(exist_ok=True)
    out = EXPECTED / "expected_public.json"
    out.write_text(json.dumps(expected, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
