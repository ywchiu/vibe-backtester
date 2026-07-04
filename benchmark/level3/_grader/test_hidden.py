"""
Level 3 — hidden 測試（GRADER ONLY，不要交給受測者）。

以 reference_engine 為 oracle，用 hidden 多空資料 + 邊界情境做 differential，
抓「剛好把 public 測試改過去、但沒真正修對」的過擬合修法。

執行：python -m pytest benchmark/level3/_grader/test_hidden.py -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import run_backtest
from reference_engine import run_backtest as ref_run

ATOL = 1e-6
DATA = Path(__file__).resolve().parents[1].parent / "data"


@pytest.fixture(scope="module")
def hidden():
    prices = pd.read_csv(DATA / "hidden_ohlcv.csv")
    signals = pd.read_csv(DATA / "signals_short_hidden.csv")["Signal"]
    return prices, signals


def test_equity_matches_reference(hidden):
    prices, signals = hidden
    got = run_backtest(prices, signals)["equity_curve"].to_numpy(dtype=float)
    exp = ref_run(prices, signals)["equity_curve"].to_numpy(dtype=float)
    np.testing.assert_allclose(got, exp, atol=ATOL)


@pytest.mark.parametrize(
    "key",
    ["total_return", "annualized_return", "volatility", "sharpe_ratio",
     "max_drawdown", "num_trades", "win_rate"],
)
def test_metric_matches_reference(hidden, key):
    prices, signals = hidden
    got = run_backtest(prices, signals)["metrics"][key]
    exp = ref_run(prices, signals)["metrics"][key]
    assert got == pytest.approx(exp, abs=ATOL)


@pytest.mark.parametrize("cost_bps", [0.0, 5.0, 30.0])
def test_cost_matches_reference(hidden, cost_bps):
    prices, signals = hidden
    got = run_backtest(prices, signals, cost_bps=cost_bps)["equity_curve"].iloc[-1]
    exp = ref_run(prices, signals, cost_bps=cost_bps)["equity_curve"].iloc[-1]
    assert got == pytest.approx(exp, abs=ATOL)


def test_short_only_matches_reference():
    prices = pd.DataFrame({"Close": [200.0, 190.0, 195.0, 180.0, 175.0, 185.0]})
    signals = pd.Series([-1, -1, -1, -1, -1, -1])
    got = run_backtest(prices, signals)["equity_curve"].to_numpy(dtype=float)
    exp = ref_run(prices, signals)["equity_curve"].to_numpy(dtype=float)
    np.testing.assert_allclose(got, exp, atol=ATOL)


def test_flip_long_short_double_turnover():
    """從 +1 翻到 -1 換手為 2，成本應為兩倍；比對 reference。"""
    prices = pd.DataFrame({"Close": [100.0, 101.0, 99.0, 102.0, 98.0]})
    signals = pd.Series([1, 1, -1, -1, 1])
    got = run_backtest(prices, signals)["metrics"]["num_trades"]
    exp = ref_run(prices, signals)["metrics"]["num_trades"]
    assert got == exp


def test_no_lookahead_hidden(hidden):
    prices, signals = hidden
    base = run_backtest(prices, signals)["equity_curve"].to_numpy(dtype=float)
    tampered = signals.copy()
    tampered.iloc[-1] = -int(tampered.iloc[-1]) if tampered.iloc[-1] != 0 else 1
    after = run_backtest(prices, tampered)["equity_curve"].to_numpy(dtype=float)
    np.testing.assert_allclose(base, after, atol=ATOL)
