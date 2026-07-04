"""
Hidden tests — 評分用，不要交給受測者。以 reference_solution 為 oracle 做
differential testing，資料用 hidden fixtures，另加邊界情境。

執行：BT_SOLUTION=<受測模組> python -m pytest benchmark/level2/tests/test_hidden.py
未設定 BT_SOLUTION 時自動 skip。
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ATOL = 1e-6
LEVEL = Path(__file__).resolve().parents[1]
DATA = LEVEL.parent / "data"

pytestmark = pytest.mark.skipif(
    os.environ.get("BT_SOLUTION", "backtester") == "backtester"
    and not os.environ.get("BT_GRADE"),
    reason="hidden 測試僅在評分模式執行（設定 BT_GRADE=1 或 BT_SOLUTION）",
)


@pytest.fixture(scope="module")
def ref():
    import reference_solution

    return reference_solution


@pytest.fixture(scope="module")
def hidden():
    prices = pd.read_csv(DATA / "hidden_ohlcv.csv")
    signals = pd.read_csv(DATA / "signals_hidden.csv")["Signal"]
    return prices, signals


def _same(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, atol=ATOL)


def test_equity_hidden(impl, ref, hidden):
    prices, signals = hidden
    got = impl.run_backtest(prices, signals)["equity_curve"].to_numpy(dtype=float)
    exp = ref.run_backtest(prices, signals)["equity_curve"].to_numpy(dtype=float)
    _same(got, exp)


@pytest.mark.parametrize(
    "key",
    ["total_return", "annualized_return", "volatility", "sharpe_ratio",
     "max_drawdown", "num_trades", "win_rate"],
)
def test_metrics_hidden(impl, ref, hidden, key):
    prices, signals = hidden
    got = impl.run_backtest(prices, signals)["metrics"][key]
    exp = ref.run_backtest(prices, signals)["metrics"][key]
    assert got == pytest.approx(exp, abs=ATOL)


@pytest.mark.parametrize("cost_bps", [0.0, 5.0, 25.0])
def test_cost_sensitivity_hidden(impl, ref, hidden, cost_bps):
    prices, signals = hidden
    got = impl.run_backtest(prices, signals, cost_bps=cost_bps)["equity_curve"].iloc[-1]
    exp = ref.run_backtest(prices, signals, cost_bps=cost_bps)["equity_curve"].iloc[-1]
    assert got == pytest.approx(exp, abs=ATOL)


def test_all_flat_signal(impl, ref):
    """全空手：資產不變、報酬 0、無交易。"""
    prices = pd.read_csv(DATA / "hidden_ohlcv.csv")
    flat = pd.Series([0] * len(prices))
    out = impl.run_backtest(prices, flat)
    assert out["equity_curve"].iloc[-1] == pytest.approx(100_000.0, abs=ATOL)
    assert out["metrics"]["num_trades"] == 0
    assert out["metrics"]["total_return"] == pytest.approx(0.0, abs=ATOL)


def test_always_long_trades_once(impl):
    """一路持有：只在第 2 天進場一次（0→1），之後不再換手。"""
    prices = pd.read_csv(DATA / "hidden_ohlcv.csv")
    always = pd.Series([1] * len(prices))
    out = impl.run_backtest(prices, always)
    assert out["metrics"]["num_trades"] == 1


def test_no_lookahead_hidden(impl):
    prices = pd.read_csv(DATA / "hidden_ohlcv.csv")
    signals = pd.read_csv(DATA / "signals_hidden.csv")["Signal"]
    base = impl.run_backtest(prices, signals)["equity_curve"].to_numpy(dtype=float)
    tampered = signals.copy()
    tampered.iloc[-1] = 1 - int(tampered.iloc[-1])
    after = impl.run_backtest(prices, tampered)["equity_curve"].to_numpy(dtype=float)
    _same(base, after)
