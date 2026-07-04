"""
Public tests — 受測者可見，離線可自我檢查。期望值來自 expected/expected_public.json。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ATOL = 1e-6
LEVEL = Path(__file__).resolve().parents[1]
DATA = LEVEL.parent / "data"
EXPECTED = LEVEL / "expected"


@pytest.fixture(scope="module")
def prices():
    return pd.read_csv(DATA / "sample_ohlcv_a.csv")


@pytest.fixture(scope="module")
def signals():
    return pd.read_csv(DATA / "signals_a.csv")["Signal"]


@pytest.fixture(scope="module")
def expected():
    return json.loads((EXPECTED / "expected_public.json").read_text())


@pytest.fixture(scope="module")
def result(impl, prices, signals):
    return impl.run_backtest(prices, signals)


def test_return_shape(result):
    assert isinstance(result, dict)
    assert "equity_curve" in result and "metrics" in result
    assert isinstance(result["equity_curve"], pd.Series)


def test_equity_curve(result, expected):
    a = result["equity_curve"].to_numpy(dtype=float)
    e = np.array(expected["equity_curve"], dtype=float)
    assert a.shape == e.shape, f"長度不符：{a.shape} vs {e.shape}"
    np.testing.assert_allclose(a, e, atol=ATOL)


def test_equity_starts_at_capital(result, expected):
    assert result["equity_curve"].iloc[0] == pytest.approx(
        expected["initial_capital"], abs=ATOL
    )


@pytest.mark.parametrize(
    "key",
    [
        "total_return",
        "annualized_return",
        "volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
    ],
)
def test_float_metric(result, expected, key):
    assert result["metrics"][key] == pytest.approx(expected["metrics"][key], abs=ATOL)


def test_num_trades(result, expected):
    assert result["metrics"]["num_trades"] == expected["metrics"]["num_trades"]


def test_transaction_cost_applied(impl, prices, signals):
    """成本必須真的扣：cost_bps=0 的最終資產應高於 cost_bps=10（此訊號有換手）。"""
    free = impl.run_backtest(prices, signals, cost_bps=0.0)["equity_curve"].iloc[-1]
    charged = impl.run_backtest(prices, signals, cost_bps=10.0)["equity_curve"].iloc[-1]
    assert free > charged, "交易成本似乎沒有被扣除"


def test_no_lookahead(impl, prices, signals):
    """最後一天的訊號永遠不會被執行；改動它不得改變任何結果。"""
    base = impl.run_backtest(prices, signals)["equity_curve"].to_numpy(dtype=float)
    tampered = signals.copy()
    tampered.iloc[-1] = 1 - int(tampered.iloc[-1])  # 翻轉最後一天訊號
    after = impl.run_backtest(prices, tampered)["equity_curve"].to_numpy(dtype=float)
    np.testing.assert_allclose(base, after, atol=ATOL,
                               err_msg="改動最後一天訊號改變了結果 → 疑似 look-ahead bias")


def test_inputs_not_mutated(impl, prices, signals):
    p, s = prices.copy(), signals.copy()
    impl.run_backtest(prices, signals)
    pd.testing.assert_frame_equal(prices, p)
    pd.testing.assert_series_equal(signals, s)
