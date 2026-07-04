"""
Level 3 — public 測試（受測者可見）。

任務：修復 backtester/ 讓所有測試通過，且**不得修改本檔**。
目前有一部分測試是 failing 的，代表 backtester/ 內藏有數個 bug。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import run_backtest

ATOL = 1e-6
DATA = Path(__file__).resolve().parents[1].parent / "data"


@pytest.fixture(scope="module")
def prices():
    return pd.read_csv(DATA / "sample_ohlcv_a.csv")


@pytest.fixture(scope="module")
def signals():
    # 多空訊號 {-1, 0, 1}
    return pd.read_csv(DATA / "signals_short_a.csv")["Signal"]


# --- 這些應該一開始就會通過 -------------------------------------------------

def test_result_structure(prices, signals):
    out = run_backtest(prices, signals)
    assert set(out) == {"equity_curve", "metrics"}
    assert isinstance(out["equity_curve"], pd.Series)


def test_equity_length(prices, signals):
    out = run_backtest(prices, signals)
    assert len(out["equity_curve"]) == len(prices)


def test_equity_starts_at_capital(prices, signals):
    out = run_backtest(prices, signals)
    assert out["equity_curve"].iloc[0] == pytest.approx(100_000.0, abs=ATOL)


def test_deterministic(prices, signals):
    a = run_backtest(prices, signals)["equity_curve"].to_numpy(dtype=float)
    b = run_backtest(prices, signals)["equity_curve"].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, atol=ATOL)


def test_metrics_keys(prices, signals):
    m = run_backtest(prices, signals)["metrics"]
    assert set(m) == {"total_return", "annualized_return", "volatility",
                      "sharpe_ratio", "max_drawdown", "num_trades", "win_rate"}


# --- 這些一開始會 failing（藏著 bug）----------------------------------------

def test_transaction_cost_applied(prices, signals):
    """交易成本要真的扣：cost_bps=0 的最終資產應高於 cost_bps=10。"""
    free = run_backtest(prices, signals, cost_bps=0.0)["equity_curve"].iloc[-1]
    charged = run_backtest(prices, signals, cost_bps=10.0)["equity_curve"].iloc[-1]
    assert free > charged


def test_no_lookahead(prices, signals):
    """改動最後一天的訊號不得改變結果（它永遠不會被執行）。"""
    base = run_backtest(prices, signals)["equity_curve"].to_numpy(dtype=float)
    tampered = signals.copy()
    tampered.iloc[-1] = -int(tampered.iloc[-1]) if tampered.iloc[-1] != 0 else 1
    after = run_backtest(prices, tampered)["equity_curve"].to_numpy(dtype=float)
    np.testing.assert_allclose(base, after, atol=ATOL)


def test_short_position_profits_when_price_falls():
    """一路做空且價格下跌時，應該獲利（最終資產 > 起始資金）。"""
    prices = pd.DataFrame({"Close": [100.0, 95.0, 90.0, 85.0, 80.0]})
    short = pd.Series([-1, -1, -1, -1, -1])
    out = run_backtest(prices, short)
    assert out["equity_curve"].iloc[-1] > 100_000.0


def test_max_drawdown_is_a_fraction(prices, signals):
    """最大回撤是比率，必須落在 [-1, 0]。"""
    mdd = run_backtest(prices, signals)["metrics"]["max_drawdown"]
    assert -1.0 <= mdd <= 0.0


def test_max_drawdown_known_value():
    """peak=120 後跌到 96：MDD = (96-120)/120 = -0.2。"""
    prices = pd.DataFrame({"Close": [100.0, 120.0, 108.0, 96.0, 102.0]})
    always_long = pd.Series([1, 1, 1, 1, 1])
    mdd = run_backtest(prices, always_long)["metrics"]["max_drawdown"]
    assert mdd == pytest.approx((96 - 120) / 120, abs=1e-4)


def test_flat_signal_win_rate_is_zero():
    """全空手時沒有持倉日，win_rate 應為 0（不得是 NaN）。"""
    prices = pd.read_csv(DATA / "sample_ohlcv_a.csv")
    flat = pd.Series([0] * len(prices))
    m = run_backtest(prices, flat)["metrics"]
    assert m["win_rate"] == pytest.approx(0.0, abs=ATOL)
    assert not np.isnan(m["win_rate"])
