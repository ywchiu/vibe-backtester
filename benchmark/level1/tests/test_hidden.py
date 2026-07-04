"""
Hidden tests — 評分用，不要交給受測者。

策略：以參考解答（reference_solution）作為 oracle，用受測者未見過的
hidden 資料與邊界情境做 differential testing。執行時需設定：

    BT_SOLUTION=<受測模組> python -m pytest benchmark/level1/tests/test_hidden.py

若未設定 BT_SOLUTION，conftest 不會把 _grader 加入路徑，這些測試會 skip，
避免受測者在本機直接跑到 hidden 測試。
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ATOL = 1e-6
LEVEL1 = Path(__file__).resolve().parents[1]
DATA = LEVEL1.parent / "data"

pytestmark = pytest.mark.skipif(
    os.environ.get("BT_SOLUTION", "indicators") == "indicators"
    and not os.environ.get("BT_GRADE"),
    reason="hidden 測試僅在評分模式執行（設定 BT_GRADE=1 或 BT_SOLUTION）",
)


@pytest.fixture(scope="module")
def ref():
    import reference_solution  # _grader 已由 conftest 加入路徑（評分模式）

    return reference_solution


@pytest.fixture(scope="module")
def hidden_close():
    return pd.read_csv(DATA / "hidden_ohlcv.csv")["Close"]


def _same_series(actual, expected):
    assert isinstance(actual, pd.Series)
    a = actual.to_numpy(dtype=float)
    e = expected.to_numpy(dtype=float)
    assert a.shape == e.shape
    np.testing.assert_allclose(a, e, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("window", [3, 5, 10, 20])
def test_sma_hidden(impl, ref, hidden_close, window):
    _same_series(impl.sma(hidden_close, window), ref.sma(hidden_close, window))


@pytest.mark.parametrize("window", [3, 5, 10, 20])
def test_ema_hidden(impl, ref, hidden_close, window):
    _same_series(impl.ema(hidden_close, window), ref.ema(hidden_close, window))


def test_daily_return_hidden(impl, ref, hidden_close):
    _same_series(impl.daily_return(hidden_close), ref.daily_return(hidden_close))


def test_cumulative_return_hidden(impl, ref, hidden_close):
    _same_series(
        impl.cumulative_return(hidden_close), ref.cumulative_return(hidden_close)
    )


def test_max_drawdown_hidden(impl, ref, hidden_close):
    assert impl.max_drawdown(hidden_close) == pytest.approx(
        ref.max_drawdown(hidden_close), abs=ATOL
    )


# --- 邊界情境 ---------------------------------------------------------------

def test_sma_window_larger_than_series(impl, ref):
    """window 大於序列長度時，SMA 全為 NaN。"""
    s = pd.Series([10.0, 11.0, 12.0])
    _same_series(impl.sma(s, 5), ref.sma(s, 5))


def test_flat_prices_zero_drawdown(impl):
    """價格恆定時最大回撤為 0。"""
    s = pd.Series([100.0] * 10)
    assert impl.max_drawdown(s) == pytest.approx(0.0, abs=ATOL)


def test_monotonic_up_zero_drawdown(impl):
    """單調上升時最大回撤為 0。"""
    s = pd.Series([10.0, 11.0, 12.5, 13.0, 20.0])
    assert impl.max_drawdown(s) == pytest.approx(0.0, abs=ATOL)


def test_single_row_cumulative_return(impl, ref):
    """單一資料點：累積報酬為 0、日報酬為 NaN。"""
    s = pd.Series([42.0])
    _same_series(impl.cumulative_return(s), ref.cumulative_return(s))
    assert np.isnan(impl.daily_return(s).iloc[0])


def test_known_drawdown_value(impl):
    """peak=120 後跌到 98：MDD = (98-120)/120。"""
    s = pd.Series([100.0, 120.0, 110.0, 98.0, 105.0])
    assert impl.max_drawdown(s) == pytest.approx((98 - 120) / 120, abs=ATOL)
