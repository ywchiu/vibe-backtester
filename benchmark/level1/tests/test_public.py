"""
Public tests — 受測者可見，可離線自我檢查。

期望值來自 expected/expected_public.json（由參考解答產生），
不需要參考解答即可執行。比對容忍：絕對誤差 1e-6。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ATOL = 1e-6
LEVEL1 = Path(__file__).resolve().parents[1]
DATA = LEVEL1.parent / "data"
EXPECTED = LEVEL1 / "expected"


@pytest.fixture(scope="module")
def close():
    return pd.read_csv(DATA / "sample_ohlcv_a.csv")["Close"]


@pytest.fixture(scope="module")
def expected():
    return json.loads((EXPECTED / "expected_public.json").read_text())


def _arr(values):
    """把含 null 的 JSON list 轉成 float ndarray（null -> nan）。"""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _check_series(actual, expected_list):
    assert isinstance(actual, pd.Series), "回傳必須是 pandas Series"
    a = actual.to_numpy(dtype=float)
    e = _arr(expected_list)
    assert a.shape == e.shape, f"長度不符：{a.shape} vs {e.shape}"
    np.testing.assert_allclose(a, e, atol=ATOL, equal_nan=True)


def test_sma(impl, close, expected):
    _check_series(impl.sma(close, expected["sma_window"]), expected["sma"])


def test_ema(impl, close, expected):
    _check_series(impl.ema(close, expected["ema_window"]), expected["ema"])


def test_daily_return(impl, close, expected):
    _check_series(impl.daily_return(close), expected["daily_return"])


def test_cumulative_return(impl, close, expected):
    result = impl.cumulative_return(close)
    _check_series(result, expected["cumulative_return"])
    assert result.iloc[0] == pytest.approx(0.0, abs=ATOL), "第一個累積報酬應為 0"


def test_max_drawdown(impl, close, expected):
    result = impl.max_drawdown(close)
    assert isinstance(result, float), "max_drawdown 必須回傳 float"
    assert result == pytest.approx(expected["max_drawdown"], abs=ATOL)


def test_inputs_not_mutated(impl, close):
    """純函式不得修改輸入。"""
    snapshot = close.copy()
    impl.sma(close, 5)
    impl.daily_return(close)
    impl.max_drawdown(close)
    pd.testing.assert_series_equal(close, snapshot)
