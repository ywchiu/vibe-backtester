"""
MA Strategy Unit Tests

Tests for:
- calculate_ma (SMA)
- calculate_ema (EMA)
- detect_crossovers (golden cross / death cross detection)
- backtest_ma_strategy (full backtest engine)
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Ensure backend is on the path (conftest.py also does this, belt-and-suspenders)
backend_dir = os.path.join(os.path.dirname(__file__), "..")
if backend_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(backend_dir))

from indicators.ma_indicator import calculate_ma, calculate_ema, detect_crossovers
from backtest.ma_backtest import backtest_ma_strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_series(values, start="2020-01-01"):
    """Create a pd.Series with a DatetimeIndex from a list of floats."""
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, dtype=float)


def _make_price_df(values, start="2020-01-01"):
    """Create a DataFrame with a 'Close' column and DatetimeIndex."""
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"Close": values}, index=dates, dtype=float)


# ===========================================================================
# 1. SMA Tests
# ===========================================================================

class TestCalculateMA:
    def test_basic_sma(self):
        """SMA of [1,2,3,4,5] with period=3 => [NaN, NaN, 2, 3, 4]."""
        prices = _make_price_series([1, 2, 3, 4, 5])
        result = calculate_ma(prices, period=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_sma_period_1(self):
        """Period=1 SMA should equal the original prices."""
        prices = _make_price_series([10, 20, 30])
        result = calculate_ma(prices, period=1)
        pd.testing.assert_series_equal(result, prices, check_names=False)

    def test_sma_period_equals_length(self):
        """Period == len(prices) should produce one valid value at the end."""
        prices = _make_price_series([2, 4, 6])
        result = calculate_ma(prices, period=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(4.0)

    def test_sma_period_exceeds_length(self):
        """Period > len(prices) should raise ValueError."""
        prices = _make_price_series([1, 2, 3])
        with pytest.raises(ValueError, match="exceeds data length"):
            calculate_ma(prices, period=5)

    def test_sma_period_zero(self):
        """Period < 1 should raise ValueError."""
        prices = _make_price_series([1, 2, 3])
        with pytest.raises(ValueError, match="must be >= 1"):
            calculate_ma(prices, period=0)

    def test_sma_empty_series(self):
        """Empty series should return an empty series."""
        prices = pd.Series([], dtype=float)
        result = calculate_ma(prices, period=3)
        assert len(result) == 0


# ===========================================================================
# 2. EMA Tests
# ===========================================================================

class TestCalculateEMA:
    def test_ema_first_values_are_nan(self):
        """First (period-1) values should be NaN."""
        prices = _make_price_series([10, 20, 30, 40, 50])
        result = calculate_ema(prices, period=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.notna(result.iloc[2])

    def test_ema_known_values(self):
        """
        EMA with span=3, adjust=False.
        multiplier k = 2/(3+1) = 0.5
        After setting first 2 to NaN, index 2 onward should be the
        ewm(span=3, adjust=False) values.
        """
        prices = _make_price_series([1, 2, 3, 4, 5])
        result = calculate_ema(prices, period=3)
        # The underlying ewm(span=3, adjust=False) is computed on all data;
        # the module then sets iloc[:2] = NaN.
        # ewm values (k=0.5): 1, 1.5, 2.25, 3.125, 4.0625
        # After masking: NaN, NaN, 2.25, 3.125, 4.0625
        assert result.iloc[2] == pytest.approx(2.25)
        assert result.iloc[3] == pytest.approx(3.125)
        assert result.iloc[4] == pytest.approx(4.0625)

    def test_ema_period_exceeds_length(self):
        prices = _make_price_series([1, 2])
        with pytest.raises(ValueError, match="exceeds data length"):
            calculate_ema(prices, period=5)

    def test_ema_period_zero(self):
        prices = _make_price_series([1, 2, 3])
        with pytest.raises(ValueError, match="must be >= 1"):
            calculate_ema(prices, period=0)

    def test_ema_empty_series(self):
        prices = pd.Series([], dtype=float)
        result = calculate_ema(prices, period=3)
        assert len(result) == 0

    def test_ema_period_1(self):
        """Period=1 EMA should equal the original prices (no NaN prefix)."""
        prices = _make_price_series([5, 10, 15])
        result = calculate_ema(prices, period=1)
        # With span=1, adjust=False, ewm produces the original series.
        # period-1 = 0 NaN values.
        pd.testing.assert_series_equal(result, prices, check_names=False)


# ===========================================================================
# 3. Crossover Detection Tests
# ===========================================================================

class TestDetectCrossovers:
    def test_single_golden_cross(self):
        """short MA crosses above long MA => one golden cross."""
        # short_ma starts below, then goes above
        short = _make_price_series([10, 10, 20, 30])
        long_ = _make_price_series([15, 15, 15, 15])
        result = detect_crossovers(short, long_)
        assert len(result) == 1
        assert result.iloc[0]["type"] == "golden_cross"

    def test_single_death_cross(self):
        """short MA crosses below long MA => one death cross."""
        short = _make_price_series([30, 20, 10, 5])
        long_ = _make_price_series([15, 15, 15, 15])
        result = detect_crossovers(short, long_)
        assert len(result) == 1
        assert result.iloc[0]["type"] == "death_cross"

    def test_no_crossover(self):
        """short MA always above long MA => no crossovers."""
        short = _make_price_series([20, 25, 30, 35])
        long_ = _make_price_series([10, 10, 10, 10])
        result = detect_crossovers(short, long_)
        assert len(result) == 0

    def test_multiple_crossovers(self):
        """
        Construct a sequence that crosses twice:
        golden cross then death cross.
        """
        short = _make_price_series([5, 5, 20, 20, 5, 5])
        long_ = _make_price_series([10, 10, 10, 10, 10, 10])
        result = detect_crossovers(short, long_)
        assert len(result) == 2
        assert result.iloc[0]["type"] == "golden_cross"
        assert result.iloc[1]["type"] == "death_cross"

    def test_crossover_uses_provided_prices(self):
        """When prices argument is supplied, the 'price' column should use it."""
        short = _make_price_series([5, 20])
        long_ = _make_price_series([10, 10])
        prices = _make_price_series([100, 200])
        result = detect_crossovers(short, long_, prices=prices)
        assert len(result) == 1
        assert result.iloc[0]["price"] == 200.0

    def test_crossover_with_nan_prefix(self):
        """NaN values in either series should be skipped gracefully."""
        short = _make_price_series([np.nan, np.nan, 5, 20])
        long_ = _make_price_series([np.nan, 10, 10, 10])
        result = detect_crossovers(short, long_)
        # Only index 2 and 3 are valid (both non-NaN). At index 2 short<long,
        # at index 3 short>long => golden cross.
        assert len(result) == 1
        assert result.iloc[0]["type"] == "golden_cross"

    def test_crossover_returns_correct_columns(self):
        """Output DataFrame should have the expected columns."""
        short = _make_price_series([5, 20])
        long_ = _make_price_series([10, 10])
        result = detect_crossovers(short, long_)
        assert set(result.columns) == {"date", "type", "price", "short_ma", "long_ma"}

    def test_empty_crossover_has_correct_columns(self):
        """Even with no crossovers, the DataFrame should have the right schema."""
        short = _make_price_series([20, 25])
        long_ = _make_price_series([10, 10])
        result = detect_crossovers(short, long_)
        assert set(result.columns) == {"date", "type", "price", "short_ma", "long_ma"}


# ===========================================================================
# 4. Backtest Engine Tests
# ===========================================================================

class TestBacktestMAStrategy:
    """Tests for the full backtest_ma_strategy function."""

    @staticmethod
    def _build_trending_data(n=60):
        """
        Build price data that produces a clear golden cross followed by
        a death cross with short_period=5, long_period=20.

        Phase 1 (downtrend): prices fall so short MA < long MA.
        Phase 2 (uptrend): prices rise sharply => golden cross.
        Phase 3 (downtrend): prices fall => death cross.
        """
        # 20 days falling from 100 to 80
        down1 = np.linspace(100, 80, 20)
        # 20 days rising from 80 to 140
        up = np.linspace(80, 140, 20)
        # 20 days falling from 140 to 90
        down2 = np.linspace(140, 90, 20)
        values = list(down1) + list(up) + list(down2)
        return _make_price_df(values[:n], start="2020-01-01")

    def test_basic_backtest_returns_expected_keys(self):
        data = self._build_trending_data()
        result = backtest_ma_strategy(
            data, short_period=5, long_period=20, initial_capital=100000
        )
        expected_keys = {
            "total_return", "cagr", "max_drawdown", "sharpe_ratio",
            "win_rate", "total_trades", "trades", "portfolio_history",
            "ma_data", "crossovers", "final_value", "initial_capital",
        }
        assert set(result.keys()) == expected_keys

    def test_initial_capital_preserved_in_output(self):
        data = self._build_trending_data()
        result = backtest_ma_strategy(
            data, short_period=5, long_period=20, initial_capital=50000
        )
        assert result["initial_capital"] == 50000.0

    def test_portfolio_history_length_matches_data(self):
        data = self._build_trending_data()
        result = backtest_ma_strategy(
            data, short_period=5, long_period=20, initial_capital=100000
        )
        assert len(result["portfolio_history"]) == len(data)

    def test_trades_alternate_buy_sell(self):
        """Trades should always alternate buy/sell starting with buy."""
        data = self._build_trending_data()
        result = backtest_ma_strategy(
            data, short_period=5, long_period=20, initial_capital=100000
        )
        trades = result["trades"]
        if len(trades) > 0:
            # First trade must be a buy
            assert trades[0]["type"] == "buy"
            for i in range(1, len(trades)):
                assert trades[i]["type"] != trades[i - 1]["type"]

    def test_ema_mode(self):
        """Backtest should work with ma_type='ema'."""
        data = self._build_trending_data()
        result = backtest_ma_strategy(
            data, short_period=5, long_period=20,
            initial_capital=100000, ma_type="ema"
        )
        assert "total_return" in result
        assert len(result["portfolio_history"]) == len(data)

    def test_no_crossover_means_capital_unchanged(self):
        """
        If prices are strictly increasing, with short_period < long_period,
        short MA should always be above long MA once both are valid.
        No crossovers => no trades => final value == initial capital.
        """
        # Strictly monotonically increasing prices
        values = list(range(50, 100))  # 50 data points
        data = _make_price_df(values)
        result = backtest_ma_strategy(
            data, short_period=5, long_period=20, initial_capital=100000
        )
        # No death cross should occur in monotone increasing data,
        # so at most a buy can happen but no completed round trip.
        assert result["total_trades"] == 0

    def test_final_value_calculation(self):
        """
        With a simple scenario where we can trace the math:
        - Flat prices => no crossovers => final_value == initial_capital.
        """
        # Constant price = 100 for 30 days
        values = [100.0] * 30
        data = _make_price_df(values)
        result = backtest_ma_strategy(
            data, short_period=5, long_period=20, initial_capital=10000
        )
        # No crossovers because short_ma == long_ma; final value = initial
        assert result["final_value"] == pytest.approx(10000.0)
        assert result["total_return"] == pytest.approx(0.0)

    def test_ma_data_contains_close_and_ma_values(self):
        data = self._build_trending_data()
        result = backtest_ma_strategy(
            data, short_period=5, long_period=20, initial_capital=100000
        )
        for entry in result["ma_data"]:
            assert "date" in entry
            assert "close" in entry
            assert "short_ma" in entry
            assert "long_ma" in entry

    # --- Validation / Edge-case tests ---

    def test_short_period_gte_long_period_raises(self):
        data = self._build_trending_data()
        with pytest.raises(ValueError, match="short_period.*must be less than"):
            backtest_ma_strategy(
                data, short_period=20, long_period=20, initial_capital=100000
            )

    def test_short_period_greater_than_long_raises(self):
        data = self._build_trending_data()
        with pytest.raises(ValueError, match="short_period.*must be less than"):
            backtest_ma_strategy(
                data, short_period=30, long_period=20, initial_capital=100000
            )

    def test_negative_initial_capital_raises(self):
        data = self._build_trending_data()
        with pytest.raises(ValueError, match="initial_capital must be > 0"):
            backtest_ma_strategy(
                data, short_period=5, long_period=20, initial_capital=-1000
            )

    def test_zero_initial_capital_raises(self):
        data = self._build_trending_data()
        with pytest.raises(ValueError, match="initial_capital must be > 0"):
            backtest_ma_strategy(
                data, short_period=5, long_period=20, initial_capital=0
            )

    def test_invalid_ma_type_raises(self):
        data = self._build_trending_data()
        with pytest.raises(ValueError, match="ma_type must be"):
            backtest_ma_strategy(
                data, short_period=5, long_period=20,
                initial_capital=100000, ma_type="wma"
            )

    def test_insufficient_data_raises(self):
        """Data length < long_period + 1 should raise."""
        data = _make_price_df([100.0] * 10)
        with pytest.raises(ValueError, match="Not enough data"):
            backtest_ma_strategy(
                data, short_period=5, long_period=20, initial_capital=100000
            )

    def test_missing_close_column_raises(self):
        dates = pd.bdate_range("2020-01-01", periods=30)
        data = pd.DataFrame({"Open": [100.0] * 30}, index=dates)
        with pytest.raises(ValueError, match="Close"):
            backtest_ma_strategy(
                data, short_period=5, long_period=20, initial_capital=100000
            )

    def test_crossover_events_match_trades(self):
        """
        Every buy/sell trade should correspond to a crossover event.
        """
        data = self._build_trending_data()
        result = backtest_ma_strategy(
            data, short_period=5, long_period=20, initial_capital=100000
        )
        crossover_dates = {c["date"] for c in result["crossovers"]}
        for trade in result["trades"]:
            assert trade["date"] in crossover_dates

    def test_buy_shares_calculation(self):
        """
        On the first buy trade, shares * price should equal the initial capital.
        """
        data = self._build_trending_data()
        result = backtest_ma_strategy(
            data, short_period=5, long_period=20, initial_capital=100000
        )
        buy_trades = [t for t in result["trades"] if t["type"] == "buy"]
        if buy_trades:
            first_buy = buy_trades[0]
            assert first_buy["value"] == pytest.approx(100000.0)
            expected_shares = 100000.0 / first_buy["price"]
            assert first_buy["shares"] == pytest.approx(expected_shares, rel=1e-3)
