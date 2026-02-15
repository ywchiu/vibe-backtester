"""
Moving Average Indicator Module

Provides functions for calculating Simple Moving Average (SMA),
Exponential Moving Average (EMA), and detecting crossover signals
(golden cross / death cross).

Pure Python + pandas/numpy implementation, no TA-Lib dependency.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def calculate_ma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        prices: Price series (typically Close prices).
        period: Number of periods for the moving average window.

    Returns:
        Series of SMA values. The first (period - 1) values will be NaN.

    Raises:
        ValueError: If period is less than 1 or greater than the data length.
    """
    if period < 1:
        raise ValueError(f"Period must be >= 1, got {period}")
    if len(prices) == 0:
        return pd.Series([], dtype=float)
    if period > len(prices):
        raise ValueError(
            f"Period ({period}) exceeds data length ({len(prices)})"
        )

    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).

    Uses the standard EMA formula with span = period.
    The first (period - 1) values will be NaN for consistency with SMA.

    Args:
        prices: Price series (typically Close prices).
        period: Number of periods for the EMA span.

    Returns:
        Series of EMA values.

    Raises:
        ValueError: If period is less than 1 or greater than the data length.
    """
    if period < 1:
        raise ValueError(f"Period must be >= 1, got {period}")
    if len(prices) == 0:
        return pd.Series([], dtype=float)
    if period > len(prices):
        raise ValueError(
            f"Period ({period}) exceeds data length ({len(prices)})"
        )

    ema = prices.ewm(span=period, adjust=False).mean()

    # Set the first (period - 1) values to NaN so the EMA only starts
    # once there are enough data points, matching SMA behavior.
    ema.iloc[: period - 1] = np.nan

    return ema


def detect_crossovers(
    short_ma: pd.Series, long_ma: pd.Series, prices: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Detect golden cross and death cross events between two MA lines.

    A golden cross occurs when the short MA crosses above the long MA.
    A death cross occurs when the short MA crosses below the long MA.

    Args:
        short_ma: Short-period moving average series.
        long_ma: Long-period moving average series.
        prices: Optional close price series to include in results.
                If None, the short_ma value at the crossover is used as price.

    Returns:
        DataFrame with columns:
            - date: Date of the crossover event
            - type: 'golden_cross' or 'death_cross'
            - price: Close price (or short_ma value) at the crossover
            - short_ma: Short MA value at crossover
            - long_ma: Long MA value at crossover
    """
    # Drop NaN rows so we only compare valid MA values
    valid_mask = short_ma.notna() & long_ma.notna()
    short_valid = short_ma[valid_mask]
    long_valid = long_ma[valid_mask]

    if len(short_valid) < 2:
        return pd.DataFrame(
            columns=["date", "type", "price", "short_ma", "long_ma"]
        )

    # Compute the difference and detect sign changes
    diff = short_valid - long_valid
    diff_sign = np.sign(diff)

    crossovers: List[Dict] = []

    prev_sign = diff_sign.iloc[0]
    for i in range(1, len(diff_sign)):
        curr_sign = diff_sign.iloc[i]

        # Skip if either sign is 0 (exactly equal, no clear cross)
        if prev_sign == 0 or curr_sign == 0:
            prev_sign = curr_sign if curr_sign != 0 else prev_sign
            continue

        if prev_sign != curr_sign:
            idx = diff_sign.index[i]
            cross_type = "golden_cross" if curr_sign > 0 else "death_cross"

            cross_price = (
                prices.loc[idx]
                if prices is not None and idx in prices.index
                else short_valid.iloc[i]
            )

            crossovers.append(
                {
                    "date": idx,
                    "type": cross_type,
                    "price": round(float(cross_price), 2),
                    "short_ma": round(float(short_valid.iloc[i]), 2),
                    "long_ma": round(float(long_valid.iloc[i]), 2),
                }
            )

        prev_sign = curr_sign

    df = pd.DataFrame(crossovers)
    if df.empty:
        df = pd.DataFrame(
            columns=["date", "type", "price", "short_ma", "long_ma"]
        )

    return df
