"""
MA Crossover Strategy Backtest Engine

Implements a moving average crossover trading strategy:
- Buy (go long) on golden cross (short MA crosses above long MA)
- Sell (close position) on death cross (short MA crosses below long MA)

Uses the indicator module for MA calculation and crossover detection,
and the existing utils/calculations.py for financial metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

import sys
import os

# Ensure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from indicators.ma_indicator import calculate_ma, calculate_ema, detect_crossovers
from utils.calculations import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_daily_returns,
    calculate_total_return,
)


def backtest_ma_strategy(
    data: pd.DataFrame,
    short_period: int,
    long_period: int,
    initial_capital: float,
    ma_type: str = "sma",
) -> Dict:
    """
    Execute an MA crossover strategy backtest.

    Strategy logic:
        - Golden cross (short MA > long MA): buy with all available cash
        - Death cross (short MA < long MA): sell entire position

    Args:
        data: DataFrame with at least a 'Close' column and a DatetimeIndex.
        short_period: Period for the short (fast) moving average.
        long_period: Period for the long (slow) moving average.
        initial_capital: Starting cash amount.
        ma_type: Type of moving average - 'sma' or 'ema'.

    Returns:
        Dictionary containing:
            - total_return: Total return percentage
            - cagr: Compound annual growth rate percentage
            - max_drawdown: Maximum drawdown percentage
            - sharpe_ratio: Sharpe ratio
            - win_rate: Win rate percentage
            - total_trades: Number of completed round-trip trades
            - trades: List of trade records
            - portfolio_history: Daily portfolio value list
            - ma_data: MA line data for charting
            - crossovers: Crossover events list

    Raises:
        ValueError: If parameters are invalid.
    """
    # --- Input validation ---
    if short_period >= long_period:
        raise ValueError(
            f"short_period ({short_period}) must be less than "
            f"long_period ({long_period})"
        )
    if short_period < 1:
        raise ValueError(f"short_period must be >= 1, got {short_period}")
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
    if ma_type not in ("sma", "ema"):
        raise ValueError(f"ma_type must be 'sma' or 'ema', got '{ma_type}'")

    if "Close" not in data.columns:
        raise ValueError("data must contain a 'Close' column")

    if len(data) < long_period + 1:
        raise ValueError(
            f"Not enough data points ({len(data)}) for long_period ({long_period}). "
            f"Need at least {long_period + 1} rows."
        )

    # --- Calculate moving averages ---
    close = data["Close"].copy()

    if ma_type == "sma":
        short_ma = calculate_ma(close, short_period)
        long_ma = calculate_ma(close, long_period)
    else:
        short_ma = calculate_ema(close, short_period)
        long_ma = calculate_ema(close, long_period)

    # --- Detect crossovers ---
    crossover_df = detect_crossovers(short_ma, long_ma, prices=close)

    # --- Simulate trades ---
    cash = initial_capital
    shares = 0.0
    position_open = False
    entry_price = 0.0
    entry_date = None

    trades: List[Dict] = []
    completed_trades: List[Dict] = []  # round-trip trades for win-rate calc

    # Build daily portfolio history
    portfolio_values = []
    dates = data.index.tolist()

    # Convert crossover dates to a dict for O(1) lookup
    crossover_map: Dict = {}
    for _, row in crossover_df.iterrows():
        crossover_map[row["date"]] = row["type"]

    for date in dates:
        price = float(close.loc[date])

        # Check for signal on this date
        signal = crossover_map.get(date)

        if signal == "golden_cross" and not position_open:
            # Buy signal
            shares = cash / price
            entry_price = price
            entry_date = date
            trades.append(
                {
                    "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                    "type": "buy",
                    "price": round(price, 2),
                    "shares": round(shares, 4),
                    "value": round(cash, 2),
                }
            )
            cash = 0.0
            position_open = True

        elif signal == "death_cross" and position_open:
            # Sell signal
            sell_value = shares * price
            profit = sell_value - (shares * entry_price)
            trades.append(
                {
                    "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                    "type": "sell",
                    "price": round(price, 2),
                    "shares": round(shares, 4),
                    "value": round(sell_value, 2),
                }
            )
            completed_trades.append(
                {
                    "entry_date": entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, "strftime") else str(entry_date),
                    "exit_date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "profit": round(profit, 2),
                    "return_pct": round((price / entry_price - 1) * 100, 2),
                }
            )
            cash = sell_value
            shares = 0.0
            position_open = False

        # Portfolio value = cash + shares * current_price
        portfolio_val = cash + shares * price
        portfolio_values.append(portfolio_val)

    # --- Calculate metrics ---
    final_value = portfolio_values[-1] if portfolio_values else initial_capital

    total_return_pct = round(
        calculate_total_return(initial_capital, final_value) * 100, 2
    )

    # Calculate years for CAGR
    total_days = (dates[-1] - dates[0]).days if len(dates) > 1 else 0
    years = total_days / 365.25 if total_days > 0 else 0

    cagr_pct = round(
        calculate_cagr(initial_capital, final_value, years) * 100, 2
    ) if years > 0 else 0.0

    portfolio_series = pd.Series(portfolio_values, index=dates)
    max_dd_pct = round(calculate_max_drawdown(portfolio_series) * 100, 2)

    daily_returns = calculate_daily_returns(portfolio_series)
    sharpe = round(calculate_sharpe_ratio(daily_returns), 2)

    # Win rate
    winning_trades = [t for t in completed_trades if t["profit"] > 0]
    total_completed = len(completed_trades)
    win_rate = round(
        (len(winning_trades) / total_completed * 100) if total_completed > 0 else 0.0,
        2,
    )

    # --- Build output data ---
    # Portfolio history for chart
    portfolio_history = [
        {
            "date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
            "value": round(v, 2),
        }
        for d, v in zip(dates, portfolio_values)
    ]

    # MA data for chart
    ma_data = []
    for date in dates:
        s_val = short_ma.loc[date] if date in short_ma.index else None
        l_val = long_ma.loc[date] if date in long_ma.index else None
        ma_data.append(
            {
                "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                "short_ma": round(float(s_val), 2) if pd.notna(s_val) else None,
                "long_ma": round(float(l_val), 2) if pd.notna(l_val) else None,
                "close": round(float(close.loc[date]), 2),
            }
        )

    # Crossovers list for chart
    crossovers_list = []
    for _, row in crossover_df.iterrows():
        crossovers_list.append(
            {
                "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                "type": row["type"],
                "price": row["price"],
                "short_ma": row["short_ma"],
                "long_ma": row["long_ma"],
            }
        )

    return {
        "total_return": total_return_pct,
        "cagr": cagr_pct,
        "max_drawdown": max_dd_pct,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "total_trades": total_completed,
        "trades": trades,
        "portfolio_history": portfolio_history,
        "ma_data": ma_data,
        "crossovers": crossovers_list,
        "final_value": round(final_value, 2),
        "initial_capital": round(initial_capital, 2),
    }
