"""
MA Strategy API Routes

Provides the endpoint for MA crossover backtest:
- POST /api/ma-backtest - Execute MA crossover strategy backtest
"""
from fastapi import APIRouter, HTTPException
from datetime import date

from api.models import (
    MABacktestRequest,
    MABacktestFrontendResponse,
    MAMetrics,
    MATradeRecord,
    MADataPoint,
    PortfolioHistoryEntry,
)
from services.data_service import get_stock_data, get_stock_info
from backtest.ma_backtest import backtest_ma_strategy


router = APIRouter(prefix="/api", tags=["MA Strategy"])


@router.post("/ma-backtest", response_model=MABacktestFrontendResponse)
async def run_ma_backtest(request: MABacktestRequest):
    """
    Execute MA crossover strategy backtest.

    Buy on golden cross (short MA crosses above long MA),
    sell on death cross (short MA crosses below long MA).

    Returns { metrics, price_data, portfolio_history, trades } to match
    the frontend's expected response shape.
    """
    # Validate short_period < long_period
    if request.short_period >= request.long_period:
        raise HTTPException(
            status_code=400,
            detail=f"短期 MA 週期 ({request.short_period}) 必須小於長期 MA 週期 ({request.long_period})",
        )

    # Validate date range
    start = date.fromisoformat(request.start_date)
    end = date.fromisoformat(request.end_date)

    if start >= end:
        raise HTTPException(status_code=400, detail="起始日期必須早於結束日期")

    if (end - start).days > 365 * 20:
        raise HTTPException(status_code=400, detail="日期範圍不能超過 20 年")

    # Fetch stock data
    data = get_stock_data(request.symbol, request.start_date, request.end_date)

    if data is None or data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"找不到股票 {request.symbol} 的資料",
        )

    # Check minimum data points
    min_required = request.long_period + 1
    if len(data) < min_required:
        raise HTTPException(
            status_code=400,
            detail=f"資料點數不足（{len(data)} 筆），長期 MA 週期需要至少 {min_required} 筆資料",
        )

    # Run backtest (ma_type is already normalized to lowercase by the validator)
    try:
        result = backtest_ma_strategy(
            data=data,
            short_period=request.short_period,
            long_period=request.long_period,
            initial_capital=request.initial_capital,
            ma_type=request.ma_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"回測計算錯誤: {str(e)}"
        )

    # Build trades with profit for sell trades
    trades_out = _build_trades_with_profit(result["trades"])

    # Build response matching frontend expectations:
    #   { metrics, price_data, portfolio_history, trades }
    metrics = MAMetrics(
        total_return=result["total_return"],
        cagr=result["cagr"],
        max_drawdown=result["max_drawdown"],
        sharpe_ratio=result["sharpe_ratio"],
        win_rate=result["win_rate"],
        trade_count=result["total_trades"],
    )

    price_data = [
        MADataPoint(
            date=d["date"],
            close=d["close"],
            short_ma=d["short_ma"],
            long_ma=d["long_ma"],
        )
        for d in result["ma_data"]
    ]

    portfolio_history = [
        PortfolioHistoryEntry(date=h["date"], value=h["value"])
        for h in result["portfolio_history"]
    ]

    return MABacktestFrontendResponse(
        metrics=metrics,
        price_data=price_data,
        portfolio_history=portfolio_history,
        trades=trades_out,
    )


def _build_trades_with_profit(raw_trades: list) -> list[MATradeRecord]:
    """
    Convert raw trade dicts to MATradeRecord list.

    For sell trades, compute profit = sell_value - buy_value of the
    preceding buy (simple paired matching).
    """
    records = []
    last_buy_value = None

    for t in raw_trades:
        profit = None
        if t["type"] == "buy":
            last_buy_value = t["value"]
        elif t["type"] == "sell":
            if last_buy_value is not None:
                profit = round(t["value"] - last_buy_value, 2)
            last_buy_value = None

        records.append(
            MATradeRecord(
                date=t["date"],
                type=t["type"],
                price=t["price"],
                shares=t["shares"],
                value=t["value"],
                profit=profit,
            )
        )

    return records
