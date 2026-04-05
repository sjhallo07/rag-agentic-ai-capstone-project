"""LangChain tools used by the RSI agent."""

from __future__ import annotations

from typing import Optional, Type

import yfinance as yf
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .calculator import RSICalculator


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class FetchPricesInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL")
    period: str = Field(
        default="6mo",
        description="Time period for historical data (e.g. 1mo, 3mo, 6mo, 1y, 2y)",
    )


class ComputeRSIInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL")
    rsi_period: int = Field(
        default=14,
        description="Number of periods for RSI calculation (default 14)",
    )
    data_period: str = Field(
        default="6mo",
        description="Historical data window to download (e.g. 3mo, 6mo, 1y)",
    )


class InterpretRSIInput(BaseModel):
    rsi_value: float = Field(description="RSI value to interpret (0-100)")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class FetchStockPricesTool(BaseTool):
    """Download historical closing prices for a ticker using yfinance."""

    name: str = "fetch_stock_prices"
    description: str = (
        "Fetch historical daily closing prices for a given stock ticker. "
        "Returns the most recent 5 closing prices and the date range."
    )
    args_schema: Type[BaseModel] = FetchPricesInput

    def _run(self, ticker: str, period: str = "6mo") -> str:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df.empty:
                return f"No data found for ticker '{ticker}'."
            close = df["Close"].squeeze()
            recent = close.tail(5)
            lines = [f"{date.date()}: {price:.2f}" for date, price in recent.items()]
            return (
                f"Closing prices for {ticker} (last 5 of {len(close)} trading days):\n"
                + "\n".join(lines)
            )
        except Exception as exc:
            return f"Error fetching data for {ticker}: {exc}"


class ComputeRSITool(BaseTool):
    """Calculate the RSI for a stock ticker."""

    name: str = "compute_rsi"
    description: str = (
        "Calculate the Relative Strength Index (RSI) for a stock ticker. "
        "Returns the current RSI value and its plain-language interpretation."
    )
    args_schema: Type[BaseModel] = ComputeRSIInput

    def _run(
        self, ticker: str, rsi_period: int = 14, data_period: str = "6mo"
    ) -> str:
        try:
            df = yf.download(ticker, period=data_period, auto_adjust=True, progress=False)
            if df.empty:
                return f"No data found for ticker '{ticker}'."
            close = df["Close"].squeeze()
            calculator = RSICalculator(period=rsi_period)
            rsi_series = calculator.compute(close)
            latest_rsi = float(rsi_series.iloc[-1])
            interpretation = RSICalculator.interpret(latest_rsi)
            return f"RSI({rsi_period}) for {ticker}: {interpretation}"
        except Exception as exc:
            return f"Error computing RSI for {ticker}: {exc}"


class InterpretRSITool(BaseTool):
    """Interpret a given RSI value without fetching any data."""

    name: str = "interpret_rsi"
    description: str = (
        "Interpret a Relative Strength Index (RSI) value. "
        "Explains whether the asset is overbought, oversold, or in neutral territory."
    )
    args_schema: Type[BaseModel] = InterpretRSIInput

    def _run(self, rsi_value: float) -> str:
        return RSICalculator.interpret(rsi_value)


def get_rsi_tools() -> list[BaseTool]:
    """Return all RSI agent tools."""
    return [
        FetchStockPricesTool(),
        ComputeRSITool(),
        InterpretRSITool(),
    ]
