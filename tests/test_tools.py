"""Tests for RSI agent tools (using mocked yfinance data)."""

from __future__ import annotations

import math
import sys
import os
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsi_agent.tools import (
    ComputeRSITool,
    FetchStockPricesTool,
    InterpretRSITool,
    get_rsi_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_close_df(prices: list[float]) -> pd.DataFrame:
    """Return a minimal DataFrame that mimics yfinance output."""
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    df = pd.DataFrame({"Close": prices}, index=dates)
    return df


# ---------------------------------------------------------------------------
# get_rsi_tools
# ---------------------------------------------------------------------------


def test_get_rsi_tools_returns_three_tools():
    tools = get_rsi_tools()
    assert len(tools) == 3


def test_get_rsi_tools_names():
    names = {t.name for t in get_rsi_tools()}
    assert names == {"fetch_stock_prices", "compute_rsi", "interpret_rsi"}


# ---------------------------------------------------------------------------
# FetchStockPricesTool
# ---------------------------------------------------------------------------


class TestFetchStockPricesTool:
    def test_returns_recent_prices(self):
        prices = [100.0 + i for i in range(30)]
        mock_df = _make_close_df(prices)

        with patch("rsi_agent.tools.yf.download", return_value=mock_df):
            tool = FetchStockPricesTool()
            result = tool._run(ticker="AAPL", period="6mo")

        assert "AAPL" in result
        assert "30" in result  # total trading days mentioned

    def test_empty_dataframe_returns_error_message(self):
        with patch("rsi_agent.tools.yf.download", return_value=pd.DataFrame()):
            tool = FetchStockPricesTool()
            result = tool._run(ticker="FAKE", period="6mo")

        assert "No data found" in result

    def test_exception_returns_error_message(self):
        with patch("rsi_agent.tools.yf.download", side_effect=RuntimeError("network error")):
            tool = FetchStockPricesTool()
            result = tool._run(ticker="AAPL", period="6mo")

        assert "Error" in result


# ---------------------------------------------------------------------------
# ComputeRSITool
# ---------------------------------------------------------------------------


class TestComputeRSITool:
    def test_computes_overbought(self):
        # Steadily rising prices -> high RSI
        prices = [100.0 + i for i in range(60)]
        mock_df = _make_close_df(prices)

        with patch("rsi_agent.tools.yf.download", return_value=mock_df):
            tool = ComputeRSITool()
            result = tool._run(ticker="AAPL", rsi_period=14, data_period="6mo")

        assert "Overbought" in result
        assert "AAPL" in result

    def test_computes_oversold(self):
        # Steadily falling prices -> low RSI
        prices = [200.0 - i for i in range(60)]
        mock_df = _make_close_df(prices)

        with patch("rsi_agent.tools.yf.download", return_value=mock_df):
            tool = ComputeRSITool()
            result = tool._run(ticker="XYZ", rsi_period=14, data_period="6mo")

        assert "Oversold" in result

    def test_empty_dataframe_returns_error_message(self):
        with patch("rsi_agent.tools.yf.download", return_value=pd.DataFrame()):
            tool = ComputeRSITool()
            result = tool._run(ticker="FAKE")

        assert "No data found" in result

    def test_exception_returns_error_message(self):
        with patch("rsi_agent.tools.yf.download", side_effect=RuntimeError("timeout")):
            tool = ComputeRSITool()
            result = tool._run(ticker="AAPL")

        assert "Error" in result


# ---------------------------------------------------------------------------
# InterpretRSITool
# ---------------------------------------------------------------------------


class TestInterpretRSITool:
    def test_interpret_overbought(self):
        tool = InterpretRSITool()
        result = tool._run(rsi_value=80.0)
        assert "Overbought" in result

    def test_interpret_oversold(self):
        tool = InterpretRSITool()
        result = tool._run(rsi_value=20.0)
        assert "Oversold" in result

    def test_interpret_neutral(self):
        tool = InterpretRSITool()
        result = tool._run(rsi_value=55.0)
        assert "Neutral" in result

    def test_interpret_nan(self):
        tool = InterpretRSITool()
        result = tool._run(rsi_value=float("nan"))
        assert "Insufficient data" in result
