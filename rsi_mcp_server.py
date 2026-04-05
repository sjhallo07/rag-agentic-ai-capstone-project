"""STDIO MCP server exposing RSI analysis tools for LLM clients."""

from __future__ import annotations

import io
import os
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rsi_agent.service_api import (  # noqa: E402
    ask_rsi_agent,
    compute_rsi_for_ticker,
    fetch_recent_stock_prices,
    interpret_rsi_value,
    search_rsi_knowledge,
)

mcp = FastMCP("rsi_analysis_tools")


@mcp.tool()
def fetch_prices(ticker: str, period: str = "6mo") -> str:
    """Fetch recent closing prices for a stock ticker.

    Args:
        ticker: Stock ticker symbol, such as AAPL.
        period: Historical period like 1mo, 3mo, 6mo, or 1y.

    Returns:
        A summary of recent closing prices.
    """
    return fetch_recent_stock_prices(ticker, period)


@mcp.tool()
def compute_rsi(ticker: str, rsi_period: int = 14, data_period: str = "6mo") -> str:
    """Compute the RSI for a stock ticker.

    Args:
        ticker: Stock ticker symbol, such as TSLA.
        rsi_period: RSI look-back period.
        data_period: Historical data window to download.

    Returns:
        The latest RSI reading with a plain-English interpretation.
    """
    return compute_rsi_for_ticker(ticker, rsi_period, data_period)


@mcp.tool()
def interpret_rsi(rsi_value: float) -> str:
    """Interpret a raw RSI value.

    Args:
        rsi_value: RSI value between 0 and 100.

    Returns:
        A plain-English interpretation of the RSI level.
    """
    return interpret_rsi_value(rsi_value)


@mcp.tool()
def search_knowledge(query: str) -> str:
    """Search the built-in RSI knowledge base.

    Args:
        query: Natural-language question or keyword about RSI.

    Returns:
        The most relevant RSI knowledge snippets.
    """
    return search_rsi_knowledge(query)


@mcp.tool()
def ask_agent(question: str, model: str = "gpt-4o-mini") -> str:
    """Answer an RSI question using the RAG-enabled OpenAI agent.

    Args:
        question: Natural-language question about RSI or a stock signal.
        model: OpenAI chat model name to use.

    Returns:
        The agent answer, or a configuration message if OpenAI is not available.
    """
    return ask_rsi_agent(question, model)


if __name__ == "__main__":
    mcp.run(transport="stdio")
