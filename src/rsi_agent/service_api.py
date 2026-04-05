"""Reusable service functions for CLI, Gradio, and MCP integrations."""

from __future__ import annotations

import os
import sys
from typing import Iterable

import yfinance as yf

from .calculator import RSICalculator
from .knowledge_base import RSI_KNOWLEDGE_DOCS

_PLACEHOLDER_VALUES = {
    "",
    "your_openai_api_key_here",
    "your_anthropic_api_key_here",
}


def get_configured_env_value(name: str) -> str | None:
    """Return an environment variable only if it is set to a non-placeholder value.

    Args:
        name: Name of the environment variable to read.

    Returns:
        The stripped environment variable value, or ``None`` if it is unset or a placeholder.
    """
    value = os.getenv(name, "").strip()
    if value in _PLACEHOLDER_VALUES:
        return None
    return value or None


def fetch_recent_stock_prices(ticker: str, period: str = "6mo") -> str:
    """Fetch recent closing prices for a stock ticker.

    Args:
        ticker: Stock ticker symbol, for example ``AAPL``.
        period: Historical period accepted by yfinance, such as ``1mo`` or ``6mo``.

    Returns:
        A human-readable summary of the most recent closing prices.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return "Please provide a stock ticker symbol."

    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as exc:  # pragma: no cover - network/provider issues
        return f"Error fetching data for {ticker}: {exc}"

    if df.empty:
        return f"No market data found for ticker '{ticker}'."

    close = df["Close"].squeeze()
    recent = close.tail(5)
    lines = [f"{date.date()}: {price:.2f}" for date, price in recent.items()]
    return (
        f"Recent closing prices for {ticker} (last 5 of {len(close)} trading days):\n"
        + "\n".join(lines)
    )


def compute_rsi_for_ticker(
    ticker: str,
    rsi_period: int = 14,
    data_period: str = "6mo",
) -> str:
    """Compute the RSI for a stock ticker.

    Args:
        ticker: Stock ticker symbol, for example ``TSLA``.
        rsi_period: RSI look-back period. Defaults to ``14``.
        data_period: Historical data period to download for the calculation.

    Returns:
        A human-readable RSI summary and interpretation.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return "Please provide a stock ticker symbol."

    try:
        df = yf.download(ticker, period=data_period, auto_adjust=True, progress=False)
    except Exception as exc:  # pragma: no cover - network/provider issues
        return f"Error computing RSI for {ticker}: {exc}"

    if df.empty:
        return f"No market data found for ticker '{ticker}'."

    close = df["Close"].squeeze()
    calculator = RSICalculator(period=rsi_period)
    rsi_series = calculator.compute(close)
    latest_rsi = float(rsi_series.iloc[-1])
    interpretation = RSICalculator.interpret(latest_rsi)
    return f"RSI({rsi_period}) for {ticker}: {interpretation}"


def interpret_rsi_value(rsi_value: float) -> str:
    """Interpret a raw RSI value.

    Args:
        rsi_value: RSI value between ``0`` and ``100``.

    Returns:
        A plain-English interpretation of the RSI level.
    """
    return RSICalculator.interpret(rsi_value)



def _score_doc(doc: str, terms: Iterable[str]) -> int:
    lower_doc = doc.lower()
    return sum(lower_doc.count(term) for term in terms)



def search_rsi_knowledge(query: str) -> str:
    """Search the built-in RSI knowledge base.

    Args:
        query: Natural-language query about RSI concepts or trading signals.

    Returns:
        A short list of the most relevant knowledge snippets.
    """
    cleaned_query = query.strip().lower()
    if not cleaned_query:
        return "Please enter a question or keyword to search the RSI knowledge base."

    terms = [term for term in cleaned_query.split() if len(term) > 2]
    if not terms:
        terms = [cleaned_query]

    scored_docs = sorted(
        RSI_KNOWLEDGE_DOCS,
        key=lambda doc: _score_doc(doc, terms),
        reverse=True,
    )
    top_docs = [doc for doc in scored_docs if _score_doc(doc, terms) > 0][:3]

    if not top_docs:
        top_docs = RSI_KNOWLEDGE_DOCS[:2]
        return "No exact keyword match found. Here are two foundational RSI references:\n\n- " + "\n\n- ".join(top_docs)

    return "Top RSI knowledge matches:\n\n- " + "\n\n- ".join(top_docs)



def ask_rsi_agent(question: str, model: str = "gpt-4o-mini") -> str:
    """Answer an RSI question using the LangChain RAG agent.

    Args:
        question: Natural-language RSI question to answer.
        model: OpenAI chat model name to use.

    Returns:
        The agent answer, or a configuration message if OpenAI is not configured.
    """
    if not question.strip():
        return "Please provide a question for the RSI agent."

    if sys.version_info >= (3, 14):
        return (
            "The LangChain-based RSI agent is currently not compatible with Python 3.14+ in this project. "
            "Use Python 3.10-3.13 for the LLM-backed agent, or continue using the non-LLM RSI tools."
        )

    api_key = get_configured_env_value("OPENAI_API_KEY")
    if not api_key:
        return (
            "OPENAI_API_KEY is not configured, so the LLM-backed RSI agent is unavailable. "
            "You can still use the other RSI tools and the knowledge-base search."
        )

    from langchain_openai import ChatOpenAI
    from .agent import RSIAgent

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    agent = RSIAgent(llm=llm, use_rag=True, verbose=False)
    return agent.run(question)
