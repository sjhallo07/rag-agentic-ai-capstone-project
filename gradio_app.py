"""Gradio UI that can also be exposed as an HTTP MCP server."""

from __future__ import annotations

import os
import sys

import gradio as gr
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

load_dotenv()

from rsi_agent.service_api import (  # noqa: E402
    ask_rsi_agent,
    compute_rsi_for_ticker,
    fetch_recent_stock_prices,
    interpret_rsi_value,
    search_rsi_knowledge,
)


def build_demo() -> gr.TabbedInterface:
    """Build the Gradio RSI tool suite."""
    fetch_prices_ui = gr.Interface(
        fn=fetch_recent_stock_prices,
        inputs=[
            gr.Textbox(label="Ticker", value="AAPL"),
            gr.Textbox(label="History Period", value="6mo"),
        ],
        outputs=gr.Textbox(label="Recent Prices", lines=8),
        title="Fetch Prices",
        description="Get recent closing prices for a ticker.",
        api_name="fetch_prices",
    )

    compute_rsi_ui = gr.Interface(
        fn=compute_rsi_for_ticker,
        inputs=[
            gr.Textbox(label="Ticker", value="TSLA"),
            gr.Number(label="RSI Period", value=14, precision=0),
            gr.Textbox(label="Data Period", value="6mo"),
        ],
        outputs=gr.Textbox(label="RSI Analysis", lines=6),
        title="Compute RSI",
        description="Calculate RSI for a ticker and explain the result.",
        api_name="compute_rsi",
    )

    interpret_rsi_ui = gr.Interface(
        fn=interpret_rsi_value,
        inputs=gr.Number(label="RSI Value", value=72.4),
        outputs=gr.Textbox(label="Interpretation", lines=4),
        title="Interpret RSI",
        description="Interpret a raw RSI value without downloading market data.",
        api_name="interpret_rsi",
    )

    knowledge_ui = gr.Interface(
        fn=search_rsi_knowledge,
        inputs=gr.Textbox(label="RSI Question", value="What is RSI divergence?"),
        outputs=gr.Textbox(label="Knowledge Base Matches", lines=12),
        title="Search Knowledge",
        description="Search the built-in RSI knowledge base for concepts and explanations.",
        api_name="search_knowledge",
    )

    ask_agent_ui = gr.Interface(
        fn=ask_rsi_agent,
        inputs=[
            gr.Textbox(
                label="Question",
                value="What is the current RSI for AAPL and what does it mean?",
            ),
            gr.Textbox(label="OpenAI Model", value="gpt-4o-mini"),
        ],
        outputs=gr.Textbox(label="Agent Answer", lines=12),
        title="Ask RSI Agent",
        description="Use the LangChain RAG RSI agent. Requires OPENAI_API_KEY in the environment.",
        api_name="ask_agent",
    )

    return gr.TabbedInterface(
        [fetch_prices_ui, compute_rsi_ui, interpret_rsi_ui, knowledge_ui, ask_agent_ui],
        ["Fetch Prices", "Compute RSI", "Interpret RSI", "Search Knowledge", "Ask RSI Agent"],
        title="RSI Tools + MCP Server",
    )


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(mcp_server=True)
