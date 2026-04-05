"""Entry-point script demonstrating the RSI agent."""

import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

load_dotenv()


def _get_configured_openai_api_key() -> str | None:
    """Return a usable OpenAI API key or ``None`` if only a placeholder is set."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        return None
    return api_key


def main() -> None:
    api_key = _get_configured_openai_api_key()
    if sys.version_info >= (3, 14):
        print(
            "Python 3.14+ detected. The LangChain-based RSI agent in this project is not currently compatible with this Python version.\n"
            "Running in demo mode (RSI calculator only, no LLM).\n"
        )
        _demo_calculator()
        return

    if not api_key:
        print(
            "OPENAI_API_KEY environment variable not set.\n"
            "Running in demo mode (RSI calculator only, no LLM).\n"
        )
        _demo_calculator()
        return

    from langchain_openai import ChatOpenAI
    from rsi_agent import RSIAgent

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    agent = RSIAgent(llm=llm, use_rag=True, verbose=True)

    queries = [
        "What is the current RSI for AAPL and is it overbought or oversold?",
        "Explain what RSI divergence means and how to trade it.",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        answer = agent.run(q)
        print(f"Answer: {answer}\n")


def _demo_calculator() -> None:
    """Demonstrate the RSI calculator with synthetic data."""
    from rsi_agent import RSICalculator

    # 30 days of steadily rising prices
    prices_up = [100 + i for i in range(30)]
    calc = RSICalculator(period=14)
    rsi_up = calc.compute(prices_up)
    latest = float(rsi_up.iloc[-1])
    print(f"Trending-up series  → RSI: {latest:.2f} | {RSICalculator.interpret(latest)}")

    # 30 days of steadily falling prices
    prices_down = [130 - i for i in range(30)]
    rsi_down = calc.compute(prices_down)
    latest = float(rsi_down.iloc[-1])
    print(f"Trending-down series → RSI: {latest:.2f} | {RSICalculator.interpret(latest)}")


if __name__ == "__main__":
    main()
