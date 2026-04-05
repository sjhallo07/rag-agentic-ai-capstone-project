"""RSI Agentic AI - A RAG-powered agent for Relative Strength Index analysis."""

from .calculator import RSICalculator

__all__ = ["RSIAgent", "RSICalculator"]


def __getattr__(name: str):
    if name == "RSIAgent":
        from .agent import RSIAgent
        return RSIAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
