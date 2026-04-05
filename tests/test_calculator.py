"""Tests for RSICalculator."""

import math

import numpy as np
import pandas as pd
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsi_agent.calculator import RSICalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trending_up(n: int = 30, start: float = 100.0, step: float = 1.0) -> list[float]:
    """Return a strictly increasing price series."""
    return [start + i * step for i in range(n)]


def _trending_down(n: int = 30, start: float = 130.0, step: float = 1.0) -> list[float]:
    """Return a strictly decreasing price series."""
    return [start - i * step for i in range(n)]


def _flat(n: int = 30, price: float = 100.0) -> list[float]:
    """Return a flat price series."""
    return [price] * n


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


def test_default_period():
    calc = RSICalculator()
    assert calc.period == RSICalculator.DEFAULT_PERIOD


def test_custom_period():
    calc = RSICalculator(period=9)
    assert calc.period == 9


def test_invalid_period_raises():
    with pytest.raises(ValueError):
        RSICalculator(period=0)
    with pytest.raises(ValueError):
        RSICalculator(period=-5)


# ---------------------------------------------------------------------------
# compute() tests
# ---------------------------------------------------------------------------


def test_compute_returns_series():
    calc = RSICalculator()
    prices = _trending_up(30)
    result = calc.compute(prices)
    assert isinstance(result, pd.Series)
    assert len(result) == 30


def test_compute_accepts_pandas_series():
    calc = RSICalculator()
    prices = pd.Series(_trending_up(30))
    result = calc.compute(prices)
    assert isinstance(result, pd.Series)


def test_compute_too_few_prices_raises():
    calc = RSICalculator()
    with pytest.raises(ValueError):
        calc.compute([100.0])


def test_compute_trending_up_high_rsi():
    """A strictly increasing series should produce RSI near 100 after warm-up."""
    calc = RSICalculator(period=14)
    prices = _trending_up(50)
    rsi = calc.compute(prices)
    # After warm-up the RSI of an ever-rising series should be very high
    assert float(rsi.iloc[-1]) > 90


def test_compute_trending_down_low_rsi():
    """A strictly decreasing series should produce RSI near 0 after warm-up."""
    calc = RSICalculator(period=14)
    prices = _trending_down(50)
    rsi = calc.compute(prices)
    assert float(rsi.iloc[-1]) < 10


def test_compute_flat_series():
    """A flat series has no gains or losses; RSI should be NaN or 50."""
    calc = RSICalculator(period=14)
    prices = _flat(30)
    rsi = calc.compute(prices)
    # First diff is NaN; a perfectly flat series after the first element yields 0/0 -> NaN
    last = float(rsi.iloc[-1])
    assert math.isnan(last) or last == pytest.approx(50.0, abs=5)


def test_compute_rsi_range():
    """RSI values should always be in [0, 100] (ignoring NaN warm-up)."""
    calc = RSICalculator(period=14)
    # Mix of ups and downs
    np.random.seed(42)
    prices = 100 + np.random.randn(100).cumsum()
    rsi = calc.compute(prices.tolist())
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_compute_known_values():
    """Spot-check RSI against a manually computed reference.

    Using a simple 5-period RSI with hand-crafted gains/losses.
    Sequence: 10 rises of +1 followed by 5 drops of -1, period=5.
    After all 10 rises the RSI should be very high (> 90).
    """
    prices = [100 + i for i in range(10)] + [110 - i for i in range(5)]
    calc = RSICalculator(period=5)
    rsi = calc.compute(prices)
    # After the 10 rises the RSI should be overbought
    assert float(rsi.iloc[9]) > 90
    # After 5 drops, RSI should have fallen
    assert float(rsi.iloc[-1]) < float(rsi.iloc[9])


# ---------------------------------------------------------------------------
# interpret() tests
# ---------------------------------------------------------------------------


def test_interpret_overbought():
    result = RSICalculator.interpret(75.0)
    assert "Overbought" in result
    assert "75.00" in result


def test_interpret_oversold():
    result = RSICalculator.interpret(25.0)
    assert "Oversold" in result
    assert "25.00" in result


def test_interpret_neutral():
    result = RSICalculator.interpret(50.0)
    assert "Neutral" in result
    assert "50.00" in result


def test_interpret_boundary_overbought():
    result = RSICalculator.interpret(70.0)
    assert "Overbought" in result


def test_interpret_boundary_oversold():
    result = RSICalculator.interpret(30.0)
    assert "Oversold" in result


def test_interpret_nan():
    result = RSICalculator.interpret(float("nan"))
    assert "Insufficient data" in result


def test_interpret_exactly_100():
    result = RSICalculator.interpret(100.0)
    assert "Overbought" in result


def test_interpret_exactly_0():
    result = RSICalculator.interpret(0.0)
    assert "Oversold" in result
