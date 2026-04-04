"""RSI (Relative Strength Index) calculator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union


class RSICalculator:
    """Calculates the Relative Strength Index (RSI) for a price series."""

    DEFAULT_PERIOD = 14

    def __init__(self, period: int = DEFAULT_PERIOD) -> None:
        if period < 1:
            raise ValueError(f"RSI period must be >= 1, got {period}")
        self.period = period

    def compute(self, prices: Union[list[float], pd.Series]) -> pd.Series:
        """Compute RSI for the given closing price series.

        Args:
            prices: A list or pandas Series of closing prices (oldest first).

        Returns:
            A pandas Series of RSI values (NaN for the initial warm-up window).

        Raises:
            ValueError: If prices has fewer than 2 elements.
        """
        series = pd.Series(prices, dtype=float)
        if len(series) < 2:
            raise ValueError("At least 2 price points are required to compute RSI.")

        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()

        # Use abs() to handle -0.0 from floating point before replacing zeros.
        avg_gain_abs = avg_gain.abs()
        avg_loss_abs = avg_loss.abs()

        # When avg_loss == 0 and avg_gain > 0, RSI = 100 (only gains).
        # When avg_loss == 0 and avg_gain == 0, RSI = NaN (no movement).
        rs = avg_gain_abs / avg_loss_abs.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # Apply RSI = 100 only where avg_loss is 0 but avg_gain is non-zero
        only_gains = (avg_loss_abs == 0) & (avg_gain_abs > 0)
        rsi = rsi.where(~only_gains, other=100.0)
        return rsi

    @staticmethod
    def interpret(rsi_value: float) -> str:
        """Return a plain-language interpretation of an RSI value.

        Args:
            rsi_value: A single RSI reading (0-100).

        Returns:
            A string describing whether the asset is overbought, oversold, or neutral.
        """
        if np.isnan(rsi_value):
            return "Insufficient data – RSI not yet available."
        if rsi_value >= 70:
            return (
                f"RSI {rsi_value:.2f} – Overbought zone. "
                "The asset may be overvalued; consider watching for a potential pullback."
            )
        if rsi_value <= 30:
            return (
                f"RSI {rsi_value:.2f} – Oversold zone. "
                "The asset may be undervalued; consider watching for a potential rebound."
            )
        return (
            f"RSI {rsi_value:.2f} – Neutral zone. "
            "No extreme momentum signal detected."
        )
