"""RSI knowledge base used for RAG-augmented responses."""

from __future__ import annotations

# Plain-text documents describing RSI concepts, used to build a vector store
# for Retrieval-Augmented Generation.
RSI_KNOWLEDGE_DOCS: list[str] = [
    # --- Overview ---
    (
        "The Relative Strength Index (RSI) is a momentum oscillator developed by J. Welles Wilder Jr. "
        "and introduced in his 1978 book 'New Concepts in Technical Trading Systems'. "
        "RSI measures the speed and magnitude of recent price changes to evaluate whether an asset "
        "is overbought or oversold. It oscillates between 0 and 100."
    ),
    # --- Calculation ---
    (
        "RSI Calculation: RSI is computed using average gains and average losses over a look-back period "
        "(default 14 periods). First, price changes (deltas) are separated into gains (positive) and "
        "losses (negative). The average gain and average loss are then smoothed using an exponential "
        "moving average. The Relative Strength (RS) is the ratio of average gain to average loss. "
        "RSI = 100 - (100 / (1 + RS))."
    ),
    # --- Overbought/Oversold ---
    (
        "RSI Overbought Signal: When RSI rises above 70, the asset is traditionally considered overbought. "
        "An overbought reading suggests that the asset's price may have risen too quickly and could be "
        "due for a correction or consolidation. However, in strong uptrends RSI can remain above 70 for "
        "extended periods, so this signal alone should not be used to trigger short positions."
    ),
    (
        "RSI Oversold Signal: When RSI falls below 30, the asset is traditionally considered oversold. "
        "An oversold reading suggests that the asset's price may have fallen too quickly and could be "
        "due for a rebound. In persistent downtrends, RSI can stay below 30 for long periods. "
        "This signal is best used with other confirmation indicators."
    ),
    # --- Divergence ---
    (
        "RSI Divergence: Bullish divergence occurs when price makes a lower low but RSI makes a higher low, "
        "signalling weakening downward momentum and a potential reversal to the upside. "
        "Bearish divergence occurs when price makes a higher high but RSI makes a lower high, "
        "signalling weakening upward momentum and a potential reversal to the downside."
    ),
    # --- Failure Swings ---
    (
        "RSI Failure Swings: A bullish failure swing happens when RSI falls below 30, bounces above 30, "
        "pulls back without revisiting the oversold territory, and then breaks its prior swing high. "
        "A bearish failure swing occurs when RSI rises above 70, falls back below 70, "
        "rallies without reaching overbought territory again, and then breaks its prior swing low."
    ),
    # --- Period Selection ---
    (
        "RSI Period Selection: The default period is 14, providing a balance between sensitivity and "
        "smoothness. Shorter periods (e.g., 7 or 9) make RSI more sensitive and produce more signals "
        "but also more noise. Longer periods (e.g., 21 or 25) produce smoother RSI readings with fewer "
        "false signals but react more slowly to price changes."
    ),
    # --- Use in Trading ---
    (
        "RSI in Trading Strategies: RSI is most reliable when used in combination with trend analysis, "
        "volume indicators, and price action. Common strategies include buying oversold pullbacks in an "
        "uptrend (RSI < 40 in a bull market) and selling overbought rallies in a downtrend "
        "(RSI > 60 in a bear market). Always use stop-loss orders when trading RSI signals."
    ),
    # --- Limitations ---
    (
        "RSI Limitations: Like all momentum oscillators, RSI is a lagging indicator and may generate "
        "false signals in trending markets. RSI thresholds of 30 and 70 are conventional defaults; "
        "experienced traders often adjust these levels depending on the asset and timeframe. "
        "RSI does not predict price targets and should never be used as the sole basis for a trade."
    ),
]
