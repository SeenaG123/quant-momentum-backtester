"""Indicators and signals. Everything works on whole DataFrames (no loops)."""

import numpy as np
import pandas as pd


def sma(prices, window):
    """Simple moving average. NaN until there are `window` observations."""
    return prices.rolling(window).mean()


def rsi(prices, window=14):
    """Wilder's RSI.

    Wilder smoothing is an EMA with alpha = 1/window.
    """
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def rolling_vol(returns, window=30):
    """Rolling std of daily returns (not annualized)."""
    return returns.rolling(window).std()


def momentum_score(prices, lookbacks=(63, 126, 189, 252)):
    """Average trailing return over several lookbacks (in trading days).

    63/126/189/252 is roughly 3/6/9/12 months. NaN until the longest
    lookback is available.
    """
    total = sum(prices / prices.shift(n) - 1 for n in lookbacks)
    return total / len(lookbacks)


def sma_crossover_signal(prices, fast=20, slow=50):
    """+1 when the fast SMA is above the slow SMA, otherwise 0 (long/flat).

    The signal on day t only uses prices up to and including day t.
    The backtester is responsible for lagging it before trading.
    """
    if fast >= slow:
        raise ValueError("fast window has to be shorter than slow window")
    fast_ma = sma(prices, fast)
    slow_ma = sma(prices, slow)
    signal = (fast_ma > slow_ma).astype(float)
    # no opinion until the slow average exists
    signal[slow_ma.isna()] = 0.0
    return signal


def rsi_signal(prices, window=14, lower=30, upper=70):
    """Mean reversion: go long when RSI < lower, hold until RSI > upper."""
    r = rsi(prices, window)
    raw = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    raw[r < lower] = 1.0
    raw[r > upper] = 0.0
    # carry the last decision forward while RSI is between the two levels
    return raw.ffill().fillna(0.0)


def inverse_vol_weights(returns, window=30):
    """w_i = (1/vol_i) / sum_j (1/vol_j). Rows sum to 1."""
    vol = rolling_vol(returns, window)
    inv = 1.0 / vol.replace(0, np.nan)
    weights = inv.div(inv.sum(axis=1), axis=0)
    return weights.fillna(0.0)
