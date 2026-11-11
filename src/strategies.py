"""Strategies. Each one returns a DataFrame of target weights per day.

Weights on day t only use prices up to and including day t. The backtester
lags them before they earn anything.
"""

import numpy as np
import pandas as pd

from src.indicators import inverse_vol_weights, momentum_score, sma_crossover_signal


def sma_trend_weights(prices, fast=20, slow=100, vol_window=30):
    """v1 strategy: long/flat SMA crossover per asset, sized by inverse vol.

    Kept as a baseline to compare the newer strategy against.
    """
    signal = sma_crossover_signal(prices, fast, slow)
    weights = inverse_vol_weights(prices.pct_change().fillna(0.0), vol_window)
    return signal * weights


def min_variance_weights(cov):
    """Long-only minimum variance weights for a covariance matrix.

    Without the long-only constraint the answer is w = inv(cov) @ 1, scaled
    to sum to 1. That can give negative weights (shorts), so if any weight is
    negative I drop the most negative asset and solve again with what's left.
    """
    cov = np.asarray(cov, dtype=float)
    keep = list(range(len(cov)))
    while True:
        sub = cov[np.ix_(keep, keep)]
        x = np.linalg.solve(sub, np.ones(len(keep)))
        x = x / x.sum()
        if (x >= -1e-12).all():
            break
        keep.pop(int(np.argmin(x)))

    w = np.zeros(len(cov))
    w[keep] = np.clip(x, 0, None)
    return w / w.sum()


def momentum_minvar_weights(
    prices,
    top_k=5,
    lookbacks=(63, 126, 189, 252),
    cov_window=60,
    hold_days=20,
    n_tranches=4,
    first_rebalance=0,
):
    """v2 strategy: hold the top_k assets by momentum, min-variance weighted.

    Based on "Adaptive Asset Allocation: A Primer" (Butler, Philbrick,
    Gordillo, Varadi, 2012):
    rank assets by recent return, keep the best half, and weight them to
    minimize portfolio variance using a short covariance window.

    Two changes from the paper:
    - momentum is the average of several lookbacks instead of one, so the
      result doesn't hinge on a single window length
    - instead of one portfolio rebalanced on one day, capital is split into
      n_tranches portfolios that rebalance every hold_days, staggered evenly.
      When I rebalanced once a month the Sharpe swung a lot depending on
      *which* day of the month I picked, which is just luck.

    first_rebalance shifts the whole schedule by that many days. It's only
    used in main.py to show how much the single-portfolio version depends on
    the rebalance day.

    This loops over rebalance dates (a few hundred), not over every day.
    """
    returns = prices.pct_change()
    score = momentum_score(prices, lookbacks)
    warmup = max(max(lookbacks), cov_window) + 1
    stagger = hold_days // n_tranches

    total = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for tranche in range(n_tranches):
        w = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        for i in range(first_rebalance + tranche * stagger, len(prices), hold_days):
            if i < warmup:
                continue
            top = list(score.iloc[i].sort_values(ascending=False).index[:top_k])
            # covariance from the last cov_window days, ending today
            window = returns.iloc[i - cov_window + 1 : i + 1][top]
            row = pd.Series(0.0, index=prices.columns)
            row[top] = min_variance_weights(window.cov().values)
            w.iloc[i] = row
        # hold each tranche's weights until its next rebalance
        total += w.ffill().fillna(0.0)

    return total / n_tranches
