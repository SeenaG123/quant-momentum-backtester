"""Performance metrics. All functions take a Series of daily returns."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns):
    """Geometric (compounded) annual return."""
    total = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return total ** (1 / years) - 1


def annualized_vol(returns):
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, rf=0.0):
    """Mean daily excess return / daily std, scaled by sqrt(252).

    rf is an annual rate. Default 0, which flatters the strategy a bit
    in years when T-bills actually paid something.
    """
    excess = returns - rf / TRADING_DAYS
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns, rf=0.0):
    """Like Sharpe but only penalizes downside moves."""
    excess = returns - rf / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    if downside == 0:
        return np.nan
    return excess.mean() / downside * np.sqrt(TRADING_DAYS)


def drawdown_series(returns):
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    return equity / peak - 1


def max_drawdown(returns):
    """Worst peak-to-trough loss, as a negative number."""
    return drawdown_series(returns).min()


def summary(returns, rf=0.0):
    return pd.Series(
        {
            "ann_return": annualized_return(returns),
            "ann_vol": annualized_vol(returns),
            "sharpe": sharpe_ratio(returns, rf),
            "sortino": sortino_ratio(returns, rf),
            "max_drawdown": max_drawdown(returns),
        }
    )
