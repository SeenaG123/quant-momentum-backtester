"""Vectorized backtest: target weights in, daily net returns out."""

import pandas as pd


def run_backtest(prices, weights, cost_bps=5.0, lag=1):
    """Backtest a matrix of target weights.

    prices   : DataFrame of adjusted closes
    weights  : DataFrame, same shape. Row t is the portfolio I want based on
               information up to the close of day t. Rows can sum to less
               than 1 (the rest is cash earning nothing).
    cost_bps : cost per unit of turnover, in basis points
    lag      : days between computing weights and holding them.
               lag=1 is the real setting (weights from today's close earn
               tomorrow's return). lag=0 is ONLY there to show how much
               lookahead bias inflates results.

    Returns a DataFrame with gross, cost, net, turnover, equity columns.
    """
    asset_ret = prices.pct_change().fillna(0.0)

    # what we actually hold during day t was decided on day t - lag
    position = weights.shift(lag).fillna(0.0)

    gross = (position * asset_ret).sum(axis=1)

    # pay costs on however much of the book changed
    turnover = position.diff().abs().sum(axis=1)
    turnover.iloc[0] = position.iloc[0].abs().sum()
    cost = turnover * cost_bps / 10_000

    net = gross - cost

    out = pd.DataFrame(
        {"gross": gross, "cost": cost, "net": net, "turnover": turnover}
    )
    out["equity"] = (1 + out["net"]).cumprod()
    return out


def buy_and_hold(prices, ticker="SPY"):
    """Benchmark: daily returns from just holding one asset."""
    return prices[ticker].pct_change().fillna(0.0)


def split_in_out(df, in_sample_frac=0.7):
    """Chronological split. No shuffling, this is time series data."""
    cut = int(len(df) * in_sample_frac)
    return df.iloc[:cut], df.iloc[cut:]
