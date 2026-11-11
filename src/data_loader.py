"""Download and clean daily price data from Yahoo Finance."""

import os

import numpy as np
import pandas as pd

# ETFs instead of single stocks. Mostly so I don't have to deal with
# survivorship bias from picking today's S&P 500 members.
DEFAULT_TICKERS = [
    "SPY",  # US large cap
    "QQQ",  # Nasdaq 100
    "IWM",  # US small cap
    "EFA",  # developed ex-US
    "EEM",  # emerging markets
    "TLT",  # long treasuries
    "IEF",  # 7-10y treasuries
    "GLD",  # gold
    "DBC",  # commodities
    "VNQ",  # real estate
]


def download_prices(tickers, start, end, cache_path=None):
    """Adjusted close prices, one column per ticker.

    If cache_path exists we just read it, so I'm not hitting Yahoo
    every time I rerun the backtest.
    """
    if cache_path is not None and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # only trust the cache if every ticker actually has data in it
        if set(tickers) <= set(cached.columns) and not cached[tickers].isna().all().any():
            return cached[tickers]

    import yfinance as yf  # imported here so tests don't need it

    # auto_adjust=True -> Close is already adjusted for splits + dividends
    # threads=False because the parallel download sometimes fails a ticker
    # with "database is locked" (yfinance's own sqlite cache)
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False
    )
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.reindex(columns=tickers)  # keep my column order

    # one retry for anything that came back empty
    for t in tickers:
        if prices[t].isna().all():
            again = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if not again.empty:
                prices[t] = again["Close"].squeeze()

    missing = [t for t in tickers if prices[t].isna().all()]
    if missing:
        # don't cache a half-broken download
        raise RuntimeError(f"No data for {missing}, try running again")

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_csv(cache_path)
    return prices


def clean_prices(prices):
    """Forward fill gaps, then drop rows before every asset has started.

    No backfill on purpose: bfill would copy a later price backwards in
    time, which is a (small) form of lookahead.
    """
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.ffill()
    prices = prices.dropna(how="any")
    return prices


def simple_returns(prices):
    return prices.pct_change()


def log_returns(prices):
    return np.log(prices / prices.shift(1))
