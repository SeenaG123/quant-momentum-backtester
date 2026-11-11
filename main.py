"""Run the whole thing: download -> choose settings in-sample -> test out-of-sample.

    python main.py
    python main.py --start 2012-01-01 --cost-bps 10
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

from src import metrics
from src.backtester import buy_and_hold, run_backtest, split_in_out
from src.data_loader import DEFAULT_TICKERS, clean_prices, download_prices
from src.strategies import momentum_minvar_weights, sma_trend_weights

# settings I let the in-sample data choose between
LOOKBACK_GRID = [
    (126,),  # the paper's 6 months
    (63, 126, 189),
    (63, 126, 189, 252),
    (21, 63, 126, 252),
]
COV_GRID = [60, 126]


def tune_in_sample(prices, cost_bps, in_frac):
    """Score each setting on the in-sample part only."""
    rows = []
    for lookbacks in LOOKBACK_GRID:
        for cov_window in COV_GRID:
            w = momentum_minvar_weights(
                prices, lookbacks=lookbacks, cov_window=cov_window
            )
            bt = run_backtest(prices, w, cost_bps=cost_bps)
            in_sample, _ = split_in_out(bt, in_frac)
            rows.append(
                {
                    "lookbacks": lookbacks,
                    "cov_window": cov_window,
                    "is_sharpe": metrics.sharpe_ratio(in_sample["net"]),
                }
            )
    grid = pd.DataFrame(rows).sort_values("is_sharpe", ascending=False)
    return grid.reset_index(drop=True)


def make_plots(bt, baseline, bench, cut_date, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt["equity"], label="Momentum + min variance (net of costs)")
    ax.plot(baseline["equity"], label="SMA crossover baseline (net of costs)")
    ax.plot((1 + bench).cumprod(), label="SPY buy & hold", alpha=0.8)
    ax.axvline(cut_date, color="gray", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(cut_date, ax.get_ylim()[0], " out-of-sample", va="bottom", color="gray")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title("Equity curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "equity_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    dd = metrics.drawdown_series(bt["net"])
    ax.fill_between(dd.index, dd, 0, alpha=0.5, label="Strategy")
    ax.plot(
        metrics.drawdown_series(bench), linewidth=0.8, color="tab:green", label="SPY"
    )
    ax.axvline(cut_date, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drawdown.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--in-frac", type=float, default=0.7)
    parser.add_argument("--cache", default="data/prices.csv")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    prices = download_prices(DEFAULT_TICKERS, args.start, args.end, args.cache)
    prices = clean_prices(prices)
    print(f"{len(prices)} days, {prices.index[0].date()} to {prices.index[-1].date()}")

    # 1. choose settings using in-sample data only
    grid = tune_in_sample(prices, args.cost_bps, args.in_frac)
    grid.to_csv(os.path.join(args.outdir, "param_grid.csv"), index=False)
    lookbacks = grid.loc[0, "lookbacks"]
    cov_window = int(grid.loc[0, "cov_window"])
    print("\nIn-sample grid:")
    print(grid.round(2).to_string(index=False))
    print(f"\nPicked lookbacks={lookbacks}, cov_window={cov_window}")

    # 2. run once with those settings and look at both halves
    weights = momentum_minvar_weights(prices, lookbacks=lookbacks, cov_window=cov_window)
    bt = run_backtest(prices, weights, cost_bps=args.cost_bps)
    baseline = run_backtest(prices, sma_trend_weights(prices), cost_bps=args.cost_bps)
    bench = buy_and_hold(prices, "SPY")

    bt_in, bt_out = split_in_out(bt, args.in_frac)
    base_in, base_out = split_in_out(baseline, args.in_frac)
    bench_in, bench_out = split_in_out(bench, args.in_frac)
    print(f"\nIn-sample ends {bt_in.index[-1].date()}, out-of-sample starts {bt_out.index[0].date()}")

    table = pd.DataFrame(
        {
            "Strategy IS": metrics.summary(bt_in["net"]),
            "Strategy OOS": metrics.summary(bt_out["net"]),
            "Strategy full": metrics.summary(bt["net"]),
            "SMA base IS": metrics.summary(base_in["net"]),
            "SMA base OOS": metrics.summary(base_out["net"]),
            "SPY IS": metrics.summary(bench_in),
            "SPY OOS": metrics.summary(bench_out),
            "SPY full": metrics.summary(bench),
        }
    )
    table.to_csv(os.path.join(args.outdir, "metrics.csv"))
    print("\n" + table.round(3).to_string())

    # 3. robustness: how fragile is the full-period Sharpe?
    def full_sharpe(**kw):
        return metrics.sharpe_ratio(run_backtest(prices, weights, **kw)["net"])

    checks = pd.Series(
        {
            "as reported (lag 1, cost as set)": metrics.sharpe_ratio(bt["net"]),
            "lag 0 (lookahead, wrong)": full_sharpe(cost_bps=args.cost_bps, lag=0),
            "lag 2 (trade a day late)": full_sharpe(cost_bps=args.cost_bps, lag=2),
            "10 bps costs": full_sharpe(cost_bps=10),
            "20 bps costs": full_sharpe(cost_bps=20),
        }
    )
    checks.to_csv(os.path.join(args.outdir, "robustness.csv"), header=["full_sharpe"])
    print("\nFull-period Sharpe under different assumptions:")
    print(checks.round(2).to_string())

    # 4. rebalance timing luck: one portfolio rebalanced every 20 days,
    #    started on 4 different days, vs. the 4-tranche version I use
    print("\nIn-sample Sharpe, single portfolio by rebalance start day:")
    for start in (0, 5, 10, 15):
        w1 = momentum_minvar_weights(
            prices,
            lookbacks=lookbacks,
            cov_window=cov_window,
            n_tranches=1,
            first_rebalance=start,
        )
        one, _ = split_in_out(run_backtest(prices, w1, cost_bps=args.cost_bps), args.in_frac)
        print(f"  start day {start:>2}: {metrics.sharpe_ratio(one['net']):.2f}")
    print(f"  4 tranches : {metrics.sharpe_ratio(bt_in['net']):.2f}")

    print(f"\nAvg daily turnover: {bt['turnover'].mean():.3f}")
    print("Avg weights:")
    print(weights.iloc[300:].mean().round(3).to_string())

    make_plots(bt, baseline, bench, bt_out.index[0], args.outdir)
    print(f"\nSaved plots + tables to {args.outdir}/")


if __name__ == "__main__":
    main()
