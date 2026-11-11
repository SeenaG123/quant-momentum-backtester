import numpy as np
import pandas as pd
import pytest

from src import metrics
from src.backtester import run_backtest, split_in_out
from src.indicators import (
    inverse_vol_weights,
    momentum_score,
    rsi,
    sma,
    sma_crossover_signal,
)
from src.strategies import (
    min_variance_weights,
    momentum_minvar_weights,
    sma_trend_weights,
)


def fake_prices(n=300, cols=("A", "B", "C"), seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, size=(n, len(cols)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=cols)


# ---------- indicators


def test_sma_matches_manual_average():
    p = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = sma(p, 3)["A"]
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_rsi_bounds_and_direction():
    up = pd.DataFrame({"A": np.arange(1.0, 60.0)})
    assert rsi(up, 14)["A"].iloc[-1] == pytest.approx(100.0)
    r = rsi(fake_prices(), 14).dropna()
    assert ((r >= 0) & (r <= 100)).all().all()


def test_crossover_needs_fast_less_than_slow():
    with pytest.raises(ValueError):
        sma_crossover_signal(fake_prices(), fast=50, slow=20)


def test_inverse_vol_weights_sum_to_one():
    rets = fake_prices().pct_change()
    w = inverse_vol_weights(rets, 30).iloc[40:]
    assert np.allclose(w.sum(axis=1), 1.0)


def test_momentum_score_is_average_of_lookbacks():
    p = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]})
    score = momentum_score(p, lookbacks=(1, 2))["A"]
    # 1-day return is 10%, 2-day return is 21% -> average 15.5%
    assert score.iloc[3] == pytest.approx(0.155)
    assert np.isnan(score.iloc[1])


# ---------- min variance


def test_min_variance_two_assets_closed_form():
    # uncorrelated assets: weight is proportional to 1/variance
    cov = np.array([[0.04, 0.0], [0.0, 0.01]])
    w = min_variance_weights(cov)
    assert w == pytest.approx([0.2, 0.8])


def test_min_variance_is_long_only_and_sums_to_one():
    rng = np.random.default_rng(2)
    for _ in range(50):
        a = rng.normal(size=(60, 5))
        a[:, 1] = a[:, 0] + 0.1 * a[:, 1]  # two very correlated assets
        w = min_variance_weights(np.cov(a, rowvar=False))
        assert (w >= 0).all()
        assert w.sum() == pytest.approx(1.0)


def test_min_variance_beats_equal_weight():
    rng = np.random.default_rng(3)
    cov = np.cov(rng.normal(size=(100, 4)) * [1, 2, 3, 4], rowvar=False)
    w = min_variance_weights(cov)
    eq = np.full(4, 0.25)
    assert w @ cov @ w <= eq @ cov @ eq


# ---------- strategies


def test_momentum_minvar_weights_shape():
    p = fake_prices(n=400, cols=list("ABCDEF"))
    w = momentum_minvar_weights(
        p, top_k=3, lookbacks=(20, 40), cov_window=30, hold_days=20, n_tranches=4
    )
    assert (w >= 0).all().all()
    assert (w.sum(axis=1) <= 1 + 1e-9).all()
    # fully invested once every tranche has rebalanced at least once
    assert w.sum(axis=1).iloc[-1] == pytest.approx(1.0)
    # nothing is held before the longest lookback is available
    assert w.iloc[:41].abs().sum().sum() == 0


def test_momentum_minvar_no_lookahead():
    """Changing prices after day 300 must not change weights up to day 299."""
    p = fake_prices(n=400, cols=list("ABCDEF"))
    kw = dict(top_k=3, lookbacks=(20, 40), cov_window=30, hold_days=20, n_tranches=4)
    base = momentum_minvar_weights(p, **kw)

    p2 = p.copy()
    p2.iloc[300:] = p2.iloc[300:] * np.linspace(1.0, 2.0, 100)[:, None]
    changed = momentum_minvar_weights(p2, **kw)

    assert np.allclose(base.iloc[:300], changed.iloc[:300])


def test_sma_strategy_no_lookahead():
    """Changing a future price must not change today's return."""
    p = fake_prices()
    base = run_backtest(p, sma_trend_weights(p, 5, 20), cost_bps=0)

    p2 = p.copy()
    p2.iloc[200:] = p2.iloc[200:] * 1.5  # mess with everything from day 200 on
    changed = run_backtest(p2, sma_trend_weights(p2, 5, 20), cost_bps=0)

    # positions held through day 199 were all decided before day 200
    assert np.allclose(base["net"].iloc[:200], changed["net"].iloc[:200])


# ---------- backtester


def test_position_is_lagged_weight():
    """Weights that turn on at day t should only earn from day t+1."""
    p = fake_prices(n=100, cols=("A",))
    w = pd.DataFrame(0.0, index=p.index, columns=p.columns)
    w.iloc[50:] = 1.0
    bt = run_backtest(p, w, cost_bps=0)
    assert (bt["gross"].iloc[:51] == 0).all()
    assert bt["gross"].iloc[51] == pytest.approx(p["A"].pct_change().iloc[51])


def test_costs_only_reduce_returns():
    p = fake_prices()
    w = sma_trend_weights(p, 5, 20)
    free = run_backtest(p, w, cost_bps=0)
    paid = run_backtest(p, w, cost_bps=10)
    assert (paid["net"] <= free["net"] + 1e-12).all()
    assert paid["equity"].iloc[-1] < free["equity"].iloc[-1]


def test_cost_on_single_trade():
    p = fake_prices(n=100, cols=("A",))
    w = pd.DataFrame(0.0, index=p.index, columns=p.columns)
    w.iloc[50:] = 1.0
    bt = run_backtest(p, w, cost_bps=10)
    # going from 0 to 100% invested is turnover of 1 -> 10 bps
    assert bt["cost"].iloc[51] == pytest.approx(0.001)
    assert bt["cost"].sum() == pytest.approx(0.001)


# ---------- metrics


def test_max_drawdown_known_path():
    rets = pd.Series([0.10, -0.50, 0.20])
    assert metrics.max_drawdown(rets) == pytest.approx(-0.5)


def test_sharpe_scaling():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 5000))
    expected = r.mean() / r.std() * np.sqrt(252)
    assert metrics.sharpe_ratio(r) == pytest.approx(expected)


def test_split_is_chronological():
    p = fake_prices(n=100)
    a, b = split_in_out(p, 0.7)
    assert len(a) == 70 and len(b) == 30
    assert a.index.max() < b.index.min()
