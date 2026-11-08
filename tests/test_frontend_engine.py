"""Tests for the web-app engine (no Streamlit needed)."""
import numpy as np
import pandas as pd
import pytest

from frontend import engine as E


def test_format_inr_indian_grouping():
    assert E.format_inr(0) == "₹0"
    assert E.format_inr(999) == "₹999"
    assert E.format_inr(1000) == "₹1,000"
    assert E.format_inr(1234567) == "₹12,34,567"
    assert E.format_inr(10000000) == "₹1,00,00,000"
    assert E.format_inr(-150000) == "-₹1,50,000"


def test_select_k_includes_extremes_and_is_sorted_by_risk():
    rng = np.random.default_rng(0)
    ret = rng.uniform(0.1, 0.4, 200)
    vol = 0.1 + 0.5 * ret + rng.normal(0, 0.01, 200)
    cost = rng.uniform(0, 0.002, 200)
    F = np.c_[-ret, vol, cost]
    sel = E.select_k(F, 5)
    assert len(sel) == len(set(sel)) == 5
    assert int(np.argmin(F[:, 1])) in sel and int(np.argmin(F[:, 0])) in sel
    assert list(F[sel, 1]) == sorted(F[sel, 1])


def test_select_k_never_exceeds_available_points():
    F = np.array([[-0.2, 0.1, 0.0], [-0.3, 0.2, 0.001]])
    assert len(E.select_k(F, 6)) == 2


def test_efficient_2d_removes_clearly_beaten_points():
    ret = np.array([0.20, 0.25, 0.20])
    vol = np.array([0.10, 0.10, 0.15])
    keep = E.efficient_2d(ret, vol)
    assert list(keep) == [False, True, False]      # point 1 beats both others on return and risk


def test_efficient_2d_ignores_tiny_differences():
    ret = np.array([0.2000, 0.2005])
    vol = np.array([0.1000, 0.1004])
    assert E.efficient_2d(ret, vol).all()           # differences are noise, keep both


def test_clean_prices_fills_short_gaps_and_rejects_missing_ticker():
    idx = pd.date_range("2024-01-01", periods=6)
    raw = pd.DataFrame({"A": [1, 2, np.nan, 4, 5, 6.0], "B": [1, 1, 1, 1, 1, 1.0], "Z": 1.0}, index=idx)
    out = E.clean_prices(raw, ["A", "B"])
    assert list(out.columns) == ["A", "B"] and out["A"].isna().sum() == 0
    with pytest.raises(ValueError):
        E.clean_prices(raw, ["A", "MISSING"])


@pytest.fixture(scope="module")
def solved():
    cfg, uni, prices = E.load_context()
    from src.data_loader import build_x_old
    tick = [t for t in uni.ticker if t in prices.columns]
    x_old = build_x_old(tick, cfg["current_portfolio"]["holdings"])
    prob = E.make_snapshot_problem(prices, cfg, x_old)
    X, F = E.optimise(prob, cfg, n_evals=2000)
    names = dict(zip(uni.ticker, uni.name))
    return prob, X, F, names


def test_snapshot_uses_only_the_latest_window(solved):
    prob, *_ = solved
    cfg, uni, prices = E.load_context()
    last = prices[prob.tickers].iloc[-cfg["data"]["estimation_days"]:]
    mu = last.pct_change().dropna().mean().to_numpy() * 252
    assert np.allclose(prob.mu, mu)                  # estimated from the frozen last window


def test_end_to_end_options_are_consistent(solved):
    prob, X, F, names = solved
    res = E.analyse(prob, X, F, 5, 1_000_000, names)
    opts = res["options"]
    assert 3 <= len(opts) <= 5
    ok, _ = E.check_rules(prob, opts)
    assert ok
    vols = [o["vol"] for o in opts]
    assert vols == sorted(vols)
    for o in opts:                                    # shown numbers equal a fresh evaluation
        f = prob.objectives(o["x"][None, :])[0]
        assert o["ret"] == pytest.approx(-f[0]) and o["vol"] == pytest.approx(f[1])
        assert o["cost_inr"] == pytest.approx(f[2] * 1_000_000)
        assert abs(o["x"].sum() - 1) < 1e-6
        # trade amounts add up: total bought equals total sold (fully invested, same capital)
        t = o["trades"]
        assert t.loc[t.Action == "Buy", "Amount (₹)"].sum() == pytest.approx(
            t.loc[t.Action == "Sell", "Amount (₹)"].sum(), rel=1e-3, abs=1.0)
    labels = [o["label"] for o in opts]
    assert labels[0] == "Safest" and labels[-1] == "Highest return" and "Balanced" in labels


def test_options_never_worse_than_today_on_both_return_and_risk(solved):
    prob, X, F, names = solved
    res = E.analyse(prob, X, F, 5, 1_000_000, names)
    cur = res["current"]
    for o in res["options"]:
        assert o["ret"] > cur["ret"] or o["vol"] < cur["vol"]


def test_same_inputs_same_answer(solved):
    prob, X, F, _ = solved
    cfg, *_ = E.load_context()
    X2, F2 = E.optimise(prob, cfg, n_evals=2000)
    assert np.allclose(np.sort(F[:, 1]), np.sort(F2[:, 1]))
