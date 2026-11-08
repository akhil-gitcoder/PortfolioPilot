"""Walk-forward (out-of-sample) evaluation helpers for PortfolioPilot.

Idea: at each rebalance date t
  1. estimate expected returns and covariance from the PAST window only  (rows t-window .. t-1),
  2. choose new weights,
  3. pay the trading cost, then hold (buy-and-hold, weights drift) for `horizon` days,
  4. measure what REALLY happened over rows t .. t+horizon  (data the estimate never saw),
  5. roll forward; the drifted weights become the next period's "current portfolio".

Everything here is deliberately small and testable.  The experiment script is
experiments/run_walkforward.py.
"""
from dataclasses import replace

import numpy as np

from src.data_loader import estimate_inputs

TRADING_DAYS = 252


def rebalance_schedule(n_rows, window, horizon):
    """List of (t, t_end): rebalance at row t, evaluate until row t_end = t + horizon.
    The first rebalance is at row `window` (the first date with a full estimation window)."""
    out, t = [], window
    while t + horizon <= n_rows - 1:
        out.append((t, t + horizon))
        t += horizon
    return out


def estimation_slice(prices, t, window):
    """Rows t-window .. t-1: strictly BEFORE the rebalance row t (no look-ahead)."""
    return prices.iloc[t - window:t]


def period_inputs(prices, t, window, cov_estimator="ledoit_wolf"):
    """(mu, cov) estimated only from the past window before row t."""
    return estimate_inputs(estimation_slice(prices, t, window), window, cov_estimator)


def period_problem(base, mu, cov, x_old):
    """Same constraints and costs as `base`, but with new estimates and a new current portfolio."""
    return replace(base, mu=mu, cov=cov, x_old=np.asarray(x_old, float))


def drift_weights(w, ratio_end):
    """Weights after holding: each position grows with its own price ratio."""
    v = np.asarray(w, float) * np.asarray(ratio_end, float)
    return v / v.sum()


def simulate_period(P, w):
    """Buy-and-hold over one period.
    P : (H+1, n) prices from the rebalance row to the end row.   w : weights at the start.
    Returns (daily portfolio returns (H,), drifted end weights)."""
    P = np.asarray(P, float)
    ratio = P / P[0]
    value = ratio @ np.asarray(w, float)
    daily = value[1:] / value[:-1] - 1.0
    return daily, drift_weights(w, ratio[-1])


def knee_index(F, rule="knee", horizon=63):
    """Pick ONE portfolio from a front F with columns [-return, volatility, cost].

    rule 'knee' : closest to the ideal point after min-max scaling the front's own columns
                  (equal importance for the three goals).
    rule 'ratio': highest (return - cost spread over the holding period) / volatility,
                  i.e. the best risk-adjusted net return."""
    F = np.asarray(F, float)
    if rule == "knee":
        lo, hi = F.min(axis=0), F.max(axis=0)
        z = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
        return int(np.argmin(np.sqrt((z ** 2).sum(axis=1))))
    if rule == "ratio":
        net = -F[:, 0] - F[:, 2] * TRADING_DAYS / horizon
        return int(np.argmax(net / np.maximum(F[:, 1], 1e-12)))
    raise ValueError(f"unknown rule {rule}")


def path_metrics(wealth):
    """Summary of a wealth series (starts at 1.0): annualised return, volatility, ratio, drawdown."""
    w = np.asarray(wealth, float)
    r = w[1:] / w[:-1] - 1.0
    days = len(r)
    ann_ret = w[-1] ** (TRADING_DAYS / days) - 1.0
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    peak = np.maximum.accumulate(w)
    return {"final_wealth": float(w[-1]), "ann_return": float(ann_ret), "ann_vol": float(ann_vol),
            "return_over_vol": float(ann_ret / ann_vol) if ann_vol > 0 else float("nan"),
            "max_drawdown": float((w / peak - 1.0).min())}
