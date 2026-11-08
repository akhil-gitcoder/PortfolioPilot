"""Tests for the walk-forward helpers (hand-checkable examples and the no-look-ahead rule)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import build_problem, load_config
from src.walk_forward import (drift_weights, knee_index, path_metrics, period_inputs,
                              rebalance_schedule, simulate_period)

ROOT = Path(__file__).resolve().parents[1]


def test_schedule_starts_after_full_window_and_never_overlaps():
    sched = rebalance_schedule(n_rows=1729, window=756, horizon=63)
    assert sched[0] == (756, 819)
    assert len(sched) == 15
    for (t, t_end), (t2, _) in zip(sched, sched[1:]):
        assert t_end == t2                          # periods are back to back
    assert sched[-1][1] <= 1728                     # never past the last row


def test_drift_weights_hand_example():
    # 50/50, one stock +10%, the other -10%  -> weights 55/45, portfolio return 0
    w = drift_weights([0.5, 0.5], [1.1, 0.9])
    np.testing.assert_allclose(w, [0.55, 0.45])
    daily, w_end = simulate_period(np.array([[100.0, 100.0], [110.0, 90.0]]), [0.5, 0.5])
    np.testing.assert_allclose(daily, [0.0], atol=1e-12)
    np.testing.assert_allclose(w_end, [0.55, 0.45])


def test_knee_picks_the_balanced_point():
    # columns: [-return, vol, cost]; the middle row is closest to the ideal after scaling
    F = np.array([[-0.30, 0.40, 0.0020],
                  [-0.20, 0.20, 0.0010],
                  [-0.05, 0.10, 0.0000]])
    assert knee_index(F, "knee") == 1
    assert knee_index(F, "ratio", horizon=63) in (1, 2)


def test_path_metrics_constant_growth():
    w = 1.0001 ** np.arange(253)                    # 252 days of +0.01% per day
    m = path_metrics(w)
    assert m["final_wealth"] == pytest.approx(1.0001 ** 252)
    assert m["max_drawdown"] == pytest.approx(0.0)


def test_estimates_do_not_look_ahead():
    cfg = load_config(ROOT / "config" / "default.json")
    _, prices, _ = build_problem(ROOT, cfg, source="synthetic")
    t, window = 800, 756
    mu1, cov1 = period_inputs(prices, t, window)
    future_changed = prices.copy()
    future_changed.iloc[t:] = future_changed.iloc[t:] * 3.0      # wreck everything from row t on
    mu2, cov2 = period_inputs(future_changed, t, window)
    np.testing.assert_allclose(mu1, mu2)
    np.testing.assert_allclose(cov1, cov2)
