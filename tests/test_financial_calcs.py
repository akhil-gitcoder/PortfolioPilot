"""Tests for return / risk / cost using the hand-worked 3-asset example (Phase 1)."""
import numpy as np
import pytest

from src.problem import PortfolioProblem


def toy_problem(**kw):
    sig = np.array([0.25, 0.15, 0.05])
    corr = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
    cov = corr * np.outer(sig, sig)
    args = dict(tickers=["A", "B", "C"], mu=np.array([0.12, 0.08, 0.05]), cov=cov,
                x_old=np.array([0.2, 0.3, 0.5]), buy_rate=np.full(3, 0.003),
                sell_rate=np.full(3, 0.003), sectors=["s1", "s2", "s3"],
                x_max=1.0, sector_cap=1.0, K=3, l_min=0.0)
    args.update(kw)
    return PortfolioProblem(**args)


def test_covariance_entries():
    p = toy_problem()
    assert p.cov[0, 1] == pytest.approx(0.01875)
    assert p.cov[0, 0] == pytest.approx(0.0625)


@pytest.mark.parametrize("x, ret, vol, cost", [
    ((0.6, 0.3, 0.1), 0.101, np.sqrt(0.0313), 0.0024),   # aggressive
    ((0.2, 0.3, 0.5), 0.073, np.sqrt(0.0074), 0.0),      # current (no trades)
    ((0.1, 0.2, 0.7), 0.063, np.sqrt(0.0035), 0.0012),   # defensive
])
def test_hand_computed_examples(x, ret, vol, cost):
    p = toy_problem()
    X = np.array([x])
    assert p.expected_return(X)[0] == pytest.approx(ret)
    assert p.volatility(X)[0] == pytest.approx(vol, rel=1e-6)
    assert p.cost(X)[0] == pytest.approx(cost)


def test_objective_signs():
    p = toy_problem()
    F = p.objectives(np.array([[0.6, 0.3, 0.1]]))
    assert F[0, 0] == pytest.approx(-0.101)      # return is negated (minimisation)


def test_buy_sell_asymmetry():
    p = toy_problem(buy_rate=np.full(3, 0.001), sell_rate=np.full(3, 0.005))
    # buy 0.1 of A (0.001*0.1), sell 0.1 of C (0.005*0.1)
    assert p.cost(np.array([[0.3, 0.3, 0.4]]))[0] == pytest.approx(0.0001 + 0.0005)


def test_variance_vs_volatility_same_ordering():
    p = toy_problem()
    X = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
    assert list(np.argsort(p.volatility(X))) == list(np.argsort(p.volatility(X) ** 2))


def test_infeasible_parameters_rejected():
    with pytest.raises(ValueError):
        toy_problem(x_max=0.2)            # 3 * 0.2 < 1
    with pytest.raises(ValueError):
        toy_problem(sector_cap=0.3)       # 3 sectors * 0.3 < 1
