"""Feasibility tests for repair and algorithms on a small synthetic problem."""
import numpy as np
import pytest

from src.algorithms.nsga2 import run_nsga2
from src.baseline.random_search import run_random_search
from src.baseline.weighted_sum_relax_round import run_weighted_sum
from src.data_loader import build_problem, load_config
from src.multiobjective.pareto import nondominated_mask
from src.problem import CV_TOL

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def problem():
    cfg = load_config(ROOT / "config" / "default.json")
    prob, _, label = build_problem(ROOT, cfg, source="synthetic")
    assert label == "synthetic"
    return prob


def test_repair_always_feasible(problem):
    rng = np.random.default_rng(0)
    X, F, CV = problem.evaluate_keys(rng.random((500, problem.n)))
    assert (CV <= CV_TOL).all(), CV.max()
    assert (X.sum(axis=1) == pytest.approx(1.0, abs=1e-8))
    assert ((X > 1e-8).sum(axis=1) <= problem.K).all()
    held = X > 1e-8
    assert (X[held] >= problem.l_min - 1e-9).all()
    assert (X <= problem.x_max + 1e-9).all()


def test_repair_handles_adversarial_keys(problem):
    n = problem.n
    for keys in [np.zeros(n), np.ones(n), np.eye(n)[0], np.r_[np.ones(5), np.zeros(n - 5)]]:
        x = problem.repair(keys)
        assert problem.violation(x)[0] <= CV_TOL


def test_nsga2_feasible_nondominated_and_reproducible(problem):
    r1 = run_nsga2(problem, pop_size=40, n_evals=800, seed=3)
    r2 = run_nsga2(problem, pop_size=40, n_evals=800, seed=3)
    assert len(r1["F"]) > 3
    assert (problem.violation(r1["X"]) <= CV_TOL).all()
    assert nondominated_mask(r1["F"]).all()
    np.testing.assert_allclose(r1["F"], r2["F"])            # same seed -> same result
    assert r1["evals"] <= 800


def test_random_search_feasible(problem):
    r = run_random_search(problem, n_evals=600, seed=1)
    assert len(r["F"]) > 0 and (problem.violation(r["X"]) <= CV_TOL).all()


def test_weighted_sum_baseline_obeys_same_constraints(problem):
    r = run_weighted_sum(problem, step=0.25)
    assert len(r["F"]) > 0
    assert (problem.violation(r["X"]) <= 1e-5).all()
    assert ((r["X"] > 1e-6).sum(axis=1) <= problem.K).all()


def test_archive_is_feasible_nondominated_and_capped(problem):
    r = run_nsga2(problem, pop_size=40, n_evals=800, seed=3, archive_size=25)
    assert 3 < len(r["F"]) <= 25
    assert (problem.violation(r["X"]) <= CV_TOL).all()
    assert nondominated_mask(r["F"]).all()
