"""Baseline 2: weighted-sum 'relax and round' - a strong, defensible analyst heuristic.

For each weight vector (l1, l2, l3) on a grid over the simplex:
  1. RELAX   : drop the cardinality (<= K) and minimum-holding rules.  The problem is
               then convex, so CVXPY solves   min  -l1*ret/Rr + l2*var/Vr + l3*cost/Cr   exactly.
  2. ROUND   : keep the (at most K) largest weights; others are set to zero.
  3. POLISH  : re-solve the same convex problem restricted to the kept stocks with
               every kept weight >= l_min (exact again, because the support is fixed).
Keep the feasible non-dominated results.

Note: the weighted sum uses VARIANCE (a quadratic, convex-solver friendly); results are
then evaluated with volatility like every other method.
Limitations (Das & Dennis, 1997): weighted sums reach only convex parts of a front and
spread points unevenly; and greedy rounding can choose the wrong stocks.
"""
import itertools
import time

import cvxpy as cp
import numpy as np

from src.multiobjective.pareto import FEAS_TOL, nondominated_mask


def _build_and_solve(problem, lam, scales, support=None):
    n = problem.n
    x = cp.Variable(n)
    b = cp.Variable(n, nonneg=True)
    s = cp.Variable(n, nonneg=True)
    S = np.zeros((len(problem.sector_names), n))
    S[problem.sector_idx, np.arange(n)] = 1.0
    cons = [cp.sum(x) == 1, x >= 0, x <= problem.x_max, S @ x <= problem.sector_cap,
            x - problem.x_old == b - s]
    if problem.turnover_cap is not None:
        cons.append(0.5 * cp.sum(b + s) <= problem.turnover_cap)
    if support is not None:
        off = np.setdiff1d(np.arange(n), support)
        if len(off):
            cons.append(x[off] == 0)
        cons.append(x[support] >= problem.l_min)
    Rr, Vr, Cr = scales
    ret = problem.mu @ x
    var = cp.quad_form(x, cp.psd_wrap(problem.cov))
    cost = problem.buy_rate @ b + problem.sell_rate @ s
    obj = -lam[0] * ret / Rr + lam[1] * var / Vr + lam[2] * cost / Cr
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    if x.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return np.clip(x.value, 0, None)


def _scales(problem):
    """Normalisation ranges from the three single-objective relaxed solves."""
    sols = [_build_and_solve(problem, lam, (1.0, 1.0, 1.0)) for lam in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
    sols = [x / x.sum() for x in sols if x is not None]
    X = np.vstack(sols)
    ret, vol, cost = problem.expected_return(X), problem.volatility(X) ** 2, problem.cost(X)
    rng = lambda v: max(v.max() - v.min(), 1e-9)
    return rng(ret), rng(vol), rng(cost)


def run_weighted_sum(problem, step=0.1):
    t0 = time.perf_counter()
    scales = _scales(problem)
    k = int(round(1 / step))
    grid = [(i / k, j / k, (k - i - j) / k) for i, j in itertools.product(range(k + 1), repeat=2) if i + j <= k]
    min_support = int(np.ceil(1 / problem.x_max - 1e-12))
    X_list = []
    for lam in grid:
        relaxed = _build_and_solve(problem, lam, scales)
        if relaxed is None:
            continue
        order = np.argsort(-relaxed)
        support = [j for j in order[:problem.K] if relaxed[j] > 1e-4]
        for j in order:                                # ensure enough stocks for the x_max cap
            if len(support) >= min_support:
                break
            if j not in support:
                support.append(j)
        polished = _build_and_solve(problem, lam, scales, support=np.array(sorted(support)))
        if polished is not None:
            X_list.append(polished / polished.sum())
    if not X_list:
        return {"X": np.zeros((0, problem.n)), "F": np.zeros((0, 3)), "evals": len(grid) * 2,
                "runtime": time.perf_counter() - t0, "n_weight_vectors": len(grid)}
    X = np.vstack(X_list)
    F = problem.objectives(X)
    feas = problem.violation(X) <= 1e-5
    X, F = X[feas], F[feas]
    nd = nondominated_mask(F)
    return {"X": X[nd], "F": F[nd], "evals": len(grid) * 2, "runtime": time.perf_counter() - t0,
            "n_weight_vectors": len(grid)}
