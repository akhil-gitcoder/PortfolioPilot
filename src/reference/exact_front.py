"""Exact Pareto front of the SIMPLE (convex) model by the epsilon-constraint method.

Simple model = budget, no shorting, x_max, sector caps, buy/sell proportional costs
(no cardinality, no minimum holding).  This problem is convex (Lobo, Fazel & Boyd, 2007),
so for each pair (eps_return, eps_cost) the convex program

    minimise  variance(x)   subject to   return(x) >= eps_return,  cost(x) <= eps_cost

is solved exactly by CVXPY.  Collecting the non-dominated results gives the true front.

ROLE: a CORRECTNESS CHECK for NSGA-II (the truth is known here).  It is NOT the baseline
that must be beaten: nothing can beat the exact front of the model it solves exactly.
"""
import time

import cvxpy as cp
import numpy as np

from src.multiobjective.pareto import nondominated_mask


def _solve(problem, eps_ret=None, eps_cost=None, objective="var"):
    n = problem.n
    x = cp.Variable(n)
    b = cp.Variable(n, nonneg=True)
    s = cp.Variable(n, nonneg=True)
    S = np.zeros((len(problem.sector_names), n))
    S[problem.sector_idx, np.arange(n)] = 1.0
    cons = [cp.sum(x) == 1, x >= 0, x <= problem.x_max, S @ x <= problem.sector_cap,
            x - problem.x_old == b - s]
    if eps_ret is not None:
        cons.append(problem.mu @ x >= eps_ret)
    if eps_cost is not None:
        cons.append(problem.buy_rate @ b + problem.sell_rate @ s <= eps_cost)
    if objective == "var":
        obj = cp.quad_form(x, cp.psd_wrap(problem.cov))
    elif objective == "neg_ret":
        obj = -(problem.mu @ x)
    else:
        obj = problem.buy_rate @ b + problem.sell_rate @ s
    p = cp.Problem(cp.Minimize(obj), cons)
    try:
        p.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    if x.value is None or p.status not in ("optimal", "optimal_inaccurate"):
        return None
    w = np.clip(x.value, 0, None)
    return w / w.sum()


def exact_front(problem, n_return=40, n_cost=12):
    t0 = time.perf_counter()
    x_maxret = _solve(problem, objective="neg_ret")
    x_minvar = _solve(problem, objective="var")
    ret_hi, ret_lo = problem.expected_return(x_maxret)[0], problem.expected_return(x_minvar)[0]
    cost_hi = problem.cost(x_maxret)[0]
    cost_lo = _solve(problem, objective="cost")
    cost_lo = problem.cost(cost_lo)[0]
    pts = []
    for er in np.linspace(ret_lo, ret_hi, n_return):
        for ec in np.linspace(cost_lo, cost_hi, n_cost):
            w = _solve(problem, eps_ret=er - 1e-9, eps_cost=ec + 1e-9)
            if w is not None:
                pts.append(w)
    X = np.vstack(pts)
    feas = problem.violation(X) <= 1e-5
    X = X[feas]
    F = problem.objectives(X)
    nd = nondominated_mask(F)
    return {"X": X[nd], "F": F[nd], "runtime": time.perf_counter() - t0}
