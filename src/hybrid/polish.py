"""Hybrid step: exact convex 'polish' of an evolved front (NSGA-II + CVXPY).

Why: NSGA-II decides WHICH stocks to hold well enough, but its weights are only
approximately optimal.  Once the set of held stocks is fixed the problem is convex,
so the best weights can be computed exactly.

For every solution x_i = (return r_i, volatility v_i, cost c_i) with held set S_i we solve,
keeping S_i fixed (weights of the held stocks >= l_min, others 0):
   (a) minimise variance          s.t. return >= r_i , cost <= c_i
   (b) maximise return            s.t. vol    <= v_i , cost <= c_i
   (c) minimise cost              s.t. return >= r_i , vol  <= v_i
Each result is at least as good as x_i in every objective (x_i itself is feasible
for all three), so the polish can only improve the front.  Results are merged with the
originals and re-filtered for non-dominance.

Keep this hybrid ONLY if it beats plain NSGA-II on hypervolume over many seeds
at a comparable wall-clock time (the polish costs solver time).
"""
import time

import cvxpy as cp
import numpy as np

from src.multiobjective.pareto import nondominated_mask


def _polish_one(problem, x, mode, r, v, c):
    n = problem.n
    held = np.where(x > 1e-8)[0]
    off = np.setdiff1d(np.arange(n), held)
    w = cp.Variable(n)
    b = cp.Variable(n, nonneg=True)
    s = cp.Variable(n, nonneg=True)
    Smat = np.zeros((len(problem.sector_names), n))
    Smat[problem.sector_idx, np.arange(n)] = 1.0
    cons = [cp.sum(w) == 1, w <= problem.x_max, Smat @ w <= problem.sector_cap,
            w - problem.x_old == b - s, w[held] >= problem.l_min]
    if len(off):
        cons.append(w[off] == 0)
    ret = problem.mu @ w
    var = cp.quad_form(w, cp.psd_wrap(problem.cov))
    cost = problem.buy_rate @ b + problem.sell_rate @ s
    tol = 1e-9
    if problem.turnover_cap is not None:
        cons.append(0.5 * cp.sum(b + s) <= problem.turnover_cap)
    if mode == "var":
        cons += [ret >= r - tol, cost <= c + tol]
        obj = var
    elif mode == "ret":
        cons += [var <= v ** 2 + tol, cost <= c + tol]
        obj = -ret
    else:
        cons += [ret >= r - tol, var <= v ** 2 + tol]
        obj = cost
    try:
        cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    if w.value is None:
        return None
    out = np.clip(w.value, 0, None)
    out[out < 1e-7] = 0.0
    return out / out.sum()


def polish_front(problem, X, F):
    """Return (X_new, F_new, seconds): the merged, re-filtered non-dominated set."""
    t0 = time.perf_counter()
    cand = [x for x in X]
    for x, f in zip(X, F):
        r, v, c = -f[0], f[1], f[2]
        for mode in ("var", "ret", "cost"):
            out = _polish_one(problem, x, mode, r, v, c)
            if out is not None:
                cand.append(out)
    Xc = np.vstack(cand)
    keep = problem.violation(Xc) <= 1e-5
    Xc = Xc[keep]
    Fc = problem.objectives(Xc)
    nd = nondominated_mask(Fc)
    return Xc[nd], Fc[nd], time.perf_counter() - t0
