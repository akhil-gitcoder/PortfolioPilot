"""NSGA-II (Deb et al., 2002) written from scratch for PortfolioPilot.

Lecture steps followed
  1. random initial population of n_pop random-key vectors
  2. decode (repair) + evaluate the 3 objectives and the constraint violation
  3. fast non-dominated sorting (constraint-aware) + crowding distance
  4. binary tournament selection: lower rank wins, ties -> larger crowding distance
  5. SBX crossover + polynomial mutation on the keys (bounded to [0, 1])
  6. merge parents + children (elitism), keep the best n_pop by (rank, crowding)
  repeat until the evaluation budget is used.

Constraint handling: repair inside the decoder + Deb's constrained dominance
for anything repair cannot fix.

Simplification: SBX / polynomial mutation here are the standard formulas
without the bound-aware correction terms; keys are clipped to [0, 1].
"""
import time

import numpy as np

from src.multiobjective.pareto import FEAS_TOL, nondominated_mask, rank_and_crowding


def _tournament(rank, crowd, n_select, rng):
    n = len(rank)
    a = rng.integers(0, n, n_select)
    b = rng.integers(0, n, n_select)
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] >= crowd[b]))
    return np.where(a_wins, a, b)


def sbx_crossover(p1, p2, eta, pc, rng):
    m, n = p1.shape
    u = rng.random((m, n))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u) + 1e-12)) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    do_cross = (rng.random((m, 1)) < pc) & (rng.random((m, n)) < 0.5)
    c1 = np.where(do_cross, c1, p1)
    c2 = np.where(do_cross, c2, p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def polynomial_mutation(C, eta, pm, rng):
    u = rng.random(C.shape)
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    mutate = rng.random(C.shape) < pm
    return np.clip(C + mutate * delta, 0, 1)


def _survivor_selection(F, CV, n_pop):
    rank, crowd = rank_and_crowding(F, CV)
    order = np.lexsort((-crowd, rank))      # sort by rank, then larger crowding first
    keep = order[:n_pop]
    return keep, rank[keep], crowd[keep]


def run_nsga2(problem, pop_size=100, n_evals=10000, seed=0, eta_c=15, eta_m=20, pc=0.9,
              pm=None, checkpoint_every=500, lamarckian=False, inject_current=False, init_X=None,
              sparse_init=False, reset_prob=0.0):
    """Run NSGA-II; returns a dict with the final feasible non-dominated set and history.

    Options (all OFF = plain NSGA-II with random-key encoding):
      lamarckian     : after repair, write the repaired weights back into the chromosome
                       (key = weight / x_max) so evolution works on the actual portfolios.
      inject_current : put the current portfolio into the initial population (domain
                       knowledge: 'do nothing' is always a candidate for a rebalancing decision).
      sparse_init    : initial keys have a random SUPPORT SIZE (stocks above the key threshold)
                       between ceil(1/x_max) and min(K, 15), instead of ~90% of stocks held.
      reset_prob     : extra 'support' mutation: with this probability per child, one key is
                       reset to 0 (drop a stock) or to a random value (add a stock).
      init_X         : optional (m, n) array of known portfolios (e.g. a baseline front) copied
                       into the initial population -> MEMETIC / seeded variant.  Must be
                       reported as a hybrid, never as pure NSGA-II.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = problem.n
    pm = 1.0 / n if pm is None else pm

    keys = rng.random((pop_size, n))
    if sparse_init:
        lo_k = int(np.ceil(1.0 / problem.x_max))
        hi_k = max(lo_k, min(problem.K, 15))
        for i in range(pop_size):
            k = rng.integers(lo_k, hi_k + 1)
            sup = rng.choice(n, k, replace=False)
            row = rng.random(n) * problem.key_threshold * 0.99     # below threshold = not held
            row[sup] = problem.key_threshold + rng.random(k) * (1 - problem.key_threshold)
            keys[i] = row
    if inject_current:
        keys[0] = np.clip(problem.x_old / problem.x_max, 0, 1)
    if init_X is not None:
        m = min(len(init_X), pop_size)
        pick = rng.choice(len(init_X), m, replace=False)
        keys[-m:] = np.clip(np.asarray(init_X)[pick] / problem.x_max, 0, 1)
    X, F, CV = problem.evaluate_keys(keys)
    if lamarckian:
        keys = np.clip(X / problem.x_max, 0, 1)
    evals = pop_size
    keep, rank, crowd = _survivor_selection(F, CV, pop_size)
    keys, X, F, CV = keys[keep], X[keep], F[keep], CV[keep]
    history = []

    def snapshot():
        feas = CV <= FEAS_TOL
        if feas.any():
            Ff = F[feas]
            history.append((evals, Ff[nondominated_mask(Ff)].copy()))

    snapshot()
    next_cp = checkpoint_every
    while evals + pop_size <= n_evals:
        parents = _tournament(rank, crowd, pop_size, rng)
        half = pop_size // 2
        p1, p2 = keys[parents[:half]], keys[parents[half:2 * half]]
        c1, c2 = sbx_crossover(p1, p2, eta_c, pc, rng)
        child = polynomial_mutation(np.vstack([c1, c2]), eta_m, pm, rng)
        if reset_prob > 0:
            hit = rng.random(len(child)) < reset_prob
            for i in np.where(hit)[0]:
                j = rng.integers(0, n)
                child[i, j] = 0.0 if (child[i, j] >= problem.key_threshold and rng.random() < 0.5) \
                    else problem.key_threshold + rng.random() * (1 - problem.key_threshold)
        if len(child) < pop_size:              # odd pop sizes
            child = np.vstack([child, rng.random((pop_size - len(child), n))])
        Xc, Fc, CVc = problem.evaluate_keys(child)
        if lamarckian:
            child = np.clip(Xc / problem.x_max, 0, 1)
        evals += len(child)

        keys_all = np.vstack([keys, child])
        X_all, F_all, CV_all = np.vstack([X, Xc]), np.vstack([F, Fc]), np.concatenate([CV, CVc])
        keep, rank, crowd = _survivor_selection(F_all, CV_all, pop_size)
        keys, X, F, CV = keys_all[keep], X_all[keep], F_all[keep], CV_all[keep]
        if evals >= next_cp:
            snapshot()
            next_cp += checkpoint_every

    snapshot()
    feas = CV <= FEAS_TOL
    Xf, Ff = X[feas], F[feas]
    nd = nondominated_mask(Ff) if len(Ff) else np.zeros(0, bool)
    return {"X": Xf[nd], "F": Ff[nd], "evals": evals, "history": history,
            "runtime": time.perf_counter() - t0, "n_feasible_final_pop": int(feas.sum())}
