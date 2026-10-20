"""Baseline 1: random search with the SAME decoder and the SAME evaluation budget as NSGA-II.

Draw random key vectors, decode them with the shared repair (so every point obeys
the constraints), evaluate, and keep the non-dominated feasible set.
Question answered: is NSGA-II's search doing anything beyond luck?
"""
import time

import numpy as np

from src.multiobjective.pareto import FEAS_TOL, nondominated_mask


def run_random_search(problem, n_evals=10000, seed=0, checkpoint_every=500, inject_current=False):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    X_nd = np.zeros((0, problem.n))
    F_nd = np.zeros((0, 3))
    history = []
    done = 0
    while done < n_evals:
        m = min(checkpoint_every, n_evals - done)
        K = rng.random((m, problem.n))
        if inject_current and done == 0:
            K[0] = np.clip(problem.x_old / problem.x_max, 0, 1)
        X, F, CV = problem.evaluate_keys(K)
        done += m
        feas = CV <= FEAS_TOL
        X_nd = np.vstack([X_nd, X[feas]])
        F_nd = np.vstack([F_nd, F[feas]])
        mask = nondominated_mask(F_nd)
        X_nd, F_nd = X_nd[mask], F_nd[mask]
        history.append((done, F_nd.copy()))
    return {"X": X_nd, "F": F_nd, "evals": done, "history": history,
            "runtime": time.perf_counter() - t0}
