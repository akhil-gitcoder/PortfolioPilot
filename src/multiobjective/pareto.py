"""Pareto tools written from scratch: dominance, constrained dominance,
fast non-dominated sorting and crowding distance.  All objectives are MINIMISED.

Constrained dominance (Deb et al., 2002):
  1. a feasible solution beats an infeasible one;
  2. between two infeasible solutions the smaller total violation wins;
  3. between two feasible solutions ordinary Pareto dominance decides.
"""
import numpy as np

FEAS_TOL = 1e-6


def dominates(a, b):
    """True if objective vector a Pareto-dominates b (a <= b everywhere, < somewhere)."""
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def dominance_matrix(F, CV=None):
    """D[i, j] = True if solution i (constrained-)dominates solution j."""
    F = np.asarray(F, float)
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    D = le & lt
    if CV is None:
        return D
    CV = np.asarray(CV, float)
    feas = CV <= FEAS_TOL
    both_feas = feas[:, None] & feas[None, :]
    D_out = both_feas & D
    D_out |= feas[:, None] & ~feas[None, :]
    both_infeas = ~feas[:, None] & ~feas[None, :]
    D_out |= both_infeas & (CV[:, None] < CV[None, :])
    return D_out


def fast_non_dominated_sort(F, CV=None):
    """Return an integer rank per solution (0 = first / best front)."""
    D = dominance_matrix(F, CV)
    n = len(D)
    dominated_count = D.sum(axis=0).astype(int)    # how many dominate j
    rank = np.full(n, -1, dtype=int)
    current = np.where(dominated_count == 0)[0]
    r = 0
    while len(current):
        rank[current] = r
        dominated_count = dominated_count - D[current].sum(axis=0)
        dominated_count[rank >= 0] = -1
        current = np.where(dominated_count == 0)[0]
        r += 1
    return rank


def crowding_distance(F):
    """Crowding distance of one front (boundary points get infinity)."""
    F = np.asarray(F, float)
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        idx = np.argsort(F[:, j])
        span = F[idx[-1], j] - F[idx[0], j]
        d[idx[0]] = d[idx[-1]] = np.inf
        if span <= 0:
            continue
        d[idx[1:-1]] += (F[idx[2:], j] - F[idx[:-2], j]) / span
    return d


def rank_and_crowding(F, CV=None):
    rank = fast_non_dominated_sort(F, CV)
    crowd = np.zeros(len(F))
    for r in np.unique(rank):
        m = rank == r
        crowd[m] = crowding_distance(F[m])
    return rank, crowd


def nondominated_mask(F):
    """Boolean mask of the non-dominated rows of F (duplicates keep one copy)."""
    F = np.asarray(F, float)
    if len(F) == 0:
        return np.zeros(0, bool)
    D = dominance_matrix(F)
    mask = ~D.any(axis=0)
    # remove exact duplicates (keep first)
    seen = set()
    for i in np.where(mask)[0]:
        key = tuple(np.round(F[i], 12))
        if key in seen:
            mask[i] = False
        seen.add(key)
    return mask
