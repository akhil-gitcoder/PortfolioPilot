"""Quality indicators for multi-objective fronts, written from scratch.

All functions assume MINIMISATION and that objectives were first normalised
to a common scale with `normalize` (using bounds shared by all methods).

  hypervolume : volume dominated by the front and bounded by a reference point (bigger = better)
  igd         : mean distance from a reference front to the obtained front (smaller = better)
  spacing     : std-dev of nearest-neighbour distances (smaller = more even)
  c_metric    : fraction of B weakly dominated by at least one point of A
"""
import numpy as np


def normalize(F, lo, hi):
    return (np.asarray(F, float) - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def _hv2d(P, ref):
    """Exact 2-D hypervolume (area). P: (n,2) points that all dominate ref."""
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    # sweep left to right; between consecutive x values the dominated height
    # is ref_y minus the lowest y seen so far
    area, best_y = 0.0, ref[1]
    xs = list(P[:, 0]) + [ref[0]]
    for i in range(len(P)):
        best_y = min(best_y, P[i, 1])
        area += (xs[i + 1] - xs[i]) * (ref[1] - best_y)
    return area


def hypervolume(F, ref):
    """Exact hypervolume for 2 or 3 objectives (slicing method for 3-D)."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if len(F) == 0:
        return 0.0
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    m = F.shape[1]
    if m == 2:
        return _hv2d(F, ref)
    if m != 3:
        raise NotImplementedError("exact hypervolume implemented for 2 or 3 objectives")
    F = F[np.argsort(F[:, 2])]
    zs = list(F[:, 2]) + [ref[2]]
    vol = 0.0
    for i in range(len(F)):
        thickness = zs[i + 1] - zs[i]
        if thickness > 0:
            vol += thickness * _hv2d(F[: i + 1, :2], ref[:2])
    return vol


def igd(reference_front, obtained):
    R, A = np.asarray(reference_front, float), np.asarray(obtained, float)
    if len(A) == 0:
        return np.inf
    d = np.sqrt(((R[:, None, :] - A[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


def spacing(F):
    F = np.asarray(F, float)
    if len(F) < 3:
        return 0.0
    d = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    return float(np.std(d.min(axis=1)))


def c_metric(A, B):
    """Fraction of points in B that are weakly dominated by some point in A."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    if len(A) == 0 or len(B) == 0:
        return 0.0
    covered = (A[:, None, :] <= B[None, :, :]).all(axis=2).any(axis=0)
    return float(covered.mean())
