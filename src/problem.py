"""PortfolioPilot problem definition: objectives, constraints, repair.

Decision variable  x in R^n : fraction of capital in each stock.

Objectives (ALL MINIMISED internally):
    f1 = -mu^T x                              (negative expected annual return)
    f2 = sqrt(x^T Sigma x)                    (annual volatility)
    f3 = sum_i buy_i*max(x_i-xold_i,0) + sell_i*max(xold_i-x_i,0)
                                              (one-off rebalancing cost, fraction of capital)

Constraints:
    sum x = 1, x >= 0, x_i <= x_max, sector sums <= sector_cap,
    #held <= K, held weights >= l_min, optional turnover cap.

Encoding used by the evolutionary algorithm ("random keys"):
    a vector of n numbers in [0,1].  `repair` turns ANY such vector into a
    portfolio that satisfies the constraints whenever that is possible.
"""
from dataclasses import dataclass, field

import numpy as np

OBJECTIVE_NAMES = ["neg_return", "volatility", "rebalancing_cost"]
HELD_EPS = 1e-8       # weight above this counts as "held"
CV_TOL = 1e-6         # total violation below this counts as feasible


@dataclass
class PortfolioProblem:
    tickers: list
    mu: np.ndarray
    cov: np.ndarray
    x_old: np.ndarray
    buy_rate: np.ndarray
    sell_rate: np.ndarray
    sectors: list
    x_max: float
    sector_cap: float
    K: int
    l_min: float
    turnover_cap: float = None
    key_threshold: float = 0.1
    sector_idx: np.ndarray = field(init=False)
    sector_names: list = field(init=False)

    def __post_init__(self):
        self.mu = np.asarray(self.mu, float)
        self.cov = np.asarray(self.cov, float)
        self.x_old = np.asarray(self.x_old, float)
        self.sector_names = sorted(set(self.sectors))
        lookup = {s: i for i, s in enumerate(self.sector_names)}
        self.sector_idx = np.array([lookup[s] for s in self.sectors])
        n = self.n
        if self.K * self.x_max < 1 - 1e-12:
            raise ValueError("Infeasible: K * x_max < 1 (cannot invest 100%).")
        if self.K * self.l_min > 1 + 1e-12:
            raise ValueError("Infeasible: K * l_min > 1.")
        if len(self.sector_names) * self.sector_cap < 1 - 1e-12:
            raise ValueError("Infeasible: sector caps sum to less than 100%.")
        if self.cov.shape != (n, n) or len(self.mu) != n or len(self.x_old) != n:
            raise ValueError("Inconsistent dimensions of mu / cov / x_old.")
        if not np.allclose(self.cov, self.cov.T, atol=1e-10):
            raise ValueError("Covariance matrix must be symmetric.")

    @property
    def n(self):
        return len(self.tickers)

    # ------------------------------------------------------------------ objectives
    def expected_return(self, X):
        return np.atleast_2d(X) @ self.mu

    def volatility(self, X):
        X = np.atleast_2d(X)
        var = np.einsum("ij,jk,ik->i", X, self.cov, X)
        return np.sqrt(np.maximum(var, 0.0))

    def cost(self, X):
        X = np.atleast_2d(X)
        d = X - self.x_old
        return (np.maximum(d, 0) * self.buy_rate + np.maximum(-d, 0) * self.sell_rate).sum(axis=1)

    def objectives(self, X):
        """Matrix of shape (m, 3): [-return, volatility, cost], all to be minimised."""
        X = np.atleast_2d(X)
        return np.column_stack([-self.expected_return(X), self.volatility(X), self.cost(X)])

    # ------------------------------------------------------------------ constraints
    def violation_components(self, X):
        """Dict of violation arrays (each shape (m,)); all zero for a feasible portfolio."""
        X = np.atleast_2d(X)
        held = X > HELD_EPS
        sector_sums = np.stack([X[:, self.sector_idx == s].sum(axis=1)
                                for s in range(len(self.sector_names))], axis=1)
        comp = {
            "budget": np.abs(X.sum(axis=1) - 1.0),
            "negative": np.maximum(-X, 0).sum(axis=1),
            "max_weight": np.maximum(X - self.x_max, 0).sum(axis=1),
            "min_holding": (np.maximum(self.l_min - X, 0) * held).sum(axis=1),
            "sector_cap": np.maximum(sector_sums - self.sector_cap, 0).sum(axis=1),
            "cardinality": np.maximum(held.sum(axis=1) - self.K, 0).astype(float),
        }
        if self.turnover_cap is not None:
            turnover = 0.5 * np.abs(X - self.x_old).sum(axis=1)
            comp["turnover"] = np.maximum(turnover - self.turnover_cap, 0)
        return comp

    def violation(self, X):
        """Total constraint violation (0 = feasible)."""
        return sum(self.violation_components(X).values())

    def is_feasible(self, X):
        return self.violation(X) <= CV_TOL

    # ------------------------------------------------------------------ repair
    def repair(self, keys):
        """Turn a random-key vector into a portfolio that satisfies the constraints.

        Steps: (1) keys below `key_threshold` mean 'not held'; (2) if more than K
        are held keep the K largest; (3) make sure enough stocks are held for the
        x_max cap to allow 100% investment; (4) weights proportional to keys;
        (5) iteratively clip to [l_min, x_max], fix sector caps and re-balance
        the leftover ('water-filling')."""
        k = np.clip(np.asarray(keys, float), 0.0, 1.0)
        order = np.argsort(-k)
        held = k >= self.key_threshold
        if held.sum() > self.K:
            held = np.zeros(self.n, bool)
            held[order[:self.K]] = True
        min_support = int(np.ceil(1.0 / self.x_max - 1e-12))
        if held.sum() < min_support:
            for j in order:
                held[j] = True
                if held.sum() >= min_support:
                    break
        held = self._ensure_sector_capacity(held, order)
        w = np.where(held, k, 0.0)
        if w.sum() <= 0:
            w[held] = 1.0
        w = w / w.sum()
        return self._project(w, held)

    def _ensure_sector_capacity(self, held, order):
        """Make sure the held stocks can absorb 100% of capital under x_max and the sector caps.
        (E.g. holding only banks with a 40% sector cap can never reach 100% invested.)
        Adds the best-keyed stock from a sector with room; if already at K, drops the
        worst-keyed stock from a sector that has more stocks than it needs."""
        S = len(self.sector_names)
        needed = int(np.ceil(self.sector_cap / self.x_max - 1e-12))   # stocks that saturate a sector
        held = held.copy()
        for _ in range(self.n):
            cnt = np.bincount(self.sector_idx[held], minlength=S)
            if np.minimum(self.sector_cap, self.x_max * cnt).sum() >= 1 - 1e-12:
                break
            has_room = self.x_max * cnt < self.sector_cap - 1e-12
            cand = [j for j in order if (not held[j]) and has_room[self.sector_idx[j]]]
            if not cand:
                break
            if held.sum() >= self.K:
                drop = [j for j in order[::-1] if held[j] and cnt[self.sector_idx[j]] > needed]
                if not drop:
                    break
                held[drop[0]] = False
            held[cand[0]] = True
        return held

    def _project(self, w, held):
        lo = np.where(held, self.l_min, 0.0)
        hi = np.where(held, self.x_max, 0.0)
        S = len(self.sector_names)
        sidx = self.sector_idx
        for _ in range(200):
            w = np.clip(w, lo, hi)
            ss = np.bincount(sidx, weights=w, minlength=S)
            for s in np.where(ss - self.sector_cap > 1e-12)[0]:
                m = sidx == s
                room = w[m] - lo[m]
                tot = room.sum()
                if tot > 0:
                    w[m] -= room * min(1.0, (ss[s] - self.sector_cap) / tot)
            diff = 1.0 - w.sum()
            if abs(diff) < 1e-12:
                break
            if diff > 0:                       # need to add weight
                ss = np.bincount(sidx, weights=w, minlength=S)
                slack = np.maximum(self.sector_cap - ss, 0.0)
                room = np.minimum(hi - w, slack[sidx])
                tot = room.sum()
                if tot <= 1e-15:
                    break
                add = room * min(1.0, diff / tot)
                sa = np.bincount(sidx, weights=add, minlength=S)
                scale = np.where(sa > slack, slack / np.maximum(sa, 1e-300), 1.0)
                w = w + add * scale[sidx]
            else:                              # need to remove weight
                room = w - lo
                tot = room.sum()
                if tot <= 1e-15:
                    break
                w = w - room * min(1.0, -diff / tot)
        return np.clip(w, lo, hi)

    def evaluate_keys(self, keys):
        """Decode a batch of key vectors -> (portfolios X, objectives F, violation CV)."""
        keys = np.atleast_2d(keys)
        X = np.vstack([self.repair(k) for k in keys])
        return X, self.objectives(X), self.violation(X)
