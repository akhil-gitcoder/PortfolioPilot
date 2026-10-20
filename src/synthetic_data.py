"""SYNTHETIC price generator - used ONLY for development and testing.

!!! Results produced from this data are NOT real market findings. !!!

Model: daily simple return of stock i =
    drift_i + beta_i * market_t + sector_factor_t(s(i)) + idio_i,t
with normal shocks.  This gives realistic-looking correlation structure
(stocks in the same sector move together) so the optimisation problem has
the right shape, but nothing here is observed market data.
"""
import numpy as np
import pandas as pd


def generate_synthetic_prices(universe: pd.DataFrame, years: int = 6, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(universe)
    n_days = int(252 * years)
    sectors = universe["sector"].to_numpy()
    sector_names = sorted(set(sectors))

    annual_mu = rng.uniform(0.06, 0.22, n)          # annual drift per stock
    beta = rng.uniform(0.7, 1.3, n)
    idio_vol = rng.uniform(0.010, 0.016, n)          # daily idiosyncratic vol
    market = rng.normal(0.0, 0.009, n_days)          # daily market factor
    sector_f = {s: rng.normal(0.0, 0.007, n_days) for s in sector_names}

    rets = np.empty((n_days, n))
    for i in range(n):
        rets[:, i] = (annual_mu[i] / 252 + beta[i] * market + sector_f[sectors[i]]
                      + rng.normal(0.0, idio_vol[i], n_days))
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return pd.DataFrame(prices, index=dates, columns=universe["ticker"].tolist())
