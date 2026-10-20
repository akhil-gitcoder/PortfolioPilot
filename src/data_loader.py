"""Load prices, estimate mu and Sigma, build the current portfolio, build the problem.

Data-leakage rule: mu and Sigma are estimated ONLY from the first
`estimation_days` rows.  Later rows are a hold-out for out-of-sample tests.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.problem import PortfolioProblem
from src.synthetic_data import generate_synthetic_prices

TRADING_DAYS = 252  # annualisation convention


def load_config(path):
    with open(path) as f:
        return json.load(f)


def load_universe(root, cfg):
    return pd.read_csv(Path(root) / cfg["data"]["universe_csv"])


def load_prices(root, cfg, source=None):
    """Return a prices DataFrame (rows = dates, columns = tickers) and a label."""
    root = Path(root)
    source = source or cfg["data"]["source"]
    uni = load_universe(root, cfg)
    if source == "synthetic":
        path = root / cfg["data"]["synthetic_prices_csv"]
        if path.exists():
            prices = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            prices = generate_synthetic_prices(uni, cfg["data"]["synthetic_years"],
                                               cfg["data"]["synthetic_seed"])
            path.parent.mkdir(parents=True, exist_ok=True)
            prices.to_csv(path)
        return prices, "synthetic"
    path = root / cfg["data"]["real_prices_csv"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` on a machine with "
            "internet access first (see README, section 'Data').")
    return pd.read_csv(path, index_col=0, parse_dates=True), "real"


def estimate_inputs(prices, estimation_days, cov_estimator="ledoit_wolf"):
    """Annualised expected returns (sample mean) and covariance from the estimation window."""
    window = prices.iloc[:estimation_days]
    rets = window.pct_change().dropna()
    mu = rets.mean().to_numpy() * TRADING_DAYS
    if cov_estimator == "ledoit_wolf":
        cov = LedoitWolf().fit(rets.to_numpy()).covariance_ * TRADING_DAYS
    elif cov_estimator == "sample":
        cov = rets.cov().to_numpy() * TRADING_DAYS
    else:
        raise ValueError(f"unknown cov_estimator {cov_estimator}")
    return mu, cov


def build_x_old(tickers, holdings):
    x_old = np.zeros(len(tickers))
    for t, w in holdings.items():
        if t not in tickers:
            raise ValueError(f"current-portfolio ticker {t} not in universe")
        x_old[tickers.index(t)] = w
    return x_old


def build_problem(root, cfg, source=None, convex=False, cov_estimator=None):
    """Create a PortfolioProblem.  convex=True drops the cardinality and min-holding rules
    (the 'simple' model used for the exact-front correctness check)."""
    prices, label = load_prices(root, cfg, source)
    uni = load_universe(root, cfg)
    tickers = [t for t in uni["ticker"] if t in prices.columns]
    prices = prices[tickers]
    uni = uni[uni["ticker"].isin(tickers)].reset_index(drop=True)
    mu, cov = estimate_inputs(prices, cfg["data"]["estimation_days"],
                              cov_estimator or cfg["data"]["cov_estimator"])
    x_old = build_x_old(tickers, cfg["current_portfolio"]["holdings"])
    c = cfg["costs"]
    mult = c.get("cost_multiplier", 1.0)
    n = len(tickers)
    con = cfg["constraints"]
    prob = PortfolioProblem(
        tickers=tickers, mu=mu, cov=cov, x_old=x_old,
        buy_rate=np.full(n, c["buy_rate"] * mult), sell_rate=np.full(n, c["sell_rate"] * mult),
        sectors=uni["sector"].tolist(), x_max=con["x_max"], sector_cap=con["sector_cap"],
        K=n if convex else con["K"], l_min=0.0 if convex else con["l_min"],
        turnover_cap=con.get("turnover_cap"), key_threshold=con.get("key_threshold", 0.1))
    return prob, prices, label
