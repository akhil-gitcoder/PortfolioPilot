"""Phase C: walk-forward out-of-sample test.

Strategies (each keeps its OWN portfolio, which drifts between rebalances):
  DoNothing    never trades (starts from the current portfolio in config)
  EqualWeight  rebalances to 1/N every period (benchmark; ignores the 10-stock limit)
  WeightedSum  baseline front -> one portfolio picked by the selection rule
  NSGA2        tuned NSGA-II + archive (no convex solver) -> pick
  Hybrid       tuned NSGA-II + archive + CVXPY polish -> pick

Per period: estimate mu / Sigma from the PAST window only, build that strategy's front,
pick one portfolio, pay the cost, hold, then measure the REAL return on the next `horizon`
days.  The evolutionary strategies are stochastic, so we run several independent paths
(different seeds) and report the spread.

Outputs: results/<tag>/walkforward_<rule>/  (per_period.csv, summary.csv, wealth.csv, stats.json)
         figures/<tag>/walkforward_wealth_<rule>.png, walkforward_pred_vs_real_<rule>.png

Synthetic-data runs are development runs only.  Real-data results come from a single
historical sample of about 15 quarters: treat differences with great caution.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms.nsga2 import run_nsga2                          # noqa: E402
from src.baseline.weighted_sum_relax_round import run_weighted_sum  # noqa: E402
from src.data_loader import build_problem, load_config             # noqa: E402
from src.hybrid.polish import polish_front                          # noqa: E402
from src.walk_forward import (TRADING_DAYS, knee_index, path_metrics, period_inputs,  # noqa: E402
                              period_problem, rebalance_schedule, simulate_period)

STRATEGIES = ["DoNothing", "EqualWeight", "WeightedSum", "NSGA2", "Hybrid"]
_CTX = {}


def _init(cfg, source, rule, horizon, evals, max_periods):
    prob, prices, label = build_problem(ROOT, cfg, source=source)
    _CTX.update(cfg=cfg, base=prob, prices=prices[prob.tickers], label=label, rule=rule,
                horizon=horizon, evals=evals, max_periods=max_periods)


def _choose(strategy, prob, seed):
    """Build the strategy's front for this period's problem and pick one portfolio."""
    cfg, rule, horizon = _CTX["cfg"], _CTX["rule"], _CTX["horizon"]
    nb, ex = cfg["nsga2"], cfg["experiment"]
    if strategy == "WeightedSum":
        r = run_weighted_sum(prob, step=ex["ws_step"])
        X, F = r["X"], r["F"]
    else:
        r = run_nsga2(prob, pop_size=nb["pop_size"], n_evals=_CTX["evals"], seed=seed,
                      eta_c=nb["eta_c"], eta_m=nb["eta_m"], pc=nb["pc"],
                      checkpoint_every=ex["checkpoint_every_evals"], inject_current=True,
                      sparse_init=True, reset_prob=nb["reset_prob"], archive_size=nb["archive_size"])
        X, F = r["X"], r["F"]
        if strategy == "Hybrid":
            X, F, _ = polish_front(prob, X, F)
    i = knee_index(F, rule, horizon)
    return X[i]


def _run_path(path):
    cfg, base, prices = _CTX["cfg"], _CTX["base"], _CTX["prices"]
    horizon, window = _CTX["horizon"], cfg["data"]["estimation_days"]
    n = base.n
    sched = rebalance_schedule(len(prices), window, horizon)
    if _CTX["max_periods"]:
        sched = sched[:_CTX["max_periods"]]
    x_start = base.x_old.copy()
    state = {s: {"x": x_start.copy(), "wealth": 1.0} for s in STRATEGIES}
    wealth = {s: {sched[0][0]: 1.0} for s in STRATEGIES}       # row index -> wealth
    rows = []
    for p, (t, t_end) in enumerate(sched):
        mu, cov = period_inputs(prices, t, window, cfg["data"]["cov_estimator"])
        P = prices.iloc[t:t_end + 1].to_numpy()
        for s in STRATEGIES:
            x_old = state[s]["x"]
            prob = period_problem(base, mu, cov, x_old)
            if s == "DoNothing":
                x_new = x_old.copy()
            elif s == "EqualWeight":
                x_new = np.full(n, 1.0 / n)
            else:
                seed = cfg["experiment"]["base_seed"] + 1000 * path + p
                x_new = _choose(s, prob, seed)
            cost = float(prob.cost(x_new)[0])
            w0 = state[s]["wealth"] * (1.0 - cost)
            daily, x_end = simulate_period(P, x_new)
            curve = w0 * np.cumprod(1.0 + daily)
            wealth[s][t] = w0
            for k, v in enumerate(curve, start=1):
                wealth[s][t + k] = float(v)
            period_ret = float(curve[-1] / state[s]["wealth"] - 1.0)
            rows.append({"path": path, "period": p, "strategy": s, "rebalance_row": t,
                         "start_date": str(prices.index[t].date()), "end_date": str(prices.index[t_end].date()),
                         "period_return": period_ret,
                         "predicted_ann_return": float(mu @ x_new),
                         "realised_ann_return": float((curve[-1] / w0) ** (TRADING_DAYS / horizon) - 1.0),
                         "cost": cost, "turnover": float(0.5 * np.abs(x_new - x_old).sum()),
                         "n_holdings": int((x_new > 1e-8).sum())})
            state[s]["x"], state[s]["wealth"] = x_end, float(curve[-1])
    wrows = [{"path": path, "strategy": s, "row": r, "date": str(prices.index[r].date()), "wealth": v}
             for s in STRATEGIES for r, v in sorted(wealth[s].items())]
    return path, rows, wrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="synthetic | real (default: config)")
    ap.add_argument("--paths", type=int, default=5, help="independent runs of the stochastic strategies")
    ap.add_argument("--horizon", type=int, default=63, help="trading days between rebalances (63 = quarter)")
    ap.add_argument("--rule", default="knee", choices=["knee", "ratio"])
    ap.add_argument("--evals", type=int, default=None)
    ap.add_argument("--max-periods", type=int, default=None, help="only for quick tests")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    cfg = load_config(ROOT / "config" / "default.json")
    evals = a.evals or cfg["nsga2"]["n_evals"]
    _, _, label = build_problem(ROOT, cfg, source=a.source)
    tag = a.tag or label
    out = ROOT / cfg["output_dir"] / tag / f"walkforward_{a.rule}"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "figures" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data source = {label}] rule={a.rule} horizon={a.horizon}d paths={a.paths} evals={evals}")

    t0 = time.perf_counter()
    with Pool(a.workers, initializer=_init,
              initargs=(cfg, a.source, a.rule, a.horizon, evals, a.max_periods)) as pool:
        res = pool.map(_run_path, list(range(a.paths)), chunksize=1)
    print(f"walk-forward finished in {time.perf_counter() - t0:.0f}s")

    per = pd.DataFrame([r for _, rows, _ in res for r in rows])
    wdf = pd.DataFrame([r for _, _, w in res for r in w])
    per.to_csv(out / "per_period.csv", index=False)
    wdf.to_csv(out / "wealth.csv", index=False)

    # ---- per-path metrics, then mean/sd across paths
    recs = []
    for (s, pth), g in wdf.groupby(["strategy", "path"]):
        m = path_metrics(g.sort_values("row").wealth.to_numpy())
        pp = per[(per.strategy == s) & (per.path == pth)]
        m.update(strategy=s, path=pth, total_cost=float(pp.cost.sum()),
                 mean_turnover=float(pp.turnover.mean()), mean_holdings=float(pp.n_holdings.mean()),
                 mean_pred_ann=float(pp.predicted_ann_return.mean()),
                 mean_real_ann=float(pp.realised_ann_return.mean()))
        recs.append(m)
    pm = pd.DataFrame(recs)
    pm.to_csv(out / "per_path_metrics.csv", index=False)
    cols = ["final_wealth", "ann_return", "ann_vol", "return_over_vol", "max_drawdown", "total_cost",
            "mean_turnover", "mean_holdings", "mean_pred_ann", "mean_real_ann"]
    summ = pm.groupby("strategy")[cols].agg(["mean", "std"]).round(4).reindex(STRATEGIES)
    summ.to_csv(out / "summary.csv")

    # ---- paired per-period comparison (mean over paths first), with honest uncertainty
    pr = per.groupby(["strategy", "period"]).period_return.mean().unstack(0)
    stats = {"data_source": label, "rule": a.rule, "horizon_days": a.horizon, "paths": a.paths,
             "n_periods": int(pr.shape[0]), "pairs": {}}
    for A, B in [("Hybrid", "WeightedSum"), ("Hybrid", "EqualWeight"), ("Hybrid", "DoNothing"),
                 ("WeightedSum", "EqualWeight"), ("NSGA2", "WeightedSum")]:
        d = (pr[A] - pr[B]).to_numpy()
        stats["pairs"][f"{A}_minus_{B}"] = {
            "mean_period_return_diff": float(d.mean()),
            "std_error": float(d.std(ddof=1) / np.sqrt(len(d))),
            "periods_A_better": int((d > 0).sum()), "n_periods": int(len(d))}
    json.dump(stats, open(out / "stats.json", "w"), indent=2)

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"DoNothing": "#7f7f7f", "EqualWeight": "#1f77b4", "WeightedSum": "#000000",
              "NSGA2": "#2ca02c", "Hybrid": "#d62728"}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for s in STRATEGIES:
        g = wdf[wdf.strategy == s].groupby("date").wealth.agg(["mean", "min", "max"])
        x = pd.to_datetime(g.index)
        ax.plot(x, g["mean"], color=colors[s], label=s, lw=1.6)
        ax.fill_between(x, g["min"], g["max"], color=colors[s], alpha=0.15)
    ax.set_ylabel("wealth (start = 1)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"Walk-forward wealth, rule={a.rule}, rebalance every {a.horizon} days [{label.upper()} data]")
    fig.tight_layout(); fig.savefig(fig_dir / f"walkforward_wealth_{a.rule}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = [s for s in STRATEGIES if s not in ("DoNothing",)]
    xx = np.arange(len(sub))
    pred = [pm[pm.strategy == s].mean_pred_ann.mean() for s in sub]
    real = [pm[pm.strategy == s].mean_real_ann.mean() for s in sub]
    ax.bar(xx - 0.2, pred, 0.4, label="predicted (from past data)", color="#9ecae1")
    ax.bar(xx + 0.2, real, 0.4, label="realised (next quarter)", color="#08519c")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xx); ax.set_xticklabels(sub)
    ax.set_ylabel("mean annualised return"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"Predicted vs realised return [{label.upper()} data]")
    fig.tight_layout(); fig.savefig(fig_dir / f"walkforward_pred_vs_real_{a.rule}.png", dpi=150); plt.close(fig)

    show = summ[[("final_wealth", "mean"), ("ann_return", "mean"), ("ann_vol", "mean"),
                 ("return_over_vol", "mean"), ("max_drawdown", "mean"), ("total_cost", "mean")]]
    print(show.to_string())
    print(json.dumps(stats["pairs"], indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
