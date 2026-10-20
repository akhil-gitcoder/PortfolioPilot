"""Main experiment: compare methods over many seeds on the realistic (non-convex) model,
plus a validation experiment on the convex model against the exact front.

Methods (all use the SAME problem, decoder and constraints):
  RandomSearch        random keys, same decoder, same evaluation budget as NSGA-II
  NSGA2               plain NSGA-II, random-key encoding
  NSGA2+current       NSGA-II with the current portfolio injected in generation 0
  NSGA2-tuned         + sparse initial supports + support-reset mutation (tuned on dev seeds 1-3)
  Hybrid              NSGA2-tuned, then: the evolved front is polished by exact convex solves (CVXPY)
  WeightedSum         relax-and-round baseline (deterministic -> run once)

Outputs (results/<tag>/): per_seed_metrics.csv, summary.csv, summary.md, stats.json,
best_front_*.csv, convergence.csv, validation_convex.json, raw/*.npz (git-ignored).
Results are labelled with the DATA SOURCE (synthetic vs real).  Synthetic results are
development results, never findings about the real market.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms.nsga2 import run_nsga2                       # noqa: E402
from src.baseline.random_search import run_random_search         # noqa: E402
from src.baseline.weighted_sum_relax_round import run_weighted_sum  # noqa: E402
from src.data_loader import build_problem, load_config           # noqa: E402
from src.hybrid.polish import polish_front                       # noqa: E402
from src.metrics import hypervolume, igd, normalize, spacing, c_metric  # noqa: E402
from src.multiobjective.pareto import nondominated_mask          # noqa: E402
from src.reference.exact_front import exact_front                # noqa: E402

STOCHASTIC = ["RandomSearch", "NSGA2", "NSGA2+current", "NSGA2-tuned", "Hybrid"]
_CTX = {}


def _init(cfg, source):
    _CTX["cfg"] = cfg
    _CTX["prob"], _, _CTX["label"] = build_problem(ROOT, cfg, source=source)


def _run_one(args):
    method, seed = args
    prob, cfg = _CTX["prob"], _CTX["cfg"]
    nb, ex = cfg["nsga2"], cfg["experiment"]
    n_evals, ck = nb["n_evals"], ex["checkpoint_every_evals"]
    kw = dict(pop_size=nb["pop_size"], n_evals=n_evals, seed=seed, eta_c=nb["eta_c"],
              eta_m=nb["eta_m"], pc=nb["pc"], checkpoint_every=ck)
    t0 = time.perf_counter()
    if method == "RandomSearch":
        r = run_random_search(prob, n_evals=n_evals, seed=seed, checkpoint_every=ck)
    elif method == "NSGA2":
        r = run_nsga2(prob, **kw)
    elif method == "NSGA2+current":
        r = run_nsga2(prob, inject_current=True, **kw)
    elif method == "NSGA2-tuned":
        r = run_nsga2(prob, inject_current=True, sparse_init=True, reset_prob=nb["reset_prob"], **kw)
    elif method == "Hybrid":
        r = run_nsga2(prob, inject_current=True, sparse_init=True, reset_prob=nb["reset_prob"], **kw)
        X, F, _ = polish_front(prob, r["X"], r["F"])
        r["X"], r["F"] = X, F
        r["history"].append((n_evals, F.copy()))      # final point includes polish
    else:
        raise ValueError(method)
    r["runtime"] = time.perf_counter() - t0
    return method, seed, r


def _metrics_row(F, lo, hi, ref, best_known_norm):
    Fn = normalize(F, lo, hi)
    return {"hv": hypervolume(Fn, ref), "igd": igd(best_known_norm, Fn),
            "spacing": spacing(Fn), "n_nd": len(F)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="synthetic | real (default: config)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--evals", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-validation", action="store_true")
    a = ap.parse_args()

    cfg = load_config(ROOT / "config" / "default.json")
    if a.seeds:
        cfg["experiment"]["seeds"] = a.seeds
    if a.evals:
        cfg["nsga2"]["n_evals"] = a.evals
    ex = cfg["experiment"]
    n_seeds = ex["seeds"]
    prob, _, label = build_problem(ROOT, cfg, source=a.source)
    tag = a.tag or label
    out = ROOT / cfg["output_dir"] / tag
    (out / "raw").mkdir(parents=True, exist_ok=True)
    print(f"[data source = {label}]  seeds={n_seeds}  evals={cfg['nsga2']['n_evals']}")

    seeds = [ex["base_seed"] + i for i in range(n_seeds)]
    jobs = [(m, s) for m in STOCHASTIC for s in seeds]
    with Pool(a.workers, initializer=_init, initargs=(cfg, a.source)) as pool:
        done = pool.map(_run_one, jobs, chunksize=1)
    runs = {(m, s): r for m, s, r in done}

    t0 = time.perf_counter()
    ws = run_weighted_sum(prob, step=ex["ws_step"])
    ws_time = time.perf_counter() - t0
    fronts = {("WeightedSum", 0): ws["F"]}
    fronts.update({k: r["F"] for k, r in runs.items()})

    # ---- pooled normalisation and best-known front
    allF = np.vstack(list(fronts.values()))
    lo, hi = allF.min(0), allF.max(0)
    ref = np.full(3, ex["hv_ref"])
    bk = allF[nondominated_mask(allF)]
    bk_n = normalize(bk, lo, hi)

    rows = []
    for (m, s), F in fronts.items():
        row = {"method": m, "seed": s, **_metrics_row(F, lo, hi, ref, bk_n)}
        row["runtime_s"] = ws_time if m == "WeightedSum" else runs[(m, s)]["runtime"]
        row["max_violation"] = float(prob.violation(
            ws["X"] if m == "WeightedSum" else runs[(m, s)]["X"]).max())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed_metrics.csv", index=False)

    summ = df.groupby("method").agg(
        hv_mean=("hv", "mean"), hv_std=("hv", "std"), hv_median=("hv", "median"),
        igd_mean=("igd", "mean"), spacing_mean=("spacing", "mean"),
        n_nd_mean=("n_nd", "mean"), runtime_mean_s=("runtime_s", "mean"),
        max_violation=("max_violation", "max")).round(5)
    summ.to_csv(out / "summary.csv")

    # ---- statistics
    stats = {"data_source": label, "n_seeds": n_seeds, "n_evals": cfg["nsga2"]["n_evals"],
             "ws_hv": float(df[df.method == "WeightedSum"].hv.iloc[0])}

    def hv_of(m):
        return df[df.method == m].hv.values
    for m in STOCHASTIC:
        stats[f"seeds_{m}_hv_greater_than_WS"] = int((hv_of(m) > stats["ws_hv"]).sum())
    pairs = [("NSGA2", "RandomSearch"), ("NSGA2+current", "NSGA2"), ("NSGA2-tuned", "NSGA2+current"),
             ("Hybrid", "NSGA2-tuned")]
    for A, B in pairs:
        u = mannwhitneyu(hv_of(A), hv_of(B), alternative="two-sided")
        stats[f"MWU_{A}_vs_{B}"] = {"U": float(u.statistic), "p": float(u.pvalue),
                                    "median_diff": float(np.median(hv_of(A)) - np.median(hv_of(B)))}
    # C-metric of WS vs each method's median-HV run and vice versa
    cm = {}
    for m in STOCHASTIC:
        sub = df[df.method == m].sort_values("hv")
        med_seed = int(sub.iloc[len(sub) // 2].seed)
        Fm = fronts[(m, med_seed)]
        cm[m] = {"C(WS,method)": c_metric(ws["F"], Fm), "C(method,WS)": c_metric(Fm, ws["F"])}
    stats["c_metric_median_run"] = cm
    json.dump(stats, open(out / "stats.json", "w"), indent=2)

    # ---- convergence (HV vs evaluations, mean over seeds)
    conv = []
    for m in STOCHASTIC:
        per_seed = {}
        for s in seeds:
            for ev, F in runs[(m, s)]["history"]:
                per_seed.setdefault(ev, []).append(hypervolume(normalize(F, lo, hi), ref))
        for ev in sorted(per_seed):
            conv.append({"method": m, "evals": ev, "hv_mean": np.mean(per_seed[ev]),
                         "hv_std": np.std(per_seed[ev])})
    pd.DataFrame(conv).to_csv(out / "convergence.csv", index=False)

    # ---- fronts (best-HV run per method) with portfolio weights
    names = prob.tickers
    for m in ["WeightedSum"] + STOCHASTIC:
        if m == "WeightedSum":
            X, F = ws["X"], ws["F"]
        else:
            s_best = int(df[df.method == m].sort_values("hv").iloc[-1].seed)
            X, F = runs[(m, s_best)]["X"], runs[(m, s_best)]["F"]
        d = pd.DataFrame(X, columns=names)
        d.insert(0, "cost", F[:, 2]); d.insert(0, "volatility", F[:, 1]); d.insert(0, "return", -F[:, 0])
        d.sort_values("return").to_csv(out / f"best_front_{m.replace('+', '_')}.csv", index=False)
    np.savez_compressed(out / "raw" / "fronts.npz", lo=lo, hi=hi,
                        **{f"{m}__{s}": F for (m, s), F in fronts.items()})
    # reference points for plots
    ref_pts = {"do_nothing": prob.objectives(prob.x_old)[0].tolist(),
               "equal_weight_1_over_N": prob.objectives(np.full(prob.n, 1 / prob.n))[0].tolist()}
    json.dump(ref_pts, open(out / "reference_points.json", "w"), indent=2)

    # ---- validation on the convex model: does NSGA-II recover the exact front?
    if not a.skip_validation:
        vprob, _, _ = build_problem(ROOT, cfg, source=a.source, convex=True)
        ex_front = exact_front(vprob)
        vruns = [run_nsga2(vprob, pop_size=cfg["nsga2"]["pop_size"], n_evals=cfg["nsga2"]["n_evals"],
                           seed=ex["base_seed"] + i, inject_current=True,
                           sparse_init=True, reset_prob=cfg["nsga2"]["reset_prob"])
                 for i in range(ex["validation_seeds"])]
        allv = np.vstack([ex_front["F"]] + [r["F"] for r in vruns])
        vlo, vhi = allv.min(0), allv.max(0)
        ex_n = normalize(ex_front["F"], vlo, vhi)
        hv_ex = hypervolume(ex_n, ref)
        val = {"exact_points": len(ex_front["F"]), "exact_hv": hv_ex,
               "nsga2_hv_over_exact_hv": [hypervolume(normalize(r["F"], vlo, vhi), ref) / hv_ex for r in vruns],
               "nsga2_igd_to_exact": [igd(ex_n, normalize(r["F"], vlo, vhi)) for r in vruns]}
        val["mean_hv_ratio"] = float(np.mean(val["nsga2_hv_over_exact_hv"]))
        val["mean_igd"] = float(np.mean(val["nsga2_igd_to_exact"]))
        json.dump(val, open(out / "validation_convex.json", "w"), indent=2)
        print("validation (convex): mean HV ratio %.4f, mean IGD %.4f" % (val["mean_hv_ratio"], val["mean_igd"]))

    md = [f"# Results ({label} data)\n",
          f"Seeds: {n_seeds}, evaluation budget: {cfg['nsga2']['n_evals']}. "
          f"Hypervolume with pooled min-max normalisation, reference point {ex['hv_ref']}.\n",
          summ.to_markdown(), "\n", "```json", json.dumps(stats, indent=2), "```"]
    (out / "summary.md").write_text("\n".join(md))
    print(summ.to_string())
    print(json.dumps({k: v for k, v in stats.items() if k.startswith(("seeds_", "ws_"))}, indent=2))


if __name__ == "__main__":
    main()
