"""PortfolioPilot entry point.

  python main.py data                   # (re)build the input data summary (synthetic unless real CSV exists)
  python main.py experiment [--seeds N] # run all methods, write results/<tag>/
  python main.py figures [--tag T]      # make figures/ from results/<tag>/
  python main.py inspect [--tag T] [--method M] [--pick knee|maxret|minvol|minpath]
                                        # print one portfolio: weights, trades, return, risk, cost
  python main.py all                    # data check + experiment + figures
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import build_problem, load_config  # noqa: E402


def cmd_data(args):
    cfg = load_config(ROOT / "config" / "default.json")
    prob, prices, label = build_problem(ROOT, cfg, source=args.source)
    print(f"data source: {label.upper()}   prices: {prices.shape}   assets: {prob.n}")
    print(f"annual return range {prob.mu.min():.3f}..{prob.mu.max():.3f}   "
          f"annual vol range {np.sqrt(np.diag(prob.cov)).min():.3f}..{np.sqrt(np.diag(prob.cov)).max():.3f}")
    print("current portfolio:", {t: w for t, w in zip(prob.tickers, prob.x_old) if w > 0})


def cmd_experiment(args):
    cmd = [sys.executable, str(ROOT / "experiments" / "run_experiments.py"), "--workers", str(args.workers)]
    if args.seeds: cmd += ["--seeds", str(args.seeds)]
    if args.evals: cmd += ["--evals", str(args.evals)]
    if args.source: cmd += ["--source", args.source]
    if args.tag: cmd += ["--tag", args.tag]
    subprocess.check_call(cmd)


def cmd_figures(args):
    from src.visualization import make_all
    made = make_all(ROOT / "results" / args.tag, ROOT / "figures" / args.tag, args.tag)
    print("figures written to figures/%s:" % args.tag, made)


def cmd_inspect(args):
    cfg = load_config(ROOT / "config" / "default.json")
    prob, _, label = build_problem(ROOT, cfg, source=None if args.tag == "default" else args.tag)
    d = pd.read_csv(ROOT / "results" / args.tag / f"best_front_{args.method.replace('+', '_')}.csv")
    if args.pick == "maxret":
        row = d.sort_values("return").iloc[-1]
    elif args.pick == "minvol":
        row = d.sort_values("volatility").iloc[0]
    elif args.pick == "minpath":
        row = d.sort_values("cost").iloc[0]
    else:   # knee: closest to the ideal point after min-max scaling
        z = (d[["return", "volatility", "cost"]] - d[["return", "volatility", "cost"]].min()) / \
            (d[["return", "volatility", "cost"]].max() - d[["return", "volatility", "cost"]].min() + 1e-12)
        dist = np.sqrt(((1 - z["return"]) ** 2) + z["volatility"] ** 2 + z["cost"] ** 2)
        row = d.loc[dist.idxmin()]
    w = row[prob.tickers].values.astype(float)
    print(f"[{label.upper()} data] method={args.method} pick={args.pick}")
    print(f"expected return {row['return']:.4f} | volatility {row['volatility']:.4f} | rebalancing cost {row['cost']:.5f}")
    t = pd.DataFrame({"ticker": prob.tickers, "sector": prob.sectors, "current": prob.x_old,
                      "target": w, "trade": w - prob.x_old})
    print(t[(t.current > 0) | (t.target > 1e-8)].sort_values("target", ascending=False).round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data"); p.add_argument("--source", default=None); p.set_defaults(f=cmd_data)
    p = sub.add_parser("experiment")
    p.add_argument("--seeds", type=int); p.add_argument("--evals", type=int)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--source"); p.add_argument("--tag")
    p.set_defaults(f=cmd_experiment)
    p = sub.add_parser("figures"); p.add_argument("--tag", default="synthetic"); p.set_defaults(f=cmd_figures)
    p = sub.add_parser("inspect"); p.add_argument("--tag", default="synthetic")
    p.add_argument("--method", default="Hybrid"); p.add_argument("--pick", default="knee",
                    choices=["knee", "maxret", "minvol", "minpath"]); p.set_defaults(f=cmd_inspect)
    p = sub.add_parser("all"); p.add_argument("--seeds", type=int); p.add_argument("--evals", type=int)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--source"); p.add_argument("--tag")
    p.set_defaults(f=lambda a: (cmd_data(a), cmd_experiment(a),
                                cmd_figures(argparse.Namespace(tag=a.tag or "synthetic"))))
    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
