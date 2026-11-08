"""Figures for PortfolioPilot, generated from results/<tag>/ by code (never hand-made)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"WeightedSum": "#000000", "RandomSearch": "#9e9e9e", "NSGA2": "#1f77b4",
          "NSGA2+current": "#17becf", "NSGA2-tuned": "#2ca02c", "Hybrid": "#d62728",
          "NSGA2-tuned+archive": "#98df8a", "Hybrid+archive": "#ff7f0e"}
ORDER = ["RandomSearch", "NSGA2", "NSGA2+current", "NSGA2-tuned", "NSGA2-tuned+archive", "Hybrid",
         "Hybrid+archive", "WeightedSum"]


def _load_front(res_dir, m):
    return pd.read_csv(res_dir / f"best_front_{m.replace('+', '_')}.csv")


def make_all(res_dir, fig_dir, label):
    res_dir, fig_dir = Path(res_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    note = f"[{label.upper()} data]"
    refs = json.load(open(res_dir / "reference_points.json"))
    dn, ew = refs["do_nothing"], refs["equal_weight_1_over_N"]
    methods = ["WeightedSum", "RandomSearch", "NSGA2", "NSGA2-tuned", "Hybrid", "Hybrid+archive"]
    fronts = {m: _load_front(res_dir, m) for m in methods}

    # 1) 2-D projections of the 3-D Pareto fronts
    pairs = [("volatility", "return"), ("cost", "return"), ("cost", "volatility")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (xa, ya) in zip(axs, pairs):
        for m in methods:
            d = fronts[m]
            ax.scatter(d[xa], d[ya], s=14 if m != "WeightedSum" else 22, alpha=0.7,
                       color=COLORS[m], label=m, marker="x" if m == "WeightedSum" else "o")
        idx = {"return": 0, "volatility": 1, "cost": 2}
        ax.scatter(-1 if False else (-dn[0] if xa == "return" else dn[idx[xa]]),
                   (-dn[0] if ya == "return" else dn[idx[ya]]), marker="*", s=160, color="gold",
                   edgecolor="k", zorder=5, label="Do nothing (current)")
        ax.scatter(-ew[0] if xa == "return" else ew[idx[xa]],
                   -ew[0] if ya == "return" else ew[idx[ya]], marker="D", s=70, color="orange",
                   edgecolor="k", zorder=5, label="Equal weight 1/N")
        ax.set_xlabel(xa + (" (annual)" if xa != "cost" else " (fraction of capital)"))
        ax.set_ylabel(ya + (" (annual)" if ya != "cost" else " (fraction of capital)"))
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Pareto fronts, best-HV run per method {note}")
    fig.tight_layout(); fig.savefig(fig_dir / "pareto_fronts_2d.png", dpi=150); plt.close(fig)

    # 2) 3-D view
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    for m in ["WeightedSum", "NSGA2", "Hybrid", "Hybrid+archive"]:
        d = fronts[m]
        ax.scatter(d["volatility"], d["cost"], d["return"], s=10, color=COLORS[m], label=m, alpha=0.7)
    ax.set_xlabel("volatility"); ax.set_ylabel("cost"); ax.set_zlabel("return")
    ax.legend(fontsize=8); ax.set_title(f"3-D Pareto fronts {note}")
    fig.tight_layout(); fig.savefig(fig_dir / "pareto_fronts_3d.png", dpi=150); plt.close(fig)

    # 3) Hypervolume boxplot over seeds, WS as horizontal line
    df = pd.read_csv(res_dir / "per_seed_metrics.csv")
    st = df[df.method != "WeightedSum"]
    ms = [m for m in ORDER if m in st.method.unique()]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([st[st.method == m].hv.values for m in ms], showmeans=True)
    ax.set_xticklabels(ms)
    ws_hv = df[df.method == "WeightedSum"].hv.iloc[0]
    ax.axhline(ws_hv, color="k", ls="--", label=f"WeightedSum baseline ({ws_hv:.3f})")
    ax.set_ylabel("hypervolume (normalised, ref 1.1)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Hypervolume over seeds {note}")
    plt.xticks(rotation=15); fig.tight_layout(); fig.savefig(fig_dir / "hv_boxplot.png", dpi=150); plt.close(fig)

    # 4) Convergence
    cv = pd.read_csv(res_dir / "convergence.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in ms:
        d = cv[cv.method == m]
        ax.plot(d.evals, d.hv_mean, color=COLORS[m], label=m)
        ax.fill_between(d.evals, d.hv_mean - d.hv_std, d.hv_mean + d.hv_std, color=COLORS[m], alpha=0.15)
    ax.axhline(ws_hv, color="k", ls="--", label="WeightedSum")
    ax.set_xlabel("function evaluations"); ax.set_ylabel("mean hypervolume"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8); ax.set_title(f"Convergence (mean ± sd over seeds) {note}")
    fig.tight_layout(); fig.savefig(fig_dir / "convergence.png", dpi=150); plt.close(fig)
    return sorted(p.name for p in fig_dir.glob("*.png"))
