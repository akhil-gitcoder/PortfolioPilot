# PortfolioPilot
**Multi-Objective Portfolio Allocation and Rebalancing under Return, Risk, and Transaction-Cost Constraints**
Optimization Techniques group project — Akhil, Ridhima.

> **STATUS:** Results are reported on SYNTHETIC data (development) and REAL NSE data
> (Yahoo Finance via yfinance, see `data/DATA.md`). Real-data results are a historical
> snapshot, not a forecast. Out-of-sample and cost-sensitivity tests are still to do.

## 1. Problem statement
Given a current portfolio `x_old` over 30 NSE stocks, choose new weights `x` that trade off
**maximise expected return**, **minimise volatility**, **minimise one-off rebalancing cost** (3 objectives, all minimised internally as `[-return, vol, cost]`).
See `problem/problem_statement.md` and `formulation/formulation.md`.

## 2. Constraints
Fully invested (`sum x = 1`), long-only, `x_i <= 0.20`, sector cap 40%, at most K=10 holdings, held weight >= 3%.
(Non-convex because of cardinality + minimum holding.) All parameters in `config/default.json`.

## 3. Data
`data/DATA.md`. Development uses a **synthetic** factor-model generator (`src/synthetic_data.py`). Real data: `scripts/download_data.py` (yfinance, run locally). Cost rates (buy 0.12%, sell 0.13%) are **placeholder assumptions** to verify against a broker charge sheet.

## 4. Methods
| File | Role |
|---|---|
| `src/algorithms/nsga2.py` | NSGA-II from scratch (Deb 2002): constrained dominance, fast non-dominated sort, crowding, SBX, polynomial mutation, random-key encoding + repair |
| `src/baseline/random_search.py` | Baseline 1: random search, same decoder & budget |
| `src/baseline/weighted_sum_relax_round.py` | Baseline 2 (the one to beat): weighted-sum convex relaxation -> round top-K -> polish (CVXPY/CLARABEL) |
| `src/hybrid/polish.py` | Hybrid: exact convex weight polish of an NSGA-II front |
| `src/reference/exact_front.py` | Exact front of the convex model (validation only) |
| `src/metrics.py`, `src/multiobjective/pareto.py` | Hypervolume (exact 2D/3D), IGD, spacing, C-metric, dominance tools |

NSGA-II variants (ablation): `NSGA2` plain; `NSGA2+current` (current portfolio injected); `NSGA2-tuned` (+ sparse initial supports + support-reset mutation); `Hybrid` (`NSGA2-tuned` + polish). Tuning used development seeds 1-3; final runs use seeds 1000-1029.

## 5. Libraries used (as allowed; with configuration)
NumPy, SciPy (Mann-Whitney U), pandas, matplotlib, scikit-learn (`LedoitWolf` covariance), **CVXPY** with solver **CLARABEL** (baseline, polish, exact reference front), yfinance (data download only). NSGA-II, metrics, repair are our own code. No pymoo.

## 6. How to run (fresh clone)
```
pip install -r requirements.txt
python -m pytest -q tests          # 22 tests
python main.py all                 # data check + 30-seed experiment + figures   (~3 min, 4 cores)
python main.py inspect --method Hybrid --pick knee   # one portfolio: weights, trades, return, risk, cost
```
Real data: `python scripts/download_data.py` then `python main.py all --source real --tag real`.

## 7. Experimental design
30 seeds, 10,000 evaluations each, population 100; WS baseline deterministic (run once). Hypervolume with pooled min-max normalisation, reference point 1.1; IGD to pooled best-known front; spacing; C-metric; Mann-Whitney U. Validation on the convex model vs the exact front (10 seeds).

## 8. Results so far (SYNTHETIC data, `results/synthetic/`)
| Method | HV mean ± sd | IGD | runtime (s) |
|---|---|---|---|
| RandomSearch | 0.505 ± 0.015 | 0.239 | 1.9 |
| NSGA2 (plain) | 0.537 ± 0.028 | 0.230 | 1.7 |
| NSGA2+current | 0.671 ± 0.018 | 0.139 | 1.9 |
| NSGA2-tuned | 0.806 ± 0.010 | 0.071 | 1.9 |
| **Hybrid** | **0.885 ± 0.007** | **0.038** | 9.9 |
| WeightedSum baseline | 0.887 | 0.059 | 1.5 |

Honest reading: plain NSGA-II beats random search (p = 5e-6) but is far below the weighted-sum baseline; each added component helps significantly (all p < 1e-10). The **Hybrid ties the baseline on hypervolume (0.885 vs 0.887; it wins on 12 of 30 seeds), is better on IGD, but is ~6.5x slower and by the C-metric the baseline covers 22% of the Hybrid front while the Hybrid covers 0% of the baseline front**. So on this synthetic data we **cannot claim** NSGA-II beats the baseline. Validation: tuned NSGA-II recovers ~85% of the exact front's hypervolume on the convex model. Next steps are in section 12.

### Results on real data (NSE, Yahoo Finance via yfinance, downloaded <DATE>)
Setup: 30 stocks, 30 seeds, 10,000 evaluations per run, estimation window = first 756
trading days, buy cost 0.119% and sell cost 0.104% (estimated from the standard charge
structure with zero brokerage; see `data/DATA.md`).

| Method | HV mean ± sd | IGD | Runtime (s) |
|---|---|---|---|
| RandomSearch | 0.516 ± 0.009 | 0.231 | 0.36 |
| NSGA2 (plain) | 0.532 ± 0.027 | 0.228 | 0.35 |
| NSGA2+current | 0.674 ± 0.021 | 0.152 | 0.36 |
| NSGA2-tuned | 0.801 ± 0.010 | 0.069 | 0.43 |
| Hybrid | 0.881 ± 0.006 | 0.037 | 2.28 |
| WeightedSum baseline | 0.877 | 0.067 | 0.62 |

The Hybrid's hypervolume is slightly higher than the weighted-sum baseline (0.881 vs 0.877),
higher in 21 of 30 seeds (sign test p = 0.043, Wilcoxon p = 0.0005), with better IGD.
The margin is small, the Hybrid is about 3.7x slower, and the p-values only reflect
seed-to-seed randomness, not market uncertainty. Tuning was done on synthetic data only.

## 9. Figures
`figures/synthetic/`: `pareto_fronts_2d.png`, `pareto_fronts_3d.png`, `hv_boxplot.png`, `convergence.png` (generated by `src/visualization.py`).

## 10. Limitations
Synthetic data only; placeholder costs; sample-mean returns are noisy; no walk-forward out-of-sample test yet; tuning on the same synthetic problem may not transfer to real data.

## 11. Reproducibility
Fixed seeds in `config/default.json`; every figure/table is generated by code.

## 12. Plan / honest open issues
(1) run on real NSE data; (2) try to win without the solver crutch: identity-preserving encoding, external archive, bigger budget; (3) walk-forward out-of-sample evaluation; (4) report + slides; (5) Streamlit UI in `frontend/` only after the CLI is final.

## 13. Contributions & references
Per-member contributions: `report/contributions.md` (to be filled with real git history). References: `literature/references.md`.
