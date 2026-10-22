# Weekly progress log

> Data status: all results so far use **SYNTHETIC** data (labelled in every file/figure).
> Authorship: each member fills their own rows and commits under their own GitHub account.

## Week 1-2 (mid-project checkpoint)

### Done
- Formulated the 3-objective constrained problem (return, volatility, rebalancing cost) -> `formulation/formulation.md`
- Data layer: Ledoit-Wolf covariance, 756-day estimation window (no look-ahead), synthetic generator, real-data download script
- Problem model + constraint repair; 22 unit tests incl. a hand-calculated example
- NSGA-II from scratch (constrained dominance, sorting, crowding, SBX, polynomial mutation)
- Baselines: random search; weighted-sum relax-and-round (CVXPY/CLARABEL)
- Hybrid: NSGA-II selects stocks, convex solver polishes weights
- Metrics: hypervolume (exact 2D/3D), IGD, spacing, C-metric, Mann-Whitney U
- 30-seed experiment, convex-model validation vs exact front, figures, `main.py` CLI

### Results (synthetic, 30 seeds, 10,000 evals)
| Method | HV |
|---|---|
| Random search | 0.505 |
| NSGA-II plain | 0.537 |
| NSGA-II tuned | 0.806 |
| Hybrid | 0.885 |
| Weighted-sum baseline | 0.887 |

Finding: the Hybrid ties the baseline; we do not claim it beats it. Tuned NSGA-II recovers ~85% of the exact front's HV on the convex model.

### Problems met
- Yahoo Finance blocked in the dev sandbox -> synthetic data + local download script
- Plain NSGA-II far below baseline -> diagnosed (random starts held ~90% of stocks) and fixed step by step
- Repair bug (sector cap could make 100% investment impossible) -> `_ensure_sector_capacity`

### Next (Week 3-4)
- Run on real NSE data; verify tickers and broker cost rates
- Try to beat the baseline outright: weight-preserving encoding, external archive, larger budget
- Walk-forward out-of-sample evaluation
- Report, slides, Streamlit UI (separate `frontend/` folder)

### Individual contributions (fill in honestly)
| Member | This week |
|---|---|
| Akhil | |
| Ridhima | |
