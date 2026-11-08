# PortfolioPilot: Final Report

Group: Akhil (241070004), Ridhima (241071062) | Course: Optimization Techniques | Professor: Akhil Kulal

## 1. Title and group details
PortfolioPilot: Multi-Objective Portfolio Allocation and Rebalancing under Return, Risk and Transaction-Cost Constraints. Group of 2 (approved by the professor).

## 2. Problem statement
See `problem/problem_statement.md`. An investor holds a concentrated NSE portfolio and must choose new weights. Return, risk and trading cost conflict, and practical limits make the problem non-convex.

## 3. Why optimization is needed
The set of possible 10-stock selections out of 30 is huge, each with continuous weights, constraints and three interacting goals. The output needed is a set of trade-offs, not one number.

## 4. Mathematical formulation
Full formulation: `formulation/formulation.md`. Minimise f1 = -mu'x, f2 = sqrt(x'Sigma x), f3 = proportional buy/sell cost, subject to full investment, long-only, 20% per stock, 40% per sector, at most 10 holdings, at least 3% per held stock.

## 5. Why this is multi-objective
Moving to higher return raises risk; moving away from the current portfolio costs money. No portfolio is best on all three, so we compute the Pareto front.

## 6. Data
30 NSE stocks, Yahoo Finance adjusted closes, 1,729 trading days (2019-01-01 to 2025-12-30), real. Synthetic data used only in development and labelled. mu = sample mean x 252 and Sigma = Ledoit-Wolf x 252 from the first 756 days; later days are held out. Cost rates (buy 0.119%, sell 0.104%) are estimates. See `data/DATA.md`.

## 7. Methods
- Baseline 1: random search with the same decoder and budget.
- Baseline 2 (the one to beat): weighted-sum relax-and-round with CVXPY, 66 weight vectors.
- NSGA-II written from scratch: constrained dominance, fast non-dominated sorting, crowding distance, SBX (eta 15, pc 0.9), polynomial mutation (eta 20), random-key encoding with a repair step.
- Improvements, each tested separately: inject current portfolio, sparse initial supports, support-reset mutation, external archive (200).
- Hybrid: NSGA-II chooses the held stocks; three convex solves per solution then set the weights exactly; results re-filtered for non-dominance.
- Tried and not adopted: Lamarckian write-back, seeding, a 30,000-evaluation budget.
- Correctness check: on the convex special case (no holdings limit) we compare with the exact front.

## 8. Experimental setup
30 independent seeds (1000-1029), 10,000 evaluations per run, population 100. Tuning used separate development seeds (1-5) on the first-756-day problem. Metrics: hypervolume (pooled min-max, reference 1.1), IGD to the pooled best-known front, spacing, C-metric, runtime. Tests: Mann-Whitney U between stochastic methods; sign and Wilcoxon tests versus the deterministic baseline.

## 9. Results (real data, from `results/real/summary.csv`)
| Method | HV (mean ± sd) | IGD | Runtime (s) |
|---|---|---|---|
| Random search | 0.516 ± 0.009 | 0.236 | 0.36 |
| NSGA-II (plain) | 0.532 ± 0.027 | 0.233 | 0.35 |
| + current portfolio | 0.674 ± 0.021 | 0.151 | 0.36 |
| NSGA-II tuned | 0.801 ± 0.010 | 0.069 | 0.42 |
| tuned + archive | 0.842 ± 0.006 | 0.050 | 0.75 |
| Hybrid | 0.881 ± 0.006 | 0.038 | 2.30 |
| Hybrid + archive | 0.907 ± 0.003 | 0.025 | 4.68 |
| Weighted-sum baseline | 0.877 | 0.068 | 0.60 |

Figures: `figures/real/pareto_fronts_2d.png`, `pareto_fronts_3d.png`, `hv_boxplot.png`, `convergence.png`.

## 10. Pareto analysis
The final front spans expected return 18% to 39%, volatility 16% to 27%, cost 0% to 0.22% of capital (estimates, first 756 days). The current portfolio (21% return, 22% volatility, zero cost) is dominated on return and volatility by 282 front portfolios. The knee portfolio (closest to the ideal point on the front's own scale) gives about 31% return at 20% volatility for 0.11% cost. Reading the front: the first cheap trades buy large risk reductions; beyond that, extra return costs both volatility and trading cost. Run `python main.py inspect --tag real --method Hybrid+archive --pick knee` for the weights and trades.

## 11. Comparison and statistical evidence
- Hybrid + archive has higher HV than the weighted-sum baseline in 30/30 seeds (sign p = 1.9e-9, Wilcoxon p = 1.9e-9), +3.4% on average. C(WS, method) = 0.156, C(method, WS) = 0.
- Hybrid alone: 21/30 seeds (sign p = 0.043, Wilcoxon p = 4.6e-4).
- Pure NSGA-II variants without the convex solver do not beat the baseline.
- NSGA-II vs random search: p = 0.014 (weak); each improvement step p about 3e-11.
- Cost: the final method is about 8 times slower than the baseline (4.7 s vs 0.6 s).

## 12. Validation
On the convex model, tuned NSGA-II reaches mean HV ratio 0.866 and IGD 0.105 against the exact front; 28 automated tests cover cost arithmetic, constraints, dominance, metrics, repair, archive, and the no-look-ahead rule of the walk-forward.

## 13. Out-of-sample (walk-forward) test
15 quarterly rebalances, rolling 756-day window ending before each rebalance, 5 paths for stochastic strategies, costs deducted.

| Strategy | Final wealth (knee) | (ratio) |
|---|---|---|
| Equal weight | 1.843 | 1.843 |
| Do nothing | 1.604 | 1.604 |
| NSGA-II | 1.630 | 1.592 |
| Hybrid | 1.570 | 1.550 |
| Weighted sum | 1.564 | 1.537 |

Hybrid minus weighted sum: +0.03% per quarter (s.e. 0.10%), not distinguishable from zero. Hybrid minus equal weight: -0.94% (s.e. 0.97%). Predicted return was about 32 to 34% a year against about 18 to 20% realised.

## 14. Interpretation
The in-sample advantage is real: the hybrid covers the trade-off surface better. But the objectives use estimated returns, and sample means are noisy, so better coverage of the estimated front did not become better realised wealth. Equal weight ignores the 10-holding limit and needs no estimates; it is hard to beat, a known result (DeMiguel et al., 2009).

## 15. Limitations
- One real dataset, 30 stocks, 7 years; the hold-out is short.
- Expected returns are plain sample means (noisy); no shrinkage or factor model.
- Costs are proportional estimates, not a broker sheet; no tax, spread or market impact.
- The starting portfolio is hypothetical.
- The hybrid is slower; a different baseline (e.g. exact mixed-integer solver) was not tested.
- Tuning was done on the same first-756-day problem used for the in-sample results, so the in-sample margin may be optimistic (final seeds were separate).

## 16. Conclusion
The hybrid NSGA-II with an archive beats the weighted-sum baseline on Pareto-front quality in every seed, at about 8 times the runtime; plain NSGA-II does not. Out of sample nothing beat equal weight, so the practical value lies in showing the trade-offs, not in a return forecast. Natural next steps: better return estimates, robust formulations, per-stock costs.

## 17. Individual contributions
See `report/contributions.md`.

## 18. Reproducibility
Fixed seeds in `config/default.json`; data snapshot committed; `python main.py all --source real --tag real`; fresh-clone steps in README sections 7 and 8.

## 19. References
See `literature/references.md`.

## AI assistance disclosure
An AI assistant (Claude) was used to help write and debug code and drafts; the team ran all experiments, and reports only results produced by the code in this repository.
