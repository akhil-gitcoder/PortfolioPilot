# References

Only sources whose existence and basic details I am confident about are listed. Check volume and page details against the publisher before final submission.

1. Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77-91. (Mean-variance model: the starting point for risk and return as conflicting objectives.)
2. Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197. (Algorithm implemented in `src/algorithms/nsga2.py`; constrained dominance.)
3. Chang, T.-J., Meade, N., Beasley, J. E., Sharaiha, Y. M. (2000). Heuristics for cardinality constrained portfolio optimisation. *Computers & Operations Research*, 27(13), 1271-1302. (Shows that the holdings limit makes the problem hard and compares heuristics.)
4. Lobo, M. S., Fazel, M., Boyd, S. (2007). Portfolio optimization with linear and fixed transaction costs. *Annals of Operations Research*, 152, 341-365. (Transaction costs in portfolio models.)
5. Ledoit, O., Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411. (Covariance estimator used, via scikit-learn.)
6. Zitzler, E., Thiele, L. (1999). Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach. *IEEE Transactions on Evolutionary Computation*, 3(4), 257-271. (Hypervolume indicator.)
7. DeMiguel, V., Garlappi, L., Uppal, R. (2009). Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915-1953. (Context for why equal weight is hard to beat out of sample.)
8. Diamond, S., Boyd, S. (2016). CVXPY: a Python-embedded modeling language for convex optimization. *Journal of Machine Learning Research*, 17(83), 1-5. (Solver interface for the baseline, the polish and the exact front.)
9. Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

## Not cited (could not verify)
Specific papers on NSGA-II applied to Indian-market portfolios, and Meghwani and Thakur's multi-objective portfolio work: do not cite until the exact title, venue and year are checked.
