# Mathematical formulation

## Sets and parameters
- Stocks $i = 1,\dots,n$ with $n = 30$ (NSE, listed in `config/universe.csv`), grouped into sectors $S_1,\dots,S_8$.
- $\mu \in \mathbb{R}^n$: expected annual returns, the sample mean of daily returns times 252, estimated from the first 756 trading days only.
- $\Sigma \in \mathbb{R}^{n\times n}$: annual covariance, the Ledoit-Wolf estimate times 252, from the same window.
- $x^{old} \in \mathbb{R}^n$: the current portfolio (weights, sum 1).
- $b_i, s_i$: proportional buy and sell cost rates (here $b_i = 0.00119$, $s_i = 0.00104$ for all $i$; estimates, see `data/DATA.md`).
- Limits: $x_{max} = 0.20$, sector cap $c = 0.40$, $K = 10$ holdings, minimum position $\ell = 0.03$.

## Decision variables
$x \in \mathbb{R}^n$, where $x_i$ is the fraction of capital in stock $i$ after rebalancing.

## Objectives (all minimised)
$$f_1(x) = -\mu^\top x \qquad \text{(negative expected annual return)}$$
$$f_2(x) = \sqrt{x^\top \Sigma x} \qquad \text{(annual volatility)}$$
$$f_3(x) = \sum_{i=1}^{n}\Big[b_i \max(x_i - x_i^{old},0) + s_i \max(x_i^{old} - x_i,0)\Big] \qquad \text{(one-off rebalancing cost, fraction of capital)}$$

The problem is $\min_x \; F(x) = (f_1(x), f_2(x), f_3(x))$ subject to the constraints below. The answer is the set of Pareto-optimal portfolios, not one portfolio.

## Constraints
$$\sum_i x_i = 1,\qquad x_i \ge 0,\qquad x_i \le x_{max}\ \ \forall i$$
$$\sum_{i \in S_k} x_i \le c \ \ \forall k$$
$$\big|\{i : x_i > 0\}\big| \le K,\qquad x_i \ge \ell \ \text{ whenever } x_i > 0$$

## Assumptions
1. One-period decision: the new portfolio is chosen once, using estimates from the past window.
2. Costs are proportional to trade value (no fixed per-order charge, spread, market impact or taxes on gains).
3. Long-only, no leverage, no cash position other than via stocks.
4. $\mu$ and $\Sigma$ are treated as known inputs; their estimation error is a limitation, tested by the walk-forward experiment.
5. Prices are Yahoo Finance adjusted closes, assumed free of errors (checked for gaps and extreme moves).
6. Current portfolio: 6 stocks (RELIANCE 20%, TCS 20%, HDFCBANK 20%, ITC 15%, SBIN 15%, LT 10%), a hypothetical concentrated holding that satisfies all constraints, so "do nothing" is a valid reference point.

## Problem classification
- Multi-objective, three objectives, constrained.
- $f_2$ is convex, $f_1$ is linear, $f_3$ is convex piecewise linear. Without the holdings limit and the minimum position size the feasible set is convex and the Pareto front can be computed exactly (ε-constraint with CVXPY); we use this as a correctness check.
- With the holdings limit and the minimum position the feasible set is non-convex and combinatorial (which stocks to hold) on top of continuous weights: a mixed problem that is NP-hard in general. This is the main problem we solve.
- Deterministic given the estimates; uncertainty enters only through the estimation of $\mu$ and $\Sigma$.

## Dominance and Pareto front
Portfolio $x$ dominates $y$ if $f_j(x) \le f_j(y)$ for all $j$ and $f_j(x) < f_j(y)$ for at least one $j$. With constraints we use Deb's rules: a feasible portfolio beats an infeasible one; between two infeasible portfolios the smaller total violation wins; between two feasible ones ordinary dominance applies. The Pareto front is the set of non-dominated feasible portfolios.

## Quality indicators
- Hypervolume: the volume of objective space dominated by the front and bounded by a reference point (1.1 after min-max normalisation using the pooled bounds of all fronts). Larger is better. Computed exactly in 3D.
- IGD: the mean distance from each point of the pooled best-known front to the nearest point of a method's front. Smaller is better.
- Spacing and the C-metric (share of one front weakly dominated by another) as secondary indicators.

## Encoding used by the evolutionary algorithm
A random-key vector $k \in [0,1]^n$. Keys below 0.1 mean "not held"; if more than $K$ are held the $K$ largest are kept; enough stocks are added so the 20% cap and the sector caps allow 100% investment; weights are proportional to keys and then projected by water-filling onto $[\ell, x_{max}]$ and the sector caps with $\sum x = 1$. Any key vector therefore decodes to a feasible portfolio.

## Convex polish used by the hybrid
For a portfolio with held set $H$ (fixed), with return $r$, volatility $v$ and cost $c$, three convex problems are solved, each with the held set fixed and all constraints above:
$$\min_x\ x^\top\Sigma x \ \text{ s.t. } \mu^\top x \ge r,\ f_3(x) \le c;\qquad \max_x\ \mu^\top x \ \text{ s.t. } x^\top\Sigma x \le v^2,\ f_3(x)\le c;\qquad \min_x\ f_3(x)\ \text{ s.t. } \mu^\top x \ge r,\ x^\top\Sigma x\le v^2.$$
The original portfolio is feasible for all three, so the result is at least as good in every objective.
