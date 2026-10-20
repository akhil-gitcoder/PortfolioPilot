# Formulation
Decision variable: x in R^n (n = 30), weights.
Objectives (minimise): f1 = -mu^T x; f2 = sqrt(x^T Sigma x); f3 = sum_i [buy_i max(x_i - xold_i, 0) + sell_i max(xold_i - x_i, 0)].
Constraints: sum x = 1; x >= 0; x_i <= 0.20; sector sum <= 0.40; |{i: x_i > 0}| <= 10; x_i >= 0.03 if held; optional turnover cap.
Inputs: mu = sample mean daily return x 252; Sigma = Ledoit-Wolf covariance x 252; both estimated on the first 756 days only (rest held out).
Simple model: drop cardinality and min holding -> convex -> exact front via epsilon-constraint (validation only).
