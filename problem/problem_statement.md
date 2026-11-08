# Problem statement

**Decision-maker:** an investor in Indian equities who already holds a concentrated portfolio of NSE stocks.

**Decision:** how to rebalance, that is, the new weight in each of 30 NSE stocks.

**Why it is hard:** three goals conflict. Expected return rises with concentration in higher-return stocks, which raises volatility; diversifying lowers volatility but usually lowers expected return; and every trade costs money, so staying put is free but may sit far from the best trade-off. Practical rules (at most 20% in one stock, at most 40% in one sector, at most 10 holdings, at least 3% in any stock held) make the feasible set non-convex, so a standard convex solver cannot handle the problem exactly.

**Why multi-objective optimization:** the investor's risk appetite and tolerance for trading cost are personal and cannot be fixed in advance. The useful output is the set of non-dominated rebalancing options with their trade-offs, from which the investor can choose.

**Why optimization is required:** the number of possible stock selections is enormous (choosing up to 10 of 30 stocks) and each selection has continuous weights, constraints and three interacting objectives; manual selection cannot map the trade-offs.

**Scope and honesty:** the model is a one-period decision using estimates from past data; its out-of-sample behaviour is tested in a walk-forward experiment. The project reports trade-offs and does not present one portfolio as "the" answer.
