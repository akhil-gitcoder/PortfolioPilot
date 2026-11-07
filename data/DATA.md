# Data

## Real data (used for all real-data results)
- Source: Yahoo Finance, downloaded with the yfinance Python library (adjusted close prices).
- Downloaded on: 07 November 2026, using scripts/download_data.py. Saved as data/prices_real.csv.
- Coverage: the 30 NSE stocks in config/universe.csv, daily prices from 2019-01-01 to 2025-12-30 (1729 trading days).
- Tickers changed from the original list: none.
- Estimation window: the first 756 trading days (up to 2022-01-20) are used to estimate expected returns and the Ledoit-Wolf covariance. The remaining 973 days are held out and not used for estimation.
- ITC.NS has a near-zero estimated annual return (0.0002) in the estimation window. This is what the data shows, not an error.
- Yahoo Finance adjusted prices can be revised over time. The committed CSV is the snapshot behind every reported real-data result.

## Transaction cost rates
- config/default.json sets buy_rate = 0.00119 (0.119%) and sell_rate = 0.00104 (0.104%).
- Estimated on 07 November 2026 from the standard Indian delivery-trade charge- Estimated on 07 November 2026 from the standard Indian TT- Estimated on 07 November 2026 from the standard Indian delivery-trade charge- Estimated on 07 November 2026 from theon- Estimated on 07 November 2026 from the standard Indian delivery-trade charge- Estimated on  - Estimated on 07 November 2026 from the standard Indian delivery-trade charge- Estimated on 07 November 2026 from the standard Indiaag- Estimated on 07 November 2026 from the standard Indian ded - Estimated on 07 November 2026 from the standard Indian delivery-trade charge- Estimated on 07 November 2026 from the standard Indian TT- Estimated on 07 November 2026 from the standard Indian delivery-trade charge- Estimated on 07 November 2026 from  i- Estimated on 07 November 2026 from the standard Indian delivery-trade charge- Estimated on 07 November 2026 from the standard Indian TT- Estimated ond - Estimated on 07 November 2026 from the standard Indian delivery-trade charge- Estimated on 07 November 2026 from the standard Indian TT- Estimated on 07 November 2026  real. Every synthetic result file and figure is labelled SYNTHETIC.
