# PortfolioPilot web app (frontend + backend)

A web page on top of the project's optimiser. Press **Rebalance** and the app freezes a snapshot
of the market, runs the final method (NSGA-II with archive + convex polish), keeps k well-spread
options from the Pareto front, and explains each one in plain English.

## Run it (no extra installs: the normal `requirements.txt` is enough)
```bash
cd PortfolioPilot
source .venv/bin/activate
python main.py serve
```
It prints an address (http://127.0.0.1:8501) and opens it in your browser. Press Ctrl+C to stop.
The first Rebalance takes about 10 to 60 seconds; the same holdings again are instant (cached).

## How it fits together
```
browser page  ──►  frontend/server.py  ──►  frontend/engine.py  ──►  src/ (NSGA-II, polish, problem)
frontend/static/index.html   REST API        snapshot, options,        the project's backend
                                             analysis text
```
- `static/index.html`: the page (HTML, CSS and JavaScript in one file, no libraries).
- `server.py`: a small web server using only the Python standard library. Endpoints:
  `GET /api/context`, `GET /api/evidence`, `POST /api/rebalance`, `POST /api/refresh`.
  It listens on 127.0.0.1 only, validates every input and runs one optimisation at a time.
- `engine.py`: all the logic (no web code): snapshot, optimise, choose k options, write the analysis.
- Tests: `tests/test_frontend_engine.py` and `tests/test_frontend_server.py`.

## What the user sees
1. Their current portfolio (editable), the amount invested, and the price source.
2. **Rebalance my portfolio**: one click = one frozen snapshot (nothing is live).
3. A chart of the options (risk across, expected return up, today marked, the area that beats today
   shaded), a comparison table, and for every option: what it is, how it differs from today and from
   the previous option, when to choose it, what to watch out for, and the exact trades in rupees.
4. An evidence tab that reads the project's own results files, and an "About and limits" tab.

## How the options are chosen
- The optimiser returns the whole Pareto front (hundreds of portfolios).
- Hidden: portfolios that barely change anything, that do not beat today by at least 0.5 points of
  return or risk, or that are worse on both return and risk and only slightly cheaper to trade into.
- From the rest, k (3 to 6) are picked: lowest risk, highest return, the balanced one (the knee), and
  the ones farthest from those, shown from safest to highest return.
- The same prices and holdings always give the same answer (fixed seed).

## Prices
By default the app uses the committed snapshot `data/prices_real.csv`. "Download latest prices"
fetches fresh prices from Yahoo Finance (needs internet) into `frontend/.cache/` (not committed).
If the download fails, the saved prices keep working.

Not investment advice. Expected returns are estimates from past prices; in the project's own
15-quarter back-test they were far too optimistic.
