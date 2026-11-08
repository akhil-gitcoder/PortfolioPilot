"""PortfolioPilot web server: the bridge between the web page and the optimiser.

Standard library only (no web framework needed).  Run with:   python main.py serve

  GET  /                 the page (frontend/static/index.html)
  GET  /api/context      stock list, example holdings, price sources
  GET  /api/evidence     the project's own results, read from results/real/
  POST /api/rebalance    {holdings: [% per stock], source}  -> options for k = 3..6
  POST /api/refresh      download the latest prices from Yahoo Finance

Listens on 127.0.0.1 only.  One rebalance runs at a time.
"""
import json
import math
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from frontend import engine as E

warnings.filterwarnings("ignore")
ROOT = E.ROOT
STATIC = ROOT / "frontend" / "static"
MAX_BODY = 100_000


class App:
    """Holds the loaded data and caches finished rebalances."""

    def __init__(self, n_evals=None):
        self.cfg, self.uni, self.bundled = E.load_context()
        self.tickers = [t for t in self.uni.ticker if t in self.bundled.columns]
        self.names = dict(zip(self.uni.ticker, self.uni.name))
        self.sector_of = dict(zip(self.uni.ticker, self.uni.sector))
        self.n_evals = n_evals
        self.lock = threading.Lock()
        self.cache = {}

    # -- data sources
    def latest(self):
        return E.load_latest_cached(self.tickers)

    def prices_for(self, source):
        if source == "latest":
            p = self.latest()
            if p is None:
                raise ValueError("No downloaded prices yet. Use 'Download latest prices' first.")
            return p
        if source == "bundled":
            return self.bundled[self.tickers]
        raise ValueError("Unknown price source.")

    def context(self):
        hold = self.cfg["current_portfolio"]["holdings"]
        sources = [{"id": "bundled", "label": "Saved snapshot", "asof": str(self.bundled.index[-1].date())}]
        lat = self.latest()
        if lat is not None:
            sources.append({"id": "latest", "label": "Latest download", "asof": str(lat.index[-1].date())})
        return {"holdings": [{"ticker": t, "stock": self.names[t], "sector": self.sector_of[t],
                              "pct": round(100 * hold.get(t, 0.0), 1)} for t in self.tickers],
                "sources": sources, "window": self.cfg["data"]["estimation_days"]}

    def evidence(self):
        sp = ROOT / "results" / "real" / "summary.csv"
        if not sp.exists():
            return {"available": False}
        s = pd.read_csv(sp)
        st = json.load(open(ROOT / "results" / "real" / "stats.json"))
        order = [m for m in ["RandomSearch", "NSGA2", "NSGA2+current", "NSGA2-tuned", "NSGA2-tuned+archive",
                             "WeightedSum", "Hybrid", "Hybrid+archive"] if m in set(s.method)]
        s = s.set_index("method").loc[order]
        out = {"available": True,
               "rows": [{"m": m, "hv": round(float(r.hv_mean), 3), "igd": round(float(r.igd_mean), 3),
                         "t": round(float(r.runtime_mean_s), 2)} for m, r in s.iterrows()],
               "wins": st.get("seeds_Hybrid+archive_hv_greater_than_WS"), "seeds": st.get("n_seeds"), "wf": None}
        wp = ROOT / "results" / "real" / "walkforward_knee" / "summary.csv"
        if wp.exists():
            w = pd.read_csv(wp, header=[0, 1], index_col=0)
            w = w[w.index.notna()]
            out["wf"] = [{"s": i, "w": round(float(w.loc[i, ("final_wealth", "mean")]), 3),
                          "p": round(float(w.loc[i, ("mean_pred_ann", "mean")]), 3),
                          "r": round(float(w.loc[i, ("mean_real_ann", "mean")]), 3)} for i in w.index]
        return out

    # -- the main action
    def rebalance(self, holdings, source):
        n = len(self.tickers)
        if not isinstance(holdings, list) or len(holdings) != n:
            raise ValueError(f"Expected {n} holdings, one per stock.")
        try:
            h = np.array([float(v) for v in holdings])
        except (TypeError, ValueError):
            raise ValueError("Holdings must be numbers.")
        if not np.all(np.isfinite(h)) or (h < 0).any() or (h > 100).any():
            raise ValueError("Each holding must be between 0 and 100 percent.")
        if h.sum() <= 0:
            raise ValueError("Enter at least one holding.")
        w = h / h.sum()
        prices = self.prices_for(source)
        key = (source, str(prices.index[-1].date()), tuple(np.round(w, 6)), self.n_evals)
        if key in self.cache:
            return self.cache[key]
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            prob = E.make_snapshot_problem(prices, self.cfg, w)
            X, F = E.optimise(prob, self.cfg, n_evals=self.n_evals)
            if len(F) < 3:
                raise ValueError("The search found too few portfolios. Try different holdings.")
            res = {"asof": str(prices.index[-1].date()), "window": self.cfg["data"]["estimation_days"],
                   "byk": E.build_payload(prob, X, F, self.names, self.sector_of)}
            self.cache[key] = res
            return res

    def refresh(self):
        E.fetch_latest_prices(self.tickers)
        return self.context()


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj=None, raw=None, ctype="application/json"):
            body = raw if raw is not None else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if "text" in ctype or "json" in ctype else ""))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, raw=(STATIC / "index.html").read_bytes(), ctype="text/html")
                if path == "/api/context":
                    return self._send(200, app.context())
                if path == "/api/evidence":
                    return self._send(200, app.evidence())
            except Exception as e:      # keep the server alive, tell the page
                return self._send(500, {"error": str(e)})
            self._send(404, {"error": "Not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self._send(413, {"error": "Request too large."})
                body = json.loads(self.rfile.read(n) or b"{}")
                if path == "/api/rebalance":
                    return self._send(200, app.rebalance(body.get("holdings"), body.get("source", "bundled")))
                if path == "/api/refresh":
                    return self._send(200, app.refresh())
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:
                return self._send(500, {"error": str(e)})
            self._send(404, {"error": "Not found"})

        def log_message(self, fmt, *args):      # quiet console
            pass
    return Handler


def create_server(port=8501, n_evals=None):
    app = App(n_evals=n_evals)
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app)), app


def serve(port=8501, open_browser=True):
    srv, _ = create_server(port)
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    print(f"PortfolioPilot is running at {url}   (press Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
