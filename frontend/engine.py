"""PortfolioPilot frontend engine: all the logic behind the web page, with no Streamlit code.

Flow when the user presses "Rebalance":
  1. SNAPSHOT  - freeze the latest prices; estimate expected returns and risk from the last
                 `estimation_days` trading days (same rule as the walk-forward test).
  2. OPTIMISE  - run the project's final method (NSGA-II with archive + convex polish).
  3. SELECT    - from the whole Pareto front keep only k well-spread options.
  4. EXPLAIN   - write a plain-English analysis for every option.

Nothing here invents numbers: every figure comes from the optimiser or from simple arithmetic
on its output.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithms.nsga2 import run_nsga2                    # noqa: E402
from src.data_loader import build_problem, load_config, load_prices, load_universe  # noqa: E402
from src.hybrid.polish import polish_front                    # noqa: E402
from src.walk_forward import knee_index, period_inputs, period_problem  # noqa: E402

DEFAULT_SEED = 1
CACHE_DIR = ROOT / "frontend" / ".cache"
LATEST_PRICES = CACHE_DIR / "prices_latest.csv"
NO_CHANGE_TURNOVER = 0.01      # a point that moves < 1% of capital counts as "no change"
HELD = 1e-8


# --------------------------------------------------------------------------- formatting
def format_inr(x):
    """Indian digit grouping: 1234567 -> '₹12,34,567'."""
    sign = "-" if x < 0 else ""
    n = int(round(abs(x)))
    s = str(n)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return f"{sign}₹{s}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# --------------------------------------------------------------------------- data
def load_context(root=ROOT):
    cfg = load_config(Path(root) / "config" / "default.json")
    uni = load_universe(root, cfg)
    prices, _ = load_prices(root, cfg, "real")
    return cfg, uni, prices


def clean_prices(raw, tickers):
    """Turn a raw download (rows = dates, columns = tickers) into the project's price table:
    keep the universe's tickers, forward-fill short gaps, drop incomplete leading rows."""
    missing = [t for t in tickers if t not in raw.columns or raw[t].isna().all()]
    if missing:
        raise ValueError("No data for: " + ", ".join(missing))
    p = raw[tickers].copy()
    p.index = pd.to_datetime(p.index)
    p = p.sort_index().ffill(limit=3).dropna(how="any")
    return p


def fetch_latest_prices(tickers, start="2019-01-01"):
    """Download fresh prices from Yahoo Finance (needs internet) and cache them.
    Returns the cleaned table.  Raises on any problem so the page can fall back."""
    import yfinance as yf
    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data (no internet?).")
    p = clean_prices(raw["Close"], list(tickers))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p.to_csv(LATEST_PRICES)
    return p


def load_latest_cached(tickers):
    if LATEST_PRICES.exists():
        p = pd.read_csv(LATEST_PRICES, index_col=0, parse_dates=True)
        if all(t in p.columns for t in tickers):
            return p[list(tickers)]
    return None


# --------------------------------------------------------------------------- snapshot + optimise
def make_snapshot_problem(prices, cfg, x_old, root=ROOT):
    """Problem built from the LAST `estimation_days` rows of `prices` (the frozen frame),
    with the user's current weights.  Mirrors the walk-forward rule: estimate from the past only."""
    base, base_prices, _ = build_problem(root, cfg, source="real")
    tickers = base.tickers
    prices = prices[tickers]
    window = cfg["data"]["estimation_days"]
    if len(prices) < window + 1:
        raise ValueError(f"Need at least {window + 1} days of prices, have {len(prices)}.")
    mu, cov = period_inputs(prices, len(prices), window, cfg["data"]["cov_estimator"])
    x_old = np.asarray(x_old, float)
    return period_problem(base, mu, cov, x_old / x_old.sum())


def optimise(problem, cfg, seed=DEFAULT_SEED, n_evals=None):
    """Run the final method (Hybrid + archive) once.  Same seed + same inputs = same answer."""
    nb = cfg["nsga2"]
    r = run_nsga2(problem, pop_size=nb["pop_size"], n_evals=n_evals or nb["n_evals"], seed=seed,
                  eta_c=nb["eta_c"], eta_m=nb["eta_m"], pc=nb["pc"], checkpoint_every=10 ** 9,
                  inject_current=True, sparse_init=True, reset_prob=nb["reset_prob"],
                  archive_size=nb["archive_size"])
    X, F, _ = polish_front(problem, r["X"], r["F"])
    ok = problem.violation(X) <= 1e-5
    return X[ok], F[ok]


# --------------------------------------------------------------------------- choosing k options
MIN_GAIN = 0.005       # an option must add >= 0.5 points of return OR cut >= 0.5 points of risk


EPS_2D = 0.001         # differences under 0.1 points of return or risk are treated as noise


def efficient_2d(ret, vol):
    """True for points that no other point clearly beats on return and risk together.
    A beats B when A is no worse than B (within a 0.1-point tolerance) on both and better
    overall by more than that tolerance."""
    n = len(ret)
    keep = np.ones(n, bool)
    for i in range(n):
        gain = (ret - ret[i]) + (vol[i] - vol)
        beat = (ret >= ret[i] - EPS_2D) & (vol <= vol[i] + EPS_2D) & (gain > 2 * EPS_2D)
        keep[i] = not beat.any()
    return keep


def candidate_mask(problem, X, F):
    """Front points worth showing to a person:
      1. a real change (moves at least 1% of capital),
      2. a meaningful gain over doing nothing (+0.5 points of return or -0.5 points of risk),
      3. not beaten on BOTH return and risk by another candidate (a portfolio that is worse on
         both and only a few rupees cheaper to trade into would just confuse the choice)."""
    cur = problem.objectives(problem.x_old[None, :])[0]
    ret, vol = -F[:, 0], F[:, 1]
    gain = (ret >= -cur[0] + MIN_GAIN) | (vol <= cur[1] - MIN_GAIN)
    turnover = np.maximum(X - problem.x_old, 0).sum(axis=1)
    m = gain & (turnover >= NO_CHANGE_TURNOVER)
    if m.sum() < 3:
        return np.ones(len(F), bool)
    idx = np.where(m)[0]
    eff = efficient_2d(ret[idx], vol[idx])
    m2 = np.zeros(len(F), bool)
    m2[idx[eff]] = True
    return m2 if m2.sum() >= 3 else m


def select_k(F, k):
    """Pick k well-spread indices of F (columns: -return, risk, cost): the lowest-risk point,
    the highest-return point, the knee (best all-round compromise), then repeatedly the point
    farthest from those already chosen.  Returned sorted from lowest to highest risk."""
    F = np.asarray(F, float)
    n = len(F)
    k = min(k, n)
    lo, hi = F.min(axis=0), F.max(axis=0)
    Z = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    chosen = []
    for i in (int(np.argmin(F[:, 1])), int(np.argmin(F[:, 0])), knee_index(F, "knee")):
        if i not in chosen and len(chosen) < k:
            chosen.append(i)
    while len(chosen) < k:
        d = np.min(np.linalg.norm(Z[:, None, :] - Z[chosen][None, :, :], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return sorted(chosen, key=lambda i: (F[i, 1], -F[i, 0]))


# --------------------------------------------------------------------------- explanations
def _label_options(F_sel, knee_pos):
    k = len(F_sel)
    before = ["Cautious", "Moderate", "Firm"]
    after = ["Growth-leaning", "Aggressive", "Very aggressive"]
    labels = []
    for p in range(k):
        if p == knee_pos:
            labels.append("Balanced")
        elif p == 0:
            labels.append("Safest")
        elif p == k - 1:
            labels.append("Highest return")
        elif p < knee_pos:
            labels.append(before[min(p - 1, 2)])
        else:
            labels.append(after[min(p - knee_pos - 1, 2)])
    return labels


def _trade_table(problem, x, amount, names):
    rows = []
    for i, t in enumerate(problem.tickers):
        cur, new = problem.x_old[i], x[i]
        if cur < HELD and new < HELD:
            continue
        d = new - cur
        action = "Buy" if d > 5e-4 else "Sell" if d < -5e-4 else "Hold"
        rows.append({"Stock": names[t], "Ticker": t, "Sector": problem.sectors[i],
                     "Now": cur, "Suggested": new if new >= HELD else 0.0,
                     "Action": action, "Amount (₹)": abs(d) * amount})
    df = pd.DataFrame(rows)
    df["_o"] = df["Action"].map({"Buy": 0, "Sell": 1, "Hold": 2})
    df = df.sort_values(["_o", "Amount (₹)"], ascending=[True, False]).drop(columns="_o")
    return df.reset_index(drop=True)


def analyse(problem, X, F, k, amount, names, exclude_trivial=True):
    """Build the k options and a plain-English analysis for each.

    Returns dict(current=..., options=[...]).  All numbers are annual fractions
    (0.12 = 12%) except rupee amounts."""
    cur_F = problem.objectives(problem.x_old[None, :])[0]
    cur = {"x": problem.x_old.copy(), "ret": -cur_F[0], "vol": cur_F[1], "cost": 0.0,
           "n_hold": int((problem.x_old > HELD).sum()), "max_pos": float(problem.x_old.max()),
           "max_sector": _max_sector(problem, problem.x_old)}
    mask = candidate_mask(problem, X, F) if exclude_trivial else np.ones(len(F), bool)
    Xc, Fc = X[mask], F[mask]
    sel = select_k(Fc, k)
    Fs = Fc[sel]
    knee_global = knee_index(Fc, "knee")
    knee_pos = int(np.argmin(np.abs(np.array(sel) - knee_global))) if knee_global in sel else \
        int(np.argmin(np.abs(Fs[:, 1] - Fc[knee_global, 1])))
    labels = _label_options(Fs, knee_pos)
    opts = []
    for p, gi in enumerate(sel):
        x = Xc[gi]
        ret, vol, cost = -Fc[gi, 0], Fc[gi, 1], Fc[gi, 2]
        o = {"n": p + 1, "label": labels[p], "x": x, "ret": ret, "vol": vol, "cost": cost,
             "cost_inr": cost * amount, "n_hold": int((x > HELD).sum()), "max_pos": float(x.max()),
             "max_sector": _max_sector(problem, x), "bad_year": ret - vol,
             "ret_per_risk": ret / vol,
             "d_ret": ret - cur["ret"], "d_vol": vol - cur["vol"],
             "better_both": (ret >= cur["ret"] - 1e-12) and (vol <= cur["vol"] + 1e-12),
             "turnover": float(np.maximum(x - problem.x_old, 0).sum()),
             "trades": _trade_table(problem, x, amount, names)}
        o["payback_days"] = (cost / o["d_ret"] * 365) if o["d_ret"] > 1e-9 and cost > 0 else None
        if p > 0:
            pr = opts[-1]
            o["step_ret"], o["step_vol"] = ret - pr["ret"], vol - pr["vol"]
            o["step_ratio"] = o["step_ret"] / o["step_vol"] if o["step_vol"] > 1e-9 else None
        o["top"] = _top_holdings(problem, x, names)
        o["text"] = _explain(o, cur, opts[-1] if opts else None, amount)
        opts.append(o)
    return {"current": cur, "options": opts, "n_front": int(len(F)), "n_candidates": int(mask.sum())}


def _max_sector(problem, x):
    best = max(problem.sector_names, key=lambda s: x[np.array(problem.sectors) == s].sum())
    return (best, float(x[np.array(problem.sectors) == best].sum()))


def _top_holdings(problem, x, names, n=3):
    idx = np.argsort(-x)[:n]
    return [(names[problem.tickers[i]], float(x[i])) for i in idx if x[i] > HELD]


def _explain(o, cur, prev, amount):
    """Plain-English analysis: what it is, how it differs from today, when to choose it, what to watch."""
    lab = o["label"]
    what = {
        "Safest": "The calmest portfolio on the list: the smallest ups and downs, with the lowest return.",
        "Cautious": "Leans toward safety. It accepts a little more risk than the safest option to earn more.",
        "Balanced": "The best all-round compromise: closest to the ideal on return, risk and cost together.",
        "Moderate": "A middle step between cautious and balanced: a bit more return, a bit more risk.",
        "Firm": "Close to balanced but still on the careful side.",
        "Growth-leaning": "Leans toward growth. It earns more but its value will swing more.",
        "Aggressive": "Pushes harder for return. Expect clearly bigger ups and downs.",
        "Very aggressive": "Close to the maximum-return end of the list. Expect large swings.",
        "Highest return": "The highest expected return on the list, with the biggest swings.",
    }[lab]
    vs = []
    if o["better_both"] and abs(o["d_vol"]) < 0.0005:
        vs.append(f"Compared with your current portfolio it has a higher expected return "
                  f"({o['d_ret'] * 100:+.1f} points) at about the same risk.")
    elif o["better_both"]:
        vs.append(f"Compared with your current portfolio it has a higher expected return "
                  f"({o['d_ret'] * 100:+.1f} points) and lower risk ({o['d_vol'] * 100:+.1f} points).")
    else:
        vs.append(f"Compared with your current portfolio: expected return {o['d_ret'] * 100:+.1f} points, "
                  f"risk {o['d_vol'] * 100:+.1f} points.")
    vs.append(f"It would cost about {format_inr(o['cost_inr'])} to trade into it ({o['cost'] * 100:.2f}% "
              f"of your money).")
    if o["payback_days"] is not None:
        vs.append(f"If the expected return holds, the extra return earns that cost back in about "
                  f"{o['payback_days']:.0f} days.")
    step = ""
    if prev is not None:
        if o.get("step_ratio") is not None:
            step = (f"Compared with Option {prev['n']}: {o['step_ret'] * 100:+.1f} points of expected return "
                    f"for {o['step_vol'] * 100:+.1f} points of extra risk, about "
                    f"{o['step_ratio']:.1f} points of return for each extra point of risk.")
        else:
            step = (f"Compared with Option {prev['n']}: {o['step_ret'] * 100:+.1f} points of expected "
                    f"return at about the same risk.")
    choose = {
        "Safest": "Choose this if avoiding big drops matters more to you than earning more.",
        "Cautious": "Choose this if you want some growth but still want a smoother ride.",
        "Balanced": "Choose this if you have no strong preference. It is the sensible default.",
        "Moderate": "Choose this if you want slightly more growth than the cautious option and can accept slightly more risk.",
        "Firm": "Choose this if you prefer balanced but want to stay a touch safer.",
        "Growth-leaning": "Choose this if you can stay calm through bad stretches and want more growth.",
        "Aggressive": "Choose this if you are comfortable with large swings and want the higher return.",
        "Very aggressive": "Choose this if you are comfortable with large swings and want to be near the top return.",
        "Highest return": "Choose this only if you can accept large swings, including bad years, for the higher return.",
    }[lab]
    watch = []
    if o["max_pos"] >= 0.18:
        watch.append(f"its largest holding is {pct(o['max_pos'], 0)} of the portfolio")
    sec, sw = o["max_sector"]
    if sw >= 0.35:
        watch.append(f"{pct(sw, 0)} sits in one sector ({sec})")
    if o["n_hold"] <= 6:
        watch.append(f"it holds only {o['n_hold']} stocks")
    if lab in ("Highest return", "Aggressive", "Very aggressive", "Growth-leaning") and o["bad_year"] < 0:
        watch.append(f"a bad year could lose about {pct(-o['bad_year'], 0)}")
    watch_txt = ("Watch out: " + "; ".join(watch) + ".") if watch else \
        "Nothing unusual to watch for beyond normal market risk."
    return {"what": what, "vs_now": " ".join(vs), "step": step, "choose": choose, "watch": watch_txt}


def check_rules(problem, options):
    """Independent check that every option obeys every rule.  Returns (ok, message)."""
    bad = [o["n"] for o in options if problem.violation(o["x"][None, :])[0] > 1e-5]
    return (not bad, "All options satisfy every rule." if not bad else f"Rule violated by option(s) {bad}")


# --------------------------------------------------------------------------- data for the web page
PAGE_KS = (3, 4, 5, 6)


def build_payload(problem, X, F, names, sector_of, ks=PAGE_KS):
    """Everything the web page needs for one rebalance, for every allowed k, as plain JSON types.

    Rupee figures are NOT baked in: the page multiplies the fractions by the amount the user
    types, and the one rupee phrase inside the text is the placeholder '{INR}'."""
    byk = {}
    for k in ks:
        res = analyse(problem, X, F, k, 1_000_000, names)
        ok, msg = check_rules(problem, res["options"])
        if not ok:
            raise RuntimeError(msg)
        cur = res["current"]
        opts = []
        for o in res["options"]:
            t = dict(o["text"])
            t["vs_now"] = t["vs_now"].replace(format_inr(o["cost_inr"]), "{INR}")
            trades = [{"a": r["Action"], "s": r["Stock"], "sec": r["Sector"],
                       "now": round(float(r["Now"]), 5), "sug": round(float(r["Suggested"]), 5)}
                      for _, r in o["trades"].iterrows()]
            opts.append({"n": o["n"], "label": o["label"], "ret": float(o["ret"]), "vol": float(o["vol"]),
                         "cost": float(o["cost"]), "n_hold": o["n_hold"], "max_pos": float(o["max_pos"]),
                         "bad": float(o["bad_year"]), "d_ret": float(o["d_ret"]), "d_vol": float(o["d_vol"]),
                         "better": bool(o["better_both"]), "text": t,
                         "top": [[a, float(b)] for a, b in o["top"]], "trades": trades})
        byk[str(k)] = {"cur": {"ret": float(cur["ret"]), "vol": float(cur["vol"]), "n_hold": cur["n_hold"],
                               "max_pos": float(cur["max_pos"])},
                       "opts": opts, "n_front": res["n_front"], "n_cand": res["n_candidates"]}
    return byk
