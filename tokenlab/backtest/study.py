"""Event study over the real unlock sample built by tokenlab.backtest.data.

For each event, daily log returns are adjusted for the crypto market using a market model
against BTC (beta estimated over days [-120, -31], no intercept in the abnormal return, so
a token's own pre-event drift is not subtracted away). Cumulative abnormal returns (CAR)
are reported for the run-up, the post-unlock window and the full window.

Run:  python -m tokenlab.backtest.study
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

EST = (-119, -31)          # beta estimation window (return days)
WINDOWS = {                # return-day ranges, inclusive; day k = ln p[k] / p[k-1]
    "pre_30": (-29, 0),    # p[-30] -> p[0]
    "pre_7": (-6, 0),      # p[-7]  -> p[0]
    "post_7": (1, 7),      # p[0]   -> p[+7]
    "post_30": (1, 30),    # p[0]   -> p[+30]
    "full": (-29, 30),     # p[-30] -> p[+30]
}


def abnormal_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide frames of abnormal / raw / BTC log returns: index event_id, columns rel_day."""
    rows = {}
    for eid, g in panel.groupby("event_id"):
        g = g.sort_values("rel_day")
        k = g["rel_day"].to_numpy()[1:]
        r = np.diff(np.log(g["price"].to_numpy()))
        rb = np.diff(np.log(g["btc"].to_numpy()))
        est = (k >= EST[0]) & (k <= EST[1])
        var = np.var(rb[est], ddof=1)
        beta = float(np.cov(r[est], rb[est])[0, 1] / var) if var > 0 else 1.0
        beta = min(max(beta, 0.0), 3.0)
        rows[eid] = {"k": k, "raw": r, "btc": rb, "ar": r - beta * rb, "beta": beta}
    return rows


def _car(x: np.ndarray, k: np.ndarray, win: tuple[int, int]) -> float:
    return float(x[(k >= win[0]) & (k <= win[1])].sum())


def _stats(v: np.ndarray, rng: np.random.Generator) -> dict:
    v = np.asarray(v, float)
    n = len(v)
    simple = np.expm1(v)                      # convert log CAR to simple return
    mean, sd = v.mean(), v.std(ddof=1)
    boot = rng.choice(v, size=(10000, n), replace=True).mean(axis=1)
    neg = int((v < 0).sum())
    # two-sided sign test
    p_sign = min(1.0, 2 * sum(math.comb(n, i) for i in range(max(neg, n - neg), n + 1)) / 2**n)
    return {
        "n": n,
        "mean_log": mean,
        "mean_pct": float(simple.mean() * 100),
        "median_pct": float(np.median(simple) * 100),
        "t_stat": float(mean / (sd / math.sqrt(n))) if sd > 0 else float("nan"),
        "ci95_log": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "share_negative": neg / n,
        "sign_test_p": p_sign,
    }


def run(verbose: bool = True) -> dict:
    events = pd.read_csv(DATA / "unlock_events.csv", parse_dates=["date"])
    panel = pd.read_csv(DATA / "event_prices.csv")
    ar = abnormal_returns(panel)
    rng = np.random.default_rng(42)

    per_event = []
    for _, e in events.iterrows():
        a = ar[e["event_id"]]
        row = {"event_id": e["event_id"], "date": e["date"].date().isoformat(), "name": e["name"],
               "category": e["category"], "pct_circ": e["pct_circ"], "usd_value": e["usd_value"],
               "beta": a["beta"]}
        for w, win in WINDOWS.items():
            row[f"car_{w}"] = _car(a["ar"], a["k"], win)
            row[f"raw_{w}"] = _car(a["raw"], a["k"], win)
            row[f"btcadj_{w}"] = _car(a["raw"] - a["btc"], a["k"], win)
        per_event.append(row)
    pe = pd.DataFrame(per_event)

    summary: dict = {"n_events": len(pe), "n_tokens": int(events["slug"].nunique()),
                     "date_min": pe["date"].min(), "date_max": pe["date"].max(),
                     "total_usd_unlocked": float(events["usd_value"].sum()),
                     "median_pct_circ": float(events["pct_circ"].median()),
                     "windows": {}, "robustness": {}, "splits": {}}
    for w in WINDOWS:
        summary["windows"][w] = _stats(pe[f"car_{w}"].to_numpy(), rng)
        summary["robustness"][w] = {
            "raw": _stats(pe[f"raw_{w}"].to_numpy(), rng),
            "btc_adjusted": _stats(pe[f"btcadj_{w}"].to_numpy(), rng),
        }

    # size split (terciles of unlock size relative to circulating supply)
    pe["size_bucket"] = pd.qcut(pe["pct_circ"], 3, labels=["small", "medium", "large"])
    for b, g in pe.groupby("size_bucket", observed=True):
        summary["splits"][f"size_{b}"] = {
            "pct_circ_range": [float(g["pct_circ"].min()), float(g["pct_circ"].max())],
            "full": _stats(g["car_full"].to_numpy(), rng),
            "post_30": _stats(g["car_post_30"].to_numpy(), rng),
        }
    for c, g in pe.groupby("category"):
        if len(g) >= 5:
            summary["splits"][f"recipient_{c}"] = {"full": _stats(g["car_full"].to_numpy(), rng)}

    # cross-sectional regression: CAR_full on ln(1 + unlock size)
    x = np.log1p(pe["pct_circ"].to_numpy())
    y = pe["car_full"].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    cov = resid.var(ddof=2) * np.linalg.inv(X.T @ X)
    summary["size_regression"] = {
        "spec": "CAR[-30,+30] = a + b * ln(1 + unlock/circulating)",
        "intercept": float(coef[0]), "slope": float(coef[1]),
        "slope_se": float(math.sqrt(cov[1, 1])), "slope_t": float(coef[1] / math.sqrt(cov[1, 1])),
        "r2": float(1 - resid.var() / y.var()),
    }

    # average cumulative abnormal return path, day -30 .. +30 (0 at day -30)
    ks = np.arange(-29, 31)
    paths = np.array([[a["ar"][a["k"] == k][0] for k in ks] for a in (ar[e] for e in pe["event_id"])])
    cum = np.cumsum(paths, axis=1)
    summary["car_path"] = {
        "rel_day": [-30] + ks.tolist(),
        "mean": [0.0] + cum.mean(axis=0).tolist(),
        "median": [0.0] + np.median(cum, axis=0).tolist(),
        "p25": [0.0] + np.percentile(cum, 25, axis=0).tolist(),
        "p75": [0.0] + np.percentile(cum, 75, axis=0).tolist(),
    }

    RESULTS.mkdir(exist_ok=True)
    pe.to_csv(RESULTS / "backtest_events.csv", index=False)
    (RESULTS / "backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(f"{summary['n_events']} events, {summary['n_tokens']} tokens, "
              f"{summary['date_min']} .. {summary['date_max']}")
        for w, s in summary["windows"].items():
            print(f"  {w:8s} mean {s['mean_pct']:+6.1f}%  median {s['median_pct']:+6.1f}%  "
                  f"t={s['t_stat']:+.2f}  negative {s['share_negative']:.0%}")
        print("  size regression:", {k: round(v, 3) for k, v in summary["size_regression"].items() if k != "spec"})
    return summary


if __name__ == "__main__":
    run()
