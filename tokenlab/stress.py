"""Stress-test runner: every design under every scenario, plus sensitivity analysis.

Collapse definition (fixed in advance, applied identically to every design):

    A simulated path COLLAPSES if, at any point after the shock, the token trades at least
    90% below its pre-shock price.  A design COLLAPSES in a scenario if more than half of
    its Monte Carlo paths collapse.

A stricter "no recovery" variant (collapsed AND still below 25% of the pre-shock price at
the end of the horizon) is reported alongside.

Run:  python -m tokenlab.stress
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import model
from .designs import DESIGNS
from .shocks import BY_KEY as SCENARIO_BY_KEY
from .shocks import CRASH_SCENARIOS, SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

COLLAPSE_DRAWDOWN = 0.90
NO_RECOVERY_LEVEL = 0.25
MAJORITY = 0.5
PATHS = 4000
BORDERLINE = 0.05       # |P(collapse) - 50%| below this is flagged as a close call
SEED = 2026


def calibrated_depth() -> float:
    f = RESULTS / "calibration.json"
    if f.exists():
        return float(json.loads(f.read_text(encoding="utf-8"))["depth"])
    return model.DEPTH


def metrics(res: model.SimResult, drawdown: float = COLLAPSE_DRAWDOWN) -> dict:
    mdd = res.max_drawdown()
    term = res.terminal_ratio()
    collapsed = mdd >= drawdown
    post = res.price[:, res.shock_day - 1:] / res.pre_shock_price()[:, None]
    hit = post <= 1 - drawdown
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), np.nan)
    s0 = res.supply[:, res.shock_day - 1]
    return {
        "p_collapse": float(collapsed.mean()),
        "p_no_recovery": float((collapsed & (term < NO_RECOVERY_LEVEL)).mean()),
        "collapsed": bool(collapsed.mean() > MAJORITY),
        "median_max_drawdown": float(np.median(mdd)),
        "median_terminal_ratio": float(np.median(term)),
        "p10_terminal_ratio": float(np.percentile(term, 10)),
        "p90_terminal_ratio": float(np.percentile(term, 90)),
        "median_days_to_collapse": float(np.nanmedian(first)) if collapsed.any() else None,
        "median_supply_growth": float(np.median(res.supply[:, -1] / s0)),
        "median_min_depth_ratio": float(np.median(res.depth[:, res.shock_day:].min(axis=1) / res.depth[:, 0])),
        "median_pre_shock_price": float(np.median(res.pre_shock_price())),
        "median_market_terminal": float(np.median(res.market[:, -1] / res.market[:, res.shock_day - 1])),
        "shock_day": int(res.shock_day),
    }


def run_matrix(paths: int = PATHS, seed: int = SEED, depth: float | None = None,
               keep_paths: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    depth = calibrated_depth() if depth is None else depth
    rows, curves = [], []
    for d in DESIGNS:
        for s in SCENARIOS:
            res = model.simulate(d, s, paths=paths, seed=seed, depth=depth)
            rows.append({"design": d.key, "design_name": d.name, "scenario": s.key,
                         "scenario_name": s.name, **metrics(res)})
            if keep_paths:
                rel = res.price / res.pre_shock_price()[:, None]
                days = np.arange(0, s.horizon, 5)
                q = np.percentile(rel[:, days], [10, 50, 90], axis=0)
                for i, day in enumerate(days):
                    curves.append({"design": d.key, "scenario": s.key, "day": int(day) - res.shock_day,
                                   "p10": q[0, i], "median": q[1, i], "p90": q[2, i]})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def count_collapsed(df: pd.DataFrame, scenarios=CRASH_SCENARIOS) -> tuple[int, list[str]]:
    sub = df[df["scenario"].isin(scenarios)]
    hit = sub.groupby("design")["collapsed"].any()
    return int(hit.sum()), sorted(hit[hit].index)


def sensitivity(paths: int = 2000, seed: int = SEED) -> dict:
    """How the headline count moves with crash severity, collapse threshold, depth and seed."""
    depth = calibrated_depth()
    crash = SCENARIO_BY_KEY["crash"]
    out: dict = {"paths": paths}

    def n_collapsed(scn, dd=COLLAPSE_DRAWDOWN, dep=depth, sd=seed):
        names = []
        for d in DESIGNS:
            res = model.simulate(d, scn, paths=paths, seed=sd, depth=dep)
            if (res.max_drawdown() >= dd).mean() > MAJORITY:
                names.append(d.key)
        return {"count": len(names), "designs": names}

    out["crash_severity"] = {
        f"{int(x * 100)}%": n_collapsed(dataclasses.replace(crash, market_drop=x))
        for x in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    }
    out["collapse_threshold"] = {
        f"{int(x * 100)}%": n_collapsed(crash, dd=x) for x in (0.80, 0.85, 0.90, 0.95)
    }
    out["market_depth"] = {
        f"x{m}": n_collapsed(crash, dep=depth * m) for m in (0.5, 0.75, 1.0, 1.5, 2.0)
    }
    out["seeds"] = {str(sd): n_collapsed(crash, sd=sd) for sd in (1, 2, 3, 4, 5)}
    return out


def run(verbose: bool = True) -> dict:
    RESULTS.mkdir(exist_ok=True)
    df, curves = run_matrix()
    df.to_csv(RESULTS / "stress_results.csv", index=False)
    curves.to_csv(RESULTS / "stress_paths.csv", index=False)

    n, names = count_collapsed(df)
    per_scn = {s.key: sorted(df[(df.scenario == s.key) & df.collapsed]["design"]) for s in SCENARIOS}
    summary = {
        "n_designs": len(DESIGNS),
        "paths": PATHS,
        "seed": SEED,
        "depth": calibrated_depth(),
        "collapse_definition": f">= {COLLAPSE_DRAWDOWN:.0%} below the pre-shock price in more than "
                               f"{MAJORITY:.0%} of paths",
        "crash_scenarios": list(CRASH_SCENARIOS),
        "collapsed_in_crashes": n,
        "collapsed_designs": names,
        "survived_designs": sorted(set(d.key for d in DESIGNS) - set(names)),
        "collapsed_by_scenario": per_scn,
        "close_calls": [
            {"design": r.design, "scenario": r.scenario, "p_collapse": r.p_collapse}
            for r in df.itertuples()
            if r.scenario in CRASH_SCENARIOS and abs(r.p_collapse - MAJORITY) < BORDERLINE
        ],
        "sensitivity": sensitivity(),
    }
    (RESULTS / "stress_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        piv = df.pivot(index="design", columns="scenario", values="p_collapse")
        piv = piv.loc[[d.key for d in DESIGNS], [s.key for s in SCENARIOS]]
        print("P(collapse) by design and scenario")
        print(piv.round(2).to_string())
        print(f"\ncollapsed in crashes: {n} of {len(DESIGNS)} -> {names}")
        print("  per scenario:", {k: len(v) for k, v in per_scn.items()})
        print("  close calls:", [(c["design"], c["scenario"], round(c["p_collapse"], 3)) for c in summary["close_calls"]])
        for k, v in summary["sensitivity"].items():
            if isinstance(v, dict):
                print(f"  {k}: " + ", ".join(f"{a}={b['count']}" for a, b in v.items()))
    return summary


if __name__ == "__main__":
    run()
