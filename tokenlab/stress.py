"""Run every design under every scenario.

A path collapses if the token trades 90% or more below its pre-shock price at any point
after the shock. A design collapses in a scenario if more than half of its paths do.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import model
from .designs import DESIGNS, REDESIGN, REDESIGN_STEPS
from .shocks import BY_KEY as SCENARIO_BY_KEY
from .shocks import CRASH_SCENARIOS, SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

COLLAPSE_DRAWDOWN = 0.90
NO_RECOVERY_LEVEL = 0.25
MAJORITY = 0.5
PATHS = 4000
BORDERLINE = 0.05   # P(collapse) within 5 points of 50% is a close call
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


def run_matrix(designs=DESIGNS, paths: int = PATHS, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    depth = calibrated_depth()
    rows, curves = [], []
    for d in designs:
        for s in SCENARIOS:
            res = model.simulate(d, s, paths=paths, seed=seed, depth=depth)
            rows.append({"design": d.key, "design_name": d.name, "scenario": s.key,
                         "scenario_name": s.name, **metrics(res)})
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
    """Collapse count in the fast crash as crash size, threshold, depth and seed change."""
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


def p_collapse(design, scenario_key: str, paths: int, seed: int = SEED) -> float:
    res = model.simulate(design, SCENARIO_BY_KEY[scenario_key], paths=paths, seed=seed, depth=calibrated_depth())
    return float((res.max_drawdown() >= COLLAPSE_DRAWDOWN).mean())


WIPEOUT_LEVEL = 0.01      # ends (last 30 days) below 1% of the pre-shock price
SUPPLY_BLOWUP = 10.0      # circulating supply ends more than 10x where it started


def step_metrics(design, paths: int, seed: int = SEED) -> dict:
    """For one redesign step: P(collapse) per crash scenario (the 90% test), plus two outcomes
    the 90% test cannot tell apart from an ordinary deep crash: ending near zero, and runaway
    dilution of circulating supply."""
    out = {"wipeout": {}, "supply_blowup": {}, "pre_shock_price": {}}
    for k in ("baseline", *CRASH_SCENARIOS):
        res = model.simulate(design, SCENARIO_BY_KEY[k], paths=paths, seed=seed, depth=calibrated_depth())
        if k in CRASH_SCENARIOS:
            out[k] = float((res.max_drawdown() >= COLLAPSE_DRAWDOWN).mean())
        out["wipeout"][k] = float((res.terminal_ratio() < WIPEOUT_LEVEL).mean())
        out["supply_blowup"][k] = float((res.supply[:, -1] / res.supply[:, 0] > SUPPLY_BLOWUP).mean())
        out["pre_shock_price"][k] = float(np.median(res.pre_shock_price() / res.price[:, 0]))
    return out


def behaviour_check(keys: list[str], paths: int = 2000) -> dict:
    """P(collapse) in the fast crash with beta, reflexivity and price-led usage set to the
    values the surviving designs use, everything else unchanged."""
    out = {}
    for d in DESIGNS:
        if d.key in keys:
            calm = dataclasses.replace(d, beta=1.2, reflexivity=0.15, usage_price_elasticity=0.0)
            out[d.key] = {"as_designed": p_collapse(d, "crash", paths), "calm": p_collapse(calm, "crash", paths)}
    return out


def run() -> dict:
    RESULTS.mkdir(exist_ok=True)
    df, curves = run_matrix()
    df.to_csv(RESULTS / "stress_results.csv", index=False)
    curves.to_csv(RESULTS / "stress_paths.csv", index=False)
    fixed, _ = run_matrix([REDESIGN])
    fixed.to_csv(RESULTS / "redesign_results.csv", index=False)

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
        "redesign": {
            "key": REDESIGN.key,
            "collapsed_in_crashes": count_collapsed(fixed)[0] > 0,
            "wipeout_level": WIPEOUT_LEVEL,
            "supply_blowup": SUPPLY_BLOWUP,
            "steps": [{"label": label, **step_metrics(d, PATHS)} for label, d in REDESIGN_STEPS],
        },
        "behaviour_check": behaviour_check(names),
        "sensitivity": sensitivity(),
    }
    (RESULTS / "stress_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    piv = pd.concat([df, fixed]).pivot(index="design", columns="scenario", values="p_collapse")
    piv = piv.loc[[d.key for d in DESIGNS] + [REDESIGN.key], [s.key for s in SCENARIOS]]
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
