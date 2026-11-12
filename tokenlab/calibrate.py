"""Calibrate the simulator's market-depth parameter against the real unlock backtest.

For each of the real events we simulate a plain token (no emissions, no revenue, no noise)
that experiences one cliff unlock of the same size relative to circulating supply, going to
the same recipient type, and measure the simulated price change over days [-30, +30].
`depth` is chosen so that the mean simulated change equals the mean cumulative abnormal
return observed in the data.

Run:  python -m tokenlab.calibrate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import model
from .model import Allocation, Scenario, TokenDesign

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVENT_DAY = 200
SIZE_CAP = 1.0   # unlocks larger than 100% of circulating supply are treated as 100%

COHORTS = {
    "insiders": {"team": 1.0},
    "privateSale": {"investor": 1.0},
    "insiders+privateSale": {"team": 0.5, "investor": 0.5},
}


def reference_design(pct_circ: float, category: str) -> TokenDesign:
    x = min(pct_circ, SIZE_CAP)
    allocs = [Allocation("Float", 1 / (1 + x), "public", tge=1.0)]
    for cohort, w in COHORTS[category].items():
        allocs.append(Allocation(cohort, w * x / (1 + x), cohort, cliff_days=EVENT_DAY, cliff_frac=1.0))
    return TokenDesign(key="ref", name="reference", description="", allocations=allocs,
                       beta=0.0, idio_vol=0.0, usage_vol=0.0, usage_growth=0.0, reflexivity=0.25)


def simulated_car(pct_circ: float, category: str, depth: float) -> float:
    scn = Scenario(key="quiet", name="quiet", description="", horizon=EVENT_DAY + 60,
                   shock_day=EVENT_DAY, market_vol=0.0)
    res = model.simulate(reference_design(pct_circ, category), scn, paths=1, seed=0, depth=depth)
    p = res.price[0]
    return float(np.log(p[EVENT_DAY + 30] / p[EVENT_DAY - 30]))


def mean_simulated_car(events: pd.DataFrame, depth: float) -> float:
    return float(np.mean([simulated_car(r.pct_circ, r.category, depth) for r in events.itertuples()]))


def run(verbose: bool = True) -> dict:
    events = pd.read_csv(RESULTS / "backtest_events.csv")
    target = float(events["car_full"].mean())
    out = {"target_mean_car_log": target, "n_events": len(events), "default_depth": model.DEPTH}

    if target >= -0.005:
        out.update(depth=model.DEPTH, calibrated=False,
                   note="No negative average abnormal return in the sample; default depth kept.")
    else:
        lo, hi = 0.02, 50.0           # simulated CAR is monotonically increasing in depth
        for _ in range(40):
            mid = (lo * hi) ** 0.5
            if mean_simulated_car(events, mid) < target:
                lo = mid
            else:
                hi = mid
        depth = (lo * hi) ** 0.5
        sim = np.array([simulated_car(r.pct_circ, r.category, depth) for r in events.itertuples()])
        out.update(depth=depth, calibrated=True, simulated_mean_car_log=float(sim.mean()),
                   simulated_median_car_log=float(np.median(sim)),
                   empirical_median_car_log=float(events["car_full"].median()),
                   corr_sim_vs_actual=float(np.corrcoef(sim, events["car_full"])[0, 1]))
        events.assign(sim_car_full=sim)[["event_id", "pct_circ", "category", "car_full", "sim_car_full"]] \
            .to_csv(RESULTS / "calibration_events.csv", index=False)

    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
