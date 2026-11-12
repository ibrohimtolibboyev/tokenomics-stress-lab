import dataclasses

import numpy as np
import pytest

from tokenlab import model
from tokenlab.calibrate import simulated_car
from tokenlab.designs import BY_KEY, DESIGNS
from tokenlab.model import Allocation, Scenario, TokenDesign
from tokenlab.shocks import BY_KEY as SCN
from tokenlab.shocks import SCENARIOS


def test_twelve_valid_designs():
    assert len(DESIGNS) == 12
    assert len({d.key for d in DESIGNS}) == 12
    for d in DESIGNS:
        d.validate()


def test_allocation_schedule_releases_everything():
    a = Allocation("x", 0.2, "investor", tge=0.1, cliff_days=100, cliff_frac=0.25, vest_days=200)
    s = a.schedule(1000)
    assert s.sum() == pytest.approx(0.2)
    assert s[0] == pytest.approx(0.02)
    assert s[1:100].sum() == 0
    assert s[100] == pytest.approx(0.18 * 0.25 + 0.18 * 0.75 / 200)


def test_bad_allocations_rejected():
    d = TokenDesign("bad", "bad", "", [Allocation("p", 0.9, "public", tge=1.0)])
    with pytest.raises(ValueError):
        d.validate()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.key)
def test_every_design_runs_and_stays_finite(scenario):
    for d in DESIGNS:
        r = model.simulate(d, scenario, paths=20, seed=3)
        for arr in (r.price, r.supply, r.depth, r.usage):
            assert np.isfinite(arr).all(), (d.key, scenario.key)
        assert (r.price > 0).all() and (r.supply > 0).all() and (r.depth > 0).all()


def test_simulation_is_reproducible():
    a = model.simulate(BY_KEY["pos_l1"], SCN["crash"], paths=10, seed=5)
    b = model.simulate(BY_KEY["pos_l1"], SCN["crash"], paths=10, seed=5)
    assert np.array_equal(a.price, b.price)


def test_crash_is_worse_than_baseline():
    for key in ("fair_launch", "memecoin", "algo_stable"):
        base = model.simulate(BY_KEY[key], SCN["baseline"], paths=300, seed=1)
        crash = model.simulate(BY_KEY[key], SCN["crash"], paths=300, seed=1)
        assert np.median(crash.max_drawdown()) > np.median(base.max_drawdown())


def test_bigger_crash_hurts_more():
    d = BY_KEY["revenue_buyback"]
    small = model.simulate(d, dataclasses.replace(SCN["crash"], market_drop=0.3), paths=300, seed=1)
    big = model.simulate(d, dataclasses.replace(SCN["crash"], market_drop=0.8), paths=300, seed=1)
    assert np.median(big.max_drawdown()) > np.median(small.max_drawdown())


def test_unlock_impact_grows_with_size_and_shrinks_with_depth():
    assert simulated_car(0.30, "privateSale", 0.4) < simulated_car(0.05, "privateSale", 0.4) < 0
    assert simulated_car(0.10, "privateSale", 0.2) < simulated_car(0.10, "privateSale", 0.8)
    # investors sell more than team members in the model
    assert simulated_car(0.10, "privateSale", 0.4) < simulated_car(0.10, "insiders", 0.4)


def test_no_supply_no_shock_token_is_flat_without_noise():
    d = TokenDesign("flat", "flat", "", [Allocation("p", 1.0, "public", tge=1.0)],
                    beta=0.0, idio_vol=0.0, usage_vol=0.0, usage_growth=0.0)
    s = Scenario("quiet", "quiet", "", horizon=200, shock_day=100, market_vol=0.0)
    r = model.simulate(d, s, paths=1)
    assert np.allclose(r.price, 1.0)
    assert np.allclose(r.supply, 1.0)


def test_algo_stable_run_mints_supply():
    r = model.simulate(BY_KEY["algo_stable"], SCN["bear"], paths=200, seed=2)
    plain = dataclasses.replace(BY_KEY["algo_stable"], stable_debt=0.0)
    p = model.simulate(plain, SCN["bear"], paths=200, seed=2)
    assert np.median(r.supply[:, -1]) > np.median(p.supply[:, -1])
    assert np.median(r.max_drawdown()) >= np.median(p.max_drawdown())
