"""Stress scenarios. Every scenario shares the same horizon (3 years) and, unless stated,
applies its shock one year after launch."""
from __future__ import annotations

from .model import Scenario

SCENARIOS: list[Scenario] = [
    Scenario(
        key="baseline",
        name="Baseline",
        description="No shock. Flat-drift market with normal crypto volatility.",
    ),
    Scenario(
        key="crash",
        name="Market crash",
        description="Crypto market falls 60% in 45 days with doubled volatility, then recovers "
                    "40% of the (log) loss over the following year.",
        market_drop=0.60, drop_days=45, recovery=0.4, recovery_days=365, vol_mult=2.0,
    ),
    Scenario(
        key="crash_at_unlock",
        name="Crash at the largest unlock",
        description="The same crash, timed to start 15 days before the design's largest cliff "
                    "unlock (identical to 'Market crash' for designs without cliffs).",
        market_drop=0.60, drop_days=45, recovery=0.4, recovery_days=365, vol_mult=2.0,
        align_to_unlock=True,
    ),
    Scenario(
        key="bear",
        name="Prolonged bear market",
        description="Slow crash: the market grinds down 75% over a year with no recovery.",
        market_drop=0.75, drop_days=365, recovery=0.0,
    ),
    Scenario(
        key="demand",
        name="Demand shock",
        description="Protocol usage falls 70% in 60 days while the wider market is unaffected.",
        demand_drop=0.70, demand_days=60,
    ),
    Scenario(
        key="liquidity",
        name="Liquidity shock",
        description="Market depth falls 70% for six months alongside a 25% market dip.",
        market_drop=0.25, drop_days=30, recovery=0.5, recovery_days=180,
        liquidity_mult=0.3, liquidity_days=180,
    ),
]

BY_KEY = {s.key: s for s in SCENARIOS}
# The three market-drawdown scenarios. A design "collapses in crashes" if it collapses in
# at least one of them; per-scenario results are always reported alongside.
CRASH_SCENARIOS = ("crash", "crash_at_unlock", "bear")
