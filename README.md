# Tokenomics Stress Lab

Independent simulations of token supply, unlocks and incentives under shocks.

A Python simulator for token economies, calibrated on real unlock data, used to stress-test
12 token designs. **Full write-up: [REPORT.md](REPORT.md).**

## Results

- **Backtest of 60 real unlock events.** 60 insider/investor cliff unlocks across 34 tokens
  (2022-2026, $13.4bn unlocked). From 30 days before to 30 days after the unlock, tokens
  returned -8.1% on average relative to the market (median -19.0%, t = -3.33); 73% of events
  were negative. Larger unlocks relative to float did worse (slope t = -3.05).
- **Stress test of 12 token designs.** 6 scenarios, 4,000 Monte Carlo paths per cell.
  **7 of 12 designs collapsed in crashes** (fell 90%+ below their pre-crash price in most
  paths): low-float/high-FDV, vote-escrow emissions DEX, liquidity-mining farm,
  algorithmic-stablecoin share token, play-to-earn, memecoin, leveraged-collateral governance.
  The 5 survivors: fair-launch capped supply, revenue buyback-and-burn, proof-of-stake L1,
  burn-and-mint, high-APY rebase.
- **How firm is the 7?** It holds in each of the three crash scenarios at the default
  settings, but two verdicts are close to the 50% line in the fast crash (liquidity farm 50%,
  vote-escrow 54%), and the count moves with the assumptions: 2 designs collapse in a 50%
  crash, 8 in a 70% crash; 8 at an 85% drawdown threshold, 2 at 95%. See the sensitivity
  tables in the report.

![Collapse probability by design and scenario](results/figures/stress_heatmap.png)

## How it works

| Step | Module | What it does |
|---|---|---|
| Data | `tokenlab/backtest/data.py` | Pulls vesting schedules (DefiLlama open emissions datasets) and daily prices (DefiLlama coins API); selects the 60 largest qualifying cliff unlocks with a fixed rule |
| Backtest | `tokenlab/backtest/study.py` | Event study: market-model abnormal returns vs BTC over days -30..+30 |
| Calibration | `tokenlab/calibrate.py` | Fits the simulator's market-depth parameter so replayed unlocks match the observed average price impact |
| Simulator | `tokenlab/model.py` | Daily-step Monte Carlo: vesting, emissions, staking, revenue and buybacks, liquidity flight, reflexive mechanisms |
| Designs | `tokenlab/designs.py` | The 12 archetypes |
| Shocks | `tokenlab/shocks.py` | Baseline, market crash, crash at the largest unlock, prolonged bear, demand shock, liquidity shock |
| Stress test | `tokenlab/stress.py` | Design x scenario matrix, collapse verdicts, sensitivity analysis |
| Report | `tokenlab/report.py` | Figures and `REPORT.md`, generated from `results/` |

## Run it

```bash
pip install -r requirements.txt

python -m tokenlab all        # backtest -> calibrate -> stress -> report (offline, ~15 min)
python -m tokenlab data       # optional: rebuild the unlock sample from the APIs (~1 hour)
python -m pytest              # tests
```

The curated sample (`data/unlock_events.csv`, `data/event_prices.csv`) is committed, so
everything except `data` runs offline and reproduces the report exactly (fixed seeds).

Test your own design:

```python
from tokenlab.model import Allocation, TokenDesign, simulate
from tokenlab.shocks import BY_KEY as SCENARIOS

design = TokenDesign(
    key="mine", name="My token", description="",
    allocations=[
        Allocation("Public", 0.40, "public", tge=1.0),
        Allocation("Investors", 0.25, "investor", cliff_days=365, cliff_frac=0.25, vest_days=730),
        Allocation("Team", 0.20, "team", cliff_days=365, cliff_frac=0.25, vest_days=730),
        Allocation("Treasury", 0.15, "treasury", vest_days=1460),
    ],
    revenue_yield=0.05, buyback_share=0.5,
)
res = simulate(design, SCENARIOS["crash"], paths=2000, seed=1)
print("P(collapse):", (res.max_drawdown() >= 0.9).mean())
```

## Caveats

The 12 designs are stylised archetypes with round-number parameters, not models of specific
tokens, and only market depth is fitted to data; the other behavioural constants are stated
assumptions. The collapse count is a property of these designs under this model. Unlock
schedules come from a single provider and may contain errors. Not investment advice.

## License

MIT
