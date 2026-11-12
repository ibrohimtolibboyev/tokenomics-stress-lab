"""The 12 token designs put through the stress lab.

These are stylised archetypes of designs that exist in the market, not models of any
specific token. Parameters are set from typical published tokenomics (allocation tables,
vesting terms, emission schedules) and are deliberately round numbers.
"""
from __future__ import annotations

from .model import Allocation as A
from .model import TokenDesign


def _insiders(investors: float, team: float, cliff: int = 365, cliff_frac: float = 0.25,
              vest: int = 730) -> list[A]:
    return [
        A("Investors", investors, "investor", cliff_days=cliff, cliff_frac=cliff_frac, vest_days=vest),
        A("Team", team, "team", cliff_days=cliff, cliff_frac=cliff_frac, vest_days=vest),
    ]


DESIGNS: list[TokenDesign] = [
    TokenDesign(
        key="fair_launch",
        name="Fair-launch capped supply",
        description="No insiders, no treasury. Fully liquid float plus a halving emission to "
                    "miners/validators. No cash flows; value is purely monetary premium.",
        allocations=[A("Public", 1.0, "public", tge=1.0)],
        beta=1.1, reflexivity=0.15, depth_mult=1.5, pol_share=0.3,
        emission_rate=0.06, emission_halflife=1460, emission_sell_frac=0.7,
    ),
    TokenDesign(
        key="revenue_buyback",
        name="Revenue buyback-and-burn",
        description="Fee-generating protocol that spends half of revenue buying and burning "
                    "tokens and pays a further share to stakers. Standard 1y cliff / 3y vest.",
        allocations=[
            A("Public", 0.30, "public", tge=1.0),
            *_insiders(0.15, 0.15, cliff_frac=0.2, vest=1095),
            A("Ecosystem", 0.25, "ecosystem", tge=0.1, vest_days=1460),
            A("Treasury", 0.15, "treasury", vest_days=1460),
        ],
        beta=1.2, reflexivity=0.15, stake_share=0.3, exit_sens=0.005,
        revenue_yield=0.08, buyback_share=0.5, staker_share=0.2,
        opex=0.02, treasury_stables=0.05,
    ),
    TokenDesign(
        key="pos_l1",
        name="Proof-of-stake L1",
        description="5% staking inflation, 60% of supply staked with a 21-day unbonding period, "
                    "half of gas fees burned. Insiders on a 1y cliff / 3y vest.",
        allocations=[
            A("Public", 0.40, "public", tge=1.0),
            *_insiders(0.20, 0.20, vest=1095),
            A("Ecosystem", 0.10, "ecosystem", tge=0.1, vest_days=1460),
            A("Foundation", 0.10, "treasury", vest_days=1460),
        ],
        beta=1.2, reflexivity=0.15, depth_mult=1.5,
        emission_rate=0.05, emission_sell_frac=0.5,
        stake_share=0.6, unbond_days=21, exit_sens=0.004,
        revenue_yield=0.03, burn_mint_share=0.5, staker_share=0.5,
        opex=0.01, treasury_stables=0.05,
    ),
    TokenDesign(
        key="burn_mint",
        name="Burn-and-mint (DePIN)",
        description="Users must buy and burn tokens to pay for a real-world service; operators "
                    "are paid with a decaying emission. Usage is only weakly tied to crypto markets.",
        allocations=[
            A("Public", 0.30, "public", tge=1.0),
            *_insiders(0.20, 0.20),
            A("Ecosystem", 0.20, "ecosystem", tge=0.1, vest_days=1460),
            A("Treasury", 0.10, "treasury", vest_days=1460),
        ],
        beta=1.2, reflexivity=0.15,
        emission_rate=0.12, emission_halflife=730, emission_sell_frac=0.7,
        revenue_yield=0.06, burn_mint_share=1.0, usage_growth=0.4, usage_beta=0.3,
        opex=0.02, treasury_stables=0.05,
    ),
    TokenDesign(
        key="ve_lock",
        name="Vote-escrow emissions DEX",
        description="High emissions directed to liquidity pools by vote-locked holders, who "
                    "receive trading fees. Locks average one year; liquidity is rented with emissions.",
        allocations=[
            A("Public", 0.35, "public", tge=1.0),
            A("Team", 0.20, "team", cliff_days=180, vest_days=1095),
            A("Investors", 0.10, "investor", cliff_days=180, vest_days=1095),
            A("Ecosystem", 0.25, "ecosystem", tge=0.05, vest_days=1460),
            A("Treasury", 0.10, "treasury", vest_days=1460),
        ],
        beta=1.3, reflexivity=0.25, pol_share=0.1,
        emission_rate=0.25, emission_halflife=730, emission_sell_frac=0.6, emission_to_lp=0.8,
        stake_share=0.45, unbond_days=365, exit_sens=0.004,
        revenue_yield=0.06, staker_share=0.8, usage_beta=0.9, usage_price_elasticity=0.2,
        opex=0.01, treasury_stables=0.03,
    ),
    TokenDesign(
        key="lowfloat_vc",
        name="Low-float / high-FDV VC token",
        description="~11% of supply liquid at launch; 45% held by investors and team with a "
                    "one-year cliff followed by two years of vesting. Little revenue, no value capture.",
        allocations=[
            A("Airdrop", 0.10, "public", tge=1.0),
            *_insiders(0.25, 0.20),
            A("Ecosystem", 0.30, "ecosystem", tge=0.05, vest_days=1460),
            A("Treasury", 0.15, "treasury", vest_days=1460),
        ],
        beta=1.4, reflexivity=0.25, usage_price_elasticity=0.1,
        revenue_yield=0.02, opex=0.03, treasury_stables=0.05,
    ),
    TokenDesign(
        key="liquidity_farm",
        name="Liquidity-mining farm token",
        description="Very high emissions paid to liquidity providers, who mostly sell. Usage and "
                    "liquidity exist only while the reward is valuable.",
        allocations=[
            A("Public", 0.50, "public", tge=1.0),
            A("Team", 0.15, "team", vest_days=365),
            A("Ecosystem", 0.25, "ecosystem", tge=0.1, vest_days=730),
            A("Treasury", 0.10, "treasury", vest_days=730),
        ],
        beta=1.4, reflexivity=0.30, pol_share=0.0, liq_flight=1.2,
        emission_rate=0.5, emission_halflife=365, emission_sell_frac=0.8, emission_to_lp=0.9,
        revenue_yield=0.03, usage_price_elasticity=0.5, usage_beta=0.9,
    ),
    TokenDesign(
        key="algo_stable",
        name="Algorithmic-stablecoin share token",
        description="Token absorbs the volatility of an under-collateralised stablecoin: each $1 "
                    "redeemed mints $1 of new tokens. Stablecoin supply starts at 30% of token market cap.",
        allocations=[
            A("Public", 0.50, "public", tge=1.0),
            *_insiders(0.20, 0.20),
            A("Treasury", 0.10, "treasury", vest_days=1460),
        ],
        beta=1.3, reflexivity=0.30, stake_share=0.3, exit_sens=0.02,
        revenue_yield=0.05, burn_mint_share=0.5,
        stable_debt=0.3, usage_price_elasticity=0.3,
        opex=0.02, treasury_stables=0.1,
    ),
    TokenDesign(
        key="rebase_pol",
        name="High-APY rebase / protocol-owned liquidity",
        description="Triple-digit staking APY paid in new tokens, 90% of supply staked with no "
                    "unbonding delay. The protocol owns its liquidity and a treasury backing ~10% of price.",
        allocations=[
            A("Public", 0.70, "public", tge=1.0),
            A("Team", 0.10, "team", vest_days=365),
            A("Investors", 0.10, "investor", vest_days=365),
            A("Treasury", 0.10, "treasury", vest_days=730),
        ],
        beta=1.3, reflexivity=0.35, usage_price_elasticity=0.45,
        depth_mult=0.7, pol_share=0.9,
        emission_rate=0.8, emission_halflife=365, emission_sell_frac=0.25,
        stake_share=0.9, unbond_days=0, exit_base=0.001, exit_sens=0.05,
        backing=0.10,
    ),
    TokenDesign(
        key="play_to_earn",
        name="Play-to-earn reward token",
        description="Uncapped reward emission to players, who sell most of it. Rewards are topped "
                    "up as price falls; player activity (and therefore token sinks) follows the token price.",
        allocations=[
            A("Public", 0.20, "public", tge=1.0),
            *_insiders(0.20, 0.20),
            A("Ecosystem", 0.30, "ecosystem", tge=0.05, vest_days=1095),
            A("Treasury", 0.10, "treasury", vest_days=1460),
        ],
        beta=1.4, reflexivity=0.25,
        emission_rate=0.12, emission_sell_frac=0.85, emission_usd_target=0.5,
        revenue_yield=0.10, burn_mint_share=0.4, usage_price_elasticity=0.5, usage_beta=0.8,
        opex=0.03, treasury_stables=0.05,
    ),
    TokenDesign(
        key="memecoin",
        name="Memecoin",
        description="Entire supply liquid from day one, liquidity pool half burned. No revenue, "
                    "no sinks; price is attention and momentum.",
        allocations=[
            A("Public", 0.90, "public", tge=1.0),
            A("Deployer", 0.10, "team", tge=1.0),
        ],
        beta=1.6, idio_vol=0.9, reflexivity=0.40,
        usage_price_elasticity=0.45, usage_beta=1.0, depth_mult=0.8, pol_share=0.5, liq_flight=1.0,
    ),
    TokenDesign(
        key="leveraged_collateral",
        name="Leveraged-collateral governance token",
        description="Half the float is staked and widely re-used as loan collateral. Price falls "
                    "trigger liquidation bands and fast unstaking (7-day unbonding).",
        allocations=[
            A("Public", 0.25, "public", tge=1.0),
            *_insiders(0.20, 0.20, cliff_frac=0.2),
            A("Ecosystem", 0.25, "ecosystem", tge=0.1, vest_days=1460),
            A("Treasury", 0.10, "treasury", vest_days=1460),
        ],
        beta=1.3, reflexivity=0.30,
        stake_share=0.5, unbond_days=7, exit_sens=0.03,
        revenue_yield=0.04, staker_share=0.5, usage_price_elasticity=0.2,
        liquidation_bands=[(0.35, 0.03), (0.50, 0.04), (0.65, 0.05)],
        opex=0.02, treasury_stables=0.05,
    ),
]

BY_KEY = {d.key: d for d in DESIGNS}
