"""Daily-step Monte Carlo simulator for a token economy.

Log price each day:

    dlnP = beta * market return + noise
         + FAIR_SPEED * (ln F - ln P)      F ~ market^beta * adoption / circ^DILUTION
         + momentum feedback + value support
         + (usd bought / P - tokens sold) / (depth * circ)

Token amounts are shares of initial total supply and the launch price is 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEPTH = 0.50            # selling x% of float moves price ~x/DEPTH %; fitted in calibrate.py
HEDGE_FRAC = 0.5        # share of a cliff's expected sales that is pre-hedged (shorted)
HEDGE_DAYS = 30         # ... over this many days before the cliff
SELL_DAYS = 20          # e-folding time of post-unlock selling
MOM_WINDOW = 30         # momentum look-back (days)
MOM_DEADBAND = 0.15     # monthly moves smaller than this do not trigger trend-following flows
FAIR_SPEED = 0.01       # daily reversion of price toward fair value (half-life ~70 days)
ADOPTION_RECOVERY = 0.5 # price-driven adoption returns at half the speed it leaves (hysteresis)
DILUTION = 0.5          # share of circulating-supply growth that lowers fair value per token
VALUE_MULTIPLE = 10.0   # fundamental value = 10x annual value accrual
VALUE_SPEED = 0.01      # daily pull toward fundamental value when price is below it
MAX_DAILY_MOVE = 0.7    # clip on |dlnP| per day
DD_WINDOW = 180         # rolling window for drawdown reference

# Share of newly unlocked tokens that each cohort sells (within ~SELL_DAYS)
UNLOCK_SELL_FRAC = {
    "investor": 0.45,
    "team": 0.25,
    "community": 0.60,   # airdrops
    "ecosystem": 0.35,   # grants / incentives budgets
    "public": 0.0,       # already liquid at launch
    "treasury": 0.0,     # goes to the treasury, sold only to fund the deficit
}


@dataclass
class Allocation:
    name: str
    share: float                 # share of total supply
    cohort: str                  # key of UNLOCK_SELL_FRAC
    tge: float = 0.0             # fraction liquid at launch
    cliff_days: int = 0          # days until the cliff
    cliff_frac: float = 0.0      # fraction released at the cliff
    vest_days: int = 0           # linear vesting of the remainder after the cliff

    def schedule(self, horizon: int) -> np.ndarray:
        """Tokens unlocked on each day 0..horizon-1 (share of total supply)."""
        out = np.zeros(horizon)
        out[0] += self.share * self.tge
        rest = self.share * (1 - self.tge)
        if self.cliff_days < horizon:
            out[self.cliff_days] += rest * self.cliff_frac
        lin = rest * (1 - self.cliff_frac)
        if self.vest_days > 0:
            a, b = self.cliff_days, min(self.cliff_days + self.vest_days, horizon)
            if a < horizon:
                out[a:b] += lin / self.vest_days
        elif self.cliff_days < horizon:
            out[self.cliff_days] += lin
        return out


@dataclass
class TokenDesign:
    key: str
    name: str
    description: str
    allocations: list[Allocation]

    # market behaviour
    beta: float = 1.2                 # sensitivity to the crypto market factor
    idio_vol: float = 0.6             # annualised idiosyncratic volatility
    reflexivity: float = 0.25         # share of the trailing-month return (beyond a dead band) that feeds back
    depth_mult: float = 1.0           # liquidity relative to the calibrated market depth
    pol_share: float = 0.2            # share of liquidity that never leaves (protocol-owned)
    liq_flight: float = 0.8           # how fast non-POL liquidity leaves in a drawdown

    # emissions (new supply)
    emission_rate: float = 0.0        # initial annual emissions, share of total supply
    emission_halflife: float = 0.0    # days; 0 = constant
    emission_cap: float = 1e9         # max cumulative emissions (share of total supply)
    emission_sell_frac: float = 0.5   # share of emissions sold on receipt
    emission_usd_target: float = 0.0  # 0..1: emissions scale up as price falls (USD-pegged rewards)
    emission_to_lp: float = 0.0       # share of emissions paying for liquidity (mercenary depth)

    # staking / locking
    stake_share: float = 0.0          # initial share of circulating supply staked/locked
    unbond_days: int = 14             # delay between exit decision and tokens being liquid
    exit_base: float = 0.0005         # baseline daily exit rate of staked tokens
    exit_sens: float = 0.01           # extra daily exit rate per unit drawdown beyond 25%
    exit_sell_frac: float = 0.7       # share of exited tokens that is sold

    # usage, revenue and value capture
    revenue_yield: float = 0.0        # annual protocol revenue / initial circulating mcap
    usage_growth: float = 0.2         # annual organic growth of adoption
    usage_beta: float = 0.6           # sensitivity of fee revenue to market returns
    usage_vol: float = 0.25           # annualised volatility of adoption
    usage_price_elasticity: float = 0.0   # adoption that exists only because of token price (P2E, farming)
    buyback_share: float = 0.0        # share of revenue buying tokens on the market (and burning)
    staker_share: float = 0.0         # share of revenue paid to stakers in USD
    burn_mint_share: float = 0.0      # share of revenue users must pay by buying+burning tokens
    backing: float = 0.0              # hard treasury backing per token / initial price (redeemable floor)

    # treasury
    opex: float = 0.0                 # annual USD operating cost / initial circulating mcap
    treasury_stables: float = 0.0     # USD reserves / initial circulating mcap

    # reflexive mechanisms
    stable_debt: float = 0.0          # algorithmic-stablecoin supply / initial circulating mcap
    run_threshold: float = 1.25       # run starts when token mcap / stablecoin supply falls below this
    run_sens: float = 0.10            # daily redemption rate as coverage goes from threshold to zero
    liquidation_bands: list[tuple[float, float]] = field(default_factory=list)
    # (drawdown level, share of circulating supply force-sold)

    def unlocks(self, horizon: int) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for a in self.allocations:
            out[a.cohort] = out.get(a.cohort, np.zeros(horizon)) + a.schedule(horizon)
        return out

    def validate(self) -> None:
        total = sum(a.share for a in self.allocations)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{self.key}: allocations sum to {total:.4f}, expected 1.0")
        for a in self.allocations:
            if a.cohort not in UNLOCK_SELL_FRAC:
                raise ValueError(f"{self.key}: unknown cohort {a.cohort!r}")


@dataclass
class Scenario:
    key: str
    name: str
    description: str
    horizon: int = 1095
    shock_day: int = 365
    market_vol: float = 0.55            # annualised market volatility
    market_drift: float = 0.0
    market_drop: float = 0.0            # total fractional market fall (0.6 = -60%)
    drop_days: int = 45
    recovery: float = 0.0               # share of the log-loss recovered ...
    recovery_days: int = 365            # ... over this many days after the drop
    vol_mult: float = 1.0               # market vol multiplier during the drop
    demand_drop: float = 0.0            # fractional fall in usage, independent of the market
    demand_days: int = 60
    liquidity_mult: float = 1.0         # depth multiplier during the liquidity shock
    liquidity_days: int = 180
    align_to_unlock: bool = False       # move the shock onto the design's largest cliff

    def market_shock_path(self, shock_day: int) -> np.ndarray:
        """Deterministic daily log-return overlay from the shock."""
        r = np.zeros(self.horizon)
        if self.market_drop > 0:
            total = np.log(1 - self.market_drop)
            a, b = shock_day, min(shock_day + self.drop_days, self.horizon)
            r[a:b] += total / self.drop_days
            if self.recovery > 0 and b < self.horizon:
                c = min(b + self.recovery_days, self.horizon)
                r[b:c] += -total * self.recovery / self.recovery_days
        return r


def _sell_schedule(design: TokenDesign, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (circulating unlocks/day, treasury unlocks/day, scheduled token sales/day)."""
    unlocks = design.unlocks(horizon)
    circ = np.zeros(horizon)
    treas = np.zeros(horizon)
    sales = np.zeros(horizon)
    post = np.exp(-np.arange(horizon) / SELL_DAYS)
    post /= post.sum()
    for cohort, u in unlocks.items():
        if cohort == "treasury":
            treas += u
            continue
        circ += u
        frac = UNLOCK_SELL_FRAC[cohort]
        if frac == 0:
            continue
        u = u.copy()
        u[0] = 0.0  # launch float is the starting state, not a sale
        # a cliff is a lump much larger than the cohort's usual daily unlock
        typical = np.median(u[u > 0]) if (u > 0).any() else 0.0
        is_cliff = u > max(10 * typical, 0.002)
        if (u > 0).sum() <= 3:
            is_cliff = u > 0
        lumps = np.where(is_cliff, u, 0.0)
        drip = u - lumps
        sales += frac * drip
        for d in np.flatnonzero(lumps):
            amt = frac * lumps[d]
            pre = min(HEDGE_DAYS, d - 1) if d > 1 else 0
            hedged = HEDGE_FRAC * amt if pre > 0 else 0.0
            if pre > 0:
                sales[d - pre:d] += hedged / pre
            tail = post[: horizon - d]
            sales[d:] += (amt - hedged) * tail / tail.sum() if tail.sum() > 0 else 0.0
    return circ, treas, sales


def largest_cliff_day(design: TokenDesign, horizon: int, after: int = 60) -> int | None:
    circ, _, _ = _sell_schedule(design, horizon)
    c = circ.copy()
    c[:after] = 0
    if c.max() < 0.01:
        return None
    return int(c.argmax())


@dataclass
class SimResult:
    design: TokenDesign
    scenario: Scenario
    shock_day: int
    price: np.ndarray        # (paths, days)
    supply: np.ndarray       # circulating supply
    depth: np.ndarray        # effective market depth
    usage: np.ndarray
    market: np.ndarray       # market index
    staked: np.ndarray
    treasury_usd: np.ndarray
    sold: np.ndarray         # tokens sold per day (all sources)

    def pre_shock_price(self) -> np.ndarray:
        return self.price[:, self.shock_day - 1]

    def terminal_ratio(self) -> np.ndarray:
        """Average price over the last 30 days relative to the pre-shock price."""
        return self.price[:, -30:].mean(axis=1) / self.pre_shock_price()

    def max_drawdown(self) -> np.ndarray:
        post = self.price[:, self.shock_day - 1:]
        return 1 - post.min(axis=1) / self.pre_shock_price()


def simulate(design: TokenDesign, scenario: Scenario, paths: int = 400, seed: int = 0,
             depth: float | None = None) -> SimResult:
    design.validate()
    T = scenario.horizon
    rng = np.random.default_rng(seed)
    d0 = (DEPTH if depth is None else depth) * design.depth_mult

    shock_day = scenario.shock_day
    if scenario.align_to_unlock:
        c = largest_cliff_day(design, T)
        if c is not None:
            shock_day = max(60, min(c - 15, T - 400))

    unlock_circ, unlock_treas, sched_sales = _sell_schedule(design, T)

    # exogenous market path
    dt = 1 / 365
    vol = np.full(T, scenario.market_vol)
    if scenario.market_drop > 0:
        vol[shock_day:shock_day + scenario.drop_days] *= scenario.vol_mult
    z_m = rng.standard_normal((paths, T))
    r_m = scenario.market_drift * dt + vol * np.sqrt(dt) * z_m   # drift is on the log scale (median path)
    r_m += scenario.market_shock_path(shock_day)
    z_i = rng.standard_normal((paths, T))
    z_u = rng.standard_normal((paths, T))

    demand = np.zeros(T)
    if scenario.demand_drop > 0:
        a, b = shock_day, min(shock_day + scenario.demand_days, T)
        demand[a:b] = np.log(1 - scenario.demand_drop) / scenario.demand_days
    liq_shock = np.ones(T)
    if scenario.liquidity_mult != 1.0:
        liq_shock[shock_day:shock_day + scenario.liquidity_days] = scenario.liquidity_mult

    # state
    C0 = unlock_circ[0]
    if C0 <= 0:
        raise ValueError(f"{design.key}: no circulating supply at launch")
    lnP = np.zeros(paths)
    C = np.full(paths, C0)
    S = np.full(paths, C0 * design.stake_share)           # staked / locked
    unbond = np.zeros((paths, max(design.unbond_days, 1)))  # exit queue (ring buffer)
    treas_tok = np.zeros(paths)
    treas_usd = np.full(paths, design.treasury_stables * C0)
    lnA = np.zeros(paths)          # adoption (idiosyncratic usage)
    lnN = np.zeros(paths)          # accumulated idiosyncratic news
    lnM = np.zeros(paths)          # market index
    debt = np.full(paths, design.stable_debt * C0)
    emitted = np.zeros(paths)
    bands_hit = np.zeros((paths, len(design.liquidation_bands)), dtype=bool)

    rev0 = design.revenue_yield * C0 / 365          # USD per day at launch
    opex = design.opex * C0 / 365
    inc0 = design.emission_rate / 365 * design.emission_to_lp  # LP incentive value/day at launch

    price = np.empty((paths, T)); supply = np.empty((paths, T)); depth_h = np.empty((paths, T))
    usage = np.empty((paths, T)); staked_h = np.empty((paths, T)); treas_h = np.empty((paths, T))
    sold_h = np.empty((paths, T)); dln_h = np.zeros((paths, T))
    market = np.exp(np.cumsum(r_m, axis=1))

    for t in range(T):
        lnP = np.clip(lnP, -25.0, 12.0)
        P = np.exp(lnP)
        lo = max(0, t - DD_WINDOW)
        peak = price[:, lo:t].max(axis=1) if t > 0 else P
        peak = np.maximum(peak, P)
        dd = 1 - P / peak
        mom = dln_h[:, max(0, t - MOM_WINDOW):t].sum(axis=1) if t > 0 else np.zeros(paths)
        mom = np.sign(mom) * np.maximum(np.abs(mom) - MOM_DEADBAND, 0.0)

        # usage and revenue
        # price-driven adoption comes back slower than it leaves
        chase = np.where(mom < 0, mom, ADOPTION_RECOVERY * mom) / MOM_WINDOW
        lnA = lnA + (design.usage_growth * dt + demand[t]
                     + design.usage_price_elasticity * chase + design.usage_vol * np.sqrt(dt) * z_u[:, t])
        lnM = lnM + r_m[:, t]
        U = np.exp(np.clip(lnA + design.usage_beta * lnM, -40, 40))
        rev = rev0 * U

        # supply: unlocks and emissions
        if t > 0:
            C = C + unlock_circ[t]
            treas_tok = treas_tok + unlock_treas[t]
        sells = np.full(paths, sched_sales[t])

        em = np.zeros(paths)
        if design.emission_rate > 0:
            decay = 0.5 ** (t / design.emission_halflife) if design.emission_halflife > 0 else 1.0
            em = np.full(paths, design.emission_rate / 365 * decay)
            if design.emission_usd_target > 0:
                em = em * np.minimum(P ** (-design.emission_usd_target), 10.0)
            em = np.minimum(em, np.maximum(design.emission_cap - emitted, 0.0))
            emitted += em
            C = C + em
            sells += em * design.emission_sell_frac
            if design.stake_share > 0:
                S = S + em * (1 - design.emission_sell_frac) * (1 - design.emission_to_lp)

        # staking exits
        if design.stake_share > 0:
            exit_rate = np.minimum(design.exit_base + design.exit_sens * np.maximum(dd - 0.25, 0), 0.25)
            leaving = S * exit_rate
            S = S - leaving
            slot = t % unbond.shape[1]
            matured = unbond[:, slot].copy() if design.unbond_days > 0 else leaving
            if design.unbond_days > 0:
                unbond[:, slot] = leaving
            sells += matured * design.exit_sell_frac
            S = S + 0.001 * np.maximum(design.stake_share * C - S, 0) * (dd < 0.25)

        # algorithmic stablecoin run
        if design.stable_debt > 0:
            coverage = C * P / np.maximum(debt, 1e-12)
            run = design.run_sens * np.clip(1 - coverage / design.run_threshold, 0, 1)
            redeemed = debt * run
            debt = debt - redeemed
            minted = redeemed / P            # $1 of new tokens per $1 redeemed
            C = C + minted
            sells += minted

        # collateral liquidations
        for j, (level, share) in enumerate(design.liquidation_bands):
            fire = (dd >= level) & ~bands_hit[:, j]
            bands_hit[:, j] |= fire
            sells += fire * share * C

        # treasury: fund the operating deficit
        buys_usd = rev * (design.buyback_share + design.burn_mint_share)
        retained = rev * max(0.0, 1 - design.buyback_share - design.burn_mint_share - design.staker_share)
        deficit = np.maximum(opex - retained, 0.0)
        treas_usd = treas_usd + np.maximum(retained - opex, 0.0)
        from_usd = np.minimum(deficit, treas_usd)
        treas_usd = treas_usd - from_usd
        need_tok = np.minimum((deficit - from_usd) / P, treas_tok)
        treas_tok = treas_tok - need_tok
        C = C + need_tok
        sells += need_tok

        # liquidity
        flight = np.maximum(1 - design.liq_flight * dd, 0.25)
        lam = d0 * liq_shock[t] * (design.pol_share + (1 - design.pol_share) * flight)
        if inc0 > 0:
            inc = em * design.emission_to_lp * P
            lam = lam * np.clip((inc / inc0) ** 0.5, 0.5, 1.5)
        lam = np.maximum(lam, 0.02)

        # fundamental value
        capture = design.buyback_share + design.staker_share + design.burn_mint_share
        V = (rev * 365 * capture * VALUE_MULTIPLE) / C + design.backing
        support = VALUE_SPEED * np.clip(np.log(np.maximum(V, 1e-12) / P), 0, 3)

        # price update
        flow = (buys_usd / P - sells) / C
        news = design.idio_vol * np.sqrt(dt) * z_i[:, t]
        lnN = lnN + news
        lnF = design.beta * lnM + lnA + lnN - DILUTION * np.log(C / C0)
        dln = (design.beta * r_m[:, t] + news + FAIR_SPEED * (lnF - lnP)
               + design.reflexivity / MOM_WINDOW * mom * np.clip(d0 / lam, 1.0, 1.5)
               + support + flow / lam)
        dln = np.clip(dln, -MAX_DAILY_MOVE, MAX_DAILY_MOVE)
        if t == 0:
            dln = np.zeros(paths)

        C = np.maximum(C - buys_usd / P, 1e-9)
        S = np.minimum(S, C)

        price[:, t] = P; supply[:, t] = C; depth_h[:, t] = lam; usage[:, t] = U
        staked_h[:, t] = S; treas_h[:, t] = treas_usd; sold_h[:, t] = sells
        dln_h[:, t] = dln
        lnP = lnP + dln

    return SimResult(design, scenario, shock_day, price, supply, depth_h, usage, market,
                     staked_h, treas_h, sold_h)
