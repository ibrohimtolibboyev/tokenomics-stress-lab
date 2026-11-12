"""Figures and REPORT.md, built from the files in results/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from .designs import DESIGNS, REDESIGN, REDESIGN_OF
from .shocks import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, CRITICAL = "#2a78d6", "#d03b3b"
BLUE_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def _save(fig, name: str) -> str:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return f"results/figures/{name}"


def pct(x: float, digits: int = 1, sign: bool = True) -> str:
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def fig_car_path(bt: dict) -> str:
    p = bt["car_path"]
    x = np.array(p["rel_day"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(x, np.expm1(p["p25"]) * 100, np.expm1(p["p75"]) * 100, color=BLUE, alpha=0.15, lw=0,
                    label="Interquartile range")
    ax.plot(x, np.expm1(p["median"]) * 100, color=BLUE, lw=1.2, ls="--", label="Median event")
    ax.plot(x, np.expm1(p["mean"]) * 100, color=BLUE, lw=2, label="Mean event")
    ax.axvline(0, color=INK2, lw=1)
    ax.axhline(0, color=AXIS, lw=1)
    ax.annotate("unlock", (0, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points", color=INK2)
    ax.set_title(f"Market-adjusted return around {bt['n_events']} real insider unlocks")
    ax.set_xlabel("Days relative to unlock")
    ax.set_ylabel("Cumulative abnormal return since day -30 (%)")
    ax.legend(frameon=False, loc="lower left")
    return _save(fig, "backtest_car_path.png")


def fig_car_scatter(ev: pd.DataFrame, bt: dict) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(ev["pct_circ"] * 100, np.expm1(ev["car_full"]) * 100, s=42, color=BLUE,
               edgecolor=SURFACE, linewidth=1.2)
    r = bt["size_regression"]
    xs = np.geomspace(ev["pct_circ"].min(), ev["pct_circ"].max(), 50)
    ax.plot(xs * 100, np.expm1(r["intercept"] + r["slope"] * np.log1p(xs)) * 100, color=INK2, lw=1.2,
            label=f"OLS fit (slope t = {r['slope_t']:.2f})")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xscale("log")
    ax.set_title("Bigger unlocks, relative to float, and the 60-day abnormal return")
    ax.set_xlabel("Unlock size, % of circulating supply (log scale)")
    ax.set_ylabel("Abnormal return, day -30 to +30 (%)")
    ax.legend(frameon=False, loc="lower left")
    return _save(fig, "backtest_size_scatter.png")


def fig_heatmap(st: pd.DataFrame) -> str:
    piv = st.pivot(index="design_name", columns="scenario_name", values="p_collapse")
    piv = piv.loc[[d.name for d in DESIGNS], [s.name for s in SCENARIOS]]
    flag = st.pivot(index="design_name", columns="scenario_name", values="collapsed").loc[piv.index, piv.columns]
    cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.imshow(piv.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v * 100:.0f}%", ha="center", va="center", fontsize=9,
                    color="#ffffff" if v > 0.5 else INK, fontweight="bold" if flag.iat[i, j] else "normal")
    ax.set_xticks(range(piv.shape[1]), [c.replace(" ", "\n", 1) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_xticks(np.arange(-.5, piv.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, piv.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Share of simulated paths that collapse (>= 90% below pre-shock price)")
    return _save(fig, "stress_heatmap.png")


def fig_crash_paths(curves: pd.DataFrame, st: pd.DataFrame) -> str:
    c = curves[curves.scenario == "crash"]
    flags = st[st.scenario == "crash"].set_index("design")["collapsed"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.2), sharex=True, sharey=True)
    for ax, d in zip(axes.ravel(), DESIGNS):
        g = c[c.design == d.key]
        col = CRITICAL if flags[d.key] else BLUE
        ax.fill_between(g.day, g.p10.clip(lower=1e-4), g.p90.clip(lower=1e-4), color=col, alpha=0.15, lw=0)
        ax.plot(g.day, g["median"].clip(lower=1e-4), color=col, lw=2)
        ax.axhline(0.1, color=INK2, lw=0.8, ls=":")
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 30)
        ax.set_title(d.name if len(d.name) <= 34 else d.name.split(" / ")[0], fontsize=9.5)
        ax.text(0.97, 0.93, "COLLAPSED" if flags[d.key] else "survived", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=INK2, fontweight="bold" if flags[d.key] else "normal")
    for ax in axes[-1]:
        ax.set_xlabel("Days from start of crash")
    for ax in axes[:, 0]:
        ax.set_ylabel("Price / pre-crash price")
    fig.suptitle("Fast market-crash scenario: median path and 10-90% band (dotted line = -90%); "
                 "label = verdict in this scenario",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return _save(fig, "stress_crash_paths.png")


def fig_sensitivity(ss: dict) -> str:
    sev = ss["sensitivity"]["crash_severity"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    xs = list(sev)
    ys = [v["count"] for v in sev.values()]
    bars = ax.bar(xs, ys, color=BLUE, width=0.6)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y + 0.15, str(y), ha="center", color=INK)
    ax.set_ylim(0, ss["n_designs"])
    ax.set_yticks(range(0, ss["n_designs"] + 1, 2))
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Designs collapsing (of {ss['n_designs']}) by size of the market crash")
    ax.set_xlabel("Market fall over 45 days")
    ax.set_ylabel("Designs collapsed")
    return _save(fig, "stress_sensitivity.png")


def _stats_row(label: str, s: dict) -> str:
    return (f"| {label} | {s['mean_pct']:+.1f}% | {s['median_pct']:+.1f}% | {s['t_stat']:+.2f} | "
            f"{s['share_negative'] * 100:.0f}% | {s['sign_test_p']:.3f} |")


def _count_table(first: str, row: str, cells: dict) -> list[str]:
    return [f"| {first} | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells),
            f"| {row} | " + " | ".join(str(v["count"]) for v in cells.values()) + " |", ""]


def build() -> str:
    bt = json.loads((RESULTS / "backtest_summary.json").read_text(encoding="utf-8"))
    ss = json.loads((RESULTS / "stress_summary.json").read_text(encoding="utf-8"))
    cal = json.loads((RESULTS / "calibration.json").read_text(encoding="utf-8"))
    ev = pd.read_csv(RESULTS / "backtest_events.csv")
    st = pd.read_csv(RESULTS / "stress_results.csv")
    fixed = pd.read_csv(RESULTS / "redesign_results.csv").set_index("scenario")
    curves = pd.read_csv(RESULTS / "stress_paths.csv")
    events = pd.read_csv(ROOT / "data" / "unlock_events.csv")

    f1, f2 = fig_car_path(bt), fig_car_scatter(ev, bt)
    f3, f4, f5 = fig_heatmap(st), fig_crash_paths(curves, st), fig_sensitivity(ss)

    name = {d.key: d.name for d in DESIGNS}
    sname = {s.key: s.name.lower() for s in SCENARIOS}
    w = bt["windows"]
    n, N = ss["collapsed_in_crashes"], ss["n_designs"]
    sens = ss["sensitivity"]
    sev = sens["crash_severity"]
    red = ss["redesign"]
    worst = max(fixed.loc[k, "p_collapse"] for k in ss["crash_scenarios"])
    L: list[str] = []
    A = L.append

    A("# Tokenomics Stress Lab\n")
    A("Independent simulations of token supply, unlocks and incentives under shocks.\n")

    A("## Summary\n")
    A(f"- **Backtest.** {bt['n_events']} real insider/investor cliff unlocks across {bt['n_tokens']} tokens "
      f"({bt['date_min']} to {bt['date_max']}, ${bt['total_usd_unlocked'] / 1e9:.1f}bn unlocked). From 30 days "
      f"before to 30 days after the unlock, tokens returned {w['full']['mean_pct']:+.1f}% on average relative to "
      f"the market (median {w['full']['median_pct']:+.1f}%, t = {w['full']['t_stat']:.2f}); "
      f"{w['full']['share_negative'] * 100:.0f}% of events were negative.")
    A(f"- **Stress test.** {N} token designs, {len(SCENARIOS)} scenarios, {ss['paths']:,} Monte Carlo paths each. "
      f"**{n} of {N} designs collapsed in crashes**: they fell at least 90% below their pre-crash price in most "
      f"paths in at least one of the three market-crash scenarios "
      f"({'; '.join(sname[k] + ': ' + str(len(ss['collapsed_by_scenario'][k])) for k in ss['crash_scenarios'])}). "
      f"Collapsed: {', '.join(name[k] for k in ss['collapsed_designs'])}. "
      f"Survived: {', '.join(name[k] for k in ss['survived_designs'])}.")
    if ss["close_calls"]:
        A("- **Close calls.** These verdicts are within 5 points of the 50% line and are coin-flips: "
          + "; ".join(f"{name[c['design']]} in {sname[c['scenario']]} ({c['p_collapse'] * 100:.1f}%)"
                      for c in ss["close_calls"]) + ".")
    A(f"- **The count depends on the crash size.** {sev['50%']['count']} designs collapse in a 50% crash, "
      f"{sev['60%']['count']} at 60% and {sev['70%']['count']} at 70% (fast crash only).")
    A(f"- **A 13th design.** Reworking the algorithmic-stablecoin token so the stablecoin is reserve-backed "
      f"takes its worst crash-scenario collapse probability from "
      f"{max(red['steps'][0][k] for k in ss['crash_scenarios']) * 100:.0f}% to {worst * 100:.0f}%"
      f"{'' if red['collapsed_in_crashes'] else ', below the 50% line'} (section 4).\n")

    A("## 1. Backtest: real unlock events\n")
    A("**Data.** Vesting schedules and cliff events are from DefiLlama's open emissions datasets, daily prices "
      "from the DefiLlama coins API, and circulating supply from CoinMarketCap's historical quotes (market cap "
      "divided by price, last quote before the unlock). An event is kept only if the two price sources agree "
      "within 15%. Selection: cliff unlocks to team/advisors or private investors, between 1% and 100% of "
      "circulating supply, full daily prices from day -120 to +30, same-token events at least 61 days apart, at "
      f"most 3 per token, then the {bt['n_events']} largest by USD value. The median event unlocked "
      f"{bt['median_pct_circ'] * 100:.1f}% of circulating supply. Tokens with no CoinMarketCap supply history "
      "are left out. The list is in the appendix and in `data/unlock_events.csv`.\n")
    A("**Method.** Daily log returns are adjusted with a market model against BTC (beta from days -120 to "
      "-31). Averages are means of cumulative log abnormal returns, converted to percent; t-statistics are "
      f"cross-sectional. The arithmetic mean of simple returns over the full window is "
      f"{w['full']['arith_mean_pct']:+.1f}%; it differs because a few events had very large gains.\n")
    A("| Window | Mean | Median | t-stat | Share negative | Sign-test p |")
    A("|---|---|---|---|---|---|")
    A(_stats_row("Day -30 to 0", w["pre_30"]))
    A(_stats_row("Day -7 to 0", w["pre_7"]))
    A(_stats_row("Day 0 to +7", w["post_7"]))
    A(_stats_row("Day 0 to +30", w["post_30"]))
    A(_stats_row("Day -30 to +30", w["full"]))
    A("")
    A(f"![Abnormal return path]({f1})\n")
    rb = bt["robustness"]["full"]
    A(f"Without the beta adjustment the full-window mean is {rb['raw']['mean_pct']:+.1f}% raw and "
      f"{rb['btc_adjusted']['mean_pct']:+.1f}% relative to BTC (t = {rb['btc_adjusted']['t_stat']:.2f}).\n")
    r = bt["size_regression"]
    A(f"**Size.** Regressing the full-window return on ln(1 + unlock / circulating supply) gives a slope of "
      f"{r['slope']:+.2f} (t = {r['slope_t']:.2f}, R2 = {r['r2']:.2f}).\n")
    A("| Unlock size (% of circulating) | Events | Mean | Median | Share negative |")
    A("|---|---|---|---|---|")
    for b in ("small", "medium", "large"):
        s = bt["splits"][f"size_{b}"]
        lo, hi = s["pct_circ_range"]
        A(f"| {lo * 100:.1f}% - {hi * 100:.1f}% | {s['full']['n']} | {s['full']['mean_pct']:+.1f}% | "
          f"{s['full']['median_pct']:+.1f}% | {s['full']['share_negative'] * 100:.0f}% |")
    A("")
    A(f"![Unlock size and abnormal return]({f2})\n")

    A("## 2. The simulator\n")
    A("`tokenlab/model.py` is a daily-step Monte Carlo model. A design sets allocations and vesting, emissions, "
      "staking, revenue and value capture, liquidity, and optional reflexive mechanisms (stablecoin "
      "redemptions that mint tokens, collateral liquidation bands, USD-pegged rewards, usage that follows the "
      "token price). Each day the log price moves with the market, idiosyncratic news, reversion toward a fair "
      "value that grows with adoption and is diluted by supply, trend-following flows, buying support below "
      "cash-flow value, and net token flow divided by market depth. Depth shrinks in drawdowns.\n")
    A(f"**Calibration.** Market depth is the one parameter fitted to data. It is set to {cal['depth']:.2f} so "
      f"that replaying the {cal['n_events']} real unlocks (same size, same recipient type) through a plain token "
      f"reproduces the observed mean return of {np.expm1(cal['target_mean_car_log']) * 100:+.1f}%. At that "
      f"depth, selling 1% of circulating supply moves the price about {1 / cal['depth']:.1f}% before feedback. "
      "The other behavioural constants are assumptions, listed at the top of `model.py`.\n")
    A("### The 12 designs\n")
    A("| # | Design | Description |")
    A("|---|---|---|")
    for i, d in enumerate(DESIGNS, 1):
        A(f"| {i} | {d.name} | {d.description} |")
    A("")
    A("### Scenarios\n")
    A("| Scenario | Description |")
    A("|---|---|")
    for s in SCENARIOS:
        A(f"| {s.name} | {s.description} |")
    A("")
    A("**Why 60%.** The crash is sized on market-wide drawdowns that have happened: Bitcoin fell about 50% in "
      "the week to 13 March 2020, total crypto market capitalisation roughly halved between mid-May and late "
      "July 2021, and it fell about 60% between early April and mid-June 2022. A 60% fall in 45 days is the "
      f"severe end of that range. The choice matters: at 50% only {sev['50%']['count']} designs collapse.\n")

    A("## 3. Stress-test results\n")
    A(f"A path collapses if the token trades {ss['collapse_definition']}. A design's verdict is **Collapsed** "
      "if that happens in at least one of the three crash scenarios. \"No recovery\" also requires the price to "
      "end the three-year horizon below 25% of its pre-shock level.\n")
    A(f"![Collapse probability by design and scenario]({f3})\n")
    A("| Design | Verdict | Crash | Crash at unlock | Prolonged bear | No recovery, crash | "
      "Median max drawdown, crash | Baseline |")
    A("|---|---|---|---|---|---|---|---|")
    g = st.set_index(["design", "scenario"])
    for d in DESIGNS:
        c, u, b = g.loc[(d.key, "crash")], g.loc[(d.key, "crash_at_unlock")], g.loc[(d.key, "baseline")]
        br = g.loc[(d.key, "bear")]
        verdict = "**Collapsed**" if d.key in ss["collapsed_designs"] else "Survived"
        A(f"| {d.name} | {verdict} | {c.p_collapse * 100:.0f}% | {u.p_collapse * 100:.0f}% | "
          f"{br.p_collapse * 100:.0f}% | {c.p_no_recovery * 100:.0f}% | {c.median_max_drawdown * 100:.0f}% | "
          f"{b.p_collapse * 100:.0f}% |")
    A("")
    A(f"![Crash paths]({f4})\n")
    A("| Scenario | Designs collapsed | Which |")
    A("|---|---|---|")
    for s in SCENARIOS:
        ks = ss["collapsed_by_scenario"][s.key]
        A(f"| {s.name} | {len(ks)} of {N} | {', '.join(name[k] for k in ks) or '-'} |")
    A("")

    A("### Sensitivity\n")
    A(f"Fast crash scenario only, {sens['paths']:,} paths per cell.\n")
    L.extend(_count_table("Market fall", "Designs collapsed", sev))
    L.extend(_count_table("Collapse threshold (drawdown)", "Designs collapsed", sens["collapse_threshold"]))
    L.extend(_count_table("Market depth vs calibrated", "Designs collapsed", sens["market_depth"]))
    L.extend(_count_table("Random seed", "Designs collapsed", sens["seeds"]))
    A(f"![Collapse count by crash size]({f5})\n")

    A("### How much is the mechanism, how much the assumed market behaviour\n")
    bc = ss["behaviour_check"]
    flips = sum(v["calm"] <= 0.5 for v in bc.values())
    A("Each design also carries three behavioural assumptions: market beta, reflexivity (trend-following) and "
      "how much of its usage follows the token price. The table reruns the collapsed designs in the fast crash "
      "with those three set to the values the survivors use (beta 1.2, reflexivity 0.15, no price-led usage) "
      f"and everything else unchanged. With them, {flips} of {len(bc)} are below the 50% line, so the verdicts "
      "rest on these assumptions at least as much as on supply mechanics.\n")
    A("| Design | P(collapse), as designed | P(collapse), survivor-level behaviour |")
    A("|---|---|---|")
    for k, v in bc.items():
        A(f"| {name[k]} | {v['as_designed'] * 100:.0f}% | {v['calm'] * 100:.0f}% |")
    A("")

    A("## 4. A 13th design\n")
    A(f"**{REDESIGN.name}.** {REDESIGN.description} It starts from the {REDESIGN_OF.name.lower()} and changes "
      "one thing at a time; beta and reflexivity are left as they were. It was built after seeing the results "
      "above and is not part of the 12.\n")
    A("| Step | Crash | Crash at unlock | Prolonged bear |")
    A("|---|---|---|---|")
    for step in red["steps"]:
        A(f"| {step['label']} | " + " | ".join(f"{step[k] * 100:.0f}%" for k in ss["crash_scenarios"]) + " |")
    A("")
    A("Across all scenarios the final design's collapse probability is "
      + ", ".join(f"{fixed.loc[s.key, 'p_collapse'] * 100:.0f}% ({s.name.lower()})" for s in SCENARIOS) + ". "
      + ("It still collapses in at least one crash scenario." if red["collapsed_in_crashes"] else
         "It stays under 50% in every scenario" + (", though narrowly in the worst one." if worst > 0.45 else "."))
      + " Removing the mint-on-redemption loop does little by itself, because in this model the run only "
      "starts once the token has already lost most of its value; the step that matters is making stablecoin "
      "demand independent of the token price.\n")

    A("## 5. Limitations\n")
    A("- The designs are archetypes with round-number parameters chosen by the author. The collapse count "
      "describes these designs under this model, not how many real tokens would fail.\n"
      "- Only market depth is fitted to data. Beta, reflexivity, price-led usage and sell fractions are "
      "assumptions, and section 3 shows the verdicts are sensitive to them.\n"
      "- Collapse is measured from the pre-shock price, so a design that has already bled before the shock has "
      "less left to lose and can look sturdier than it is.\n"
      "- The count moves with the collapse threshold, crash size and seed.\n"
      f"- The backtest is {bt['n_events']} events. Unlock dates are public in advance, so abnormal returns mix "
      "anticipation with selling; t-statistics ignore clustering of events in time.\n"
      "- The sample takes the largest unlocks among tokens with price and supply data, which tilts it toward "
      "larger, surviving tokens. Schedules come from one provider and circulating supply from another; either "
      "can be wrong.\n"
      "- Not investment advice.\n")

    A("## Appendix: the unlock events\n")
    A("| ID | Date | Token | Recipients | % of circulating | USD value | Abnormal return -30..+30 |")
    A("|---|---|---|---|---|---|---|")
    m = events.merge(ev[["event_id", "car_full"]], on="event_id")
    for e in m.itertuples():
        A(f"| {e.event_id} | {str(e.date)[:10]} | {e.symbol} ({e.name}) | {e.category} | {e.pct_circ * 100:.1f}% | "
          f"${e.usd_value / 1e6:,.0f}m | {np.expm1(e.car_full) * 100:+.1f}% |")
    A("")

    text = "\n".join(L)
    (ROOT / "REPORT.md").write_text(text, encoding="utf-8")
    print("wrote REPORT.md")
    return text


if __name__ == "__main__":
    build()
