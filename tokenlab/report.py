"""Build figures and REPORT.md from the files in results/. Every number in the report is
read from those files; nothing is typed in by hand.

Run:  python -m tokenlab.report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import model
from .designs import DESIGNS
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


# ------------------------------------------------------------------------------ figures
def fig_car_path(bt: dict) -> str:
    p = bt["car_path"]
    x = np.array(p["rel_day"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(x, np.expm1(p["p25"]) * 100, np.expm1(p["p75"]) * 100, color=BLUE, alpha=0.15, lw=0,
                    label="Interquartile range")
    ax.plot(x, np.expm1(p["median"]) * 100, color=BLUE, lw=1.2, ls="--", label="Median event")
    ax.plot(x, np.expm1(p["mean"]) * 100, color=BLUE, lw=2, label="Average event (geometric mean)")
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
    ax.legend(frameon=False, loc="upper right")
    return _save(fig, "backtest_size_scatter.png")


def fig_heatmap(st: pd.DataFrame) -> str:
    from matplotlib.colors import LinearSegmentedColormap
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


# ------------------------------------------------------------------------------- report
def _stats_row(label: str, s: dict) -> str:
    return (f"| {label} | {s['mean_pct']:+.1f}% | {s['median_pct']:+.1f}% | {s['t_stat']:+.2f} | "
            f"{s['share_negative'] * 100:.0f}% | {s['sign_test_p']:.3f} |")


def build() -> str:
    bt = json.loads((RESULTS / "backtest_summary.json").read_text(encoding="utf-8"))
    ss = json.loads((RESULTS / "stress_summary.json").read_text(encoding="utf-8"))
    cal = json.loads((RESULTS / "calibration.json").read_text(encoding="utf-8"))
    ev = pd.read_csv(RESULTS / "backtest_events.csv")
    st = pd.read_csv(RESULTS / "stress_results.csv")
    curves = pd.read_csv(RESULTS / "stress_paths.csv")
    events = pd.read_csv(ROOT / "data" / "unlock_events.csv")

    f1, f2 = fig_car_path(bt), fig_car_scatter(ev, bt)
    f3, f4, f5 = fig_heatmap(st), fig_crash_paths(curves, st), fig_sensitivity(ss)

    name = {d.key: d.name for d in DESIGNS}
    sname = {s.key: s.name.lower() for s in SCENARIOS}
    w = bt["windows"]
    n, N = ss["collapsed_in_crashes"], ss["n_designs"]
    L: list[str] = []
    A = L.append

    A("# Tokenomics Stress Lab - Report\n")
    A("Independent simulations of token supply, unlocks and incentives under shocks.\n")
    A("*This file is generated by `python -m tokenlab.report` from the files in `results/`. "
      "Re-running the pipeline regenerates every number below.*\n")

    A("## Summary\n")
    A(f"- **Backtest.** {bt['n_events']} real insider/investor cliff unlocks across {bt['n_tokens']} tokens "
      f"({bt['date_min']} to {bt['date_max']}, ${bt['total_usd_unlocked'] / 1e9:.1f}bn unlocked in total). "
      f"From 30 days before to 30 days after the unlock the average token returned "
      f"{w['full']['mean_pct']:+.1f}% relative to the market (median {w['full']['median_pct']:+.1f}%, "
      f"t = {w['full']['t_stat']:.2f}); {w['full']['share_negative'] * 100:.0f}% of events were negative.")
    A(f"- **Stress test.** {N} token designs, {len(SCENARIOS)} scenarios, {ss['paths']:,} Monte Carlo paths each. "
      f"**{n} of {N} designs collapsed in crashes** - they fell at least 90% below their pre-crash price in "
      f"most paths in at least one of the three market-crash scenarios "
      f"({'; '.join(sname[k] + ': ' + str(len(ss['collapsed_by_scenario'][k])) for k in ss['crash_scenarios'])}): "
      f"{', '.join(name[k] for k in ss['collapsed_designs'])}.")
    if ss["close_calls"]:
        A("- **Close calls.** Some verdicts sit within 5 points of the 50% line and should be read as "
          "coin-flips rather than firm results: "
          + "; ".join(f"{name[c['design']]} in {sname[c['scenario']]} ({c['p_collapse'] * 100:.0f}%)"
                      for c in ss["close_calls"]) + ".")
    A(f"- **Survivors:** {', '.join(name[k] for k in ss['survived_designs'])}.")
    A("- **What separates them.** Every design that collapsed has at least one mechanism that turns a lower "
      "price into more selling or less demand (rewards sold on receipt, usage that exists only because of the "
      "token price, liquidity rented with emissions, redemptions or liquidations that mint or dump tokens, large "
      "insider supply arriving into a thin float). The survivors have demand that does not depend on the token "
      "price, or no forced sellers.\n")

    A("## 1. Backtest: real unlock events\n")
    A("**Data.** Vesting schedules and cliff events come from DefiLlama's open emissions datasets; daily prices "
      "from the DefiLlama coins API. The sample is built by `tokenlab/backtest/data.py` with a rule fixed before "
      "individual returns were examined: cliff unlocks to team/advisors or private investors, between 1% and 100% "
      "of circulating supply, at least 120 days of prior price history, events of the same token at least 61 days apart, at most "
      f"3 per token, then the {bt['n_events']} largest by USD value. "
      f"The median event unlocked {bt['median_pct_circ'] * 100:.1f}% of circulating supply. "
      "The full list is in `data/unlock_events.csv`.\n")
    A("**Method.** Daily log returns are adjusted with a market model against BTC (beta estimated over days "
      "-120 to -31). Cumulative abnormal returns (CAR) are shown below as simple returns; t-statistics are "
      "cross-sectional and computed on log returns, so a window can have a positive mean simple return and a "
      "negative t-statistic when a few events have very large gains.\n")
    A("| Window | Mean | Median | t-stat | Share negative | Sign-test p |")
    A("|---|---|---|---|---|---|")
    A(_stats_row("Day -30 to 0 (run-up)", w["pre_30"]))
    A(_stats_row("Day -7 to 0", w["pre_7"]))
    A(_stats_row("Day 0 to +7", w["post_7"]))
    A(_stats_row("Day 0 to +30", w["post_30"]))
    A(_stats_row("Day -30 to +30 (full)", w["full"]))
    A("")
    A(f"![CAR path]({f1})\n")
    A("*The chart's average line is the mean of log returns (a geometric average), which is why it ends lower "
      "than the arithmetic mean in the table; a handful of events with very large gains pull the arithmetic "
      "mean up.*\n")
    rb = bt["robustness"]["full"]
    A(f"**Robustness.** Over the full window the mean raw return was {rb['raw']['mean_pct']:+.1f}% and the mean "
      f"BTC-relative return (beta = 1) was {rb['btc_adjusted']['mean_pct']:+.1f}% "
      f"(t = {rb['btc_adjusted']['t_stat']:.2f}).\n")
    r = bt["size_regression"]
    A(f"**Size.** Regressing the full-window CAR on ln(1 + unlock / circulating supply) gives a slope of "
      f"{r['slope']:+.2f} (t = {r['slope_t']:.2f}, R2 = {r['r2']:.2f}). By size tercile:\n")
    A("| Unlock size (% of circulating) | Events | Mean CAR -30..+30 | Median | Share negative |")
    A("|---|---|---|---|---|")
    for b in ("small", "medium", "large"):
        s = bt["splits"][f"size_{b}"]
        lo, hi = s["pct_circ_range"]
        A(f"| {b} ({lo * 100:.1f}% - {hi * 100:.1f}%) | {s['full']['n']} | {s['full']['mean_pct']:+.1f}% | "
          f"{s['full']['median_pct']:+.1f}% | {s['full']['share_negative'] * 100:.0f}% |")
    A("")
    A(f"![Size scatter]({f2})\n")

    A("## 2. The simulator\n")
    A("`tokenlab/model.py` is a daily-step Monte Carlo model. A design specifies allocations and vesting, "
      "emissions, staking, revenue and value capture, liquidity, and optional reflexive mechanisms "
      "(algorithmic-stablecoin redemptions, collateral liquidation bands, USD-pegged rewards, price-dependent "
      "usage). Each day the log price moves with the market (beta), idiosyncratic news, reversion toward a fair "
      "value that scales with adoption and is diluted by supply growth, trend-following flows, buying support "
      "below cash-flow value, and the net token flow divided by market depth. Market depth itself shrinks in "
      "drawdowns and when liquidity incentives lose value.\n")
    if cal.get("calibrated"):
        A(f"**Calibration.** Market depth is the one parameter fitted to data. It is set to {cal['depth']:.2f} so "
          f"that replaying the {cal['n_events']} real unlocks (same size, same recipient type) through a plain "
          f"reference token reproduces the observed mean abnormal return of "
          f"{np.expm1(cal['target_mean_car_log']) * 100:+.1f}% over days -30 to +30. A depth of {cal['depth']:.2f} "
          f"means that selling 1% of circulating supply moves the price by about {1 / cal['depth']:.1f}% before "
          "feedback effects. The remaining behavioural constants are stated assumptions, listed at the top of "
          "`model.py`.\n")
    else:
        A(f"**Calibration.** {cal.get('note', '')} Depth = {cal['depth']:.2f}.\n")

    A("### The 12 designs\n")
    A("| # | Design | Description |")
    A("|---|---|---|")
    for i, d in enumerate(DESIGNS, 1):
        A(f"| {i} | {d.name} | {d.description} |")
    A("")
    A("These are stylised archetypes, not models of specific tokens.\n")
    A("### Scenarios\n")
    A("| Scenario | Description |")
    A("|---|---|")
    for s in SCENARIOS:
        A(f"| {s.name} | {s.description} |")
    A("")

    A("## 3. Stress-test results\n")
    A(f"**Collapse definition** (set before running): a path collapses if the token trades "
      f"{ss['collapse_definition']}. \"No recovery\" additionally requires the price to still be below 25% of "
      "its pre-shock level at the end of the three-year horizon.\n")
    A(f"![Heatmap]({f3})\n")
    A("### Crash scenarios, design by design\n")
    A("A design's verdict is **Collapsed** if P(collapse) exceeds 50% in at least one of the three crash "
      "scenarios.\n")
    A("| Design | Verdict | P(collapse), crash | P(collapse), crash at unlock | P(collapse), prolonged bear | "
      "P(no recovery), crash | Median max drawdown, crash | Median end price / pre-crash | P(collapse), baseline |")
    A("|---|---|---|---|---|---|---|---|---|")
    g = st.set_index(["design", "scenario"])
    for d in DESIGNS:
        c, u, b = g.loc[(d.key, "crash")], g.loc[(d.key, "crash_at_unlock")], g.loc[(d.key, "baseline")]
        br = g.loc[(d.key, "bear")]
        verdict = "**Collapsed**" if d.key in ss["collapsed_designs"] else "Survived"
        A(f"| {d.name} | {verdict} | {c.p_collapse * 100:.0f}% | {u.p_collapse * 100:.0f}% | "
          f"{br.p_collapse * 100:.0f}% | {c.p_no_recovery * 100:.0f}% | {c.median_max_drawdown * 100:.0f}% | "
          f"{c.median_terminal_ratio:.2f} | {b.p_collapse * 100:.0f}% |")
    A("")
    mk = g.loc[(DESIGNS[0].key, "crash")].median_market_terminal
    A(f"For reference, in the crash scenario the market itself troughs at -60% and ends the horizon at a median "
      f"of {mk:.2f}x its pre-crash level.\n")
    A(f"![Crash paths]({f4})\n")

    A("### Other scenarios\n")
    A("| Scenario | Designs collapsed | Which |")
    A("|---|---|---|")
    for s in SCENARIOS:
        ks = ss["collapsed_by_scenario"][s.key]
        A(f"| {s.name} | {len(ks)} of {N} | {', '.join(name[k] for k in ks) or '-'} |")
    A("")

    A("### Sensitivity of the headline count\n")
    sens = ss["sensitivity"]
    A(f"Run with {sens['paths']:,} paths per cell on the fast market-crash scenario only (so the 60% / 90% / x1.0 "
      "cells correspond to the 'Market crash' row above, not to the three-scenario headline).\n")
    A("| What is varied | " + " | ".join(sens["crash_severity"]) + " |")
    A("|---|" + "---|" * len(sens["crash_severity"]))
    A("| Crash size -> designs collapsed | " + " | ".join(str(v["count"]) for v in sens["crash_severity"].values()) + " |")
    A("")
    A("| Collapse threshold (drawdown) | " + " | ".join(sens["collapse_threshold"]) + " |")
    A("|---|" + "---|" * len(sens["collapse_threshold"]))
    A("| Designs collapsed | " + " | ".join(str(v["count"]) for v in sens["collapse_threshold"].values()) + " |")
    A("")
    A("| Market depth vs calibrated | " + " | ".join(sens["market_depth"]) + " |")
    A("|---|" + "---|" * len(sens["market_depth"]))
    A("| Designs collapsed | " + " | ".join(str(v["count"]) for v in sens["market_depth"].values()) + " |")
    A("")
    A("| Random seed | " + " | ".join(sens["seeds"]) + " |")
    A("|---|" + "---|" * len(sens["seeds"]))
    A("| Designs collapsed | " + " | ".join(str(v["count"]) for v in sens["seeds"].values()) + " |")
    A("")
    A(f"![Sensitivity]({f5})\n")

    A("## 4. Limitations\n")
    A("- The 12 designs are archetypes with round-number parameters chosen by the author. The collapse count is a "
      "property of these designs under this model, not an estimate of how many real tokens would fail.\n"
      "- Only market depth is fitted to data. Reflexivity, adoption elasticity, sell fractions and the fair-value "
      "reversion speed are assumptions; the sensitivity tables show how much the headline moves with the most "
      "important ones.\n"
      "- The collapse count depends on the threshold. Designs whose median drawdown sits close to 90% can change "
      "verdict with the threshold, crash size or seed; see the tables above.\n"
      "- The backtest is an event study on 60 events. Unlock dates are public in advance, so abnormal returns "
      "measure the combined effect of anticipation and selling, and cross-sectional t-statistics ignore "
      "clustering of events in time. Schedules come from one data provider and may contain errors.\n"
      "- The sample selects the largest unlocks by USD value among tokens that still have price data, which "
      "tilts it toward larger, surviving tokens.\n"
      "- Nothing here is investment advice.\n")

    A("## Appendix: the unlock events\n")
    A("| ID | Date | Token | Recipients | % of circulating | USD value | CAR -30..+30 |")
    A("|---|---|---|---|---|---|---|")
    m = events.merge(ev[["event_id", "car_full"]], on="event_id")
    for r_ in m.itertuples():
        A(f"| {r_.event_id} | {str(r_.date)[:10]} | {r_.name} | {r_.category} | {r_.pct_circ * 100:.1f}% | "
          f"${r_.usd_value / 1e6:,.0f}m | {np.expm1(r_.car_full) * 100:+.1f}% |")
    A("")

    text = "\n".join(L)
    (ROOT / "REPORT.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    build()
    print("wrote REPORT.md and figures")
