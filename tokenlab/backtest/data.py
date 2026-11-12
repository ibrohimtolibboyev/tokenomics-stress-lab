"""Build the real-world unlock event sample used by the backtest.

Sources (both public, no API key):
  * DefiLlama open emissions datasets  - vesting schedules and cliff unlock events
      https://defillama-datasets.llama.fi/emissionsIndex
      https://defillama-datasets.llama.fi/emissions/<protocol-slug>
  * DefiLlama coins API                - daily token prices (keyed by CoinGecko id)
      https://coins.llama.fi/chart/coingecko:<id>

Selection rule (fixed before looking at any returns):
  1. Cliff unlocks to insiders (team / advisors) or private-sale investors.
  2. Event date between START_DATE and END_DATE.
  3. Unlock size between MIN_PCT_CIRC and MAX_PCT_CIRC of the circulating supply on the
     day before. (An "unlock" larger than the whole float means the schedule's float
     figure is incomplete or the event is effectively a launch; both are excluded.)
  4. Not a token-generation event: >= MIN_HISTORY_DAYS of prior price history.
  5. Complete daily prices for the token and for BTC over [-120, +30] days.
  6. Events of the same token must be >= MIN_GAP_DAYS apart (largest kept first),
     and at most MAX_PER_TOKEN events per token.
  7. Keep the N_EVENTS largest remaining events by USD value at the unlock date.

Run:  python -m tokenlab.backtest.data
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"

HEADERS = {"User-Agent": "tokenomics-stress-lab"}
INDEX_URL = "https://defillama-datasets.llama.fi/emissionsIndex"
PROTOCOL_URL = "https://defillama-datasets.llama.fi/emissions/{slug}"
PRICE_URL = "https://coins.llama.fi/chart/coingecko:{gid}?start={start}&span={span}&period=1d"

START_DATE = dt.date(2022, 1, 1)
END_DATE = dt.date(2026, 9, 30)
INSIDER_CATEGORIES = {"insiders", "privateSale"}
MIN_PCT_CIRC = 0.01
MAX_PCT_CIRC = 1.0
MIN_HISTORY_DAYS = 120
PRE_DAYS = 120
POST_DAYS = 30
MIN_GAP_DAYS = 61
MAX_PER_TOKEN = 3
N_EVENTS = 60
MIN_PRICE_CONFIDENCE = 0.9
# Tokens dropped for data-integrity reasons (documented, not return-based):
EXCLUDED_SLUGS = {
    # schedule is denominated in the post-redenomination token while the price series is
    # the old token, so token counts and USD values do not match
    "omni-network",
}
# stablecoins / wrapped assets are not meaningful for an unlock study
EXCLUDED_GECKO_IDS = {"bitcoin", "ethereum", "tether", "usd-coin"}

DAY = 86400


def _get(url: str, cache: Path, tries: int = 4):
    """GET JSON with an on-disk cache and simple retry."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(r.text, encoding="utf-8")
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code in (400, 403, 404):
                break
        except requests.RequestException as e:  # network hiccup: retry
            last = repr(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _date(ts: float) -> dt.date:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()


def _midnight(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())


def candidate_events(index: dict) -> pd.DataFrame:
    """All insider/investor cliff unlocks in the date window, one row per token-day."""
    rows = []
    for p in index["data"]:
        gid = p.get("gecko_id")
        if not gid or gid in EXCLUDED_GECKO_IDS or p.get("protocolSlug") in EXCLUDED_SLUGS:
            continue
        per_day: dict[dt.date, dict] = {}
        for ev in p.get("unlockEvents") or []:
            d = _date(ev["timestamp"])
            if not (START_DATE <= d <= END_DATE):
                continue
            for a in ev.get("cliffAllocations") or []:
                if a.get("category") not in INSIDER_CATEGORIES or not a.get("amount"):
                    continue
                slot = per_day.setdefault(d, {"tokens": 0.0, "recipients": set(), "cats": set()})
                slot["tokens"] += float(a["amount"])
                slot["recipients"].add(str(a.get("recipient")))
                slot["cats"].add(a["category"])
        for d, s in per_day.items():
            rows.append(
                {
                    "slug": p["protocolSlug"],
                    "name": p["name"],
                    "gecko_id": gid,
                    "date": d,
                    "tokens_unlocked": s["tokens"],
                    "recipients": "; ".join(sorted(s["recipients"])),
                    "category": "+".join(sorted(s["cats"])),
                    "max_supply": p.get("maxSupply"),
                }
            )
    return pd.DataFrame(rows)


def circulating_before(proto: dict, d: dt.date) -> float:
    """Unlocked supply (excluding non-circulating buckets) on the day before `d`."""
    nonc = set((proto.get("categories") or {}).get("noncirculating") or [])
    cutoff = _midnight(d)
    total = 0.0
    for series in proto["documentedData"]["data"]:
        if series["label"] in nonc:
            continue
        last = 0.0
        for pt in series["data"]:
            if pt["timestamp"] >= cutoff:
                break
            last = pt.get("unlocked") or 0.0
        total += last
    return total


def fetch_prices(gid: str, start: dt.date, end: dt.date) -> pd.Series:
    """Daily prices sampled at 00:00 UTC, indexed by date."""
    span = (end - start).days + 1
    cache = RAW / "prices" / f"{gid}_{start}_{end}.json"
    js = _get(PRICE_URL.format(gid=gid, start=_midnight(start), span=span), cache)
    coin = js.get("coins", {}).get(f"coingecko:{gid}")
    if not coin or coin.get("confidence", 0) < MIN_PRICE_CONFIDENCE:
        return pd.Series(dtype=float)
    s = {}
    for pt in coin["prices"]:
        # keep only points sampled close to midnight (API returns nearest observation)
        d = _date(pt["timestamp"] + 6 * 3600)
        if abs(pt["timestamp"] - _midnight(d)) <= 6 * 3600 and pt["price"] > 0:
            s.setdefault(d, float(pt["price"]))
    return pd.Series(s).sort_index()


def _window_ok(px: pd.Series, d: dt.date) -> bool:
    need = [d + dt.timedelta(days=k) for k in range(-PRE_DAYS, POST_DAYS + 1)]
    have = sum(1 for x in need if x in px.index)
    # tolerate a few missing days (filled forward later) but both endpoints must exist
    return have >= len(need) - 3 and need[0] in px.index and d in px.index and need[-1] in px.index


def build(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    DATA.mkdir(exist_ok=True)
    index = _get(INDEX_URL, RAW / "emissionsIndex.json")
    cand = candidate_events(index)
    if verbose:
        print(f"candidate insider cliff unlock days: {len(cand)} across {cand.slug.nunique()} tokens")

    # --- size relative to circulating supply ---------------------------------
    keep = []
    for slug, grp in cand.groupby("slug"):
        try:
            proto = _get(PROTOCOL_URL.format(slug=slug), RAW / "emissions" / f"{slug}.json")
        except RuntimeError:
            continue
        for _, r in grp.iterrows():
            circ = circulating_before(proto, r["date"])
            if circ <= 0:
                continue
            pct = r["tokens_unlocked"] / circ
            if MIN_PCT_CIRC <= pct <= MAX_PCT_CIRC:
                keep.append({**r.to_dict(), "circ_before": circ, "pct_circ": pct})
    ev = pd.DataFrame(keep)
    if verbose:
        print(f"events {MIN_PCT_CIRC:.0%}-{MAX_PCT_CIRC:.0%} of circulating supply: {len(ev)} across {ev.slug.nunique()} tokens")

    # --- prices --------------------------------------------------------------
    btc_start = START_DATE - dt.timedelta(days=PRE_DAYS + 5)
    btc_end = END_DATE + dt.timedelta(days=POST_DAYS + 5)
    btc = pd.concat(
        [
            fetch_prices("bitcoin", a, min(a + dt.timedelta(days=399), btc_end))
            for a in (btc_start + dt.timedelta(days=400 * i) for i in range(((btc_end - btc_start).days // 400) + 1))
        ]
    )
    btc = btc[~btc.index.duplicated()].sort_index()

    priced = []
    price_rows = []
    for _, r in ev.iterrows():
        d = r["date"]
        a = d - dt.timedelta(days=PRE_DAYS + MIN_HISTORY_DAYS - PRE_DAYS + 2)
        b = d + dt.timedelta(days=POST_DAYS + 1)
        try:
            px = fetch_prices(r["gecko_id"], a, b)
        except RuntimeError:
            continue
        if px.empty or not _window_ok(px, d) or not _window_ok(btc, d):
            continue
        p0 = px[d]
        priced.append({**r.to_dict(), "price_at_unlock": p0, "usd_value": p0 * r["tokens_unlocked"],
                       "mcap_before": p0 * r["circ_before"]})
        price_rows.append((r["slug"], d, px))
    ev = pd.DataFrame(priced)
    if verbose:
        print(f"events with complete price windows: {len(ev)} across {ev.slug.nunique()} tokens")

    # --- de-overlap within token, cap per token, take the largest ------------
    ev = ev.sort_values("usd_value", ascending=False).reset_index(drop=True)
    chosen: list[int] = []
    per_token: dict[str, list[dt.date]] = {}
    for i, r in ev.iterrows():
        dates = per_token.setdefault(r["slug"], [])
        if len(dates) >= MAX_PER_TOKEN:
            continue
        if any(abs((r["date"] - x).days) < MIN_GAP_DAYS for x in dates):
            continue
        dates.append(r["date"])
        chosen.append(i)
        if len(chosen) == N_EVENTS:
            break
    sel = ev.loc[chosen].sort_values("date").reset_index(drop=True)
    sel.insert(0, "event_id", [f"E{i + 1:02d}" for i in range(len(sel))])
    if verbose:
        print(f"selected: {len(sel)} events across {sel.slug.nunique()} tokens")

    # --- tidy price panel for the selected events ----------------------------
    lookup = {(s, d): px for s, d, px in price_rows}
    panel = []
    for _, r in sel.iterrows():
        px = lookup[(r["slug"], r["date"])]
        for k in range(-PRE_DAYS, POST_DAYS + 1):
            day = r["date"] + dt.timedelta(days=k)
            panel.append({"event_id": r["event_id"], "rel_day": k, "date": day,
                          "price": px.get(day, float("nan")), "btc": btc.get(day, float("nan"))})
    panel = pd.DataFrame(panel)
    panel[["price", "btc"]] = panel.groupby("event_id")[["price", "btc"]].ffill()

    cols = ["event_id", "date", "name", "slug", "gecko_id", "category", "recipients", "tokens_unlocked",
            "circ_before", "pct_circ", "price_at_unlock", "usd_value", "mcap_before"]
    sel[cols].to_csv(DATA / "unlock_events.csv", index=False)
    panel.to_csv(DATA / "event_prices.csv", index=False)
    return sel[cols], panel


if __name__ == "__main__":
    events, _ = build()
    pd.set_option("display.width", 200)
    print(events[["event_id", "date", "name", "category", "pct_circ", "usd_value"]].to_string(index=False))
