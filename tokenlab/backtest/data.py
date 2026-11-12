"""Build the unlock event sample.

Vesting schedules and daily prices come from DefiLlama, circulating supply from
CoinMarketCap's historical quotes (market cap / price).

Selection:
  1. cliff unlocks to insiders or private-sale investors, START_DATE..END_DATE
  2. unlock is 1%-100% of circulating supply just before the event
  3. daily prices for the token and BTC over [-120, +30] days
  4. same-token events at least 61 days apart, at most 3 per token
  5. the 60 largest by USD value
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"

HEADERS = {"User-Agent": "tokenomics-stress-lab"}
INDEX_URL = "https://defillama-datasets.llama.fi/emissionsIndex"
PRICE_URL = "https://coins.llama.fi/chart/coingecko:{gid}?start={start}&span={span}&period=1d"
CMC = "https://api.coinmarketcap.com/data-api/v3"
CMC_MAP_URL = CMC + "/map/all?listing_status=active,inactive,untracked&start={start}&limit=9999"
CMC_HIST_URL = CMC + "/cryptocurrency/historical?id={cid}&convertId=2781&timeStart={a}&timeEnd={b}"

START_DATE = dt.date(2022, 1, 1)
END_DATE = dt.date(2026, 9, 30)
SUPPLY_FROM = dt.date(2021, 12, 1)
SUPPLY_TO = dt.date(2026, 11, 1)
INSIDER_CATEGORIES = {"insiders", "privateSale"}
MIN_PCT_CIRC = 0.01
MAX_PCT_CIRC = 1.0
PRE_DAYS = 120
POST_DAYS = 30
MIN_GAP_DAYS = 61
MAX_PER_TOKEN = 3
N_EVENTS = 60
MIN_PRICE_CONFIDENCE = 0.9
MAX_SUPPLY_AGE = 7      # days between the supply quote and the unlock
PRICE_TOLERANCE = 0.15  # max log gap between the CoinMarketCap and DefiLlama price
# schedule is in post-redenomination tokens, price series is the old token
EXCLUDED_SLUGS = {"omni-network"}
EXCLUDED_GECKO_IDS = {"bitcoin", "ethereum", "tether", "usd-coin"}


def _get(url: str, cache: Path, tries: int = 4):
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    err = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(r.text, encoding="utf-8")
                time.sleep(0.3)
                return r.json()
            err = f"HTTP {r.status_code}"
            if r.status_code in (400, 403, 404):
                break
        except requests.RequestException as e:
            err = repr(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}: {err}")


def _date(ts: float) -> dt.date:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()


def _midnight(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())


def candidate_events(index: dict) -> pd.DataFrame:
    """Insider/investor cliff unlocks in the date window, one row per token-day."""
    rows = []
    for p in index["data"]:
        gid = p.get("gecko_id")
        if not gid or gid in EXCLUDED_GECKO_IDS or p.get("protocolSlug") in EXCLUDED_SLUGS:
            continue
        days: dict[dt.date, dict] = {}
        for ev in p.get("unlockEvents") or []:
            d = _date(ev["timestamp"])
            if not START_DATE <= d <= END_DATE:
                continue
            for a in ev.get("cliffAllocations") or []:
                if a.get("category") not in INSIDER_CATEGORIES or not a.get("amount"):
                    continue
                s = days.setdefault(d, {"tokens": 0.0, "recipients": set(), "cats": set()})
                s["tokens"] += float(a["amount"])
                s["recipients"].add(str(a.get("recipient")))
                s["cats"].add(a["category"])
        for d, s in days.items():
            rows.append({"slug": p["protocolSlug"], "gecko_id": gid, "date": d,
                         "tokens_unlocked": s["tokens"],
                         "recipients": "; ".join(sorted(s["recipients"])),
                         "category": "+".join(sorted(s["cats"]))})
    return pd.DataFrame(rows)


def cmc_map() -> list[dict]:
    coins, start = [], 1
    while True:
        js = _get(CMC_MAP_URL.format(start=start), RAW / "cmc" / f"map_{start}.json")
        page = js["data"]["cryptoCurrencyMap"]
        coins += page
        if len(page) < 9999:
            return coins
        start += 9999


def cmc_supply(cid: int) -> pd.DataFrame:
    """Close price and circulating supply (market cap / price), indexed by date."""
    url = CMC_HIST_URL.format(cid=cid, a=_midnight(SUPPLY_FROM), b=_midnight(SUPPLY_TO))
    js = _get(url, RAW / "cmc" / f"{cid}.json")
    rows = {}
    for q in (js.get("data") or {}).get("quotes") or []:
        v = q["quote"]
        if v.get("close") and v.get("marketCap"):
            rows[dt.date.fromisoformat(q["timeClose"][:10])] = (v["close"], v["marketCap"] / v["close"])
    return pd.DataFrame.from_dict(rows, orient="index", columns=["close", "circ"]).sort_index()


def cmc_candidates(p: dict, coins: list[dict]) -> list[tuple[dict, pd.DataFrame]]:
    """CoinMarketCap listings that share the entry's slug, name or ticker, best guess first.
    The right one is picked later by comparing prices."""
    sym = ((p.get("tokenPrice") or [{}])[0].get("symbol") or "").upper()
    gid, name = p["gecko_id"], p["name"].lower()
    cands = [c for c in coins
             if c["slug"] == gid or c["name"].lower() == name or (sym and c["symbol"].upper() == sym)]
    cands.sort(key=lambda c: (c["slug"] != gid, c.get("rank") or 10**9))
    out = []
    for c in cands[:5]:
        try:
            sup = cmc_supply(c["id"])
        except RuntimeError:
            continue
        if not sup.empty:
            out.append((c, sup))
    return out


def fetch_prices(gid: str, start: dt.date, end: dt.date) -> pd.Series:
    """Daily prices at 00:00 UTC, indexed by date."""
    span = (end - start).days + 1
    js = _get(PRICE_URL.format(gid=gid, start=_midnight(start), span=span),
              RAW / "prices" / f"{gid}_{start}_{end}.json")
    coin = js.get("coins", {}).get(f"coingecko:{gid}")
    if not coin or coin.get("confidence", 0) < MIN_PRICE_CONFIDENCE:
        return pd.Series(dtype=float)
    out = {}
    for pt in coin["prices"]:
        d = _date(pt["timestamp"] + 6 * 3600)
        if abs(pt["timestamp"] - _midnight(d)) <= 6 * 3600 and pt["price"] > 0:
            out.setdefault(d, float(pt["price"]))
    return pd.Series(out).sort_index()


def _window_ok(px: pd.Series, d: dt.date) -> bool:
    need = [d + dt.timedelta(days=k) for k in range(-PRE_DAYS, POST_DAYS + 1)]
    have = sum(x in px.index for x in need)
    return have >= len(need) - 3 and need[0] in px.index and d in px.index and need[-1] in px.index


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = _get(INDEX_URL, RAW / "emissionsIndex.json")
    cand = candidate_events(index)
    print(f"candidates: {len(cand)} unlock days, {cand.slug.nunique()} tokens")

    protocols = {p["protocolSlug"]: p for p in index["data"] if p.get("protocolSlug")}
    coins = cmc_map()
    sized, no_supply = [], []
    for slug, grp in cand.groupby("slug"):
        listings = cmc_candidates(protocols[slug], coins)
        if not listings:
            no_supply.append(slug)
            continue
        for r in grp.to_dict("records"):
            options = []
            for coin, sup in listings:
                before = sup[sup.index < r["date"]]
                if before.empty or (r["date"] - before.index[-1]).days > MAX_SUPPLY_AGE:
                    continue
                circ = before["circ"].iloc[-1]
                if MIN_PCT_CIRC <= r["tokens_unlocked"] / circ <= MAX_PCT_CIRC:
                    options.append({"name": coin["name"], "symbol": coin["symbol"], "cmc_id": coin["id"],
                                    "circ_before": circ, "pct_circ": r["tokens_unlocked"] / circ,
                                    "supply_date": before.index[-1], "cmc_close": before["close"].iloc[-1]})
            if options:
                sized.append({**r, "options": options})
    print(f"no CoinMarketCap supply history: {len(no_supply)} tokens: {', '.join(no_supply)}")
    print(f"{MIN_PCT_CIRC:.0%}-{MAX_PCT_CIRC:.0%} of circulating: {len(sized)} events")

    btc_a = START_DATE - dt.timedelta(days=PRE_DAYS + 5)
    btc_b = END_DATE + dt.timedelta(days=POST_DAYS + 5)
    chunks = [btc_a + dt.timedelta(days=400 * i) for i in range((btc_b - btc_a).days // 400 + 1)]
    btc = pd.concat([fetch_prices("bitcoin", a, min(a + dt.timedelta(days=399), btc_b)) for a in chunks])
    btc = btc[~btc.index.duplicated()].sort_index()

    priced, series, mismatched = [], {}, []
    for r in sized:
        d = r["date"]
        try:
            px = fetch_prices(r["gecko_id"], d - dt.timedelta(days=PRE_DAYS + 2), d + dt.timedelta(days=POST_DAYS + 1))
        except RuntimeError:
            continue
        if px.empty or not _window_ok(px, d) or not _window_ok(btc, d):
            continue
        # the CoinMarketCap close is the end of supply_date, DefiLlama samples at 00:00
        match = next((o for o in r.pop("options")
                      if (ref := px.get(o["supply_date"] + dt.timedelta(days=1)))
                      and abs(math.log(ref / o["cmc_close"])) <= PRICE_TOLERANCE), None)
        if match is None:
            mismatched.append(f"{r['slug']} {d}")
            continue
        priced.append({**r, **match, "price_at_unlock": px[d], "usd_value": px[d] * r["tokens_unlocked"],
                       "mcap_before": px[d] * match["circ_before"]})
        series[(r["slug"], d)] = px
    print(f"price mismatch between sources, dropped: {len(mismatched)}: {', '.join(mismatched)}")
    ev = pd.DataFrame(priced).sort_values("usd_value", ascending=False).reset_index(drop=True)
    print(f"with complete prices: {len(ev)} events, {ev.slug.nunique()} tokens")

    chosen, taken = [], {}
    for i, r in ev.iterrows():
        dates = taken.setdefault(r["slug"], [])
        if len(dates) >= MAX_PER_TOKEN or any(abs((r["date"] - x).days) < MIN_GAP_DAYS for x in dates):
            continue
        dates.append(r["date"])
        chosen.append(i)
        if len(chosen) == N_EVENTS:
            break
    sel = ev.loc[chosen].sort_values(["date", "slug"]).reset_index(drop=True)
    sel.insert(0, "event_id", [f"E{i + 1:02d}" for i in range(len(sel))])
    print(f"selected: {len(sel)} events, {sel.slug.nunique()} tokens")

    panel = []
    for r in sel.to_dict("records"):
        px = series[(r["slug"], r["date"])]
        for k in range(-PRE_DAYS, POST_DAYS + 1):
            day = r["date"] + dt.timedelta(days=k)
            panel.append({"event_id": r["event_id"], "rel_day": k, "date": day,
                          "price": px.get(day, float("nan")), "btc": btc.get(day, float("nan"))})
    panel = pd.DataFrame(panel)
    panel[["price", "btc"]] = panel.groupby("event_id")[["price", "btc"]].ffill()

    cols = ["event_id", "date", "symbol", "name", "slug", "gecko_id", "cmc_id", "category", "recipients",
            "tokens_unlocked", "circ_before", "pct_circ", "price_at_unlock", "usd_value", "mcap_before"]
    DATA.mkdir(exist_ok=True)
    sel[cols].to_csv(DATA / "unlock_events.csv", index=False)
    panel.to_csv(DATA / "event_prices.csv", index=False)
    return sel[cols], panel


if __name__ == "__main__":
    events, _ = build()
    pd.set_option("display.width", 200)
    print(events[["event_id", "date", "symbol", "category", "pct_circ", "usd_value"]].to_string(index=False))
