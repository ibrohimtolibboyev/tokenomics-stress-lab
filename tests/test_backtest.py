from pathlib import Path

import numpy as np
import pandas as pd

from tokenlab.backtest import study

DATA = Path(__file__).resolve().parents[1] / "data"


def test_sample_has_sixty_distinct_real_events():
    ev = pd.read_csv(DATA / "unlock_events.csv", parse_dates=["date"])
    assert len(ev) == 60
    assert ev["event_id"].is_unique
    assert not ev.duplicated(["slug", "date"]).any()
    assert ev["pct_circ"].between(0.01, 1.0).all()
    assert ev.groupby("slug").size().max() <= 3
    # same-token events never have overlapping [-30, +30] windows
    for _, g in ev.sort_values("date").groupby("slug"):
        assert (g["date"].diff().dropna().dt.days >= 61).all()


def test_price_panel_is_complete():
    panel = pd.read_csv(DATA / "event_prices.csv")
    assert panel["event_id"].nunique() == 60
    assert (panel.groupby("event_id").size() == 151).all()
    assert panel[["price", "btc"]].notna().all().all()
    assert (panel[["price", "btc"]] > 0).all().all()


def test_abnormal_return_of_the_benchmark_is_zero():
    k = np.arange(-120, 31)
    btc = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.03, len(k))))
    panel = pd.DataFrame({"event_id": "X", "rel_day": k, "price": 2 * btc, "btc": btc})
    ar = study.abnormal_returns(panel)["X"]
    assert abs(ar["beta"] - 1) < 1e-9
    assert np.allclose(ar["ar"], 0)
