"""Tests for sdk/cost_model.py — the FROZEN cost ladder + borrow filter + re-priced net_of_cost.
Pure-logic tests use synthetic data (no live deps); a network-marked smoke test hits real caches."""
import numpy as np
import pandas as pd
import pytest

from sdk import cost_model as cm
from sdk.signal_kit import net_of_cost


def test_ladder_frozen_values():
    # Guard against accidental edits to the frozen ladder (pre-reg §2).
    assert cm.LADDER_CENTRAL[10] == 5 and cm.LADDER_CENTRAL[1] == 100
    assert cm.LADDER_CONSERVATIVE[10] == 8 and cm.LADDER_CONSERVATIVE[1] == 160
    assert cm.BORROW_INFEASIBLE_CAP == 0.20
    # conservative strictly dearer than central at every decile
    assert all(cm.LADDER_CONSERVATIVE[d] > cm.LADDER_CENTRAL[d] for d in range(1, 11))


def test_ladder_cost_bps_decile_ordering():
    # 10 names with monotone dollar volume -> dearest cost on the least liquid.
    dv = {f"T{i}": float(10 ** i) for i in range(10)}  # T0 illiquid ... T9 liquid
    costs = cm.ladder_cost_bps(list(dv), cm.LADDER_CENTRAL, dv_map=dv)
    assert costs["T0"] > costs["T9"]
    assert costs["T0"] == cm.LADDER_CENTRAL[1] and costs["T9"] == cm.LADDER_CENTRAL[10]


def test_ladder_unmapped_treated_liquid():
    costs = cm.ladder_cost_bps(["SPY", "QQQ"], cm.LADDER_CENTRAL, dv_map={})
    assert costs["SPY"] == cm.LADDER_CENTRAL[10]  # non-equity -> most liquid bucket


def test_borrow_zeroes_infeasible_short_and_charges_per_name():
    idx = pd.date_range("2020-01-01", periods=5)
    # long LIQ, short ILLIQ(non-shortable) -> the short must be zeroed
    W = pd.DataFrame({"LIQ": [0.5] * 5, "ILLIQ": [-0.5] * 5}, index=idx)
    rets = pd.DataFrame({"LIQ": [0.0, 0.01, 0.0, 0.0, 0.0],
                         "ILLIQ": [0.0, 0.10, 0.0, 0.0, 0.0]}, index=idx)  # short would've "earned" -0.05
    dv = {"LIQ": 1e9, "ILLIQ": 1e6}
    shortable = frozenset({"LIQ"})  # ILLIQ NOT shortable
    rec = {}
    net = cm.make_net_of_cost(cm.LADDER_CENTRAL, dv_map=dv, shortable=shortable, record=rec)(W, rets)
    # day-2 gross = 0.5*0.01 (LIQ) + 0 (ILLIQ zeroed) = 0.005, NOT 0.005 - 0.05
    assert net.iloc[1] > 0  # the fake short profit is removed, not banked
    assert rec["short_infeasible_share"] == 1.0  # all short pos-days were non-shortable


def test_reprice_is_dearer_than_flat_for_illiquid_book():
    idx = pd.date_range("2020-01-01", periods=10)
    W = pd.DataFrame({"A": [0.5, 0.0] * 5, "B": [0.5, 0.0] * 5}, index=idx)  # high turnover
    rets = pd.DataFrame({"A": np.zeros(10), "B": np.zeros(10)}, index=idx)
    dv = {"A": 1e6, "B": 2e6}  # both illiquid -> dear ladder
    shortable = frozenset({"A", "B"})
    flat = net_of_cost(W, rets, cost_bps=8.0)
    ladder = cm.make_net_of_cost(cm.LADDER_CENTRAL, dv_map=dv, shortable=shortable)(W, rets)
    assert ladder.sum() < flat.sum()  # illiquid names cost MORE than flat 8bps -> lower net


def test_borrow_verdict_from_trades():
    shortable = frozenset({"GOODSHORT"})
    trades = [
        {"ticker": "GOODSHORT", "position_value": -1000, "hold_days": 10},
        {"ticker": "NOSHORT", "position_value": -1000, "hold_days": 90},  # 90% non-shortable
    ]
    v = cm.borrow_verdict(trades, shortable=shortable)
    assert v["borrow_feasible"] is False and v["short_infeasible_share"] == 0.9
    # long-only book is always borrow-feasible
    assert cm.borrow_verdict([{"ticker": "X", "position_value": 1000, "hold_days": 5}],
                             shortable=shortable)["borrow_feasible"] is True


@pytest.mark.network
def test_live_caches_load():
    s = cm.shortable_set()
    assert len(s) > 1000  # the Alpaca shortable snapshot
    dv = cm.dollar_volume_map()
    assert len(dv) > 1000  # SEP universe dollar-volume map


def test_harness_deployability_filter_demotes(monkeypatch):
    """The live-gate filter (pre-reg §3): a borrow-infeasible short leg is NOT deployable
    (candidate=False -> borrow-only path, no signal() call needed)."""
    from sdk import cost_model as cm
    from sdk import harness as H
    cm.shortable_set.cache_clear()
    monkeypatch.setattr(cm, "shortable_set", lambda *a, **k: frozenset({"OK"}))
    trades = [{"ticker": "NOSHORT", "position_value": -1000, "hold_days": 50},
              {"ticker": "OK", "position_value": -100, "hold_days": 5}]
    out = H._deployability_filter(spec=None, panel=None, search_trades=trades, candidate=False)
    assert out["deployable"] is False
    assert out["borrow_feasible"] is False
    assert any("un-borrowable" in r or "DEPLOYABILITY" in r for r in out["reasons"])
    # a long-only book is always deployable on the borrow axis
    ok = H._deployability_filter(spec=None, panel=None,
                                search_trades=[{"ticker": "X", "position_value": 1000, "hold_days": 5}],
                                candidate=False)
    assert ok["deployable"] is True
