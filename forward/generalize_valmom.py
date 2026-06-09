"""Stage-2 CROSS-MARKET GENERALIZATION for the value×mom 0.994 candidate.

The candidate was discovered on MID-cap (DSR 0.994, OOS Sharpe 1.31). Its pre-registration declares a BROAD
factor claim ("value+momentum everywhere" — AQR). Confirmation = the SAME mechanism must show POSITIVE OOS
(untouched 2022+ holdout) in the OTHER cap tiers too. One lucky tier + negatives = mid-cap overfit, reject.

Runs the candidate's exact signal on Large + Small cap universes. Reports every tier (report-ALL discipline).
"""
import importlib
import sys

sys.path.insert(0, "/root/hephaestus")
import numpy as np
import pandas as pd
from sdk.adapters import sep_panel, sf1, us_universe

M = importlib.import_module("strategies.auto_value_momentum_complementary_combination_smith3_68617")
HOLDOUT = "2022-01-01"


def sharpe(r, ann=252):
    r = pd.Series(r).dropna()
    return round(float(r.mean() / r.std() * np.sqrt(ann)), 2) if len(r) > 20 and r.std() > 0 else None


def build_panel(marketcap, per_sector_n=120):
    """Mirror the candidate's _build_universe at a CHOSEN cap tier."""
    tic_sector = {}
    for s in M.SECTORS:
        try:
            ts = us_universe(sector=s, category="Domestic Common Stock",
                             marketcap=marketcap, include_delisted=True, top_n=per_sector_n)
        except Exception:
            ts = []
        for t in (ts or []):
            tic_sector.setdefault(t, s)
    if len(tic_sector) < 100:
        try:
            ts = us_universe(category="Domestic Common Stock", marketcap=marketcap, include_delisted=True, top_n=1200)
        except Exception:
            ts = []
        for t in (ts or []):
            tic_sector.setdefault(t, M.SECTORS[hash(t) % len(M.SECTORS)])
    tickers = sorted(tic_sector)
    px = sep_panel(tickers, M.START, field="closeadj").sort_index()
    px = px.reindex(columns=[c for c in tickers if c in px.columns])
    try:
        bvps = M._pit_panel(sf1(list(px.columns), ["bvps"], dimension="ARQ"), "bvps", px.index, px.columns)
    except Exception:
        bvps = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    panel = pd.concat({"px": px, "bvps": bvps}, axis=1)
    panel.attrs["sector"] = tic_sector
    M._SECTOR_MAP = tic_sector
    return panel


def run(marketcap):
    panel = build_panel(marketcap)
    r = pd.Series(M.signal(panel, **M.SPEC.default_params)[0]).dropna()
    return {"tier": marketcap, "n_names": int(panel["px"].shape[1]),
            "IS_sharpe": sharpe(r[r.index < HOLDOUT]), "OOS_sharpe": sharpe(r[r.index >= HOLDOUT]),
            "full_sharpe": sharpe(r)}


if __name__ == "__main__":
    import json
    res = [run(mc) for mc in ["Large", "Small"]]
    print(json.dumps(res, indent=2))
    oos = [x["OOS_sharpe"] for x in res if x["OOS_sharpe"] is not None]
    pos = sum(1 for s in oos if s > 0)
    print(f"\nDISCOVERY tier (Mid): OOS Sharpe 1.31 (DSR 0.994)")
    print(f"BREADTH on UNTOUCHED tiers: {pos}/{len(oos)} OOS-positive -> "
          f"{'GENERALISES (broad factor)' if pos == len(oos) and len(oos) >= 2 else 'DOES NOT GENERALISE -> mid-cap overfit, REJECT'}")
