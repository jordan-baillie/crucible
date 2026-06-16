"""
Storage-confirmed commodity scarcity premium — hedging-pressure x inventory x carry,
with a 2-of-3 confirmation gate (energy CL/NG + grains ZC/ZS/ZW).

Mechanism (broad, structural): speculators are paid to absorb commercial hedgers'
positions; the size of that insurance premium is governed by physical storage. We read
the same scarcity state THREE independent ways and require >=2 to agree:
  HP    = z(-(commercial_net/OI))         commercials net short  -> long is paid
  SCAR  = z(-(stocks - trailing-5yr same-calendar-week mean)/std)  below-norm -> long
  CARRY = z(front/second - 1)             backwardation          -> long
COMPOSITE = equal-weight mean (NO fitted weights). 2-of-3 GATE: long-eligible iff
COMPOSITE>0 AND >=2 components>0; short-eligible iff COMPOSITE<0 AND >=2 components<0;
conflicted roots -> flat. Book = long top-2 / short bottom-2 eligible by COMPOSITE,
inverse-20d-vol weighted within each leg, dollar-neutral; shrinks honestly if <2 a side.
Front contract, within-contract chained returns (NEVER diffed across a roll), ~8bps cost.

PIT discipline: every conditioning input is taken as-of its RELEASE date (COT Friday,
EIA Wed/Thu, USDA quarterly publish date) and the whole weight matrix is lagged 1 day
(the +1 lag is applied here -> W.shift(1) -> net_of_cost). No look-ahead.

Adapters only (owned Databento GLBX one-time pull + free CFTC/EIA/USDA/FRED feeds), per
research-wiki/DATA_CATALOG.md. The only novel code is the signal; costs/trades/regime
labels go through the mandatory kit.
"""

from sdk.harness import StrategySpec
from sdk.adapters import (fred_series, fut_curve, cot_positioning, eia_series, usda_nass)
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------------
NAME = "commod_scarcity_2of3"
START = "2010-01-01"
SEARCH_ROOTS = ["CL", "NG", "ZC", "ZS", "ZW"]

# disjoint generalization complexes (different CME roots, share NO tickers with search).
# Commodity universes are inherently small (asset-class reality, not equity ~300-name
# slices); each runs same-night. Storage is NOT available for these -> the frozen signal
# runs on HP+carry there, isolating the storage contribution (gate degrades to 2-of-2).
GEN = {
    "precious":          ["GC", "SI", "PL", "PA"],   # COMEX/NYMEX metals
    "livestock":         ["LE", "HE", "GF"],         # CME livestock
    "industrial_energy": ["HG", "HO", "RB"],         # copper + refined energy
}

ROOT_SECTOR = {
    "CL": "Energy", "NG": "Energy", "HO": "Energy", "RB": "Energy",
    "ZC": "Grains", "ZS": "Grains", "ZW": "Grains",
    "GC": "Metals", "SI": "Metals", "PL": "Metals", "PA": "Metals", "HG": "Metals",
    "LE": "Livestock", "HE": "Livestock", "GF": "Livestock",
}

# storage sources (only energy/grains in the search universe carry inventory)
EIA_STOCK = {"CL": "PET.WCESTUS1.W", "NG": "NG.NW2_EPG0_SWO_R48_BCF.W"}
FRED_STOCK = {"CL": "WCESTUS1", "NG": "NGNW2BCFR48S"}      # free EIA-on-FRED fallback
USDA_COMMODITY = {"ZC": "CORN", "ZS": "SOYBEANS", "ZW": "WHEAT"}

DEFAULT = dict(z_window=156, n_per_side=2, vol_lb=20, scar_lb_years=5,
               min_confirm=2, cost_bps=8.0, rebalance="W",
               drop_hp=False, drop_scar=False, drop_carry=False)


# ============================== data plumbing (adapters only) =====================
def _col(df, name):
    if df is None:
        return None
    low = {str(c).lower(): c for c in df.columns}
    return df[low[name.lower()]].astype(float) if name.lower() in low else None


def _front_returns(fc):
    """Within-contract chained front return; never diff across a roll."""
    low = {str(c).lower(): c for c in fc.columns}
    for cand in ("ret_1", "return_1", "ret1", "front_return", "roll_adj_ret", "ret"):
        if cand in low:
            return fc[low[cand]].astype(float)
    close = None
    for cand in ("close_1", "c1", "front", "px_1", "settle_1", "close"):
        if cand in low:
            close = fc[low[cand]].astype(float)
            break
    if close is None:
        close = fc.iloc[:, 0].astype(float)
    ret = close.pct_change()
    for cand in ("days_to_roll_1", "dtr_1", "days_to_roll", "dtr"):
        if cand in low:                                   # roll boundary -> drop the gap
            ret = ret.mask(fc[low[cand]].astype(float).diff() > 0, np.nan)
            break
    return ret


def _as_series(obj):
    s = None
    if isinstance(obj, pd.Series):
        s = obj
    elif isinstance(obj, pd.DataFrame) and obj.shape[1] >= 1:
        low = {str(c).lower(): c for c in obj.columns}
        pick = next((low[k] for k in ("value", "stocks", "stk", "level", "val") if k in low), None)
        s = obj[pick] if pick is not None else obj.iloc[:, 0]
    if s is None:
        return pd.Series(dtype=float)
    s = pd.Series(s)
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    return s.sort_index()


def _usda_series(df):
    """USDA NASS STOCKS: parse comma 'Value', index by PUBLISH/RELEASE date (not survey period)."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    low = {str(c).lower(): c for c in df.columns}
    vcol = low.get("value")
    if vcol is None:
        return pd.Series(dtype=float)
    val = pd.to_numeric(df[vcol].astype(str).str.replace(",", "", regex=False).str.strip(),
                        errors="coerce")
    dcol = next((low[k] for k in ("release_date", "load_time", "published_date",
                                  "published", "report_date", "date") if k in low), None)
    if dcol is None:
        for c in df.columns:                              # last resort: a parseable date col
            try:
                pd.to_datetime(df[c]); dcol = c; break
            except Exception:
                continue
    if dcol is None:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(df[dcol], errors="coerce")
    s = pd.Series(val.values, index=idx).dropna()
    s = s[~s.index.isna()]
    return s.groupby(level=0).last().sort_index()


def _fred_stock(fred_id, start):
    try:
        return _as_series(fred_series({fred_id: "stk"}, start))
    except Exception:
        return pd.Series(dtype=float)


def _stocks_for(root, start):
    s = pd.Series(dtype=float)
    if root in EIA_STOCK:                                 # energy: EIA (key) -> FRED fallback
        try:
            try:
                raw = eia_series(EIA_STOCK[root], start=start)
            except TypeError:
                raw = eia_series(EIA_STOCK[root])
            s = _as_series(raw)
        except Exception:
            s = pd.Series(dtype=float)
        if s.dropna().empty and root in FRED_STOCK:
            s = _fred_stock(FRED_STOCK[root], start)
    elif root in USDA_COMMODITY:                          # grains: USDA NASS quarterly
        try:
            s = _usda_series(usda_nass(USDA_COMMODITY[root], statisticcat_desc="STOCKS"))
        except Exception:
            s = pd.Series(dtype=float)
    if len(s):
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.index = pd.to_datetime(s.index)
    return s


def _load_cot(roots, start):
    for call in (lambda: cot_positioning(list(roots), start_year=int(str(start)[:4])),
                 lambda: cot_positioning(list(roots))):
        try:
            return call()
        except Exception:
            continue
    return None


def _cot_for(cot, root):
    empty = pd.Series(dtype=float)
    df = None
    if isinstance(cot, dict):
        df = cot.get(root)
    elif isinstance(cot, pd.DataFrame):
        low = {str(c).lower(): c for c in cot.columns}
        if "root" in low:
            df = cot[cot[low["root"]].astype(str) == root]
        elif isinstance(cot.columns, pd.MultiIndex) and root in cot.columns.get_level_values(0):
            df = cot[root]
    if df is None or len(df) == 0:
        return empty, empty
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        low = {str(c).lower(): c for c in df.columns}
        dcol = low.get("date") or low.get("report_date") or low.get("release_date")
        if dcol is not None:
            df = df.set_index(pd.to_datetime(df[dcol]))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    low = {str(c).lower(): c for c in df.columns}
    cn = low.get("comm_net") or low.get("commercial_net") or low.get("comm")
    oi = low.get("oi") or low.get("open_interest")
    comm_net = df[cn].astype(float) if cn else empty
    oival = df[oi].astype(float) if oi else empty
    return comm_net.sort_index(), oival.sort_index()


def _build_panel(roots, start=START):
    """Wide panel: MultiIndex columns (root, field) for field in
       {ret, c1, c2, comm_net, oi, stocks}; weekly/quarterly inputs ffilled from release."""
    cot = _load_cot(roots, start)
    frames = {}
    for r in roots:
        try:
            fc = fut_curve(r, n_contracts=2)
        except Exception:
            continue
        if fc is None or len(fc) == 0:
            continue
        fc = fc.copy()
        if not isinstance(fc.index, pd.DatetimeIndex):
            low = {str(c).lower(): c for c in fc.columns}
            dcol = low.get("date")
            if dcol is None:
                continue
            fc = fc.set_index(pd.to_datetime(fc[dcol]))
        fc = fc[fc.index >= pd.Timestamp(start)].sort_index()
        c1 = _col(fc, "close_1")
        if c1 is None:
            for cand in ("c1", "front", "settle_1", "close"):
                c1 = _col(fc, cand)
                if c1 is not None:
                    break
        if c1 is None:
            continue
        c2 = _col(fc, "close_2")
        if c2 is None:
            c2 = _col(fc, "c2")
        comm_net, oi = _cot_for(cot, r)
        stocks = _stocks_for(r, start)
        df = pd.DataFrame(index=fc.index)
        df["ret"] = _front_returns(fc).reindex(fc.index)
        df["c1"] = c1
        df["c2"] = c2 if c2 is not None else np.nan
        df["comm_net"] = comm_net.reindex(fc.index, method="ffill") if len(comm_net) else np.nan
        df["oi"] = oi.reindex(fc.index, method="ffill") if len(oi) else np.nan
        df["stocks"] = stocks.reindex(fc.index, method="ffill") if len(stocks) else np.nan
        frames[r] = df
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def load_data():
    return _build_panel(SEARCH_ROOTS, start=START)


def load_gen_data(label):
    return _build_panel(GEN[label], start=START)


# ============================== signal (the only novel code) ======================
def _field(panel, field, roots):
    return pd.concat({r: panel[(r, field)] for r in roots}, axis=1)[roots]


def _ts_z(df, window):
    """Per-root trailing time-series z (level-robust: each root vs its own history)."""
    mp = max(26, window // 3)
    m = df.rolling(window, min_periods=mp).mean()
    s = df.rolling(window, min_periods=mp).std()
    return (df - m) / s.replace(0, np.nan)


def _scarcity(stocks, years):
    """SCAR raw = -(stocks - trailing-`years`yr same-calendar-week mean)/std, strictly
       earlier history only (PIT). Below seasonal norm -> positive scarcity -> long."""
    out = pd.DataFrame(np.nan, index=stocks.index, columns=stocks.columns)
    if len(stocks) == 0:
        return out
    idx = stocks.index
    woy = idx.isocalendar().week.to_numpy().astype(int)
    for col in stocks.columns:
        s = stocks[col]
        if s.dropna().empty:
            continue
        vals = s.values.astype(float)
        res = np.full(len(idx), np.nan)
        for i in range(len(idx)):
            if not np.isfinite(vals[i]):
                continue
            lo = idx[i] - pd.DateOffset(years=years)
            mask = (woy == woy[i]) & (idx < idx[i]) & (idx >= lo)
            hist = vals[mask]
            hist = hist[np.isfinite(hist)]
            if hist.size >= 2 and hist.std() > 0:
                res[i] = -(vals[i] - hist.mean()) / hist.std()
        out[col] = res
    return out


def _legw(names, vol):
    """Inverse-trailing-vol weights within a leg, normalized to sum 1 (equal-weight fallback)."""
    if len(names) == 0:
        return {}
    iv = {}
    for r in names:
        v = vol.get(r, np.nan)
        iv[r] = 1.0 / v if (pd.notna(v) and v > 0) else np.nan
    tot = np.nansum(list(iv.values()))
    if not np.isfinite(tot) or tot <= 0:
        return {r: 1.0 / len(names) for r in names}
    return {r: (iv[r] / tot if np.isfinite(iv[r]) else 0.0) for r in names}


def _book(HP, SCAR, CARRY, vol_w, n_side, min_conf, roots):
    W = pd.DataFrame(0.0, index=HP.index, columns=roots)
    vol_w = vol_w.reindex(HP.index)
    for dt in HP.index:
        comps = pd.DataFrame({"HP": HP.loc[dt], "SCAR": SCAR.loc[dt], "CARRY": CARRY.loc[dt]})
        comp_mean = comps.mean(axis=1, skipna=True)            # composite = mean of available
        n_pos = (comps > 0).sum(axis=1)
        n_neg = (comps < 0).sum(axis=1)
        n_avail = comps.notna().sum(axis=1)
        long_elig = (comp_mean > 0) & (n_pos >= min_conf) & (n_avail >= 1)
        short_elig = (comp_mean < 0) & (n_neg >= min_conf) & (n_avail >= 1)
        longs = comp_mean[long_elig].sort_values(ascending=False).head(n_side)
        shorts = comp_mean[short_elig].sort_values(ascending=True).head(n_side)
        v = vol_w.loc[dt]
        for r, w in _legw(list(longs.index), v).items():
            W.at[dt, r] = 0.5 * w                              # dollar-neutral: long leg = +0.5
        for r, w in _legw(list(shorts.index), v).items():
            W.at[dt, r] = -0.5 * w                             #                short leg = -0.5
    return W


def signal(panel, **params):
    if panel is None or len(panel) == 0 or not isinstance(panel.columns, pd.MultiIndex):
        return pd.Series(dtype=float, name=NAME), []
    p = {**DEFAULT, **params}
    z_window = int(p["z_window"]); n_side = int(p["n_per_side"]); vol_lb = int(p["vol_lb"])
    scar_years = int(p["scar_lb_years"]); min_conf = int(p["min_confirm"])
    cost_bps = float(p["cost_bps"])
    freq = "2W-FRI" if str(p["rebalance"]).upper() in ("2W", "BIWEEKLY", "2W-FRI") else "W-FRI"

    roots = list(panel.columns.get_level_values(0).unique())
    sector_map = {r: ROOT_SECTOR.get(r, "Commodity") for r in roots}

    # daily within-contract front returns; roll gaps -> 0 (catalog rule: never diff a roll)
    rets = _field(panel, "ret", roots).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol = rets.rolling(vol_lb, min_periods=max(5, vol_lb // 2)).std()

    # weekly (COT-Friday cadence) snapshots — already release-date ffilled in the panel
    c1 = _field(panel, "c1", roots).resample(freq).last()
    c2 = _field(panel, "c2", roots).resample(freq).last()
    comm = _field(panel, "comm_net", roots).resample(freq).last()
    oi = _field(panel, "oi", roots).resample(freq).last()
    stk = _field(panel, "stocks", roots).resample(freq).last()
    vol_w = vol.resample(freq).last()

    # three trailing-156-week per-root z-scores of one scarcity state
    HP = _ts_z(-(comm / oi.replace(0, np.nan)), z_window)        # commercials net short -> long
    CARRY = _ts_z(c1 / c2 - 1.0, z_window)                       # backwardation -> long
    SCAR = _ts_z(_scarcity(stk, scar_years), z_window)           # below seasonal norm -> long
    if bool(p["drop_hp"]):    HP = HP * np.nan
    if bool(p["drop_scar"]):  SCAR = SCAR * np.nan
    if bool(p["drop_carry"]): CARRY = CARRY * np.nan

    W_weekly = _book(HP, SCAR, CARRY, vol_w, n_side, min_conf, roots)

    # expand weekly target to daily, then apply the +1 day execution lag here (our job)
    W_daily = W_weekly.reindex(rets.index, method="ffill")
    W_held = W_daily.shift(1).fillna(0.0)[roots]

    daily = net_of_cost(W_held, rets, cost_bps=cost_bps, name=NAME)   # turnover-cost on lagged W
    trades = trades_from_weights(W_held, rets, sector_map)            # kit stamps entry_regime
    return daily, trades


# ============================== soft expectations (pre-registered falsifiers) ======
def _search_sharpe(returns, holdout):
    r = pd.Series(returns).dropna()
    r = r[r.index < pd.Timestamp(holdout)]
    if len(r) < 30 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252.0))


def _exp_gate(ctx):
    """The 2-of-3 confirmation must not underperform the ungated composite (the storage/
       positioning confirmation is signal, not noise). Free: uses declared grid variants."""
    g = ctx.get("grid", {})
    if "default" not in g or "ungated" not in g:
        return {"pass": False, "observed": "variants missing"}
    sd = _search_sharpe(g["default"], ctx["holdout_start"])
    su = _search_sharpe(g["ungated"], ctx["holdout_start"])
    return {"pass": sd >= su, "observed": round(sd - su, 3)}


def _drop_check(component):
    def chk(ctx):
        r, _ = signal(ctx["panel"], **{f"drop_{component}": True})   # 1 extra signal() call
        s = _search_sharpe(r, ctx["holdout_start"])
        return {"pass": s > 0.0, "observed": round(s, 3)}
    return chk


EXPECTATIONS = [
    {"name": "gate_adds_value",
     "claim": "2-of-3 confirmation gate search Sharpe >= ungated composite search Sharpe",
     "check": _exp_gate},
    {"name": "carry_not_sole_driver",
     "claim": "dropping CARRY keeps search Sharpe > 0 (no single leg carries the composite)",
     "check": _drop_check("carry")},
    {"name": "hp_not_sole_driver",
     "claim": "dropping hedging-pressure keeps search Sharpe > 0",
     "check": _drop_check("hp")},
    {"name": "scar_not_sole_driver",
     "claim": "dropping the storage/scarcity component keeps search Sharpe > 0",
     "check": _drop_check("scar")},
]

# honest search burden (default = primary; the rest are robustness, not selection)
GRID = {
    "default":      {},
    "long1_short1": {"n_per_side": 1},
    "z_window_104": {"z_window": 104},
    "biweekly":     {"rebalance": "2W"},
    "ungated":      {"min_confirm": 0},
}

PRE_REG = (
    "Structural commodity insurance/scarcity premium: speculators are paid to absorb "
    "commercial hedgers' positions, magnitude governed by physical storage. Harvested as a "
    "dollar-neutral cross-section of 5 retail-micro-tradable roots (energy CL,NG + grains "
    "ZC,ZS,ZW). Weekly on CFTC-COT Friday-release cadence, every input as-of its release date "
    "(COT Friday, EIA Wed/Thu, USDA quarterly publish date), whole book lagged 1 day. Per root "
    "form three trailing-156-week z-scores: HP=z(-(commercial_net/OI)), "
    "SCAR=z(-(stocks - trailing-5yr same-calendar-week mean)/std), CARRY=z(front/second-1). "
    "COMPOSITE = equal-weight mean (NO fitted weights). 2-of-3 GATE: long-eligible iff "
    "COMPOSITE>0 AND >=2 components>0; short-eligible iff COMPOSITE<0 AND >=2 components<0; "
    "conflicted roots dropped flat. Book = long top-2 / short bottom-2 eligible by COMPOSITE, "
    "inverse-20d-vol weighted within each leg, dollar-neutral; shrinks honestly if <2 a side. "
    "Front contract, within-contract chained returns (never diffed across rolls), ~8bps "
    "turnover cost. FROZEN spec; grid is honest robustness only. Pre-registered falsifiers "
    "(soft checks): gated must not underperform ungated, and dropping ANY single component must "
    "keep search Sharpe>0. scope=broad: carry/hedging-pressure is a universal commodity "
    "mechanism -> must generalise to disjoint CME complexes (precious metals, livestock, "
    "industrial+refined energy) on HP+carry where storage is absent, isolating the storage "
    "contribution. EIA-key dependency degrades gracefully: energy SCAR falls back to FRED "
    "stocks; if both unavailable energy runs HP+carry (gate -> 2-of-2). Holdout 2022-01-01, "
    "write-once."
)

SPEC = StrategySpec(
    id=NAME,
    family="commodity_carry_hedging_pressure_storage",
    title="Storage-confirmed commodity scarcity premium (HP x inventory x carry, 2-of-3 gate)",
    markets=SEARCH_ROOTS,
    data_desc=("Databento GLBX front+2nd futures curve (carry/returns) + CFTC COT commercial "
               "net/OI (hedging pressure) + EIA weekly energy stocks & USDA quarterly grain "
               "stocks (scarcity, FRED fallback); all release-date PIT, 1-day lagged."),
    pre_registration=PRE_REG,
    load_data=load_data,
    signal=signal,
    default_params=DEFAULT,
    grid=GRID,
    scope="broad",
    generalization_universes=list(GEN.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=4,
    expectations=EXPECTATIONS,
)