"""
Inventory-conditioned commodity convenience-yield premium
==========================================================
Storage-theory risk premium: get paid to bear stockout risk when inventories
are scarce (backwardation).  This is the FUNDAMENTALS-CONFIRMED form of
commodity carry -- the term-structure carry signal is GATED by a real
physical-inventory deviation (EIA energy stocks / USDA grain stocks), which
was unowned until data task #60.  Prior commodity books used price-or-
positioning signals only (basis-momentum, COT, seasonality, skew, price-only
carry -- all FAILED); the novelty here is the SIGN-AGREEMENT gate between the
price basis and the actual storage variable the theory invokes.

Mechanism (frozen, weekly EIA-cadence rebalance, 7 storable roots):
  C_i = annualised front-back basis  (close_1 - close_2)/close_2   (backw.>0 -> long)
  I_i = -seasonal z-score of latest RELEASED stocks vs same-week/quarter norm
        (low inventory -> high I_i -> long), PIT release-dated + ffilled.
  S_i = z(C_i) + z(I_i)  cross-sectionally; a root is ELIGIBLE only when
        sign(C_i)==sign(I_i) (basis AND fundamentals must concur).
  LONG the sign-agreed backwardated/low-stock roots, SHORT the agreed
  contango/high-stock roots, inverse-vol within leg, dollar-neutral,
  vol-target 10% annual.  Standalone (no reflexive trend overlay).

SCOPE NOTE (decision, see pre_registration): the proposal requested
scope='broad', but the owned-inventory data boundary caps the universe at 7
storable futures -- the harness broad battery (>=3 DISJOINT ~150-400 name
universes) is impossible with 7 contracts.  The universality claim (the
premium must appear in BOTH the energy and grain complexes, not one only --
the BAB single-universe trap) is therefore enforced as two MACHINE-CHECKED
soft expectations, which is the disjoint-sub-universe test in falsifiable
form.  scope='local' -> forward-validation + MCPT + 2022 holdout confirm.

NOTE ON ADAPTERS: fut_curve / eia_series / usda_nass are the owned/free
commodity-curve + physical-inventory adapters delivered by data task #60
(Databento contract-months; EIA weekly stocks; USDA NASS Grain Stocks) per
research-wiki/DATA_CATALOG.md.  They are required for this proposal; all other
data plumbing uses the mandatory kit.  No raw downloads, no hand-rolled
look-ahead-prone logic -- the only novel code is the signal itself.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
# owned/free commodity-curve + physical-inventory adapters (DATA_CATALOG.md, task #60)
from sdk.adapters import fut_curve, eia_series, usda_nass

# ----------------------------------------------------------------------------
SPEC_ID = "inv_cond_convenience_yield_v1"

ENERGY = ["CL", "NG", "HO", "RB"]
GRAINS = ["ZC", "ZS", "ZW"]
ROOTS  = ENERGY + GRAINS
SECTOR = {**{r: "Energy" for r in ENERGY}, **{r: "Grains" for r in GRAINS}}

# EIA weekly stock series (verified IDs, task #60)
EIA_SERIES = {
    "CL": "PET.WCESTUS1.W",            # crude commercial stocks
    "RB": "PET.WGTSTUS1.W",            # total motor gasoline
    "HO": "PET.WDISTUS1.W",            # distillate (~heating oil)
    "NG": "NG.NW2_EPG0_SWO_R48_BCF.W", # working gas in storage (lower-48)
}
USDA_COMMODITY = {"ZC": "CORN", "ZS": "SOYBEANS", "ZW": "WHEAT"}

# nominal contract-spacing annualisation of the front-back basis (months/yr basis)
ANN = {**{r: 12.0 for r in ENERGY}, **{r: 6.0 for r in GRAINS}}
# CONSERVATIVE release lag added to the inventory period date -> NO look-ahead.
# (extra lag is only mildly stale, never forward-looking; >= true EIA/USDA delay)
RELEASE_LAG = {**{r: 7 for r in ENERGY}, **{r: 30 for r in GRAINS}}

FUT_START = "2008-01-01"
INV_START = "1995-01-01"   # long burn-in for the seasonal norm

DEFAULTS = dict(
    leg_frac=1.0,        # take ALL sign-agreeing roots per leg (robust for tiny universe)
    min_per_leg=1,       # need >=1 long AND >=1 short to keep the book dollar-neutral
    vol_target=0.10,     # 10% annualised book vol
    vol_lb=63,           # ~3-month trailing vol window
    cost_bps=8.0,        # conservative round-trip on turnover (>= ~3-5 futures ticks)
    use_agreement=True,  # storage-theory sign-agreement gate (False = price-only carry baseline)
    max_lev=4.0,
)

GRID = {
    "default":   {},
    "third":     {"leg_frac": 0.34},   # concentrate to strongest tercile per leg
    "half":      {"leg_frac": 0.50},
    "vol_lb_42": {"vol_lb": 42},
}

GEN_UNIVERSES = {"energy_complex": ENERGY, "grain_complex": GRAINS}


# ----------------------------- helpers --------------------------------------
def _col(df, *cands):
    """Resolve a column by exact then case-insensitive match."""
    for c in cands:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _front_returns(fc):
    """Front-contract daily return; roll days (days_to_roll jumps up) masked to 0
    so the close_1 contract-switch jump never becomes a fake return."""
    cc = _col(fc, "close_1", "c1")
    r = fc[cc].astype(float).pct_change()
    drc = _col(fc, "days_to_roll_1", "days_to_roll", "dtr_1")
    if drc is not None:
        roll = fc[drc].astype(float).diff() > 0
        r = r.mask(roll.reindex(r.index).fillna(False), 0.0)
    return r


def _seasonal_z(s, bucket):
    """PIT seasonal z-score: within each seasonal bucket (week-of-year / quarter)
    use ONLY prior-year same-bucket obs (expanding mean/std shifted by 1)."""
    df = pd.DataFrame({"v": np.asarray(s.values, dtype=float),
                       "b": np.asarray(bucket)}, index=s.index).sort_index()
    g = df.groupby("b")["v"]
    mu = g.transform(lambda x: x.expanding(min_periods=2).mean().shift(1))
    sd = g.transform(lambda x: x.expanding(min_periods=2).std().shift(1))
    z = (df["v"] - mu) / sd.replace(0.0, np.nan)
    return z.dropna()


def _seasonal_dev_series(raw, bucket_fn, lag_days):
    """raw: period-dated stock level Series -> seasonal-deviation z dated at ~release."""
    s = raw.sort_index().dropna()
    s = s[~s.index.duplicated(keep="last")]
    z = _seasonal_z(s, bucket_fn(s.index))          # season computed from the PERIOD date
    z = z.copy()
    z.index = z.index + pd.Timedelta(days=lag_days)  # then shift to ~release date (PIT)
    return z


def _woy(idx):
    return idx.isocalendar().week.to_numpy()


def _qtr(idx):
    return np.asarray(idx.quarter)


def _eia_raw(root):
    df = eia_series({EIA_SERIES[root]: "v"}, start=INV_START)
    s = df["v"] if isinstance(df, pd.DataFrame) else df
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _usda_to_series(raw):
    """Defensive parse of usda_nass STOCKS output -> national-total Series indexed
    by release date.  Handles comma-strings ('1,234,567') and suppressed values."""
    if isinstance(raw, pd.Series):
        vals = (pd.to_numeric(raw.astype(str).str.replace(",", "", regex=False),
                              errors="coerce") if raw.dtype == object
                else pd.to_numeric(raw, errors="coerce"))
        return vals.dropna().sort_index()
    df = pd.DataFrame(raw).copy()
    ac = _col(df, "agg_level_desc")
    if ac is not None:
        df = df[df[ac].astype(str).str.upper().str.contains("NATIONAL")]
    vcol = _col(df, "Value", "value", "val")
    if vcol is None:
        num = df.select_dtypes("number")
        if num.shape[1] == 0:
            raise ValueError("usda_nass: no numeric value column")
        vals = num.iloc[:, 0]
    else:
        vals = pd.to_numeric(
            df[vcol].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce")
    dcol = _col(df, "release_date", "released", "load_time", "date", "end_date",
                "week_ending")
    if dcol is not None:
        idx = pd.to_datetime(df[dcol], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        raise ValueError("usda_nass: no release-date column")
    out = pd.Series(np.asarray(vals, dtype=float), index=idx).dropna()
    out = out[out.index.notna()]
    out = out.groupby(out.index).max()      # national total >= on/off-farm sub-rows
    return out.sort_index()


def _usda_raw(root):
    raw = usda_nass(USDA_COMMODITY[root], statisticcat_desc="STOCKS")
    return _usda_to_series(raw)


# ----------------------------- data -----------------------------------------
def load_data() -> pd.DataFrame:
    """Panel signal() consumes: per root  R__c1, R__c2, R__ret, R__invz
       (invz = PIT release-dated, ffilled seasonal-deviation z of physical stocks)."""
    cols, have = {}, []
    for r in ROOTS:
        try:
            fc = fut_curve(r, n_contracts=2, start=FUT_START).sort_index()
            c1c, c2c = _col(fc, "close_1", "c1"), _col(fc, "close_2", "c2")
            if c1c is None or c2c is None:
                continue
            cols[f"{r}__c1"] = fc[c1c].astype(float)
            cols[f"{r}__c2"] = fc[c2c].astype(float)
            cols[f"{r}__ret"] = _front_returns(fc)
            have.append(r)
        except Exception:
            continue  # drop a root with no/thin curve coverage (gate0 sanity)

    panel = pd.DataFrame(cols).sort_index()
    panel.index = pd.to_datetime(panel.index)
    panel = panel[~panel.index.duplicated(keep="last")]
    didx = panel.index

    for r in have:
        try:
            if r in EIA_SERIES:
                z = _seasonal_dev_series(_eia_raw(r), _woy, RELEASE_LAG[r])
            else:
                z = _seasonal_dev_series(_usda_raw(r), _qtr, RELEASE_LAG[r])
            zf = z.sort_index().reindex(didx.union(z.index)).ffill().reindex(didx)
        except Exception:
            zf = pd.Series(np.nan, index=didx)   # no inventory -> root never eligible
        panel[f"{r}__invz"] = zf
    return panel


def load_gen_data(label) -> pd.DataFrame:
    """Sub-complex panel (energy / grain).  Used by the cross-complex soft checks
    and available if this is ever promoted to a broad battery."""
    panel = load_data()
    roots = GEN_UNIVERSES[label]
    keep = [c for c in panel.columns if c.split("__")[0] in roots]
    return panel[keep]


# ----------------------------- signal ---------------------------------------
def _select(cvec, ivec, svec, valid, p):
    """Return (longs, shorts) root lists."""
    if p["use_agreement"]:
        longs = [r for r in valid if cvec[r] > 0 and ivec[r] > 0]
        shorts = [r for r in valid if cvec[r] < 0 and ivec[r] < 0]
        longs.sort(key=lambda r: svec[r], reverse=True)
        shorts.sort(key=lambda r: svec[r])
        if p["leg_frac"] < 1.0:
            if longs:
                longs = longs[:max(1, int(np.ceil(p["leg_frac"] * len(longs))))]
            if shorts:
                shorts = shorts[:max(1, int(np.ceil(p["leg_frac"] * len(shorts))))]
    else:
        # price-only carry baseline: rank by CARRY, no inventory gate
        ranked = sorted(valid, key=lambda r: cvec[r], reverse=True)
        n = len(ranked)
        k = min(max(1, int(np.floor(p["leg_frac"] * n))), n // 2)
        if k < 1:
            return [], []
        longs, shorts = ranked[:k], ranked[-k:]
    return longs, shorts


def signal(panel, **params):
    p = {**DEFAULTS, **params}
    roots = sorted({c.split("__")[0] for c in panel.columns})

    def field(suf):
        cc = [f"{r}__{suf}" for r in roots if f"{r}__{suf}" in panel.columns]
        df = panel[cc].copy()
        df.columns = [c.split("__")[0] for c in cc]
        return df

    c1, c2, ret, invz = field("c1"), field("c2"), field("ret"), field("invz")
    idx = panel.index

    common = [r for r in c1.columns if r in invz.columns and r in c2.columns]
    if len(common) < 2:
        return pd.Series(0.0, index=idx, name=SPEC_ID), []

    C = ((c1 - c2) / c2)[common].replace([np.inf, -np.inf], np.nan)
    ann = pd.Series({r: ANN.get(r, 12.0) for r in common})
    C = C.mul(ann, axis=1)
    Ii = -invz[common]                       # low inventory -> high favourability
    zC, zI = xs_zscore(C), xs_zscore(Ii)
    S = zC + zI

    retc = ret.reindex(columns=common)
    vol_df = retc.rolling(p["vol_lb"], min_periods=max(10, p["vol_lb"] // 3)).std()

    # weekly rebalance: last trading day of each ISO week (EIA cadence)
    tmp = pd.Series(idx.values, index=idx.to_period("W"))
    rebal_dates = pd.DatetimeIndex(sorted(tmp.groupby(level=0).last().values))

    rows = {}
    for t in rebal_dates:
        cvec, ivec, svec, vt = C.loc[t], Ii.loc[t], S.loc[t], vol_df.loc[t]
        valid = [r for r in common
                 if pd.notna(cvec[r]) and pd.notna(ivec[r]) and pd.notna(svec[r])]
        longs, shorts = _select(cvec, ivec, svec, valid, p)

        w0 = pd.Series(0.0, index=common)
        if (len(longs) >= p["min_per_leg"]) and (len(shorts) >= p["min_per_leg"]):
            def legw(names, gross):
                iv = {r: 1.0 / vt.get(r, np.nan) for r in names
                      if pd.notna(vt.get(r, np.nan)) and vt.get(r, np.nan) > 0}
                tot = sum(iv.values())
                return {r: v / tot * gross for r, v in iv.items()} if tot > 0 else {}
            wl, ws = legw(longs, 0.5), legw(shorts, -0.5)
            if wl and ws:
                for r, v in {**wl, **ws}.items():
                    w0[r] = v

        # vol-target the dollar-neutral book
        if w0.abs().sum() > 0:
            win = retc.loc[:t].tail(p["vol_lb"]).fillna(0.0)
            bv = win.mul(w0, axis=1).sum(axis=1).std() * np.sqrt(252)
            scale = 0.0 if (not np.isfinite(bv) or bv <= 1e-9) \
                else min(p["vol_target"] / bv, p["max_lev"])
            w0 = w0 * scale
        rows[t] = w0

    Wd = pd.DataFrame(rows).T.reindex(columns=common).sort_index()
    W = Wd.reindex(idx).ffill().fillna(0.0)
    price_ok = c1.reindex(columns=common).reindex(idx).notna()
    W = W.where(price_ok, 0.0)
    W = W.shift(1).fillna(0.0)               # 1-day execution lag (signals set at close t)

    retm = retc.reindex(idx).fillna(0.0)
    daily = net_of_cost(W, retm, cost_bps=p["cost_bps"], name=SPEC_ID)
    trades = trades_from_weights(W, retm, SECTOR)
    return daily, trades


# -------------------------- soft expectations -------------------------------
def _sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 20 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _subpanel(panel, roots):
    keep = [c for c in panel.columns if c.split("__")[0] in roots]
    return panel[keep]


def _chk_energy(ctx):
    hs = pd.Timestamp(ctx["holdout_start"])
    r, _ = signal(_subpanel(ctx["panel"], ENERGY))      # one extra signal() call
    sr = _sharpe(r[r.index < hs])
    return {"pass": bool(sr > 0), "observed": round(sr, 3)}


def _chk_grain(ctx):
    hs = pd.Timestamp(ctx["holdout_start"])
    r, _ = signal(_subpanel(ctx["panel"], GRAINS))      # one extra signal() call
    sr = _sharpe(r[r.index < hs])
    return {"pass": bool(sr > 0), "observed": round(sr, 3)}


def _chk_inventory_adds_value(ctx):
    hs = pd.Timestamp(ctx["holdout_start"])
    base = _sharpe(pd.Series(ctx["search"]))            # gated default (free)
    r_off, _ = signal(ctx["panel"], use_agreement=False)  # one extra signal() call
    off = _sharpe(r_off[r_off.index < hs])
    return {"pass": bool(base > off), "observed": round(base - off, 3)}


EXPECTATIONS = [
    {"name": "energy_complex_positive",
     "claim": "convenience-yield book is Sharpe-positive STANDALONE in the ENERGY "
              "sub-block (CL/NG/HO/RB) over the search window (universality test #1)",
     "check": _chk_energy},
    {"name": "grain_complex_positive",
     "claim": "convenience-yield book is Sharpe-positive STANDALONE in the GRAIN "
              "sub-block (ZC/ZS/ZW) over the search window (universality test #2; a "
              "one-complex-only result is an artifact, cf. BAB)",
     "check": _chk_grain},
    {"name": "inventory_conditioning_adds_value",
     "claim": "sign-agreement inventory gate beats price-only carry: Sharpe(gated) > "
              "Sharpe(carry-only) over the search window (the core mechanism claim)",
     "check": _chk_inventory_adds_value},
]


# ------------------------------- spec ---------------------------------------
SPEC = StrategySpec(
    id=SPEC_ID,
    family="commodity_convenience_yield",
    title="Inventory-conditioned commodity convenience-yield premium "
          "(term-structure carry GATED by physical-stock deviation)",
    markets=ROOTS,
    data_desc="Databento fut_curve(root, n_contracts=2) front-back basis on 7 storable "
              "roots (energy CL/NG/HO/RB, grains ZC/ZS/ZW) + EIA weekly stocks + USDA "
              "NASS quarterly Grain Stocks; PIT release-dated seasonal deviation.",
    pre_registration=(
        "Storage theory: low inventory -> backwardation -> a convenience-yield premium "
        "for bearing stockout risk; high inventory -> contango -> negative carry. We "
        "LONG roots where the term-structure basis (C>0) AND the real seasonal inventory "
        "deviation (I>0, i.e. stocks below the same-week/quarter norm) BOTH favour long, "
        "and SHORT roots where both favour short; sign-disagreement roots are flat. "
        "Inverse-vol within leg, dollar-neutral, vol-target 10%, weekly EIA-cadence "
        "rebalance, signals lagged 1 day, ~8bps round-trip on turnover. Inventory is "
        "RELEASE-DATED with a conservative extra lag (EIA +7d, USDA +30d) and a PIT "
        "seasonal z built from prior-year same-bucket obs only (no calendardate/no "
        "look-ahead). Tested STANDALONE (no reflexive trend overlay). "
        "SCOPE: the proposal asked for scope='broad', but owned physical-inventory data "
        "bounds the universe to 7 storable futures -- the harness broad battery (>=3 "
        "DISJOINT ~150-400 name universes) cannot be satisfied. The universality claim "
        "is therefore enforced as TWO machine-checked soft expectations "
        "(energy_complex_positive, grain_complex_positive) -- the disjoint-sub-universe "
        "test in falsifiable form -- so a one-complex-only artifact is caught. "
        "Validation path: MCPT (mandatory) + 2022 holdout + live forward-validation. "
        "Falsifiable mechanism: the sign-agreement inventory gate must BEAT price-only "
        "carry (checked); if not, the inventory-conditioning thesis is wrong."
    ),
    load_data=load_data,
    signal=signal,
    default_params=dict(DEFAULTS),
    grid=GRID,
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=7,
    expectations=EXPECTATIONS,
)