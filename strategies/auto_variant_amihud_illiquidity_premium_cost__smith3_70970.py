"""
Amihud illiquidity premium — LONG-ONLY cost-gated mid-illiquid band + single liquid ETF beta-hedge.

VARIANT of the elite cost-gated Amihud construction: the multi-name small/mid stock SHORT leg
(unmodeled borrow fees / locate failures / short-side impact, untradeable at $5K) is replaced by
ONE liquid index-ETF hedge (IWM primary, SPY fallback) sized to the long book's trailing-60d beta.

FIX vs failed run: sep_panel(field="close") crashed inside pyarrow column projection — the vendored
SEP parquet does not expose every raw SEP column under that name. We now load the unadjusted price
via a fallback chain ("closeunadj" -> "close" -> "closeadj") and volume via ("volume",), taking the
first field the store actually serves. Dollar volume = unadjusted price x raw share volume as
pre-registered; if only closeadj exists the dollar-volume proxy degrades gracefully (stated below)
rather than crashing the whole run. No other logic changed.

PRE-REGISTERED (before any run) — faithful to the parent proposal's frozen PRIMARY config:
  * Universe: Sharadar mid-cap (search) / small-cap sector slices (generalization), delisted INCLUDED.
  * Amihud_i = trailing-252d mean( |ret| / dollar_volume ), dollar volume from UNADJUSTED close x raw volume.
  * Band: 60th–80th illiquidity percentile (capacity-tradeable band, NOT the extreme quintile);
    hysteresis keep-band 50th–85th; monthly rebalance.
  * Long leg UNCHANGED from the parent's frozen PRIMARY: EQUAL-WEIGHTED ~50-name book,
    sector x size-tercile balanced. (No inverse-vol resizing — that was not pre-registered.)
  * SELF-CONSISTENT cost gate, as the proposal states it: include a name ONLY if its EXPECTED
    PREMIUM NET OF ITS OWN MODELED IMPACT stays positive — i.e. round-trip impact 2 x Amihud x $T
    must be <= the pre-registered expected gross premium accrued over the expected hold
    (expected gross premium frozen at 300 bps/yr; expected hold 126 trading days from the
    hysteresis design -> round-trip impact budget = 300 * 126/252 = 150 bps). $T default $2.5K;
    $10K / $50K declared in the grid.
  * Net returns = base 8 bps on all turnover (incl. ETF hedge) MINUS name-specific impact
    drag Amihud x $T on every unit of long-leg turnover.  Flat-15bps x-check in the grid.
  * Hedge: short IWM at trailing-60d book beta, clipped [0, 1.0] -> gross <= 2x. The HEDGED
    series is what the rails judge; "unhedged_diag" grid entry is the long-only diagnostic
    (the premium must not be residual small-cap beta).
  * No look-ahead: every input at date d uses data through d only; weights decided at close d
    are shift(1)-lagged before net_of_cost / trades_from_weights (the lag is OUR responsibility).
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2000-01-01"
HEDGE_CANDIDATES = ["IWM", "SPY"]          # single liquid ETF; SPY only if IWM unloadable
HEDGE_SECTOR = "Index Hedge (ETF)"

# Generalization universes: SMALL-cap (different cap tier from the Mid search universe — disjoint
# by construction) split into three sector partitions that share NO tickers with each other.
GEN_SECTORS = {
    "small_cyclical":  ["Consumer Cyclical", "Industrials", "Basic Materials", "Energy"],
    "small_defensive": ["Consumer Defensive", "Healthcare", "Utilities", "Real Estate"],
    "small_techfin":   ["Technology", "Financial Services", "Communication Services"],
}

DEFAULTS = dict(
    amihud_lb=252, min_obs=126,          # trailing Amihud window / min observations
    price_floor=2.0, adv_floor=2.0e5,    # raw price >= $2, 63d median $vol >= $200k
    band_lo=0.60, band_hi=0.80,          # entry band (illiquidity percentile)
    keep_lo=0.50, keep_hi=0.85,          # hysteresis keep band
    n_target=50, min_book=15,            # book size / stand-aside floor
    trade_size_usd=2500.0,               # pre-registered $T (deployable slice)
    exp_premium_bps_ann=300.0,           # pre-registered expected GROSS premium (bps/yr)
    exp_hold_days=126,                   # expected hold from hysteresis design (trading days)
    beta_lb=60, beta_cap=1.0,            # hedge beta window / clip -> gross <= 2x
    cost_bps=8.0,                        # base cost on all turnover
)

GRID = {
    "default": {},
    "band_deeper":      {"band_lo": 0.70, "band_hi": 0.90, "keep_lo": 0.60, "keep_hi": 0.93},
    "T_10k":            {"trade_size_usd": 10000.0},
    "T_50k":            {"trade_size_usd": 50000.0},
    "unhedged_diag":    {"beta_cap": 0.0},            # long-only diagnostic (beta-residue check)
    "flat_cost_xcheck": {"flat_cost_xcheck": True},   # flat 15 bps, no own-impact term (sanity)
}

_SECTOR_MAPS = {}  # fallback cache: tuple(tickers) -> sector_map (in case .attrs are dropped)


# ----------------------------------------------------------------------------- data loading

def _sep_field(tickers, start, candidates):
    """Load the first SEP field the parquet store actually serves (column-projection-safe).
    Returns (field_name, DataFrame)."""
    last_err = None
    for f in candidates:
        try:
            df = sep_panel(tickers, start, field=f)
            if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                return f, df
        except Exception as e:                          # missing column -> try next candidate
            last_err = e
            continue
    raise RuntimeError(f"none of SEP fields {candidates} loadable: {last_err}")


def _load_hedge(start=START):
    """One liquid index ETF series. SEP first; yfinance acceptable ONLY here (single live ETF,
    survivorship-irrelevant)."""
    for tk in HEDGE_CANDIDATES:
        for loader in (lambda t: sep_panel([t], start, field="closeadj"),
                       lambda t: yf_panel([t], start)):
            try:
                px = loader(tk)
                if tk in px.columns:
                    s = px[tk].dropna()
                    if len(s) > 2000:
                        return tk, s
            except Exception:
                continue
    raise RuntimeError("no hedge ETF series loadable (tried IWM, SPY via SEP and yfinance)")


def _build_panel(tickers, sector_map, start=START):
    adj = sep_panel(tickers, start, field="closeadj")   # returns (split+div adjusted)
    # UNADJUSTED close for true dollar volume; fall back to closeadj only if the store has no
    # unadjusted price (then $vol is approximate around splits — acceptable degradation vs crash).
    _, raw = _sep_field(tickers, start, ("closeunadj", "close", "closeadj"))
    _, vol = _sep_field(tickers, start, ("volume",))    # raw share volume
    tks = [t for t in adj.columns if t in raw.columns and t in vol.columns]
    h_tk, h_px = _load_hedge(start)
    panel = pd.concat(
        {"adj": adj[tks], "raw": raw[tks], "vol": vol[tks],
         "hedge": pd.DataFrame({h_tk: h_px}).reindex(adj.index)},
        axis=1,
    )
    smap = {t: sector_map.get(t, "Unknown") for t in tks}
    panel.attrs["sector_map"] = smap
    panel.attrs["hedge_ticker"] = h_tk
    _SECTOR_MAPS[tuple(sorted(tks))] = smap
    return panel


def load_data():
    """Search universe: MID-cap, sector-spread (~600-700 names), delisted included."""
    tks, smap = sector_universe(marketcap="Mid", top_n_per_sector=60)
    return _build_panel(tks, smap)


def load_gen_data(label):
    """Generalization: SMALL-cap sector slices — disjoint cap tier vs search, disjoint sectors
    vs each other (~150-250 names each)."""
    if label not in GEN_SECTORS:
        raise KeyError(f"unknown generalization universe: {label}")
    want = set(GEN_SECTORS[label])
    tks, smap = sector_universe(marketcap="Small", top_n_per_sector=60)
    sel = [t for t in tks if smap.get(t) in want]
    return _build_panel(sel, smap)


# ----------------------------------------------------------------------------- selection helper

def _bucket_select(cands, illiq_d, adv_d, smap, n_slots):
    """Sector x size-tercile balanced pick (quotas replace the old long-short neutralization):
    round-robin across buckets, most-illiquid-first within each bucket."""
    if n_slots <= 0 or not cands:
        return []
    s_adv = adv_d.reindex(cands)
    try:
        terc = pd.qcut(s_adv.rank(method="first"), 3, labels=False, duplicates="drop")
    except ValueError:
        terc = pd.Series(0, index=cands)
    buckets = {}
    for t in cands:
        b = terc.get(t, 0)
        b = int(b) if pd.notna(b) else 0
        buckets.setdefault((smap.get(t, "Unknown"), b), []).append(t)
    for k in buckets:
        buckets[k].sort(key=lambda t: -float(illiq_d.get(t, 0.0)))
    picked, keys = [], sorted(buckets)
    while len(picked) < n_slots and any(buckets.values()):
        for k in keys:
            if buckets[k]:
                picked.append(buckets[k].pop(0))
                if len(picked) >= n_slots:
                    break
    return picked


# ----------------------------------------------------------------------------- signal

def signal(panel, **params):
    p = dict(DEFAULTS)
    p.update(params)

    adj, raw, vol = panel["adj"], panel["raw"], panel["vol"]
    hedge = panel["hedge"]
    h_tk = hedge.columns[0]
    smap = panel.attrs.get("sector_map") or _SECTOR_MAPS.get(tuple(sorted(adj.columns)), {})

    rets = adj.pct_change()
    h_ret = hedge[h_tk].pct_change()

    # --- trailing Amihud illiquidity (return per $ traded) and liquidity floors (all data <= t)
    dv = (raw * vol).replace(0.0, np.nan)                       # daily dollar volume, unadjusted
    illiq = (rets.abs() / dv).rolling(p["amihud_lb"], min_periods=p["min_obs"]).mean()
    adv = dv.rolling(63, min_periods=40).median()
    eligible = (raw >= p["price_floor"]) & (adv >= p["adv_floor"]) & illiq.notna()

    # --- SELF-CONSISTENT cost gate, AS PRE-REGISTERED: include a name only if its expected
    # premium NET of its own modeled impact stays positive — round-trip impact 2*Amihud*$T must
    # not exceed the expected gross premium accrued over the pre-registered expected hold.
    impact_rt_bps = 2.0 * illiq * p["trade_size_usd"] * 1e4
    premium_budget_bps = p["exp_premium_bps_ann"] * (p["exp_hold_days"] / 252.0)
    cost_ok = impact_rt_bps <= premium_budget_bps               # expected net premium > 0

    # cross-sectional illiquidity percentile among eligible names only
    ill_pct = illiq.where(eligible).rank(axis=1, pct=True)

    cols = list(adj.columns) + [h_tk]
    rb_dates = rets.index.to_series().groupby(rets.index.to_period("M")).max().values

    held, w_rows = [], {}
    for d in rb_dates:
        elig_d = eligible.loc[d] & cost_ok.loc[d]
        if int(elig_d.sum()) < 2 * p["min_book"]:               # warm-up / thin cross-section
            held = []
            w_rows[d] = pd.Series(0.0, index=cols)
            continue
        pct = ill_pct.loc[d]

        # hysteresis: holdings persist while inside the (wider) keep band and still cost-gated
        keep = set(pct.index[(pct >= p["keep_lo"]) & (pct <= p["keep_hi"]) & elig_d])
        held = [t for t in held if t in keep]

        # fill remaining slots from the core entry band, sector x size balanced
        entry = [t for t in pct.index[(pct >= p["band_lo"]) & (pct <= p["band_hi"]) & elig_d]
                 if t not in held]
        held = held + _bucket_select(entry, illiq.loc[d], adv.loc[d], smap,
                                     p["n_target"] - len(held))

        row = pd.Series(0.0, index=cols)
        if len(held) < p["min_book"]:                           # can't build a real book: stand aside
            held = []
            w_rows[d] = row
            continue

        # EQUAL-WEIGHTED long book (parent's frozen PRIMARY — unchanged), gross long = 1.0
        w = pd.Series(1.0 / len(held), index=held)

        # hedge: trailing-60d beta of the current long book vs the ETF (data through d only)
        win = rets.loc[:d, held].tail(p["beta_lb"]).fillna(0.0)
        book = (win * w.reindex(held)).sum(axis=1)
        hb = h_ret.reindex(book.index)
        var = hb.var()
        beta = float(book.cov(hb) / var) if (var and np.isfinite(var) and var > 0) else 0.7
        beta = float(np.clip(beta, 0.0, p["beta_cap"]))

        row.loc[held] = w.reindex(held).values
        row.loc[h_tk] = -beta                                   # gross <= 1.0 + beta_cap = 2x
        w_rows[d] = row

    W = pd.DataFrame(w_rows).T.sort_index()
    Wd = W.reindex(rets.index).ffill().fillna(0.0)

    R = pd.concat([rets, h_ret.rename(h_tk)], axis=1)[cols]
    WL = Wd.shift(1)   # weights decided at close t -> held/traded from t+1 (OUR lag responsibility)

    flat = bool(p.get("flat_cost_xcheck", False))
    base_bps = 15.0 if flat else p["cost_bps"]
    net = net_of_cost(WL, R, cost_bps=base_bps, name="amihud_illiq_lo_etfhedge")

    if not flat:
        # name-specific impact drag: one-way Amihud x $T (fractional) on every unit of stock turnover
        dW = WL[adj.columns].fillna(0.0).diff().abs()
        one_way = (illiq * p["trade_size_usd"]).shift(1).reindex(dW.index).fillna(0.0)
        net = (net - (dW * one_way).sum(axis=1)).rename("amihud_illiq_lo_etfhedge")

    trades = trades_from_weights(WL, R, {**smap, h_tk: HEDGE_SECTOR})
    return net.dropna(), trades


# ----------------------------------------------------------------------------- spec

SPEC = StrategySpec(
    id="amihud_illiq_longonly_etfhedge_v1",
    family="illiquidity",
    title=("Amihud illiquidity premium — cost-gated mid-illiquid band, LONG-ONLY equal-weight + "
           "single liquid ETF beta-hedge (borrow risk eliminated, $5K-deployable)"),
    markets=["US mid-cap equities (search)", "US small-cap sector slices (generalization)",
             "IWM/SPY (hedge only)"],
    data_desc=("Sharadar SEP closeadj + unadjusted close (closeunadj/close fallback chain) + "
               "volume, survivorship-clean (delisted incl.), 2000+; one liquid hedge ETF (IWM "
               "primary, SPY fallback; SEP else yfinance — single live ETF, survivorship-"
               "irrelevant). No DATA-GATED sources."),
    pre_registration=(
        "Liquidity risk premium (Amihud 2002): compensation for bearing illiquidity, structurally "
        "self-protected by limits to arbitrage — NOT a forecast. Frozen PRIMARY (parent config "
        "unchanged where not explicitly varied): trailing-252d Amihud, 60-80th illiquidity-"
        "percentile band (capacity-tradeable, not the extreme quintile), 50-85th keep-band "
        "hysteresis, monthly rebalance, sector x size-tercile balanced ~50-name EQUAL-WEIGHTED "
        "long book. SELF-CONSISTENT cost gate as proposed: include a name only if its expected "
        "premium net of its OWN modeled impact stays positive — round-trip 2*Amihud*$T <= "
        "expected gross premium over the expected hold (frozen: 300 bps/yr premium, 126-day hold "
        "-> 150 bps round-trip budget) at pre-registered $T=$2.5K; $10K/$50K declared in grid "
        "BEFORE any run. Net of base 8bps + own-impact drag Amihud*$T per unit of long-leg "
        "turnover. Hedge: short IWM at trailing-60d book beta clipped [0,1], gross <= 2x. The "
        "HEDGED series is judged; unhedged_diag must show the premium is the illiquid tilt, not "
        "small-cap beta (hedged alpha must carry the Sharpe). Expect: net premium positive in the "
        "band, monotonically weakening toward liquid (not reversing); graceful no-cliff "
        "degradation across $T; present across sectors; survives in small AND mid tiers with mid "
        "deployable. MCPT vs within-size permutation null + write-once holdout on the hedged net "
        "series. Standalone first; <=25% trend tail-overlay only after a holdout PASS."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid=GRID,
    scope="broad",
    generalization_universes=list(GEN_SECTORS),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)