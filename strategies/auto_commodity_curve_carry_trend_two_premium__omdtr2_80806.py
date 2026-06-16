"""
commod_curvecarry_x_trend_v1
=============================
Two-premium commodity book on the OWNED native futures-curve complex.

PRIMARY (pre-registered, the VERDICT): a DAILY cross-sectional convenience-yield /
curve-carry book. Per root, annualized roll-yield = log(close_1/close_2) * 365/days_to_roll_1
from fut_curve(root, n_contracts=2). Daily returns are computed WITHIN a contract month
(roll days zeroed, never diffed across a roll). Rank the cross-section; LONG the most
backwardated tercile, SHORT the most contango tercile; inverse-vol equal-risk legs;
vol-target 10% ann; weekly rebalance; ~8 bps micro-futures cost. PA dropped (thin rank-2).

STORAGE TILT (SIZING ONLY, never a trigger -> trade count stays high): scale a leg up when
the storage state CONFIRMS its curve (low inventory supports backwardation/long; high
inventory supports contango/short), using FRED-mirrored EIA petroleum stocks (PIT, release-
lagged ~7 bdays). Sign is always preserved (tilt in [0.5,1.5]); degrades to NEUTRAL where no
storage series exists (all non-energy roots and every generalization universe). NOTE: native
EIA/USDA adapters are key-pending (CEO journal 2026-06-15), so the tilt is sourced from the
sanctioned fred_series and is robust-by-omission.

TREND OVERLAY (deployable variant ONLY, obeying the 2026-06-08 anti-blend lesson): a small
~22% risk-budget canonical 12-1 TSMOM sleeve on the same complex as a crisis-alpha tail hedge.
default_params trade CARRY STANDALONE (trend_weight=0). The 'combined' grid variant is the
deployable book. A soft expectation falsifies the overlay if it dilutes Sharpe or fails to cut DD.

SCOPE = 'broad'. Carry is a universal convenience-yield mechanism, so it must generalize.
Search universe = ENERGY+METALS (dense rank-2). Generalization universes = GRAINS / SOFTS /
LIVESTOCK -> strictly DISJOINT sub-complexes (share no roots with search or each other), per
the broad-scope contract. >=60% (2 of 3) must be OOS-positive on holdout or the candidate is
rejected as an overfit one-complex outlier.

NO external side effects. OWNED/FREE data only.
"""

from sdk.harness import StrategySpec
from sdk.adapters import fut_curve, fred_series, inv_vol_position
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# --------------------------------------------------------------------------------------
# Universe: commodity futures roots (Databento owned complex). "Sector" = commodity group
# (gives the trade ledger its sector-spread; deployment-sanity needs >1 group).
# --------------------------------------------------------------------------------------
SECTOR_MAP = {
    "CL": "Energy", "HO": "Energy", "RB": "Energy", "NG": "Energy", "BZ": "Energy",
    "GC": "Metals", "SI": "Metals", "HG": "Metals", "PL": "Metals",
    "ZC": "Grains", "ZW": "Grains", "ZS": "Grains", "ZL": "Grains", "ZM": "Grains",
    "KC": "Softs",  "SB": "Softs",  "CC": "Softs",  "CT": "Softs",
    "LE": "Livestock", "HE": "Livestock", "GF": "Livestock",
}

# Search universe = energy + metals (dense rank-2; PA palladium dropped per gate-0 thinness).
SEARCH_ROOTS = ["CL", "HO", "RB", "NG", "BZ", "GC", "SI", "HG", "PL"]

# Generalization universes: disjoint from search AND from each other (broad-scope contract).
GEN_UNIVERSES = {
    "grains":    ["ZC", "ZW", "ZS", "ZL", "ZM"],
    "softs":     ["KC", "SB", "CC", "CT"],
    "livestock": ["LE", "HE", "GF"],
}

# FRED-mirrored EIA weekly petroleum stocks for the storage-confirmation tilt (energy only).
# Wrapped in try/except: a wrong/missing id just degrades that root to a NEUTRAL tilt.
STORAGE_FRED = {
    "CL": "WCESTUS1",   # crude oil ending stocks ex-SPR (Thousand Barrels), weekly ~Wed
    "RB": "WGTSTUS1",   # total gasoline stocks
    "HO": "WDISTUS1",   # distillate fuel oil stocks
}

_FRED_START = "2000-01-01"


# --------------------------------------------------------------------------------------
# Data helpers (all the fut_curve-schema parsing lives here; signal() stays schema-free)
# --------------------------------------------------------------------------------------
def _extract_curve(cur):
    """Pull (front_close, second_close, days_to_roll_front) from a fut_curve frame, defensively."""
    cols = list(cur.columns)
    lc = {str(c).lower(): c for c in cols}

    def pick(cands):
        for k in cands:
            if k in lc:
                return cur[lc[k]]
        return None

    c1 = pick(["close_1", "c1", "front", "px_1", "settle_1", "f1", "close1", "price_1", "p1", "near"])
    c2 = pick(["close_2", "c2", "second", "px_2", "settle_2", "f2", "close2", "price_2", "p2", "deferred", "next"])
    dte = pick(["days_to_roll_1", "dte_1", "dte1", "dte", "days_to_expiry_1", "dtr_1",
                "days_to_roll", "days_to_expiry", "ttm_1", "expiry_days_1"])

    # fallback: take the first two price-like columns / first dte-like column
    if c1 is None or c2 is None:
        plike = sorted([c for c in cols if any(t in str(c).lower()
                        for t in ("close", "settle", "price", "px"))], key=lambda x: str(x))
        if c1 is None and len(plike) >= 1:
            c1 = cur[plike[0]]
        if c2 is None and len(plike) >= 2:
            c2 = cur[plike[1]]
    if dte is None:
        dlike = [c for c in cols if any(t in str(c).lower() for t in ("dte", "days", "expir", "roll", "ttm"))]
        if dlike:
            dte = cur[dlike[0]]
    return c1, c2, dte


def _storage_z(root, dates):
    """PIT storage state z-score vs trailing 3y; release-lagged 7 bdays. None if unavailable."""
    fid = STORAGE_FRED.get(root)
    if not fid:
        return None
    try:
        s = fred_series({fid: "v"}, _FRED_START)["v"].astype(float).dropna()
        if s.empty:
            return None
    except Exception:
        return None
    idx = pd.DatetimeIndex(dates)
    s.index = pd.to_datetime(s.index)
    s = s.reindex(idx.union(s.index)).sort_index().ffill().reindex(idx)
    s = s.shift(7)  # PIT buffer: weekly stocks (week-ending) are released ~next Wed -> no look-ahead
    mu = s.rolling(756, min_periods=120).mean()
    sd = s.rolling(756, min_periods=120).std()
    z = (s - mu) / sd
    return z.clip(-3.0, 3.0)


def _build_panel(roots):
    """Wide panel with MultiIndex columns ('ret'|'carry'|'stor_z', root). Used by load_data + load_gen_data."""
    rets, carries, stors = {}, {}, {}
    for r in roots:
        try:
            cur = fut_curve(r, n_contracts=2)
        except Exception:
            continue
        if cur is None or len(cur) == 0:
            continue
        cur = cur.copy()
        cur.index = pd.to_datetime(cur.index)
        cur = cur.sort_index()
        c1, c2, dte = _extract_curve(cur)
        if c1 is None or c2 is None:
            continue
        c1 = pd.to_numeric(c1, errors="coerce")
        c2 = pd.to_numeric(c2, errors="coerce")

        # within-contract daily return: NEVER diff across a roll
        if dte is not None:
            dte = pd.to_numeric(dte, errors="coerce")
            roll = dte.diff() > 0          # days-to-roll jumped UP -> new front contract
            gap = dte.clip(lower=21.0)     # floor to tame near-expiry annualization blow-ups
        else:                              # crude fallback if the curve frame carries no dte
            roll = c1.pct_change().abs() > 0.15
            gap = pd.Series(30.0, index=c1.index)
        rr = c1.pct_change().where(~roll).fillna(0.0)

        # annualized roll-yield: backwardation (c1>c2) -> positive carry -> LONG
        cy = np.log(c1 / c2) * (365.0 / gap)
        cy = cy.replace([np.inf, -np.inf], np.nan)

        rets[r] = rr
        carries[r] = cy
        stors[r] = _storage_z(r, c1.index)

    if len(rets) < 2:
        raise RuntimeError("commod_curvecarry: insufficient curve coverage for a cross-section")

    ret_df = pd.DataFrame(rets).sort_index()
    ret_df.index = pd.to_datetime(ret_df.index)
    cols = ret_df.columns
    carry_df = pd.DataFrame(carries).reindex(index=ret_df.index, columns=cols)
    stor_df = pd.DataFrame(index=ret_df.index, columns=cols, dtype=float)
    for r in cols:
        z = stors.get(r)
        if z is not None:
            stor_df[r] = pd.Series(z).reindex(ret_df.index)

    panel = pd.concat({"ret": ret_df, "carry": carry_df, "stor_z": stor_df}, axis=1)
    panel.index.name = "date"
    return panel.sort_index()


# --------------------------------------------------------------------------------------
# Signal helpers (the ONLY novel code: carry rank, storage sizing-tilt, trend overlay)
# --------------------------------------------------------------------------------------
def _tercile_signal(cz, frac):
    """+1 to the top-`frac` (most backwardated), -1 to the bottom-`frac` (most contango)."""
    out = pd.DataFrame(0.0, index=cz.index, columns=cz.columns)
    for dt, row in cz.iterrows():
        v = row.dropna()
        n = len(v)
        if n < 2:
            continue
        k = max(1, int(np.floor(n * frac)))
        out.loc[dt, v.nlargest(k).index] = 1.0
        out.loc[dt, v.nsmallest(k).index] = -1.0
    return out


def _storage_tilt(sig, stor, strength):
    """Sizing-only multiplier in [0.5,1.5]; >1 when inventory CONFIRMS the position's curve."""
    sgn = np.sign(sig)
    sz = np.tanh(stor.reindex_like(sig).astype(float))      # bounded storage state
    confirm = -sgn * sz                                     # long+low-inv or short+high-inv -> positive
    tilt = (1.0 + strength * confirm).clip(lower=0.5, upper=1.5)
    return tilt.fillna(1.0)                                 # no storage data -> NEUTRAL (sign preserved)


def _trend_signal(ret, lb, skip):
    """Canonical 12-1 TSMOM: sign of cumulative return over [t-lb, t-skip]."""
    lr = np.log1p(ret.fillna(0.0))
    mom = lr.rolling(max(lb - skip, 5)).sum().shift(skip)
    return np.sign(mom).fillna(0.0)


DEFAULT_PARAMS = dict(
    tercile=1.0 / 3.0, target_vol=0.10, vol_lb=63, max_pos=20, cost_bps=8.0,
    storage_tilt=True, tilt_strength=0.5,
    trend_weight=0.0,                 # PRIMARY/VERDICT = carry STANDALONE
    trend_lb=252, trend_skip=21,
    name="commod_curvecarry_x_trend",
)


def load_data():
    return _build_panel(SEARCH_ROOTS)


def load_gen_data(label):
    return _build_panel(GEN_UNIVERSES[label])


def signal(panel, **params):
    p = {**DEFAULT_PARAMS, **params}
    ret = panel["ret"].copy()
    carry = panel["carry"].copy()
    try:
        stor = panel["stor_z"].copy()
    except Exception:
        stor = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)

    # cross-sectional carry rank -> long backwardated / short contango terciles
    cz = xs_zscore(carry, winsor=(0.05, 0.95))
    sig = _tercile_signal(cz, p["tercile"])

    # storage-confirmation SIZING tilt (sign always preserved -> trade count unchanged)
    sig_carry = sig * _storage_tilt(sig, stor, p["tilt_strength"]) if p["storage_tilt"] else sig

    # inverse-vol, vol-targeted 10% ann, weekly rebalance. inv_vol_position returns ALREADY-LAGGED
    # positions -> do NOT shift again before net_of_cost / trades_from_weights.
    W_carry = inv_vol_position(sig_carry, ret, target_vol=p["target_vol"],
                               vol_lb=p["vol_lb"], max_pos=p["max_pos"], rebalance="W")

    if p["trend_weight"] > 0:
        tr = _trend_signal(ret, p["trend_lb"], p["trend_skip"])
        W_trend = inv_vol_position(tr, ret, target_vol=p["target_vol"],
                                   vol_lb=p["vol_lb"], max_pos=p["max_pos"], rebalance="W")
        tw = p["trend_weight"]
        W = W_carry.mul(1.0 - tw).add(W_trend.mul(tw), fill_value=0.0)   # ~22% risk to the trend tail
    else:
        W = W_carry

    daily = net_of_cost(W, ret, cost_bps=p["cost_bps"], name=p["name"])   # W already lagged
    trades = trades_from_weights(W, ret, SECTOR_MAP)                       # kit stamps entry_regime
    return daily, trades


# --------------------------------------------------------------------------------------
# Soft expectations (machine-checkable mechanism claims; recorded, do not block gates)
# --------------------------------------------------------------------------------------
def _sharpe(r):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 5 and r.std() > 0) else 0.0


def _maxdd(r):
    r = pd.Series(r).dropna()
    if r.empty:
        return 0.0
    c = (1.0 + r).cumprod()
    return float((c / c.cummax() - 1.0).min())


def _exp_carry_sharpe(ctx):
    """Pre-condition for the overlay: carry STANDALONE search Sharpe > 0.3."""
    try:
        sh = _sharpe(ctx.get("search"))
        return {"pass": bool(sh > 0.3), "observed": round(sh, 3)}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


def _exp_trend_helps(ctx):
    """Anti-blend lesson: combined book must CUT drawdown WITHOUT diluting Sharpe (>=90%)."""
    try:
        g = ctx.get("grid", {}) or {}
        if "combined" not in g or "default" not in g:
            return {"pass": False, "observed": "missing default/combined grid variants"}
        s0, s1 = _sharpe(g["default"]), _sharpe(g["combined"])
        d0, d1 = _maxdd(g["default"]), _maxdd(g["combined"])
        ok = (d1 > d0) and (s1 >= 0.9 * s0)   # d* are negative; d1>d0 == shallower DD
        return {"pass": bool(ok), "observed": f"dd {d0:.3f}->{d1:.3f}; sharpe {s0:.2f}->{s1:.2f}"}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


def _exp_storage_sizing(ctx):
    """Storage tilt is SIZING ONLY -> trade count must be ~unchanged vs tilt-off (<10% drift)."""
    try:
        panel, hs = ctx.get("panel"), ctx.get("holdout_start")
        if panel is None or hs is None:
            return {"pass": False, "observed": "no panel/holdout in ctx"}
        pre = panel.loc[panel.index < pd.Timestamp(hs)]
        n_on = len(ctx.get("trades") or [])
        _, tr_off = signal(pre, storage_tilt=False)           # single extra signal() call, search slice
        n_off = len(tr_off)
        if n_off == 0 or n_on == 0:
            return {"pass": False, "observed": f"on={n_on} off={n_off}"}
        rel = abs(n_on - n_off) / n_off
        return {"pass": bool(rel < 0.10), "observed": f"on={n_on} off={n_off} rel={rel:.1%}"}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


SPEC = StrategySpec(
    id="commod_curvecarry_x_trend_v1",
    family="commodity_carry",
    title="Commodity curve-carry x trend two-premium book - native daily roll-yield cross-section "
          "(storage-confirmation sizing tilt) + small commodity-trend crisis-alpha overlay",
    markets=["commodity_futures"],
    data_desc="OWNED Databento commodity-futures curve via fut_curve(root, n_contracts=2): native "
              "within-contract daily roll-yield (roll days zeroed, never diffed across a roll). "
              "FRED-mirrored EIA weekly petroleum stocks via fred_series for the PIT (release-lagged 7bd) "
              "storage-confirmation SIZING tilt; degrades to neutral where unavailable.",
    pre_registration=(
        "PRIMARY (verdict): daily cross-sectional convenience-yield curve-carry. carry = "
        "log(close_1/close_2)*365/days_to_roll_1; LONG top-tercile (backwardated), SHORT bottom-tercile "
        "(contango); inverse-vol equal-risk legs; vol-target 10% ann; weekly rebalance; ~8bps cost; PA dropped "
        "(thin rank-2). Returns are WITHIN-contract only. STORAGE TILT is SIZING ONLY (in [0.5,1.5], sign always "
        "preserved so trade count stays high): up-size when inventory confirms the curve (low EIA stocks support "
        "backwardation/long; high support contango/short), PIT on release date; neutral where no series exists. "
        "TREND OVERLAY (deployable variant ONLY): ~22% risk-budget 12-1 TSMOM crisis-alpha tail on the same complex, "
        "added because carry+trend are complementary (opposite tails) - NOT a reflexive 50/50; default_params trade "
        "carry STANDALONE so a 0-Sharpe carry leg cannot manufacture a false-fail. SCOPE=broad: carry is a universal "
        "convenience-yield mechanism, so it must generalize. Search universe = ENERGY+METALS; generalization universes "
        "= GRAINS / SOFTS / LIVESTOCK (strictly DISJOINT sub-complexes, no shared roots) - a real premium shows broad "
        "(even if weak) positivity, an overfit one shows one lucky complex; >=60% (2/3) OOS-positive required. Must "
        "clear MCPT (long-short, absolute null) before conviction; then forward-validate the combined book in paper. "
        "DATA NOTE: native EIA/USDA adapters are key-pending, so the storage tilt is sourced from sanctioned "
        "fred_series (energy roots) and is robust-by-omission; USDA grain-stocks confirmation is intentionally left "
        "as neutral pending the keyed adapter. Soft expectations falsify the overlay/storage mechanism stories."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},                              # carry STANDALONE (primary)
        "combined": {"trend_weight": 0.22},         # deployable two-premium book
        "no_storage_tilt": {"storage_tilt": False},
        "tercile_25": {"tercile": 0.25},
    },
    scope="broad",
    generalization_universes=list(GEN_UNIVERSES.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=10,
    expectations=[
        {"name": "carry_standalone_sharpe",
         "claim": "carry-standalone search Sharpe > 0.3 (pre-condition for adding the trend tail)",
         "check": _exp_carry_sharpe},
        {"name": "trend_overlay_cuts_tail_not_sharpe",
         "claim": "combined (trend_weight=0.22) has shallower max-drawdown than carry-standalone AND "
                  "Sharpe >= 90% of carry-standalone (anti-blend)",
         "check": _exp_trend_helps},
        {"name": "storage_tilt_is_sizing_only",
         "claim": "trade count with storage tilt is within 10% of tilt-off (tilt sizes, never triggers)",
         "check": _exp_storage_sizing},
    ],
)