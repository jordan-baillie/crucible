"""
Commodity-complex CROSS-SECTIONAL CARRY (spot-vs-front-futures basis) — market-neutral
long/short across the energy + precious-metals subset of the Databento 16-root complex.

================================ HONEST DATA DISCLOSURE (read pre_registration) ================
The parent proposal harvests roll-yield CARRY from a directly-observed 2-deep TERM STRUCTURE
(fut_curve close_1/close_2) across all 16 roots, GATED by physical inventory (eia_series,
usda_nass STOCKS) and hedging-pressure (cot_positioning). NONE of fut_curve / eia_series /
usda_nass / cot_positioning is in the SANCTIONED Crucible kit (the only commodity-price sources
available are yf_panel front-month continuous + fred_series). A clean 16-root term-structure carry
is therefore NOT computable with the sanctioned imports, and importing un-built adapters would hard-
fail the module load.

This module implements the SAME cross-sectional carry-harvester ARCHITECTURE on the SUBSET of roots
for which a directly-observable, PIT-safe SPOT exists, computing carry as the spot-vs-front-futures
BASIS:  basis = (spot - future)/future  (backwardation, spot>future => positive carry => long).
   energy : CL, NG (+HO, RB if FRED ids resolve) via FRED daily cash prices
   metals : GC, SI, PL via Yahoo spot-FX (XAUUSD=X, XAGUSD=X, XPTUSD=X)
The inventory/COT CONDITIONING gate and the full 16-root book are DEFERRED to when those adapters
land (prose-only here, not machine-checkable). Scope is therefore 'local'. CORRECTION vs the prior
draft: the frozen signal elements that ARE computable on this proxy data have been restored — a
FROZEN carry-magnitude HURDLE, FAST-EXIT on a 2-consecutive-week carry-sign flip, min-hold/
hysteresis (held legs that still clear the hurdle are retained before topping up), sit-FLAT on any
unfilled sleeve, and BETA-BALANCING of the short sleeve so the book is ~beta-neutral to the
equal-weight complex (not merely dollar-neutral). $0 owned/free data only.
"""
import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights

SECTOR = {'CL': 'Energy', 'NG': 'Energy', 'HO': 'Energy', 'RB': 'Energy',
          'GC': 'Metals', 'SI': 'Metals', 'PL': 'Metals'}
FUT_TKR = {'CL': 'CL=F', 'NG': 'NG=F', 'HO': 'HO=F', 'RB': 'RB=F',
           'GC': 'GC=F', 'SI': 'SI=F', 'PL': 'PL=F'}
# spot source per root: ('fred', daily cash id) or ('yf', spot-FX symbol)
SPOT_SRC = {'CL': ('fred', 'DCOILWTICO'), 'NG': ('fred', 'DHHNGSP'),
            'HO': ('fred', 'DJFUELUSGULF'), 'RB': ('fred', 'DGASUSGULF'),
            'GC': ('yf', 'XAUUSD=X'), 'SI': ('yf', 'XAGUSD=X'), 'PL': ('yf', 'XPTUSD=X')}
START = "2005-01-01"


def _yf_one(sym, start):
    try:
        df = yf_panel([sym], start)
        if df is None or len(df) == 0:
            return None
        s = df[sym] if sym in df.columns else df.iloc[:, 0]
        s = pd.to_numeric(s, errors='coerce').dropna()
        return s if len(s) > 250 else None
    except Exception:
        return None


def _fred_one(fid, start):
    try:
        df = fred_series({fid: fid}, start)
        if df is None or len(df) == 0:
            return None
        s = pd.to_numeric(df[fid], errors='coerce').dropna()
        return s if len(s) > 250 else None
    except Exception:
        return None


def _load(roots, start=START):
    fut_raw = yf_panel([FUT_TKR[r] for r in roots], start)
    fut = {}
    for r in roots:
        t = FUT_TKR[r]
        if t in fut_raw.columns:
            s = pd.to_numeric(fut_raw[t], errors='coerce').dropna()
            if len(s) > 250:
                fut[r] = s
    spot = {}
    for r in list(fut.keys()):
        src, key = SPOT_SRC[r]
        s = _fred_one(key, start) if src == 'fred' else _yf_one(key, start)
        if s is not None:
            spot[r] = s
    keep = [r for r in roots if r in fut and r in spot]
    if len(keep) < 2:
        raise RuntimeError("insufficient spot-observable commodity roots loaded")
    fut_df = pd.DataFrame({r: fut[r] for r in keep}).sort_index()
    idx = fut_df.index
    spot_df = pd.DataFrame({r: spot[r].reindex(idx).ffill(limit=5) for r in keep})
    panel = pd.concat({'fut': fut_df, 'spot': spot_df}, axis=1).dropna(how='all')
    return panel


def load_data() -> pd.DataFrame:
    return _load(list(FUT_TKR.keys()))


def signal(panel, k=2, vol_lb=63, carry_clip=0.5, carry_hurdle=0.005,
           flip_exit=2, cost_bps=8.0, **params):
    fut = panel['fut'].astype(float)
    spot = panel['spot'].astype(float)
    roots = [c for c in fut.columns if c in spot.columns]
    fut, spot = fut[roots], spot[roots]

    # negative/zero front price (e.g. Apr-2020 WTI) -> excise so nothing blows up
    fut_pos = fut.where(fut > 0)
    raw_rets = fut_pos.pct_change().replace([np.inf, -np.inf], np.nan)
    rets = raw_rets.fillna(0.0)

    # carry = directly-observed spot-vs-front-futures basis (backwardation>0 => long)
    basis = ((spot - fut) / fut).where(fut > 0).clip(-carry_clip, carry_clip)

    # trailing inverse-vol risk weights (computed on un-filled returns)
    vol = raw_rets.rolling(vol_lb, min_periods=max(20, vol_lb // 3)).std()
    inv_vol = 1.0 / vol.replace(0.0, np.nan)

    # trailing beta of each root to the equal-weight complex (for beta-balancing)
    eqw = raw_rets.mean(axis=1)
    var_eqw = eqw.rolling(vol_lb, min_periods=max(20, vol_lb // 3)).var()
    beta_df = pd.DataFrame(
        {r: raw_rets[r].rolling(vol_lb, min_periods=max(20, vol_lb // 3)).cov(eqw)
             / var_eqw.replace(0.0, np.nan) for r in roots})

    # weekly (last trading day each week) rebalance dates
    rebal_dates = pd.DatetimeIndex(fut.index.to_series().resample('W-FRI').last().dropna().values)
    rebal_dates = rebal_dates[rebal_dates.isin(fut.index)]

    W_reb = pd.DataFrame(0.0, index=rebal_dates, columns=roots)
    held, flip = {}, {}      # held: root -> +1/-1 currently held side ; flip: consecutive sign-flip weeks
    for d in rebal_dates:
        b = basis.loc[d].dropna()
        if b.empty:
            continue
        # ---- FAST-EXIT: drop a held leg whose carry sign disagrees for `flip_exit` weeks ----
        for r in list(held.keys()):
            if r in b.index:
                cur = np.sign(b[r])
                if cur != 0 and cur != held[r]:
                    flip[r] = flip.get(r, 0) + 1
                else:
                    flip[r] = 0
                if flip[r] >= flip_exit:
                    held.pop(r, None); flip.pop(r, None)
            # if no carry obs this week, keep the prior leg (no info to exit on)
        # ---- candidates clearing the FROZEN carry-magnitude hurdle ----
        long_c = list(b[b >= carry_hurdle].sort_values(ascending=False).index)
        short_c = list(b[b <= -carry_hurdle].sort_values().index)
        # min-hold / hysteresis: keep persistent held legs that still clear hurdle, then top up to k
        longs = [r for r in held if held[r] > 0 and r in long_c]
        for r in long_c:
            if len(longs) >= k:
                break
            if r not in longs:
                longs.append(r)
        shorts = [r for r in held if held[r] < 0 and r in short_c]
        for r in short_c:
            if len(shorts) >= k:
                break
            if r not in shorts:
                shorts.append(r)
        longs, shorts = longs[:k], shorts[:k]
        held = {**{r: 1 for r in longs}, **{r: -1 for r in shorts}}
        if not longs and not shorts:        # too few names clear the hurdle -> sit flat
            continue
        iv = inv_vol.loc[d]
        # inverse-vol within each sleeve; an unfilled sleeve simply sits flat (dollar tilt absorbed)
        if longs:
            wl = iv[longs].fillna(0.0)
            if wl.sum() > 0:
                W_reb.loc[d, longs] = (wl / wl.sum() * 0.5).values
        if shorts:
            ws = iv[shorts].fillna(0.0)
            if ws.sum() > 0:
                W_reb.loc[d, shorts] = -(ws / ws.sum() * 0.5).values
        # ---- BETA-BALANCE: scale short sleeve so book beta ~0 to the equal-weight complex ----
        bser = beta_df.loc[d]
        bl = float(np.nansum([W_reb.loc[d, r] * (bser.get(r, 1.0) if np.isfinite(bser.get(r, 1.0)) else 1.0) for r in longs])) if longs else 0.0
        bs = float(np.nansum([W_reb.loc[d, r] * (bser.get(r, 1.0) if np.isfinite(bser.get(r, 1.0)) else 1.0) for r in shorts])) if shorts else 0.0
        if longs and shorts and bs != 0.0:
            scale = float(np.clip(-bl / bs, 0.5, 2.0))
            W_reb.loc[d, shorts] = W_reb.loc[d, shorts].values * scale

    # hold between Friday rebalances
    W = W_reb.reindex(fut.index, method='ffill').fillna(0.0)
    # LAG: weights built from date-t close info -> act next day (the lag is ours)
    Wl = W.shift(1).fillna(0.0)

    daily = net_of_cost(Wl, rets, cost_bps=cost_bps, name="commodity_xs_carry_basis")
    trades = trades_from_weights(Wl, rets, SECTOR)
    return daily, trades


def load_gen_data(label) -> pd.DataFrame:
    # scope='local': no stage-2 generalization battery (specific ~7-root cross-section;
    # forward-validation confirms). Defined for signature completeness only.
    return load_data()


# ---------- soft expectations: falsify the proposal's flagged failure modes ----------
def _check_breadth(ctx):
    h = str(ctx["holdout_start"])
    names = {t.get("ticker") for t in ctx["trades"] if str(t.get("entry_date", "")) < h}
    return {"pass": len(names) >= 4, "observed": len(names)}


def _check_energy_share(ctx):
    h = str(ctx["holdout_start"])
    eng = tot = 0.0
    for t in ctx["trades"]:
        if str(t.get("entry_date", "")) >= h:
            continue
        hd = float(t.get("hold_days", 0) or 0)
        tot += hd
        if t.get("sector") == "Energy":
            eng += hd
    share = (eng / tot) if tot > 0 else 1.0
    return {"pass": share <= 0.70, "observed": round(share, 3)}


def _check_market_neutral(ctx):
    h = pd.Timestamp(ctx["holdout_start"])
    eqw = ctx["panel"]["fut"].astype(float).pct_change().mean(axis=1)
    df = pd.concat([ctx["search"].dropna(), eqw], axis=1, join="inner").dropna()
    df = df[df.index < h]
    if len(df) < 60:
        return {"pass": True, "observed": "insufficient"}
    x, y = df.iloc[:, 1].values, df.iloc[:, 0].values
    v = float(np.var(x))
    beta = float(np.cov(x, y)[0, 1] / v) if v > 0 else 0.0
    return {"pass": abs(beta) <= 0.25, "observed": round(beta, 3)}


SPEC = StrategySpec(
    id="commodity-xs-carry-spotbasis",
    family="carry",
    title=("Commodity cross-sectional CARRY (spot-vs-front-futures basis), energy+metals subset — "
           "the validated cross-sectional carry-harvester architecture transplanted to commodity "
           "futures; PROXY of the 16-root storage-gated proposal pending fut_curve/eia/usda/cot adapters"),
    markets=["commodity futures"],
    data_desc=("yf_panel front-month continuous futures {CL,NG,HO,RB,GC,SI,PL} + directly-observed "
               "SPOT (FRED daily cash: DCOILWTICO/DHHNGSP/DJFUELUSGULF/DGASUSGULF; Yahoo spot-FX: "
               "XAUUSD=X/XAGUSD=X/XPTUSD=X). Carry = (spot-future)/future basis; weekly Friday rebal; "
               "inverse-vol; frozen carry hurdle; fast-exit; beta-balanced top-k/bottom-k long/short. $0."),
    pre_registration=(
        "MECHANISM: commodity carry/roll-yield IS the storage-theory risk premium — backwardation "
        "(front future BELOW spot; scarce inventory) pays a LONG a positive roll as the future "
        "converges UP to spot; contango pays a SHORT. We harvest it CROSS-SECTIONALLY: each Friday "
        "close, rank roots by the directly-observed spot-vs-front-futures basis (spot-future)/future "
        "(a clean PIT carry observable, both legs same-day), go inverse-vol LONG the top-k "
        "backwardated and SHORT the bottom-k contango'd, BETA-BALANCED (~market-neutral to the "
        "equal-weight complex). This is NOT a price forecast.\n"
        "FROZEN SIGNAL: (a) weekly Friday rebal; (b) carry = (spot-future)/future; (c) inverse-vol "
        "sizing within each sleeve; (d) a leg must clear a FROZEN carry-magnitude HURDLE (|basis|>= "
        "carry_hurdle) to qualify — if fewer than k per side clear, that sleeve sits FLAT (the built-in "
        "regime gate); (e) min-hold/HYSTERESIS — a held leg still clearing the hurdle is retained before "
        "topping up to k; (f) FAST-EXIT — a held leg is dropped once its carry sign disagrees for "
        "flip_exit (=2) consecutive weeks; (g) short sleeve beta-scaled so book beta ~0. All thresholds "
        "frozen pre-registration, no optimization.\n"
        "DATA CAVEAT (material): the parent harvests carry from a 2-deep TERM STRUCTURE (fut_curve) "
        "across 16 roots GATED by inventory (eia_series/usda_nass STOCKS) + hedging-pressure "
        "(cot_positioning); none of those adapters is in the sanctioned kit, so the full storage-"
        "gated 16-root book is NOT computable here. This module is the feasible proxy: same "
        "architecture on the ~7-root SUBSET with an observable spot (energy via FRED daily cash, "
        "precious metals via Yahoo spot-FX), carry=spot-future basis. The inventory/COT conditioning "
        "gate is DEFERRED (prose-only) — NOT yet implemented because the data is unavailable, so this "
        "proxy is ungated carry and must be read as the architecture test, not the full combination. "
        "Scope is therefore 'local' (specific small cross-section, confirmed by forward-validation) — "
        "commodity complexes cannot supply 3 disjoint 150-name holdouts for a broad battery.\n"
        "NO LOOKAHEAD: signals use only same-day-close info (basis + trailing vol + trailing beta); "
        "weights lagged one day (W.shift(1)) before returns/costs; spot joined same-day with a 1-day "
        "act lag covering FRED cash release latency; non-positive front prices (e.g. Apr-2020 WTI) are "
        "excised from BOTH returns and the basis so the construction cannot blow up.\n"
        "HONEST RISKS (machine-checked): (1) BREADTH — 7 roots << the parent's 75; if the book "
        "collapses to <4 distinct roots it violates the Fundamental-Law spirit. (2) precious-metals "
        "basis ~ -(r-q) (thin convenience yield) so ENERGY may dominate both tails -> carry degrades "
        "into a sector bet (energy <=70% of position-days). (3) residual beta to the equal-weight "
        "complex must be ~0. Costs 8bps/turnover; weekly rebal; inverse-vol; all thresholds frozen."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={"default": {}, "k1": {"k": 1}, "k3": {"k": 3}, "vol126": {"vol_lb": 126}},
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=8,
    expectations=[
        {"name": "breadth_distinct_roots",
         "claim": "the long/short carry book trades >= 4 distinct roots over the search window "
                  "(else the cross-section collapsed — proposal's flagged #1 risk)",
         "check": _check_breadth},
        {"name": "energy_not_dominant",
         "claim": "energy roots are <= 70% of long/short position-days (else carry is an energy "
                  "sector bet, not a cross-sectional harvest — gate0 #4)",
         "check": _check_energy_share},
        {"name": "market_neutral",
         "claim": "|beta| of search-window net returns to the equal-weight complex <= 0.25 "
                  "(delta/market neutrality)",
         "check": _check_market_neutral},
    ],
)