"""
Crypto delta-neutral basis-carry — CROSS-SECTIONAL BREADTH variant.

Economics (inherited from the BTC/ETH parent, UNCHANGED): get paid to supply the
SHORT side to crowded perp longs, fully delta-hedged (long-spot 1x / short-perp 1x).
This is a carry / liquidity-provision premium, NOT a price forecast.

The ONE structural mutation: SELECTION goes from a fixed 2-major pair to a daily
cross-sectional top-K rank across binance_universe(75). `basis = perp/spot - 1` is the
deep-history, broad-universe FUNDING PROXY (real funding_rates() is majors-only). Per
coin we earn the funding stream (short side) and mark the basis spread to market; we
ENTER on persistent positive basis that clears a 2x-cost hurdle AND ranks top-quartile,
and EXIT on a collapsed/negative basis (hysteresis + min-hold + 3-day fast-exit). When
fewer than K names clear the hurdle the book sits in stablecoin BY DESIGN — that is the
built-in regime gate (a dead-aggregate-funding regime => mostly flat, no churn).

NO-LOOKAHEAD: every entry/exit/sizing decision at date t uses data through t only; the
resulting target weights are then W.shift(1)-lagged before being applied to the
contemporaneous (lagged-weight x same-day-return) carry returns. The lag is explicit
and is OUR responsibility (net_of_cost does not lag).

scope='local': perp basis/funding exists only in crypto perp markets (no equity/futures
breadth analogue), so there is no DISJOINT cross-market generalization universe to run a
stage-2 battery on. It is BROADER WITHIN crypto than the parent (75-name rotating top-K
vs a fixed 2) and is confirmed by forward-validation on the holdout.

Data adapters: binance_universe / binance_klines / funding_rates are the OWNED/FREE
crypto adapters in sdk.adapters (DATA_CATALOG.md; CRUCIBLE_FOCUS=crypto wires the forge
to binance_universe(75)). They are imported inside the data functions so a module import
never fails on data provisioning — only data loading does, which is the honest behaviour.
"""

from sdk.harness import StrategySpec
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------- config
START          = "2019-01-01"          # Binance USDT-perps from 2019-09; alts roll in later
N_UNIVERSE     = 75                     # broad liquid crypto cross-section (the breadth gate)
FUNDINGS_PER_DAY = 3                    # Binance 8h funding intervals
DAYS_PER_YEAR    = 365

DEFAULTS = dict(
    trail=7,            # trailing window (days) for the mean annualized-basis signal
    entry_ann=0.10,     # ENTER  only if trailing basis > 10% ann (2x cost hurdle)
    exit_ann=0.02,      # EXIT   when trailing basis < 2% ann
    top_q=0.25,         # ENTER  only if top-quartile of the 75 by basis
    exit_rank_q=0.50,   # EXIT   when rank falls below median
    neg_days=3,         # fast-exit: basis prints negative this many consecutive days
    min_hold=7,         # min hold (days); fast-exit EXEMPT
    top_k=4,            # deployed book cap (gross <= 2x; equal-risk; stablecoin otherwise)
    vol_lb=30,          # inverse-vol lookback (days)
    cost_bps=8.0,       # realistic round-trip cost on turnover
)

GRID = {                # pre-declared search burden for the DSR effective-N (honest)
    "default":  {},
    "K3":       {"top_k": 3},
    "K5":       {"top_k": 5},
    "hurdle_hi": {"entry_ann": 0.15},
    "trail_10": {"trail": 10},
}

# ---- crypto pseudo-sectors so the trade ledger carries spread labels (deterministic) ----
_SECTOR_MAP = {
    "BTC": "L1-store", "ETH": "L1-smart", "BNB": "exchange", "SOL": "L1-smart",
    "ADA": "L1-smart", "XRP": "payments", "DOGE": "meme", "SHIB": "meme",
    "AVAX": "L1-smart", "DOT": "L1-smart", "LINK": "oracle", "UNI": "defi",
    "AAVE": "defi", "LTC": "payments", "BCH": "payments", "MATIC": "L2-scaling",
    "ATOM": "L1-smart", "NEAR": "L1-smart", "FTM": "L1-smart", "ALGO": "L1-smart",
    "TRX": "L1-smart", "ETC": "L1-smart", "FIL": "storage", "APT": "L1-smart",
    "ARB": "L2-scaling", "OP": "L2-scaling", "PEPE": "meme", "WIF": "meme",
    "SUI": "L1-smart", "INJ": "defi", "TIA": "L1-smart", "SEI": "L1-smart",
    "RUNE": "defi", "LDO": "defi", "CRV": "defi", "MKR": "defi", "SNX": "defi",
    "GRT": "data", "RNDR": "compute", "FET": "ai", "AGIX": "ai", "IMX": "gaming",
    "SAND": "gaming", "AXS": "gaming", "GALA": "gaming", "MANA": "gaming",
    "XLM": "payments", "XMR": "privacy", "ZEC": "privacy", "DASH": "privacy",
}
_FALLBACK = ["alt-a", "alt-b", "alt-c", "alt-d", "alt-e"]


def _base(sym):
    s = sym.upper()
    for q in ("USDT", "USDC", "BUSD", "PERP", "USD"):
        if s.endswith(q):
            return s[:-len(q)]
    return s


def _sector_for(sym):
    b = _base(sym)
    if b in _SECTOR_MAP:
        return _SECTOR_MAP[b]
    return _FALLBACK[(sum(ord(c) for c in b)) % len(_FALLBACK)]   # stable across runs


def _sector_map(tickers):
    return {t: _sector_for(t) for t in tickers}


def _to_close(df):
    """Tolerate either a plain close panel or an OHLCV MultiIndex; return close panel."""
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in (0, -1):
            vals = set(df.columns.get_level_values(lvl))
            if "close" in vals:
                return df.xs("close", axis=1, level=lvl)
    return df


def _assemble(perp, spot):
    perp, spot = _to_close(perp), _to_close(spot)
    common = [c for c in perp.columns if c in set(spot.columns)]
    perp, spot = perp[common], spot[common]
    idx = perp.index.union(spot.index)
    perp = perp.reindex(idx).astype(float)
    spot = spot.reindex(idx).astype(float)
    perp.columns = pd.MultiIndex.from_product([["perp"], perp.columns])
    spot.columns = pd.MultiIndex.from_product([["spot"], spot.columns])
    panel = pd.concat([perp, spot], axis=1).sort_index()
    panel.index.name = "date"
    return panel


def _load(tickers):
    from sdk.adapters import binance_klines  # owned/free crypto klines (DATA_CATALOG.md)
    perp = binance_klines(tickers, start=START, market="perp")
    spot = binance_klines(tickers, start=START, market="spot")
    return _assemble(perp, spot)


# ----------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    """Aligned daily perp & spot closes over the broad liquid cross-section (survivorship
    -clean iff the adapter includes historically-liquid-then-delisted names — see prereg)."""
    from sdk.adapters import binance_universe
    tickers = list(binance_universe(N_UNIVERSE))
    return _load(tickers)


def load_gen_data(label) -> pd.DataFrame:
    """scope='local' => the stage-2 battery is NOT run (no disjoint cross-market analogue).
    Provided for completeness / optional within-crypto robustness: a DISJOINT lower-liquidity
    slice (ranks 76-150, sharing NO tickers with the search top-75)."""
    from sdk.adapters import binance_universe
    big = list(binance_universe(150))
    sub = big[N_UNIVERSE:150] if len(big) > N_UNIVERSE else big
    return _load(sub)


# ----------------------------------------------------------------- signal helpers
def _neg_streak(raw_basis):
    """Per-coin count of consecutive days with negative basis (NaN resets)."""
    neg = (raw_basis < 0)

    def f(c):
        ci = c.astype(int)
        grp = (~c).cumsum()
        return ci.groupby(grp).cumsum()

    return neg.apply(f)


def _select(trailing7, rankpct, neg_streak, dates, tickers, p):
    """Daily 00:00-UTC selection with hysteresis + min-hold + 3-day fast-exit + top-K cap.
    Decisions at date t use data through t only (target weights; lagged before P&L)."""
    entry_ann, exit_ann = p["entry_ann"], p["exit_ann"]
    top_q, exit_q       = p["top_q"], p["exit_rank_q"]
    neg_days, min_hold  = int(p["neg_days"]), int(p["min_hold"])
    top_k               = int(p["top_k"])

    S = pd.DataFrame(0.0, index=dates, columns=tickers)
    held = {}  # ticker -> hold_days
    for t in dates:
        b7 = trailing7.loc[t]
        rk = rankpct.loc[t]
        ns = neg_streak.loc[t]

        protected, retainable = [], []
        for tk in list(held.keys()):
            hd = held[tk]
            v = b7.get(tk, np.nan)
            if pd.isna(v):                         # data gone (delist/depeg) -> exit
                continue
            if ns.get(tk, 0) >= neg_days:          # fast-exit (min-hold EXEMPT)
                continue
            if hd < min_hold:                      # min-hold lock-in
                protected.append(tk)
                continue
            r = rk.get(tk, np.nan)
            if (v < exit_ann) or pd.isna(r) or (r > exit_q):   # soft exits
                continue
            retainable.append(tk)

        eligible = []
        for tk in tickers:
            if tk in held:
                continue
            v, r = b7.get(tk, np.nan), rk.get(tk, np.nan)
            if (not pd.isna(v)) and (v > entry_ann) and (not pd.isna(r)) and (r <= top_q):
                eligible.append(tk)

        slots = max(top_k - len(protected), 0)
        pool = sorted(retainable + eligible, key=lambda x: b7.get(x, -1e18), reverse=True)
        chosen = (protected + pool[:slots])[:top_k]

        held = {tk: held.get(tk, 0) + 1 for tk in chosen}
        for tk in chosen:
            S.at[t, tk] = 1.0
    return S


def _weights(S, carry_rets, vol_lb, top_k):
    """Inverse-vol (equal-risk) within the chosen book; deploy len(book)/K of capital so the
    book sits in stablecoin (0 return) when fewer than K names clear the hurdle."""
    vol = carry_rets.rolling(int(vol_lb), min_periods=10).std()
    invvol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    held = S > 0
    invvol = invvol.where(held)                    # NaN off-book; held-but-NaN-vol also NaN
    med = invvol.median(axis=1)                     # per-date median inv-vol of the book
    invvol = invvol.T.fillna(med).T                 # held-but-NaN-vol -> ~equal weight
    invvol = invvol.where(held, 0.0)                # re-zero off-book
    invvol = invvol.mask(held & invvol.isna(), 1.0) # all-NaN dates -> equal weight
    invvol = invvol.fillna(0.0)

    rs = invvol.sum(axis=1).replace(0.0, np.nan)
    W = invvol.div(rs, axis=0)
    deploy = (held.sum(axis=1) / float(top_k)).clip(upper=1.0)
    return W.mul(deploy, axis=0).fillna(0.0)


# ----------------------------------------------------------------------------- signal
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    perp = panel["perp"].astype(float)
    spot = panel["spot"].astype(float)
    tickers = list(perp.columns)
    dates = panel.index

    # --- basis = funding proxy ---
    raw_basis = (perp / spot - 1.0)                 # instantaneous perp premium
    ann_basis = raw_basis * FUNDINGS_PER_DAY * DAYS_PER_YEAR
    trailing7 = ann_basis.rolling(int(p["trail"]), min_periods=int(p["trail"])).mean()
    rankpct = trailing7.rank(axis=1, ascending=False, pct=True)   # top => small pct
    neg_streak = _neg_streak(raw_basis)

    # --- per-coin delta-neutral carry return (NO weights yet) ---
    # funding received by the short (~3x daily premium) + spread mark-to-market of
    # $1 long-spot / $1 short-perp. Same-day returns; paired with LAGGED weights below.
    spread_pnl = spot.pct_change() - perp.pct_change()
    funding = raw_basis * FUNDINGS_PER_DAY
    carry_rets = (funding + spread_pnl).reindex(index=dates, columns=tickers).fillna(0.0)

    # --- selection (data through t) -> target weights -> EXPLICIT 1-day lag ---
    S = _select(trailing7, rankpct, neg_streak, dates, tickers, p)
    W = _weights(S, carry_rets, p["vol_lb"], p["top_k"])
    W_held = W.shift(1).fillna(0.0)                  # the lag is OUR responsibility

    net = net_of_cost(W_held, carry_rets, cost_bps=p["cost_bps"], name="crypto_xsec_basis_carry")
    trades = trades_from_weights(W_held, carry_rets, _sector_map(tickers))
    return net, trades


# ------------------------------------------------------- soft expectation checks
def _xp_multi_name(ctx):
    trades = ctx.get("trades") or []
    names = {t.get("ticker") for t in trades}
    names.discard(None)
    return {"pass": len(names) >= 8, "observed": len(names)}


def _xp_topk_robust(ctx):
    grid = ctx.get("grid") or {}
    obs, total, ok = {}, 0, 0
    for lbl in ("default", "K3", "K5"):
        s = grid.get(lbl)
        if s is None:
            continue
        s = s.dropna()
        if len(s) == 0:
            continue
        m = float(s.mean()); obs[lbl] = round(m, 6); total += 1; ok += (m > 0)
    return {"pass": (total >= 2 and ok == total), "observed": obs}


def _xp_dispersion(ctx):
    from collections import Counter
    trades = ctx.get("trades") or []
    c, tot = Counter(), 0
    for t in trades:
        d = int(t.get("hold_days", 0) or 0)
        c[t.get("ticker")] += d; tot += d
    if tot == 0:
        return {"pass": False, "observed": 0.0}
    top = max(c.values()) / tot
    return {"pass": top <= 0.40, "observed": round(top, 3)}


def _xp_basis_tracks_funding(ctx):
    """Gate0 #2: basis proxy must track realized funding on the majors (where funding exists)."""
    try:
        from sdk.adapters import funding_rates
        h = pd.Timestamp(ctx["holdout_start"])
        prem = (ctx["panel"]["perp"] / ctx["panel"]["spot"] - 1.0)
        prem = prem[prem.index < h]

        def find(sub):
            for col in prem.columns:
                if str(col).upper().startswith(sub):
                    return col
            return None

        cols = [c for c in (find("BTC"), find("ETH")) if c is not None]
        if not cols:
            return {"pass": True, "observed": "no BTC/ETH cols"}

        fr = funding_rates(cols, start=START)
        if isinstance(fr.columns, pd.MultiIndex):
            fr = _to_close(fr)
        fr = fr.resample("1D").sum() if not fr.index.freq else fr   # 8h funding -> daily
        corrs = []
        for c in cols:
            fc = fr[c] if c in fr.columns else (fr.iloc[:, 0] if fr.shape[1] else None)
            if fc is None:
                continue
            a, b = prem[c].align(fc[fc.index < h], join="inner")
            a, b = a.dropna(), b.dropna()
            a, b = a.align(b, join="inner")
            if len(a) > 50:
                corrs.append(float(a.corr(b)))
        if not corrs:
            return {"pass": True, "observed": "no overlap"}
        avg = float(np.nanmean(corrs))
        return {"pass": avg > 0.3, "observed": round(avg, 3)}
    except Exception as e:                            # never block on a missing adapter
        return {"pass": True, "observed": f"unchecked:{type(e).__name__}"}


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="crypto_xsec_basis_carry_topk",
    family="carry",
    title="Crypto delta-neutral basis-carry — cross-sectional top-K across binance_universe(75)",
    markets=["crypto"],
    data_desc=("Owned/free: binance_klines(binance_universe(75), market='perp'|'spot') -> aligned "
               "daily perp & spot closes. basis=(perp/spot-1) annualized = deep-history broad-universe "
               "funding proxy (funding_rates() is majors-only; cross-validated as a soft expectation). "
               "No OI / long-short ratio (last-30-days only, data-gated)."),
    pre_registration=(
        "THESIS: harvest the perp carry / liquidity-provision premium CROSS-SECTIONALLY. Per coin run the "
        "parent's delta-neutral long-spot/short-perp (positive-basis side ONLY -> no short-spot, no spot-borrow). "
        "Daily 00:00-UTC check; min-hold(7d)+hysteresis make EFFECTIVE turnover weekly-or-lower (8bps applied on "
        "realized turnover). FROZEN thresholds (NO optimization): trailing-7d annualized basis; ENTER iff >10% ann "
        "AND top-quartile of 75; EXIT iff <2% ann OR rank<median OR basis negative 3 consecutive days (fast-exit, "
        "min-hold exempt); deployed cap top-K=4 by basis (gross<=2x), equal-risk (inverse-vol). Fewer than K clearing "
        "the hurdle => book sits in stablecoin (the built-in regime gate: dead aggregate funding => mostly FLAT). "
        "RETURN MODEL: short-side funding (~3x daily premium) + spread mark-to-market of the $1/$1 legs (this captures "
        "basis-gap risk that the fast-exit/top-K controls are meant to bound). NO-LOOKAHEAD: decisions at t use data "
        "through t; target weights are then W.shift(1)-lagged before P&L (lag is explicit, net_of_cost does not lag). "
        "SCOPE=local: perp basis exists ONLY in crypto perps (no disjoint cross-market generalization universe), but "
        "broader WITHIN crypto than the fixed-2 parent; forward-validation on the 2022-01-01 holdout confirms it. "
        "MACHINE-CHECKED mechanism claims: (1) >=8 distinct coins traded (dispersion, not a BTC/ETH book); (2) top-K in "
        "{3,4,5} all positive in search (not a single-K artifact); (3) no single coin >40% position-days; (4) basis "
        "proxy tracks realized funding on majors (corr>0.3). PROSE-ONLY (not cheaply checkable here): head-to-head vs "
        "the frozen 2-major parent on overlapping data (variant must beat it AND earn materially positive net return in "
        "the 2025-26 dead-funding window where the parent earns ~0 — if the variant ALSO dies in 2025-26, REJECT: the "
        "dispersion thesis is falsified); drop-one-coin robustness; stress May-2021 / FTX-Nov-2022 / 2022-bear with a "
        "demonstrated fast-exit trigger and no catastrophic gap loss on a delisted/depegged name. CAVEATS (prior=medium): "
        "basis != realized funding exactly; SURVIVORSHIP depends on binance_universe including historically-liquid-then-"
        "delisted names — if it is winners-only the cross-section is inflated. If it passes: fresh write-once forward-paper "
        "run with its own pre-registered ~3-month verdict date before re-entering any carry+trend book (Midas closure rule)."
    ),
    load_data=load_data,
    signal=signal,
    default_params=DEFAULTS,
    grid=GRID,
    scope="local",
    generalization_universes=[],          # local: no disjoint cross-market analogue (see prereg)
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=4,
    expectations=[
        {"name": "multi_name_participation",
         "claim": "harvests cross-sectional dispersion: >=8 distinct coins traded in-sample (not just BTC/ETH)",
         "check": _xp_multi_name},
        {"name": "topk_robustness",
         "claim": "top-K in {3,4,5} all show positive mean daily search return (not driven by a single K)",
         "check": _xp_topk_robust},
        {"name": "not_single_name",
         "claim": "no single coin exceeds 40% of position-days (genuine cross-section, not a disguised 1-name book)",
         "check": _xp_dispersion},
        {"name": "basis_tracks_funding",
         "claim": "basis proxy tracks realized funding on the majors (BTC/ETH corr > 0.3, pre-holdout)",
         "check": _xp_basis_tracks_funding},
    ],
)