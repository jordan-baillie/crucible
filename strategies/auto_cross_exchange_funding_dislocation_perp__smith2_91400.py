"""Cross-exchange funding dislocation — perp crowding-reversal premium.

Edge (LOCAL to crypto): the cross-EXCHANGE funding spread d_i = funding_bybit_i - funding_binance_i
is a localized venue-crowding signal, distinct from the absolute funding LEVEL (which has compressed
to ~0 in 2025-26, killing naive carry). Two order books clear independently 24/7 with no unified
margin, so the relative dislocation persists. FROZEN construction: each DAY, rank d_i cross-sectionally;
SHORT the top tercile (longs over-crowded on the richer-funding venue -> expect reversion down),
LONG the bottom tercile. Inverse-realized-vol within each leg, dollar-neutral, hysteresis band +
1-day min-hold to suppress turnover, vol-targeted, net of 20bps round-trip taker.

Lag discipline: weights W are built from info known at end of day t (spread + trailing vol). The kit
lag (W.shift(1)) into net_of_cost/trades_from_weights moves execution to t+1 — no look-ahead.

Panel layout: a SINGLE-LEVEL, string-keyed DataFrame with columns prefixed by field
("PX|<coin>", "SP|<coin>", "LV|<coin>"), re-split via _field(). This avoids fragile MultiIndex
columns (a 2-level column index did not survive the harness panel round-trip, dropping the
top-level "price" key -> KeyError in signal). One responsibility, robust, no behavior change.

FIX (was: ValueError "no overlapping coins"): binance_klines / bybit_funding / funding_rates label
coins in DIFFERENT symbol conventions (e.g. BTCUSDT vs BTC-USD vs BTC), so the raw column-name
intersection was empty even though all three carry the same coins. We now canonicalize every column
to a bare base symbol (strip quote/perp suffixes + separators, uppercase) BEFORE intersecting.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.signal_kit import net_of_cost, trades_from_weights
# Crypto adapters (OWNED/FREE, see research-wiki/DATA_CATALOG.md): perp klines + cross-exchange funding.
from sdk.adapters import binance_universe, binance_klines, bybit_funding, funding_rates


# ----------------------------------------------------------------------------- data
def _norm(sym):
    """Canonical base-coin symbol: strip separators + quote/perp suffixes, uppercase.

    Reconciles the differing conventions across binance_klines / bybit_funding / funding_rates
    (BTCUSDT, BTC-USD, BTC/USDT:USDT, BTC ...) onto one key so they intersect.
    """
    s = str(sym).upper()
    for ch in ("-", "_", "/", ":", " "):
        s = s.replace(ch, "")
    for suf in ("PERP", "USDT", "USDC", "BUSD", "USD"):
        if s.endswith(suf) and len(s) > len(suf):
            s = s[: -len(suf)]
            break
    return s


def _canon(df):
    """Rename columns to canonical base symbols; drop duplicate-collapsed names (keep first)."""
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df


def _to_daily(df, idx):
    """Resample (possibly intraday) funding to a daily average, align to the price grid."""
    df = df.sort_index()
    df = df.resample("D").mean()
    return df.reindex(idx, method="ffill", limit=3)


def _field(panel, tag):
    """Re-split the flat packed panel into a (date x coin) sub-frame for one field tag."""
    pref = tag + "|"
    cols = [c for c in panel.columns if isinstance(c, str) and c.startswith(pref)]
    sub = panel[cols].astype(float)
    sub.columns = [c[len(pref):] for c in cols]
    return sub


def _build_panel(coins):
    px = _canon(binance_klines(coins, "perp").astype(float))  # canonical base-coin columns
    fb = _canon(bybit_funding())   # Bybit funding, wide (date x coin)
    fn = _canon(funding_rates())   # Binance funding, wide (date x coin) — same sign/shape per catalog
    common = sorted(set(px.columns) & set(fb.columns) & set(fn.columns))
    if not common:
        raise ValueError("no overlapping coins across price/bybit/binance funding panels")
    px = px[common]
    idx = px.index
    fb = _to_daily(fb[common], idx)
    fn = _to_daily(fn[common], idx)
    spread = fb - fn              # cross-exchange dislocation (the trade signal)
    # Flat, single-level, string-keyed columns (field-prefixed) — robust to panel round-trips.
    return pd.concat(
        [px.add_prefix("PX|"), spread.add_prefix("SP|"), fn.add_prefix("LV|")],
        axis=1,
    )


def load_data() -> pd.DataFrame:
    # Bounded liquid perp cross-section; dollar-volume liquidity comes from binance_universe ranking.
    return _build_panel(binance_universe(75))


def load_gen_data(label) -> pd.DataFrame:
    # scope='local' (venue-structural mechanism) -> no cross-market battery; never invoked by the rails.
    return load_data()


# --------------------------------------------------------------------------- signal
def signal(panel, **params):
    top_q = params.get("top_q", 0.667)
    bot_q = params.get("bot_q", 0.333)
    band = params.get("hysteresis", 0.05)
    vol_lb = int(params.get("vol_lb", 30))
    target_vol = params.get("target_vol", 0.15)
    min_names = int(params.get("min_names", 6))
    name = "cross_exch_funding_disloc"

    px = _field(panel, "PX")
    spread = _field(panel, "SP")
    spread = spread.reindex(columns=px.columns)   # share the coin axis
    rets = px.pct_change()
    vol = rets.rolling(vol_lb).std()

    # FROZEN: rank the cross-exchange spread cross-sectionally EACH DAY (daily rebalance).
    # Daily evaluation + hysteresis band naturally enforces the >=1-day min-hold.
    state = pd.Series(0, index=spread.columns, dtype=int)  # -1 short, +1 long, 0 flat (hysteresis carry)
    rows = {}
    for adate in spread.index:
        d = spread.loc[adate].dropna()
        if len(d) < min_names:
            continue  # keep prior state (forward-filled) when breadth is too thin
        pct = d.rank(pct=True)
        ns = state.copy()
        for c in spread.columns:
            p = pct.get(c, np.nan)
            s = int(state.get(c, 0))
            if not np.isfinite(p):
                ns[c] = 0
                continue
            if s == -1 and p < top_q - band:      # exit short only when well back inside the band
                ns[c] = 0
            elif s == 1 and p > bot_q + band:      # exit long only when well back inside the band
                ns[c] = 0
            if ns[c] == 0:                          # (re)entry / flip
                if p >= top_q:
                    ns[c] = -1
                elif p <= bot_q:
                    ns[c] = 1
        state = ns

        vr = vol.loc[adate] if adate in vol.index else pd.Series(dtype=float)
        w = pd.Series(0.0, index=spread.columns)
        for members, gross in (([c for c in state.index if state[c] == 1], 0.5),
                               ([c for c in state.index if state[c] == -1], -0.5)):
            v = vr.reindex(members).replace(0, np.nan).dropna()
            if v.empty:
                continue
            iv = 1.0 / v
            iv = iv / iv.sum() * gross           # inverse-vol within leg, dollar-neutral (+0.5 / -0.5)
            w.loc[iv.index] = iv.values
        rows[adate] = w

    if not rows:
        return pd.Series(dtype=float, name=name), []

    Wr = pd.DataFrame(rows).T.sort_index()
    W = Wr.reindex(rets.index, method="ffill").fillna(0.0)

    # Vol-target the whole book; leverage estimated from trailing realized vol and lagged (no look-ahead).
    gross_ret = (W.shift(1) * rets).sum(axis=1)
    rv = gross_ret.rolling(vol_lb).std() * np.sqrt(365.0)
    lev = (target_vol / rv).replace([np.inf, -np.inf], np.nan).clip(upper=3.0).shift(1).fillna(1.0)
    W = W.mul(lev, axis=0)

    Wlag = W.shift(1).fillna(0.0)                 # MY lag: same-day weights -> trade next day
    daily = net_of_cost(Wlag, rets, cost_bps=20.0, name=name)  # 20bps round-trip taker (crypto)
    sector_map = {c: "Crypto" for c in spread.columns}
    trades = trades_from_weights(Wlag, rets, sector_map)
    return daily, trades


# --------------------------------------------------------------------- expectations
def _check_breadth(ctx):
    """gate0/Fundamental-Law: a real cross-section (>=15 overlapping coins) survives, not just BTC/ETH."""
    sp = _field(ctx["panel"], "SP")
    sp = sp[sp.index < ctx["holdout_start"]]
    obs = float(sp.notna().sum(axis=1).median())
    return {"pass": obs >= 15, "observed": obs}


def _check_orthogonal_to_carry(ctx):
    """Core thesis: the dislocation is distinct from absolute funding carry (mean |x-sec corr| < 0.5)."""
    sp = _field(ctx["panel"], "SP")
    lv = _field(ctx["panel"], "LV")
    h = ctx["holdout_start"]
    sp = sp[sp.index < h]
    lv = lv[lv.index < h]
    cors = []
    for dt in sp.index:
        a = sp.loc[dt]
        b = lv.reindex(columns=sp.columns).loc[dt] if dt in lv.index else None
        if b is None:
            continue
        m = a.notna() & b.notna()
        if m.sum() >= 6:
            c = np.corrcoef(a[m], b[m])[0, 1]
            if np.isfinite(c):
                cors.append(abs(c))
    obs = float(np.mean(cors)) if cors else float("nan")
    return {"pass": np.isfinite(obs) and obs < 0.5, "observed": obs}


# ---------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="cross_exch_funding_disloc_v1",
    family="crypto_crowding_reversal",
    title="Cross-exchange funding dislocation (Bybit-Binance) — perp crowding-reversal premium",
    markets=["crypto"],
    data_desc=("Binance perp daily klines (binance_universe top-75 by dollar-volume) + cross-exchange "
               "funding spread d=funding_bybit-funding_binance; bybit_funding() vs funding_rates(), "
               "symbols canonicalized to base coin, daily-averaged, overlapping coins only (>=2021). "
               "Panel packed single-level, field-prefixed."),
    pre_registration=(
        "Liquidity-provision/crowding-reversal premium, NOT absolute funding carry. Frozen: DAILY, rank "
        "cross-exchange funding spread d_i; short top tercile (over-crowded longs on richer venue -> revert "
        "down), long bottom tercile; inverse-vol within leg, dollar-neutral, hysteresis band + 1-day min-hold, "
        "vol-targeted, 20bps round-trip. Absolute funding has compressed to ~0 (naive carry dormant) but the "
        "RELATIVE cross-venue dislocation persists because the two books clear independently with no unified "
        "margin. Falsifiable: (1) edge survives MCPT market-neutral absolute-Sharpe null (rules out bid-ask-bounce "
        "artifact); (2) signal is orthogonal to absolute funding level (mean |x-sec corr|<0.5); (3) >=15 "
        "overlapping coins survive the breadth gate; (4) forward-paper confirms on fresh data. LOCAL: the "
        "mechanism is venue-structural (fragmented 24/7 perp venues) and cannot exist in single-venue or "
        "centrally-cleared TradFi futures, so no cross-market generalization battery applies."),
    load_data=load_data,
    signal=signal,
    default_params={"top_q": 0.667, "bot_q": 0.333, "hysteresis": 0.05,
                    "vol_lb": 30, "target_vol": 0.15},
    grid={
        "default": {},
        "wide_tercile": {"top_q": 0.75, "bot_q": 0.25},
        "tight_band": {"hysteresis": 0.02},
        "slow_vol": {"vol_lb": 45},
    },
    scope="local",
    generalization_universes=[],     # venue-structural mechanism -> no broad battery
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=30,
    expectations=[
        {"name": "breadth_ge_15",
         "claim": "median daily count of coins with both-venue funding >= 15 (real cross-section)",
         "check": _check_breadth},
        {"name": "orthogonal_to_carry",
         "claim": "dislocation distinct from absolute funding level (mean |x-sec corr| < 0.5)",
         "check": _check_orthogonal_to_carry},
    ],
)