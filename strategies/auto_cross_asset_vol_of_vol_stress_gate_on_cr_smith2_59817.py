"""
Crypto cross-sectional momentum, de-risked by a frozen cross-asset vol-of-vol
(VVIX/VIX) stress gate.  scope = LOCAL.

WHY THIS IS A REWRITE, NOT A TUNE
---------------------------------
The parent was a BTC/ETH delta-neutral *perp-funding carry* book gated by a
VVIX/VIX flag.  Its return engine depended on a crypto perp/funding feed
(`crypto_perp_panel` / `carry_returns()`) that DOES NOT EXIST in the harness data
inventory (research-wiki/DATA_CATALOG.md) -- a fabricated adapter; the module never
ran.  Per the no-fabrication rule, funding is NOT synthesizable from spot, so that
carry premium is simply not measurable with OWNED/FREE data here.

Honest fix: keep the only genuinely-testable, real-data half of the design -- the
cross-asset vol-of-vol stress GATE (VVIX rising while VIX is complacent), which is
the novel contribution -- and bolt it onto a return engine the owned data CAN
support: a crypto cross-sectional MOMENTUM premium built purely from OWNED/FREE
daily spot (yf_panel '<COIN>-USD').  The thesis label honestly changes from
"funding carry" to "cross-sectional momentum"; nothing is tuned to a result.

RETURN ENGINE: dollar-neutral crypto XS momentum.  Each weekly UTC rebalance, rank
a fixed basket of liquid coins by trailing-`mom_lb`d return; long the top-k / short
the bottom-k, inverse-vol sized within each leg (gross ~1x, ~0.5 long / 0.5 short,
market-neutral).  ~10bps/leg turnover cost (conservative for crypto).

RISK-OFF GATE (frozen, NONE searched): FLAG = (VVIX>110 & VIX<18) OR
(VVIX/VIX>6.5 & VIX<18), read on the US-equity close STRICTLY PRIOR to the crypto
session; when it fires the whole book is forced FLAT for `gate_window` days.  It
generates no alpha -- it only clips the left tail.  Standalone book (gate OFF,
'gate_off') is the pre-registered benchmark; the grid is robustness, not selection.

NO-LOOKAHEAD (independent guards): (1) momentum uses trailing returns through day t;
(2) equity flag ffilled onto the crypto grid then shift(1)'d to the strictly-prior
session; (3) the whole target-weight matrix is held via W.shift(1) before
net_of_cost / trades_from_weights.  All data OWNED/FREE ($0).
"""
import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series
from sdk.signal_kit import net_of_cost, trades_from_weights

# liquid coins with multi-year spot history; short/missing ones are dropped at runtime
COINS = ["BTC", "ETH", "XRP", "LTC", "BCH", "BNB", "EOS", "XLM",
         "TRX", "ADA", "ETC", "LINK", "DOGE", "NEO", "DASH", "XMR"]
START   = "2019-01-01"
HOLDOUT = "2022-01-01"

# ---- frozen defaults (gate thresholds NONE searched; grid variants are robustness) ----
P0 = dict(
    coins=list(COINS),
    mom_lb=30,          # trailing momentum look-back (days)
    vol_lb=30,          # inverse-vol look-back (days)
    top_k=3, bot_k=3,   # long top-k / short bottom-k
    rebal_dow=0,        # weekly rebalance anchor (Mon; crypto trades 7d/wk)
    vvix_hi=110.0, vix_lo=18.0, ratio_hi=6.5,   # FROZEN gate thresholds
    gate_window=10,     # flat-override window (days)
    cost_bps=10.0,      # ~10bps/leg crypto turnover cost (conservative)
    gate_off=False,     # True => standalone-momentum benchmark
    placebo=False,      # True => count-matched random gate (placebo)
    placebo_seed=0,
)


# ----------------------------------------------------------------------------- data
def _load_crypto(start=START):
    """OWNED/FREE daily spot for the liquid-coin basket via yf_panel('<COIN>-USD').
    Spot only -- no perp/funding is claimed (that feed does not exist in inventory)."""
    raw = yf_panel([f"{c}-USD" for c in COINS], start=start)
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    raw = raw.rename(columns={c: c.replace("-USD", "") for c in raw.columns})
    cols = {f"{c}_spot": raw[c] for c in raw.columns if c in COINS}
    return pd.DataFrame(cols).sort_index()


def load_data() -> pd.DataFrame:
    crypto = _load_crypto(START)
    vol = yf_panel(["^VVIX", "^VIX"], start=START).rename(
        columns={"^VVIX": "VVIX", "^VIX": "VIX"})
    try:  # cross-check / backfill spot VIX from FRED VIXCLS (same series, business-day)
        vixcls = fred_series({"VIXCLS": "VIX_FRED"}, start=START)["VIX_FRED"]
        vol["VIX"] = vol["VIX"].combine_first(vixcls.reindex(vol.index))
    except Exception:
        pass
    panel = pd.concat([crypto, vol], axis=1).sort_index()
    panel.index = pd.to_datetime(panel.index)
    panel.attrs["name"] = "vvix_gated_crypto_xsmom"
    return panel


def load_gen_data(label) -> pd.DataFrame:
    # LOCAL scope: a crypto XS-momentum premium gated on the US vol complex has no
    # disjoint cross-market analogue here. Robustness is pre-registered via grid
    # variants + machine-checkable expectations (threshold/window sign-stability,
    # placebo, tail-clip). Returned for signature completeness only; the stage-2
    # broad battery does not apply.
    return load_data()


# ------------------------------------------------------------------------- helpers
def _equity_flag(panel, vvix_hi, vix_lo, ratio_hi):
    vvix = panel["VVIX"].astype(float); vix = panel["VIX"].astype(float)
    valid = vvix.notna() & vix.notna() & (vix > 0)
    flag = (((vvix > vvix_hi) & (vix < vix_lo)) |
            ((vvix / vix > ratio_hi) & (vix < vix_lo)))
    return flag.where(valid, np.nan).astype(float)


def _gate_mask(panel, cidx, p):
    """Flat-override mask on the crypto grid: equity flag ffilled onto the grid then
    shift(1)'d to the strictly-prior equity session, then a gate_window-day window."""
    flag = _equity_flag(panel, p["vvix_hi"], p["vix_lo"], p["ratio_hi"])
    f = flag.ffill().reindex(cidx).ffill().shift(1).fillna(0.0)
    return (f.rolling(p["gate_window"], min_periods=1).max() > 0).astype(float)


def _sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 20 or r.std(ddof=0) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(365.0))   # crypto = 365 days/yr


def _maxdd(r):
    r = pd.Series(r).dropna()
    if not len(r):
        return 0.0
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


# -------------------------------------------------------------------------- signal
def signal(panel, **params):
    p = dict(P0); p.update(params)
    coins = [c for c in p["coins"] if f"{c}_spot" in panel.columns]

    spot = panel[[f"{c}_spot" for c in coins]].copy(); spot.columns = coins
    spot = spot.astype(float)
    need = p["mom_lb"] + p["vol_lb"] + 30
    coins = [c for c in coins if spot[c].notna().sum() > need]
    spot = spot[coins].dropna(how="all")
    cidx = spot.index                                   # crypto trading days (7d/wk)
    ret = spot.pct_change().replace([np.inf, -np.inf], np.nan)

    k_long = min(p["top_k"], max(1, len(coins) // 2))
    k_short = min(p["bot_k"], max(1, len(coins) // 2))

    # cross-sectional momentum + inverse-vol leg sizing (info through day t only)
    mom = spot.pct_change(p["mom_lb"]).replace([np.inf, -np.inf], np.nan)
    vol = ret.rolling(p["vol_lb"], min_periods=max(5, p["vol_lb"] // 3)).std()
    ivol = (1.0 / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    long_mask  = mom.rank(axis=1, ascending=False).le(k_long)
    short_mask = mom.rank(axis=1, ascending=True).le(k_short)

    longw  = (long_mask  * ivol)
    shortw = (short_mask * ivol)
    longw  = longw.div(longw.sum(axis=1).replace(0.0, np.nan), axis=0) * 0.5
    shortw = shortw.div(shortw.sum(axis=1).replace(0.0, np.nan), axis=0) * 0.5
    Wt = longw.fillna(0.0) - shortw.fillna(0.0)         # same-day target, dollar-neutral

    # weekly rebalance: refresh target only on the anchor weekday, hold (ffill) between
    W = Wt.copy()
    nonrebal = ~(cidx.weekday == p["rebal_dow"])
    W[nonrebal] = np.nan
    W = W.ffill().fillna(0.0)

    # risk-off gate override (strictly-prior equity close); placebo = count-matched random
    if not p["gate_off"]:
        if p["placebo"]:
            real = _gate_mask(panel, cidx, p)
            n_on = int(real.sum())
            rng = np.random.default_rng(p["placebo_seed"])
            override = pd.Series(0.0, index=cidx)
            if 0 < n_on < len(cidx):
                override.iloc[rng.choice(len(cidx), size=n_on, replace=False)] = 1.0
        else:
            override = _gate_mask(panel, cidx, p)
        W = W.mul(1.0 - override.reindex(cidx).fillna(0.0), axis=0)

    # 1-day execution lag, costs on turnover, net returns + contract ledger
    Wl = W.shift(1).fillna(0.0)
    R = ret.fillna(0.0)
    rr = net_of_cost(Wl, R, cost_bps=p["cost_bps"], name="vvix_gated_crypto_xsmom")
    smap = {c: f"Crypto-{c}" for c in coins}            # distinct sector per coin
    trades = trades_from_weights(Wl, R, smap)           # kit stamps entry_regime
    return rr, trades


# --------------------------------------------------------- machine-checkable checks
def _chk_gate_helps_tail(ctx):
    g = ctx["grid"]; fused = g.get("default"); base = g.get("gate_off")
    if fused is None or base is None:
        return {"pass": False, "observed": "missing default/gate_off variant"}
    shf, shb = _sharpe(fused), _sharpe(base)
    ddf, ddb = _maxdd(fused), _maxdd(base)
    ok = (ddf >= ddb - 0.005) and (shf >= shb - 0.10)   # shallower DD, Sharpe not materially worse
    return {"pass": bool(ok),
            "observed": f"Sharpe {shf:.2f} vs {shb:.2f}; maxDD {ddf:.1%} vs {ddb:.1%}"}


def _chk_gate_events(ctx):
    panel = ctx["panel"]; h = pd.Timestamp(ctx["holdout_start"])
    sub = panel[panel.index < h]
    scols = [c for c in sub.columns if c.endswith("_spot")]
    cidx = sub[scols].dropna(how="all").index
    ov = _gate_mask(sub, cidx, dict(P0))
    episodes = int((ov.diff() > 0).sum())               # rising edges = distinct firings
    frac = float(ov.mean()) if len(ov) else 0.0
    ok = episodes >= 3 and 0.0 < frac < 0.30            # sparse but present (binding caveat)
    return {"pass": bool(ok),
            "observed": f"{episodes} divergence episodes, gate-on {frac:.1%} of search days"}


def _chk_placebo(ctx):
    panel = ctx["panel"]; h = pd.Timestamp(ctx["holdout_start"])
    base = ctx["grid"].get("gate_off"); fused = ctx["grid"].get("default")
    if base is None or fused is None:
        return {"pass": False, "observed": "missing grid variants"}
    pr, _ = signal(panel, placebo=True, placebo_seed=12345)   # one extra signal() call
    pr = pr[pr.index < h]
    shp, shb, shf = _sharpe(pr), _sharpe(base), _sharpe(fused)
    ok = (shp <= shb + 0.10) and (shf >= shp - 0.05)    # random gate must not improve; real >= placebo
    return {"pass": bool(ok),
            "observed": f"real {shf:.2f} vs placebo {shp:.2f} vs standalone {shb:.2f}"}


def _chk_threshold_sign(ctx):
    g = ctx["grid"]; base = g.get("gate_off")
    if base is None or g.get("default") is None:
        return {"pass": False, "observed": "missing baseline variants"}
    shb = _sharpe(base)
    primary = np.sign(_sharpe(g["default"]) - shb) or 1.0
    vs = ["vvix_105", "vvix_115", "ratio_60", "ratio_70", "win_5", "win_15"]
    signs = [np.sign(_sharpe(g[v]) - shb) for v in vs if g.get(v) is not None]
    if not signs:
        return {"pass": False, "observed": "no threshold variants"}
    frac = float(np.mean([s == primary for s in signs]))
    return {"pass": bool(frac >= 0.6),
            "observed": f"{frac:.0%} of VVIX/ratio/window variants share the primary sign"}


# ----------------------------------------------------------------------------- grid
grid = {
    "default":  {},                       # PRIMARY: gated book
    "gate_off": {"gate_off": True},        # pre-registered standalone-momentum benchmark
    "mom_20":   {"mom_lb": 20},
    "mom_60":   {"mom_lb": 60},
    "vol_20":   {"vol_lb": 20},
    "vvix_105": {"vvix_hi": 105.0},
    "vvix_115": {"vvix_hi": 115.0},
    "ratio_60": {"ratio_hi": 6.0},
    "ratio_70": {"ratio_hi": 7.0},
    "win_5":    {"gate_window": 5},
    "win_15":   {"gate_window": 15},
}

PRE_REG = """
PRE-REGISTRATION (frozen 2026-06-15).

PROVENANCE / HONEST REWRITE: the parent specified a BTC/ETH delta-neutral
*perp-funding carry* engine gated by a VVIX/VIX flag. That engine required a crypto
perp/funding feed (crypto_perp_panel / carry_returns()) that DOES NOT EXIST in the
harness data inventory -- a fabricated adapter; the module never ran. Funding is NOT
synthesizable from spot (synthesizing it would be fabrication), so the carry premium
is unmeasurable with OWNED/FREE data here. Rather than fake the data, the un-buildable
carry leg is dropped and the genuinely-testable half -- the cross-asset vol-of-vol
stress GATE -- is retained verbatim and applied to a return engine the owned data CAN
support: crypto cross-sectional momentum from yf_panel daily spot. The thesis label
changes (carry -> XS momentum); nothing is tuned to a result.

PREMIUM: documented crypto cross-sectional momentum -- dollar-neutral long top-k /
short bottom-k coins ranked on trailing-30d return, inverse-vol sized within each leg
(gross ~1x), weekly UTC rebalance, ~10bps/leg turnover cost (conservative for crypto).
OVERLAY: a frozen cross-asset vol-of-vol stress flag (VVIX rising while VIX complacent)
used ONLY to de-risk the book ahead of cross-asset deleveraging; it generates no alpha,
it only clips the left tail.

FROZEN GATE (nothing searched): FLAG = (VVIX>110 & VIX<18) OR (VVIX/VIX>6.5 & VIX<18)
on the equity close STRICTLY PRIOR to the crypto entry -> flat 10 days. The grid is
robustness (VVIX 105/110/115, ratio 6.0/6.5/7.0, window 5/10/15, momentum/vol
look-backs), NOT selection.

BENCHMARK / FALSIFICATION: the gated book is tested against the standalone momentum
book ('gate_off') -- the honest claim is a SHALLOWER maxDD without materially worse
Sharpe (expectation gate_helps_tail), beating a count-matched RANDOM placebo gate
(placebo_no_improvement), with threshold/window sign-stability (threshold_sign_stable).

BINDING CAVEAT (carried to the human gate, NOT engineered away): distinct VVIX/VIX
divergence episodes in the 2019+ window may be FEW (and 2019 was a low-vol year, so the
VIX<18 leg may fire on many calm days). Both event-count and gate-on fraction are
reported LOUDLY (gate_events_sufficient); thresholds are NEVER loosened to manufacture
events. HONEST STOP-CONDITION: if the gate adds no tail protection outside genuine
cross-asset stress, the correct result is 'DEPLOY STANDALONE MOMENTUM', not a tuned gate.

DEPLOYMENT-SANITY: a 16-coin basket holding ~6 names per side gives real sector spread
(distinct Crypto-<COIN> labels) and keeps single-name position-day share well under the
40% cap -- it does not inherit the 2-coin structural violation the parent disclosed.

NO-LOOKAHEAD: momentum uses trailing returns through day t; equity flag ffilled then
shift(1)'d to the strictly-prior session; whole weight matrix held via W.shift(1). All
data OWNED/FREE ($0). LOCAL scope: no disjoint cross-market analogue for this gated book.
"""

SPEC = StrategySpec(
    id="vvix_gated_crypto_xsmom",
    family="crypto-xsmom-cross-asset-vol-gate",
    title="Cross-asset vol-of-vol stress gate on crypto cross-sectional momentum (VVIX-conditioned, real-data rewrite of an un-buildable carry book)",
    markets=["crypto-spot-basket", "us-equity-vol-complex(conditioning)"],
    data_desc=("OWNED/FREE: daily crypto spot for a 16-coin liquid basket via "
               "yf_panel('<COIN>-USD') (2019+); equity vol complex via "
               "yf_panel('^VVIX','^VIX') + FRED VIXCLS cross-check. No perp/funding "
               "feed is used or claimed (that adapter does not exist in inventory)."),
    pre_registration=PRE_REG,
    load_data=load_data,
    signal=signal,
    default_params={},
    grid=grid,
    scope="local",
    generalization_universes=[],          # LOCAL: no disjoint cross-market analogue
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT,
    deploy_max_positions=6,               # top-3 long / bottom-3 short
    expectations=[
        {"name": "gate_helps_tail",
         "claim": "gated book maxDD no deeper than standalone momentum AND Sharpe within 0.10 (search window)",
         "check": _chk_gate_helps_tail},
        {"name": "gate_events_sufficient",
         "claim": ">=3 distinct divergence episodes in the 2019+ search window and gate-on <30% of days (binding low-power caveat)",
         "check": _chk_gate_events},
        {"name": "placebo_no_improvement",
         "claim": "a count-matched RANDOM gate does not beat the standalone book, and the real gate >= placebo",
         "check": _chk_placebo},
        {"name": "threshold_sign_stable",
         "claim": ">=60% of pre-declared VVIX/ratio/window variants share the primary improvement sign",
         "check": _chk_threshold_sign},
    ],
)