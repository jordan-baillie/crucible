"""
Illiquidity-Premium x Trend Crisis-Alpha -- CRYPTO-BREADTH-DIVERSIFIED VARIANT
==============================================================================
Two-premium combination book (scope='local'):

  Leg A  -- Amihud illiquidity premium, US small-cap cross-section (the GATED CORE,
            100% of book risk).  Illiquid names earn a premium that *crashes* exactly
            when liquidity evaporates -> pro-cyclical, needs a crisis hedge.
  Leg B  -- Time-series-trend crisis-alpha SLEEVE (25% of book risk total), built as
            ONE canonical TS-trend rule applied per-market across 7 markets, each at an
            EQUAL 1/7 per-market risk weight:
              * 5 TradFi ETFs : SPY, EFA, TLT, GLD, DBC   (the parent sleeve, untouched)
              * 2 crypto      : BTC-USD, ETH-USD          (THE MUTATION)
            The parent sleeve is all-TradFi -> its 5 members converge toward high
            correlation in a global liquidity crisis (Mar-2020), so its EFFECTIVE breadth
            is < 5 exactly when it must hedge.  BTC/ETH trend is driven by crypto-native
            deleveraging / 24-7 liquidation cascades -> structurally TradFi-independent,
            so it adds genuine, decorrelated crisis-alpha breadth.

EXECUTION (now faithful to the frozen parent design):
  * Every per-market trend weight is routed through the parent's pre-registered
    PROPORTIONAL 20%-DRIFT NO-TRADE BAND + a MINIMUM-TRADE FLOOR before it is lagged and
    charged costs.  In this return-series harness the absolute $50 floor is approximated
    by a small relative min-weight-change floor (no notional book is available here).
  * Costs net: 8bps equity/ETF turnover, 20bps crypto round-trip taker. Every signal
    lagged 1 day.

HONESTY CAVEATS (stated, not hidden):
  * This is a clean KIT reconstruction of the Amihud leg (sep_panel volume+price ->
    |ret|/$vol -> xs_zscore -> inv_vol_position), NOT a literal stored "byte-frozen"
    stream -- no such artefact exists inside this harness.  Sign: LONG high illiquidity.
  * The CONTRACT trade ledger is emitted from the Amihud equity book ONLY (the gated
    core with real sectors/regimes).  The trend+crypto sleeve is a 25% RISK OVERLAY that
    is reflected in daily_returns but not in the ledger.
  * Crypto is reindexed to the equity business-day calendar (weekend crypto bars dropped)
    so the whole panel shares one clean daily index -- no NaN-propagation lookahead.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel, inv_vol_position
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

SID = "illiq_x_trend_crypto_breadth_v1"
START = "2010-01-01"
ETFS = ["SPY", "EFA", "TLT", "GLD", "DBC"]          # 5 TradFi cross-asset trend markets
CRYPTO = ["BTC-USD", "ETH-USD"]                      # 2 TradFi-independent trend markets
N_SLEEVE = len(ETFS) + len(CRYPTO)                   # 7 markets, equal 1/7 risk weight each

DEFAULTS = dict(
    lb_illiq=21,            # Amihud averaging window (trading days)
    lookback_trend=100,     # single canonical TS-trend lookback (all 7 markets)
    vol_lb=60,              # trailing vol window for per-market inverse-vol scaling
    target_vol=0.10,        # annualised target vol (Amihud inv_vol + per-market trend)
    sleeve_weight=0.25,     # trend sleeve = 25% of book risk vs Amihud 100%
    eq_cost_bps=8.0,
    etf_cost_bps=8.0,
    crypto_cost_bps=20.0,   # conservative crypto round-trip taker
    drift_band=0.20,        # parent's pre-registered 20%-drift no-trade band
    min_trade=0.02,         # minimum-trade floor (relative-weight proxy for the $50 floor)
    max_pos=80,             # Amihud cross-section breadth
    rebalance="W",          # weekly rebalance (kit handles the holding)
)

_LEG_CACHE = {}


# --------------------------------------------------------------------------- #
# DATA                                                                        #
# --------------------------------------------------------------------------- #
def _build_panel(tickers, sector_map):
    """One clean DataFrame on the equity business-day calendar; sector_map in .attrs."""
    eq_close = sep_panel(tickers, START, field="closeadj")
    master = eq_close.index                                   # equity trading days = master
    eq_vol = sep_panel(tickers, START, field="volume").reindex(master)
    etf = yf_panel(ETFS, START).reindex(master).ffill(limit=2)
    crypto = yf_panel(CRYPTO, START).reindex(master).ffill(limit=2)   # weekends dropped
    panel = pd.concat(
        {"eq_close": eq_close, "eq_vol": eq_vol, "etf": etf, "crypto": crypto}, axis=1
    )
    panel.attrs["sector_map"] = dict(sector_map)
    return panel


def load_data() -> pd.DataFrame:
    tickers, sector_map = sector_universe("Small", 40)        # ~sector-spread small caps
    return _build_panel(tickers, sector_map)


def load_gen_data(label) -> pd.DataFrame:
    # scope='local' -> the stage-2 generalization battery is NOT run; both core legs'
    # standalone validation is settled upstream. Provided for signature completeness.
    return load_data()


# --------------------------------------------------------------------------- #
# EXECUTION: parent's proportional no-trade band + minimum-trade floor         #
# --------------------------------------------------------------------------- #
def _apply_band(w, drift_band, min_trade):
    """Causal proportional no-trade band: re-deploy the held weight to the new target
    ONLY when the target drifts more than `drift_band` (relative to the current held
    weight) AND the change clears `min_trade` (the minimum-trade floor). This is the
    parent's pre-registered 20%-drift band + $50 floor, applied per market. Purely
    backward-looking (held weight at i depends only on info through i)."""
    vals = np.asarray(w.values, dtype=float)
    held = np.empty_like(vals)
    cur = 0.0
    for i in range(len(vals)):
        tgt = vals[i]
        if np.isnan(tgt):
            held[i] = cur
            continue
        thresh = max(drift_band * abs(cur), min_trade)
        if abs(tgt - cur) > thresh:
            cur = tgt
        held[i] = cur
    return pd.Series(held, index=w.index)


# --------------------------------------------------------------------------- #
# LEG BUILDERS (the only novel code = the signals)                            #
# --------------------------------------------------------------------------- #
def _trend_block(prices, p, cost_bps, name):
    """Canonical per-market TS-trend, inverse-vol scaled to target_vol, equal 1/7 weight,
    routed through the parent's 20%-drift no-trade band + min-trade floor, LAGGED 1 day,
    charged `cost_bps` on turnover. Returns (net Series, per-market contrib)."""
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    sig = np.sign(prices / prices.shift(p["lookback_trend"]) - 1.0)          # +1/-1 trend
    vol = rets.rolling(p["vol_lb"], min_periods=max(20, p["vol_lb"] // 2)).std()
    dtv = p["target_vol"] / np.sqrt(252.0)
    w = (sig * (dtv / vol)).clip(-3.0, 3.0) / float(N_SLEEVE)                 # equal 1/7
    # parent's pre-registered no-trade band + minimum-trade floor, per market (causal)
    w = w.apply(lambda col: _apply_band(col, p["drift_band"], p["min_trade"]))
    w = w.shift(1)                                                            # lag execution
    net = net_of_cost(w, rets, cost_bps=cost_bps, name=name)
    return net, (w * rets)


def _legs(panel, p):
    """Amihud net, ETF-sleeve net, crypto-sleeve net, per-market contribs, W, eq_ret, smap."""
    eq_close, eq_vol = panel["eq_close"], panel["eq_vol"]
    smap = panel.attrs.get("sector_map", {})

    # --- Leg A: Amihud illiquidity (LONG high illiquidity) -----------------
    eq_ret = eq_close.pct_change().replace([np.inf, -np.inf], np.nan)
    dvol = (eq_close * eq_vol).replace(0.0, np.nan)
    illiq = (eq_ret.abs() / dvol).replace([np.inf, -np.inf], np.nan)
    illiq = illiq.rolling(p["lb_illiq"], min_periods=max(5, p["lb_illiq"] // 2)).mean()
    sig = xs_zscore(illiq)                                                    # +z = illiquid
    # inv_vol_position returns weekly-held, ALREADY-LAGGED inverse-vol positions; the kit's
    # weekly rebalance is the Amihud leg's own (parent) no-trade cadence + cost is charged.
    W = inv_vol_position(sig, eq_ret, p["target_vol"], 63, p["max_pos"], p["rebalance"])
    amihud = net_of_cost(W, eq_ret, cost_bps=p["eq_cost_bps"], name="amihud")

    # --- Leg B: trend crisis-alpha sleeve (5 ETF + 2 crypto, differential cost) ----
    etf_net, etf_c = _trend_block(panel["etf"], p, p["etf_cost_bps"], "etf_trend")
    cry_net, cry_c = _trend_block(panel["crypto"], p, p["crypto_cost_bps"], "crypto_trend")
    return amihud, etf_net, cry_net, etf_c, cry_c, W, eq_ret, smap


def _legs_cached(panel, p):
    key = (id(panel), tuple(sorted(p.items())))
    if key not in _LEG_CACHE:
        _LEG_CACHE[key] = _legs(panel, p)
    return _LEG_CACHE[key]


def _vol_match_blend(amihud, sleeve, weight):
    """Scale the sleeve to the Amihud leg's TRAILING vol (lagged -> no lookahead),
    then add it at `weight` of book risk."""
    va = amihud.rolling(126, min_periods=60).std()
    vs = sleeve.rolling(126, min_periods=60).std()
    scale = (va / vs).shift(1).replace([np.inf, -np.inf], np.nan).clip(upper=5.0).fillna(0.0)
    return amihud.add(weight * (sleeve * scale), fill_value=0.0)


# --------------------------------------------------------------------------- #
# SIGNAL                                                                       #
# --------------------------------------------------------------------------- #
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    amihud, etf_net, cry_net, _ec, _cc, W, eq_ret, smap = _legs_cached(panel, p)
    sleeve = etf_net.add(cry_net, fill_value=0.0)                 # full 7-market sleeve
    combined = _vol_match_blend(amihud, sleeve, p["sleeve_weight"])
    combined = combined.reindex(amihud.index).dropna()
    combined.name = SID
    # CONTRACT ledger = the Amihud gated core (real sectors + regime stamps). The
    # trend/crypto sleeve is a 25% risk overlay (in daily_returns, not in the ledger).
    trades = trades_from_weights(W, eq_ret, smap)
    return combined, trades


# --------------------------------------------------------------------------- #
# SOFT EXPECTATIONS (machine-checkable pre-registered mechanism claims)        #
# --------------------------------------------------------------------------- #
def _check_independence(ctx):
    """Claim: crypto-leg daily corr to BOTH Amihud leg AND the ETF sleeve <= +0.3."""
    panel, hs = ctx["panel"], pd.Timestamp(ctx["holdout_start"])
    a, e, c, *_ = _legs_cached(panel, DEFAULTS)
    m = pd.concat({"a": a, "e": e, "c": c}, axis=1).dropna()
    m = m[m.index < hs]
    if len(m) < 60 or m["c"].std() == 0:
        return {"pass": False, "observed": "insufficient_overlap"}
    obs = max(float(m["c"].corr(m["a"])), float(m["c"].corr(m["e"])))
    return {"pass": bool(obs <= 0.30), "observed": round(obs, 3)}


def _neff(df):
    """Effective breadth = 1 / HHI of normalised correlation eigenvalues."""
    C = np.nan_to_num(df.corr().values, nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    s = lam.sum()
    if s <= 0:
        return float("nan")
    wn = lam / s
    return float(1.0 / np.sum(wn ** 2))


def _check_breadth_gain(ctx):
    """Claim: adding the crypto leg RAISES the trend sleeve's effective breadth."""
    panel, hs = ctx["panel"], pd.Timestamp(ctx["holdout_start"])
    _a, _e, _c, ec, cc, *_ = _legs_cached(panel, DEFAULTS)
    ec, cc = ec[ec.index < hs], cc[cc.index < hs]
    n_base = _neff(ec)                          # 5 TradFi markets only
    n_full = _neff(pd.concat([ec, cc], axis=1))  # + BTC/ETH
    gain = round(float(n_full - n_base), 3)
    return {"pass": bool(n_full > n_base), "observed": gain}


def _check_sharpe_dilution(ctx):
    """Claim: net-Sharpe dilution of the crypto-augmented book vs the crypto-free
    parent (Amihud + ETF-only sleeve) is <= 5% over the search window."""
    panel, hs = ctx["panel"], pd.Timestamp(ctx["holdout_start"])
    a, e, c, *_ = _legs_cached(panel, DEFAULTS)
    parent = _vol_match_blend(a, e, DEFAULTS["sleeve_weight"])
    child = _vol_match_blend(a, e.add(c, fill_value=0.0), DEFAULTS["sleeve_weight"])
    parent = parent[parent.index < hs].dropna()
    child = child[child.index < hs].dropna()

    def shp(x):
        return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0

    sp, sc = shp(parent), shp(child)
    dil = (sp - sc) / abs(sp) if sp != 0 else 1.0
    return {"pass": bool(dil <= 0.05), "observed": round(float(dil), 3)}


# --------------------------------------------------------------------------- #
# SPEC                                                                         #
# --------------------------------------------------------------------------- #
SPEC = StrategySpec(
    id=SID,
    family="multi_premium_combination",
    title="Illiquidity-Premium x Trend Crisis-Alpha (crypto-breadth-diversified): "
          "Amihud small-cap leg + 5-ETF TS-trend sleeve + additive BTC/ETH TS-trend "
          "breadth source, sized as a 25% crisis-hedge overlay.",
    markets=["us_equity_smallcap", "etf_cross_asset", "crypto_btc_eth"],
    data_desc="Sharadar SEP closeadj+volume (small-cap cross-section, owned, "
              "survivorship-clean) for the Amihud leg; yfinance daily closes for "
              "SPY/EFA/TLT/GLD/DBC and BTC-USD/ETH-USD (free) for the trend sleeve.",
    pre_registration=(
        "HYPOTHESIS: the Amihud illiquidity premium (long high-illiquidity small caps) is "
        "pro-cyclical and crashes when liquidity evaporates; a time-series-trend sleeve earns "
        "in those exact stress regimes, so the COMBINATION dominates either leg. The parent "
        "crisis-hedge sleeve is all-TradFi (SPY/EFA/TLT/GLD/DBC) whose members co-correlate in "
        "a global liquidity crisis (Mar-2020 'sell everything for cash') -> its EFFECTIVE breadth "
        "is < 5 precisely when it must hedge (a Fundamental-Law fragility). MUTATION: add ONE "
        "small canonical TS-trend leg on BTC/ETH -- a trend market whose drivers (crypto-native "
        "deleveraging, 24-7 liquidation cascades) are structurally independent of TradFi forced "
        "selling -- to raise the crisis-alpha leg's effective breadth with a decorrelated source. "
        "FROZEN DESIGN: single canonical TS-trend rule (lookback=100, trailing-60d inverse-vol to "
        "10% target) applied per-market across all 7 markets at an EQUAL 1/7 risk weight, EACH "
        "routed through the parent's 20%-drift no-trade band + minimum-trade floor; the whole "
        "sleeve is sized to 25% of book risk vs the Amihud leg's 100% (crypto = ~2/7 of 25% -> "
        "additive, never dominant). Costs net: 8bps equity/ETF turnover, 20bps crypto round-trip "
        "taker. Every signal lagged 1 day. SUCCESS (vs the crypto-free parent, post-2018 overlap, "
        "all net): (i) crypto-leg daily corr <= +0.3 to BOTH other legs [independence], (ii) the "
        "sleeve's effective-breadth proxy rises with crypto, (iii) net-Sharpe dilution <= 5%. If "
        "the crypto leg's stress-window sign dies under throttling/cost, it adds no independent "
        "crisis-alpha and is DROPPED -- itself a fundable finding -- leaving the validated parent "
        "intact. HONESTY: this is a clean KIT reconstruction of the Amihud leg (not a literal "
        "byte-frozen artefact); the CONTRACT ledger is the Amihud gated core only (the ETF/crypto "
        "sleeve is a risk overlay carrying no equity-name deployment exposure); crypto is reindexed "
        "to the equity business-day calendar; the $50 floor is approximated by a relative "
        "min-weight-change floor in this return-series harness. Claims (i)-(iii) are declared "
        "machine-checkable below."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "trend_slow": {"lookback_trend": 200},
        "trend_fast": {"lookback_trend": 63},
        "illiq_3m": {"lb_illiq": 63},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {"name": "crypto_leg_independence",
         "claim": "crypto-leg daily corr to BOTH the Amihud leg and the ETF sleeve <= +0.30",
         "check": _check_independence},
        {"name": "effective_breadth_gain",
         "claim": "adding BTC/ETH raises the sleeve's effective breadth (1/HHI of leg-corr eigvals)",
         "check": _check_breadth_gain},
        {"name": "sharpe_dilution_within_5pct",
         "claim": "net-Sharpe dilution of crypto-augmented book vs crypto-free parent <= 5%",
         "check": _check_sharpe_dilution},
    ],
)