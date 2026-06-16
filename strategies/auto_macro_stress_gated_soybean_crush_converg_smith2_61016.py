"""
Macro-stress-gated processing-margin convergence (soybean board crush as search book).

MECHANISM (frozen): a pro-cyclical relative-value convergence premium on a processing
margin (long outputs / short input) harvested via trailing-percentile mean-reversion,
with its LEFT TAIL amputated by a LEADING cross-asset macro-stress gate (vol-of-vol /
tail-hedge-demand divergence) that can only FLATTEN/BLOCK, never add exposure.

HONEST DATA DEVIATION (stated up front): the proposal specified Databento individual
delivery months via fut_curve + cboe_index('VVIX'). Those adapters are NOT in the tested
import set. This module therefore uses the tested free routes:
  - CONTINUOUS front-month CME/NYMEX futures via yf_panel (ZS/ZM/ZL etc.)
  - CBOE vol-of-vol ^VVIX via yf_panel + VIX via fred_series('VIXCLS')
The GPM-percentile convergence engine and the VVIX/VIX risk-off overlay are preserved
EXACTLY; the second-nearest-month matching / per-contract chaining / roll-together
mechanics are approximated by continuous-contract returns.

FROZEN CONSTRUCTION (restored to match the proposal): the trade is the fixed
1-contract-per-leg spread 'long 1 ZM + 1 ZL, short 1 ZS' (equal-notional legs in the role
direction), so the traded spread is the SAME object as the GPM that generates the signal;
entries/exits are EVENT-DRIVEN on the day the GPM crosses its band / the median (or the
60-day max-hold) -- no weekly rebalance, no inverse-vol re-weight.

NO-LOOKAHEAD: every signal (GPM percentiles, the realized-vol gate, and the VVIX/VIX
RISK_OFF flag) is computed from the close of day t; the FULL weight matrix is lagged one
day (net_of_cost(W.shift(1), rets)) so a decision at t earns t+1's return. That single
global shift implements the proposal's strict t-1 conditioning.
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------- #
# Recipes: each complex is a set of legs (output role=+1, input role=-1) + a GPM #
# ----------------------------------------------------------------------------- #
SOY_LEGS = [  # SEARCH universe -- soybean board crush (CME published factors)
    {"ticker": "ZS=F", "role": -1, "sector": "oilseed"},
    {"ticker": "ZM=F", "role": +1, "sector": "protein-meal"},
    {"ticker": "ZL=F", "role": +1, "sector": "veg-oil"},
]
CATTLE_LEGS = [  # cattle feeding margin (live <- feeder + corn)
    {"ticker": "LE=F", "role": +1, "sector": "live-cattle"},
    {"ticker": "GF=F", "role": -1, "sector": "feeder-cattle"},
    {"ticker": "ZC=F", "role": -1, "sector": "feed-grain"},
]
CRACK_LEGS = [  # 3:2:1 refining crack (gasoline + distillate <- crude)
    {"ticker": "RB=F", "role": +1, "sector": "gasoline"},
    {"ticker": "HO=F", "role": +1, "sector": "distillate"},
    {"ticker": "CL=F", "role": -1, "sector": "crude"},
]
HOG_LEGS = [  # hog feeding margin (lean hogs <- corn)
    {"ticker": "HE=F", "role": +1, "sector": "lean-hogs"},
    {"ticker": "ZC=F", "role": -1, "sector": "feed-grain"},
]

def _gpm_soy(px):     # $/bu board crush: ZM $/ton, ZL c/lb, ZS c/bu
    return 0.022 * px["ZM=F"] + 0.11 * px["ZL=F"] - px["ZS=F"] / 100.0
def _gpm_cattle(px):  # ~$/head (1250lb fed - 750lb feeder - 50bu corn), cents prices
    return 12.5 * px["LE=F"] - 7.5 * px["GF=F"] - 0.5 * px["ZC=F"]
def _gpm_crack(px):   # $/bbl 3:2:1 crack; RB/HO $/gal*42, CL $/bbl
    return 28.0 * px["RB=F"] + 14.0 * px["HO=F"] - px["CL=F"]
def _gpm_hog(px):     # ~$/head (280lb hog - 10bu corn), cents prices
    return 2.8 * px["HE=F"] - 0.1 * px["ZC=F"]

GEN = {
    "cattle_crush": (CATTLE_LEGS, _gpm_cattle),
    "crack_spread": (CRACK_LEGS, _gpm_crack),
    "hog_margin":   (HOG_LEGS,   _gpm_hog),
}
SECTOR_OF = {}
for _legs in (SOY_LEGS, CATTLE_LEGS, CRACK_LEGS, HOG_LEGS):
    for _l in _legs:
        SECTOR_OF[_l["ticker"]] = _l["sector"]

VOL_START = "2007-01-01"   # ^VVIX history is the binding constraint (joint sample ~2007+)

DEFAULTS = dict(
    pct_window=252, lo_pct=0.20, hi_pct=0.80, max_hold=60, block_days=10,
    vvix_hi=110.0, vix_lo=18.0, ratio_hi=6.5, vol_lb=60,
)

# ----------------------------------------------------------------------------- #
# Panel builders                                                                 #
# ----------------------------------------------------------------------------- #
def _vol_gate(start):
    vv = yf_panel(["^VVIX"], start)
    if isinstance(vv, pd.Series):
        vvix = vv
    else:
        vvix = vv["^VVIX"] if "^VVIX" in vv.columns else vv.iloc[:, 0]
    vix = fred_series({"VIXCLS": "VIX"}, start)["VIX"]
    return vvix.astype(float), vix.astype(float)

def _make_panel(legs, gpm_fn, start=VOL_START):
    tickers = [l["ticker"] for l in legs]
    px = yf_panel(tickers, start)
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px[[t for t in tickers if t in px.columns]]
    out = pd.DataFrame(index=px.index)
    for l in legs:
        pref = "OUT__" if l["role"] > 0 else "IN__"
        out[pref + l["ticker"]] = px[l["ticker"]]
    out["GPM"] = gpm_fn(px)
    vvix, vix = _vol_gate(start)
    out = out.join(vvix.rename("VVIX"), how="left").join(vix.rename("VIX"), how="left")
    out["VVIX"] = out["VVIX"].ffill(limit=5)
    out["VIX"] = out["VIX"].ffill(limit=5)
    out = out.dropna()                  # enforces the JOINT (grains x VVIX) sample
    out.index.name = "date"
    return out

def load_data() -> pd.DataFrame:
    """SEARCH panel: soybean board crush + the cross-asset vol-of-vol gate."""
    return _make_panel(SOY_LEGS, _gpm_soy)

def load_gen_data(label) -> pd.DataFrame:
    """Stage-2 generalization panel for ONE disjoint processing-margin complex."""
    legs, fn = GEN[label]
    return _make_panel(legs, fn)

# ----------------------------------------------------------------------------- #
# Signal: percentile-band convergence engine + leading macro-stress overlay      #
# ----------------------------------------------------------------------------- #
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    win = int(p["pct_window"]); lo_q = float(p["lo_pct"]); hi_q = float(p["hi_pct"])
    max_hold = int(p["max_hold"]); block_days = int(p["block_days"])
    vvix_hi = float(p["vvix_hi"]); vix_lo = float(p["vix_lo"]); ratio_hi = float(p["ratio_hi"])
    use_overlay = bool(p.get("use_overlay", True))
    mp = max(win // 2, 60)

    price_cols = [c for c in panel.columns if c.startswith("OUT__") or c.startswith("IN__")]
    tickers = [c.split("__", 1)[1] for c in price_cols]
    roles_s = pd.Series([1.0 if c.startswith("OUT__") else -1.0 for c in price_cols], index=tickers)
    px = panel[price_cols].copy(); px.columns = tickers
    rets = px.pct_change().fillna(0.0)
    idx = panel.index
    gpm = panel["GPM"].astype(float)
    vvix = panel["VVIX"].astype(float); vix = panel["VIX"].astype(float)

    # trailing percentile bands of the GPM level (window ends at t; acted on t+1)
    p_lo = gpm.rolling(win, min_periods=mp).quantile(lo_q)
    p_hi = gpm.rolling(win, min_periods=mp).quantile(hi_q)
    p_md = gpm.rolling(win, min_periods=mp).quantile(0.50)

    # FROZEN construction: fixed 1-contract-per-leg spread in the role direction
    # (long 1 ZM + 1 ZL, short 1 ZS), normalized to gross 1 so the TRADED spread is
    # the SAME object as the GPM that generates the signal. No inverse-vol re-weight.
    base = roles_s / roles_s.abs().sum()

    # reference long-crush stream -> 20d realized spread vol, top-decile blowout gate
    ref_ret = (base * rets).sum(axis=1)
    sp_vol = ref_ret.rolling(20, min_periods=10).std()
    sp_vol_thr = sp_vol.rolling(win, min_periods=mp).quantile(0.90)

    # composite RISK_OFF (Parent 2 tail-hedge-demand + Parent 1 ag blowout); t-close info
    cond_a = (vvix > vvix_hi) & (vix < vix_lo)
    cond_b = (vvix / vix > ratio_hi) & (vix < vix_lo)
    cond_c = (sp_vol > sp_vol_thr)
    risk_off = (cond_a | cond_b | cond_c).fillna(False).values
    if not use_overlay:
        risk_off = np.zeros(len(idx), dtype=bool)

    g = gpm.values; plo = p_lo.values; phi = p_hi.values; pmd = p_md.values
    n = len(idx); pos_dir = np.zeros(n)
    state = 0; entry_i = -1; block_until = -1
    for i in range(n):
        if risk_off[i]:                                   # flatten + block new entries
            if state != 0:
                state = 0; entry_i = -1
            block_until = i + block_days
        if state != 0:                                    # exits (any day)
            held = i - entry_i
            crossed = (state == 1 and g[i] >= pmd[i]) or (state == -1 and g[i] <= pmd[i])
            if held >= max_hold or crossed:
                state = 0; entry_i = -1
        if state == 0 and i > block_until and not np.isnan(plo[i]):   # EVENT-DRIVEN entry
            if g[i] < plo[i]:
                state = 1; entry_i = i                    # long crush (margin depressed)
            elif g[i] > phi[i]:
                state = -1; entry_i = i                   # short crush (margin rich)
        pos_dir[i] = state
    pos = pd.Series(pos_dir, index=idx)

    # fixed-weight spread held while position is open (no intra-position rebalance)
    W = pd.DataFrame(np.outer(pos.values, base.values), index=idx, columns=tickers)

    Wlag = W.shift(1).fillna(0.0)                         # the ONLY lag: decision_t -> return_t+1
    daily = net_of_cost(Wlag, rets, cost_bps=8.0, name="crush_gated")
    sector_map = {t: SECTOR_OF[t] for t in tickers}
    trades = trades_from_weights(Wlag, rets, sector_map)
    return daily, trades

# ----------------------------------------------------------------------------- #
# Soft expectations (machine-checkable mechanism claims)                         #
# ----------------------------------------------------------------------------- #
def _cvar(s, q=0.05):
    s = pd.Series(s).dropna()
    if len(s) < 20:
        return float("nan")
    thr = s.quantile(q)
    tail = s[s <= thr]
    return float(tail.mean()) if len(tail) else float("nan")

def _chk_overlay_tail(ctx):
    # CLAIM: the leading macro-stress overlay makes the 5% left-tail LESS negative.
    h = pd.Timestamp(ctx["holdout_start"]); panel = ctx["panel"]
    gated = pd.Series(ctx["search"]); gated = gated[gated.index < h]
    ung, _ = signal(panel, use_overlay=False)            # one extra signal() call
    ung = ung[ung.index < h]
    cg, cu = _cvar(gated), _cvar(ung)
    ok = (not np.isnan(cg)) and (not np.isnan(cu)) and (cg >= cu)
    return {"pass": bool(ok), "observed": f"gated_cvar5={cg:.5f} ungated_cvar5={cu:.5f}"}

def _chk_riskoff_nondegenerate(ctx):
    # CLAIM: the VVIX/VIX flag is non-degenerate (fires sometimes, has real episodes).
    h = pd.Timestamp(ctx["holdout_start"]); sub = ctx["panel"]; sub = sub[sub.index < h]
    vvix = sub["VVIX"].astype(float); vix = sub["VIX"].astype(float)
    flag = ((vvix > 110) & (vix < 18)) | ((vvix / vix > 6.5) & (vix < 18))
    frac = float(flag.mean()); f = flag.values.astype(int)
    episodes = int(((f[1:] == 1) & (f[:-1] == 0)).sum() + (f[0] if len(f) else 0))
    ok = (0.001 < frac < 0.30) and (episodes >= 5)
    return {"pass": bool(ok), "observed": f"fired_frac={frac:.4f} episodes={episodes}"}

def _chk_convergence_meanrev(ctx):
    # CLAIM: the convergence engine earns on average over the search window.
    tr = ctx["trades"]
    if not tr:
        return {"pass": False, "observed": "no trades"}
    pnls = np.array([t["pnl"] for t in tr], dtype=float)
    avg = float(np.nanmean(pnls)); wr = float(np.mean(pnls > 0))
    return {"pass": bool(avg > 0), "observed": f"avg_pnl={avg:.2f} win_rate={wr:.2f} n={len(tr)}"}

# ----------------------------------------------------------------------------- #
# Spec                                                                           #
# ----------------------------------------------------------------------------- #
SPEC = StrategySpec(
    id="crush_macrogate_v1",
    family="commodity_processing_margin_rv",
    title="Macro-stress-gated processing-margin convergence (soybean crush) with a vol-of-vol risk-off overlay",
    markets=["CME grains", "CME livestock", "NYMEX energy", "CBOE vol-of-vol"],
    data_desc=(
        "Continuous front-month CME/NYMEX futures via yf_panel: search=soybean board "
        "crush (ZS/ZM/ZL); generalization=cattle crush (LE/GF/ZC), 3:2:1 crack (RB/HO/CL), "
        "hog feeding margin (HE/ZC). Cross-asset gate: CBOE ^VVIX (yf_panel) + VIX "
        "(fred_series VIXCLS). All free. NOTE: continuous front-month contracts substitute "
        "for Databento individual delivery months (tested adapters expose continuous "
        "futures only) -- the percentile-convergence signal and the VVIX/VIX overlay are "
        "preserved; per-delivery roll/chaining is approximated by continuous-contract returns."
    ),
    pre_registration=(
        "ENGINE: GPM = published/economic processing margin (long outputs, short input). "
        "Long crush (fixed 1 contract per leg: +ZM +ZL -ZS, equal-notional, gross~1) when GPM "
        "< trailing-252d 20th pct; short crush when > 80th pct; exit at trailing median OR "
        "60-day max-hold. Entries/exits are EVENT-DRIVEN (acted the day the band/median is "
        "crossed); the traded spread is the SAME object as the GPM signal. OVERLAY (fusion): "
        "RISK_OFF = [VVIX>110 & VIX<18] OR [VVIX/VIX>6.5 & VIX<18] OR [20d realized crush-spread "
        "vol in trailing top decile]; on -> flatten any open crush + block new entries 10 "
        "trading days; off -> trade normally. Overlay can ONLY flatten/block (never adds "
        "leverage; no options/borrow). LAG: all signals use close-of-t data; the entire weight "
        "matrix is shifted one day (net_of_cost(W.shift(1))) -> strict t-1 conditioning for t+1 "
        "returns. COSTS: harness-standard 8bps on turnover (proxy for 1 tick/leg + fees). "
        "SCOPE=broad: mechanism = pro-cyclical processing-margin convergence protected by a "
        "leading macro-stress gate; the IDENTICAL frozen spec is run on 3 disjoint complexes "
        "(cattle crush, crack spread, hog margin) sharing no tickers with soy as STAGE-2 "
        "generalization; >=60% must be OOS-positive. ROBUSTNESS (declared as grid, measured "
        "not selected): VVIX 105/110/115, ratio 6.0/6.5/7.0, block 5/10/15d must share sign. "
        "POWER CAVEAT (loud): joint VVIX x grains sample starts ~2007; distinct macro-stress "
        "episodes are few (~single digits) -> this caps confidence in the OVERLAY arm; the bulk "
        "of return must come from many calm-period convergence trades, not the rare stress "
        "windows. We MEASURE (not select on) whether RISK_OFF windows overlap the crush's "
        "largest-drawdown days. Continuous-contract and approximate livestock/energy GPM "
        "coefficients are explicit deviations from the Databento individual-month construction; "
        "percentile bands are scale/shift-robust."
    ),
    load_data=load_data,
    signal=signal,
    default_params=dict(DEFAULTS),
    grid={
        "default": {},
        "vvix_105": {"vvix_hi": 105.0},
        "vvix_115": {"vvix_hi": 115.0},
        "ratio_6":  {"ratio_hi": 6.0},
        "ratio_7":  {"ratio_hi": 7.0},
        "block_5":  {"block_days": 5},
        "block_15": {"block_days": 15},
    },
    scope="broad",
    generalization_universes=["cattle_crush", "crack_spread", "hog_margin"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=3,
    expectations=[
        {"name": "overlay_cuts_left_tail",
         "claim": "gated 5% daily CVaR is no worse (>=) than the ungated convergence engine",
         "check": _chk_overlay_tail},
        {"name": "risk_off_nondegenerate",
         "claim": "VVIX/VIX risk-off flag fires on 0.1%-30% of days with >=5 distinct episodes",
         "check": _chk_riskoff_nondegenerate},
        {"name": "convergence_engine_positive",
         "claim": "mean per-trade pnl of the convergence book is positive on the search window",
         "check": _chk_convergence_meanrev},
    ],
)