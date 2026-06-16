from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1, yf_panel, fred_series, trend_returns, inv_vol_position
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel
import numpy as np, pandas as pd

# ======================================================================================
# Processing-margin CONVERGENCE book (soy crush + cattle feedlot), run as ONE equal-risk
# market-neutral futures-spread portfolio with a depth-tail entry, hysteresis exit band,
# min-hold, and a SHORT-HORIZON REALISED-CONVERGENCE CONFIRMATION entry gate.
#
# HONEST THESIS NOTE (read pre_registration): an earlier draft labelled the entry gate a
# "term-structure roll-carry" gate (near-slot vs second-deferred-slot GPM). That is NOT what
# this code can compute -- true term-structure carry needs per-EXPIRY contract months
# (fut_curve / Databento), which is NOT in the harness's tested adapter list. The only futures
# source available here is yf_panel = continuous FRONT-MONTH, from which the term structure
# cannot be recovered. So the gate is RE-STATED HONESTLY for exactly what the data supports and
# what the code does: a short-horizon REALISED-CONVERGENCE CONFIRMATION -- enter a depth-tail
# dislocation only once the board margin has ALREADY begun drifting back toward fair over the
# last ~10 days (do not fade a falling knife; harvest the convergence as it is realised, not
# predicted). NO term-structure carry is claimed anywhere. The soft-expectations FALSIFY the
# turnover/Sharpe mechanism story instead of asserting it in prose. The ONLY novel code below is
# the depth + confirmation state machine; sizing/cost/ledger use the kit.
# ======================================================================================

# ---- FROZEN complex recipes: margin = sum(coef*price); products (+) / inputs (-) = LONG-CRUSH ----
RECIPES = {
    # SEARCH universe
    'soy':    {'coef': {'ZM=F': 0.022, 'ZL=F': 0.11, 'ZS=F': -0.01}},           # $/bu board crush
    'cattle': {'coef': {'LE=F': 12.5, 'GF=F': -7.5, 'ZC=F': -0.52}},            # $/head feedlot margin
    # GENERALIZATION universes (DISJOINT from soy+cattle search roots) -- physically-anchored
    # convergence spreads tested by the IDENTICAL frozen signal:
    'energy_crack':      {'coef': {'RB=F': 28.0, 'HO=F': 14.0, 'CL=F': -1.0}},  # 3:2:1 refining crack $/bbl
    'wheat_convergence': {'coef': {'KE=F': 1.0, 'ZW=F': -1.0}},                 # KC vs Chicago wheat
    'brent_wti':         {'coef': {'BZ=F': 1.0, 'CL=F': -1.0}},                 # Brent-WTI grade spread
}
for _c in RECIPES:
    RECIPES[_c]['tickers'] = set(RECIPES[_c]['coef'])

SEARCH_COMPLEXES = ['soy', 'cattle']
GEN_UNIVERSES = ['energy_crack', 'wheat_convergence', 'brent_wti']
START = "2005-01-01"

DEFAULTS = dict(
    complexes=None,            # None -> auto-detect complexes FULLY present in the panel
    lookback=252,             # depth-percentile window (trading days)
    depth_lo=0.15, depth_hi=0.85, exit_pct=0.50,          # entry tails + median hysteresis exit
    trend_lb=10, trend_hurdle=0.005,                       # realised-convergence confirmation: 10d margin drift; 0.5%/10d ~ 2x round-trip cost
    trend_flip_days=3, min_hold=10, max_hold=60,           # fast-exit / min-hold / max-hold (trading days)
    vol_lb=20, vol_gate_q=0.90, use_vol_gate=True,         # no new entry if 20d spread vol in top decile
    trend_gate=True,                                       # single-knob ablation for the depth-only head-to-head
    target_dvol=0.01, max_size=4.0, cost_bps=8.0, reb='W-FRI',
)


# ----------------------------- data adapters -----------------------------
def load_data() -> pd.DataFrame:
    tickers = sorted({t for c in SEARCH_COMPLEXES for t in RECIPES[c]['tickers']})
    return yf_panel(tickers, start=START)


def load_gen_data(label) -> pd.DataFrame:
    return yf_panel(sorted(RECIPES[label]['tickers']), start=START)


# ----------------------------- helpers (only the signal is novel) -----------------------------
def _rolling_pctile(s, lb):
    # fraction of trailing lb obs <= current (inclusive), as-of close -> no look-ahead
    return s.rolling(lb, min_periods=lb).apply(lambda a: float((a <= a[-1]).mean()), raw=True)


def _margin_and_gross(panel, coef):
    legs = list(coef)
    px = panel[legs].astype(float)
    valid = px.notna().all(axis=1)
    margin = sum(coef[l] * px[l] for l in legs).where(valid)
    gross = sum(abs(coef[l]) * px[l].abs() for l in legs).where(valid)
    return px, margin, gross, valid


def _state_machine(margin, gross, p):
    idx = margin.index
    spread_ret = (margin.diff() / gross.shift(1)).fillna(0.0)
    rv = spread_ret.rolling(p['vol_lb'], min_periods=p['vol_lb']).std()
    rv_thr = rv.rolling(p['lookback'], min_periods=p['lookback'] // 2).quantile(p['vol_gate_q'])
    depth = _rolling_pctile(margin, p['lookback'])
    # short-horizon REALISED drift of the board margin (NOT term-structure carry; see note):
    # "has the dislocation already begun converging back toward fair over the last ~10 days?"
    trend = (margin - margin.shift(p['trend_lb'])) / gross.shift(p['trend_lb'])

    dep, trn, rvv, rvt = depth.values, trend.values, rv.values, rv_thr.values
    val = margin.notna().values
    h, gate = p['trend_hurdle'], p['trend_gate']
    pos = np.zeros(len(idx))
    state = held = flip = 0
    for i in range(len(idx)):
        if (not val[i]) or np.isnan(dep[i]):
            pos[i] = state                       # hold through data gaps
            continue
        if state == 0:
            volok = True
            if p['use_vol_gate'] and not np.isnan(rvt[i]):
                volok = not (rvv[i] > rvt[i])
            # confirmation: go long a deep-low margin only once it has begun rising back to fair;
            # short a deep-high margin only once it has begun falling (don't fade a falling knife)
            clong = (not gate) or (not np.isnan(trn[i]) and trn[i] > h)
            cshort = (not gate) or (not np.isnan(trn[i]) and trn[i] < -h)
            if volok and dep[i] < p['depth_lo'] and clong:
                state, held, flip = 1, 0, 0
            elif volok and dep[i] > p['depth_hi'] and cshort:
                state, held, flip = -1, 0, 0
        else:
            held += 1
            if gate and not np.isnan(trn[i]):
                against = (trn[i] < 0) if state == 1 else (trn[i] > 0)
                flip = flip + 1 if against else 0
            else:
                flip = 0
            exit_now = False
            if gate and flip >= p['trend_flip_days']:
                exit_now = True                  # confirmation reversed -> reversion thesis broke, fast-exit (exempt from min-hold)
            elif held >= p['min_hold']:
                if state == 1 and dep[i] >= p['exit_pct']:
                    exit_now = True              # converged back to median
                elif state == -1 and dep[i] <= (1.0 - p['exit_pct']):
                    exit_now = True
                elif held >= p['max_hold']:
                    exit_now = True
            if exit_now:
                state, held, flip = 0, 0, 0
        pos[i] = state
    return pd.Series(pos, index=idx), spread_ret, rv


def _complex_book(panel, name, p):
    coef = RECIPES[name]['coef']
    if not RECIPES[name]['tickers'] <= set(panel.columns):
        return None
    px, margin, gross, valid = _margin_and_gross(panel, coef)
    if int(valid.sum()) < p['lookback'] + p['trend_lb'] + 40:
        return None                              # not enough history to form the signal honestly
    pos, spread_ret, rv = _state_machine(margin, gross, p)
    size = (p['target_dvol'] / rv.replace(0.0, np.nan)).clip(upper=p['max_size']).fillna(0.0)  # inverse-vol equal-risk
    denom = gross.replace(0.0, np.nan)
    W = pd.DataFrame(0.0, index=panel.index, columns=list(coef))
    for l in coef:
        wl = coef[l] * px[l] / denom             # signed notional share, sum|.|=1 (long-crush convention)
        W[l] = (pos * size * wl).fillna(0.0)
    return W


# ----------------------------- the signal -----------------------------
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    if isinstance(panel, pd.Series):
        panel = panel.to_frame()
    cols = set(panel.columns)
    if p['complexes'] is not None:
        names = [c for c in p['complexes'] if RECIPES[c]['tickers'] <= cols]
    else:
        names = [c for c in RECIPES if RECIPES[c]['tickers'] <= cols]   # auto-detect (no subset collisions by design)

    parts, sector_map = [], {}
    for name in names:
        W = _complex_book(panel, name, p)
        if W is None:
            continue
        parts.append(W)
        for l in W.columns:
            sector_map[l] = name
    if not parts:
        return pd.Series(dtype=float, name='proc_margin_conv'), []

    W = pd.concat(parts, axis=1).fillna(0.0)                 # legs disjoint across complexes -> no overlap
    Wk = W.resample(p['reb']).last().reindex(W.index, method='ffill').fillna(0.0)  # weekly rebalance
    Wl = Wk.shift(1).fillna(0.0)                              # 1-day execution lag is OUR responsibility

    legs = list(W.columns)
    R = panel[legs].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    R, Wl = R.align(Wl, join='inner', axis=0)

    rets = net_of_cost(Wl, R, cost_bps=p['cost_bps'], name='proc_margin_conv')    # cost on EVERY leg's turnover
    trades = trades_from_weights(Wl, R, sector_map)                                # kit stamps entry_regime
    return rets, trades


# ----------------------------- soft expectations (machine-checkable mechanism claims) -----------------------------
def _sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 30 or r.std() == 0:
        return float('nan')
    return float(r.mean() / r.std() * np.sqrt(252.0))


def _chk_conf_sharpe(ctx):
    d = _sharpe(ctx['search'])
    g = ctx['grid'].get('no_trend_gate')
    n = _sharpe(g) if g is not None else float('nan')
    ok = bool(np.isfinite(d) and np.isfinite(n) and d >= n)
    return {"pass": ok, "observed": "default=%.3f no_gate=%.3f" % (d, n)}


def _chk_conf_turnover(ctx):
    h = str(ctx['holdout_start'])
    nd = sum(1 for t in ctx['trades'] if str(t.get('entry_date', '')) < h)   # confirmation-gated (search window)
    _, ung = signal(ctx['panel'], trend_gate=False)                          # one extra signal() call
    nn = sum(1 for t in ung if str(t.get('entry_date', '')) < h)             # sliced to search window
    return {"pass": bool(nd <= nn), "observed": "gated=%d ungated=%d" % (nd, nn)}


def _chk_subbooks(ctx):
    s = _sharpe(ctx['grid'].get('soy_only'))
    c = _sharpe(ctx['grid'].get('cattle_only'))
    ok = bool(np.isfinite(s) and np.isfinite(c) and s > 0 and c > 0)
    return {"pass": ok, "observed": "soy=%.3f cattle=%.3f" % (s, c)}


# ----------------------------- spec -----------------------------
SPEC = StrategySpec(
    id="proc_margin_convergence",
    family="futures_convergence",
    title="Processing-margin convergence book (soy + cattle crush): depth-tail entry + realised-convergence confirmation, hysteresis + min-hold, cross-complex equal-risk",
    markets=["futures"],
    data_desc=("yfinance FRONT-MONTH continuous daily Close (no per-expiry term structure available). Search legs: soy crush "
               "ZS/ZM/ZL, cattle feedlot LE/GF/ZC. Gen legs: energy crack CL/RB/HO, wheat KE/ZW, Brent-WTI BZ/CL. Board margin "
               "GPM = sum(frozen recipe coef * price); spread P&L = inverse-vol-scaled signed basket of the legs (market-neutral)."),
    pre_registration=(
        "MECHANISM: liquidity/insurance provision to commercial hedging flow inside a physical processing complex (soy crush, "
        "cattle feedlot). We harvest the CONVERGENCE of a DISLOCATED board margin back toward fair value, as a market-neutral "
        "inter-commodity SPREAD, not a directional bet. Entry only when (a) the board margin is in its deep-dislocation tail "
        "(252d depth percentile) AND (b) a short-horizon REALISED-CONVERGENCE CONFIRMATION holds: the margin has ALREADY begun "
        "drifting back toward fair over the last ~10 days (do not fade a falling knife; harvest the convergence as it is realised, "
        "not predicted).\n"
        "HONEST DATA LIMITATION (disclosed, not spun): an earlier draft called gate (b) a 'term-structure roll-carry' gate "
        "(near-slot vs second-deferred-slot GPM). That is NOT implementable here -- true term-structure carry needs per-EXPIRY "
        "contract months (fut_curve/Databento), which is NOT in the harness's tested adapter list; the only futures source is "
        "yf_panel = continuous FRONT-MONTH, from which the term structure CANNOT be recovered. The gate is therefore RE-STATED for "
        "exactly what the code computes: the realised 10-day drift of the board margin (hurdle 0.5%/10d ~ 2x round-trip cost). NO "
        "term-structure carry is claimed; gate (b) is a short-horizon momentum-of-convergence TIMING filter, and its mechanism "
        "effect is CHECKED (soft expectations), not asserted.\n"
        "FROZEN PARAMS: depth percentile vs trailing 252d; long-crush entry depth<15th, short-crush entry depth>85th; HYSTERESIS "
        "exit at the 50th-pct median; min-hold 10d, max-hold 60d; 20d realised-spread-vol top-decile entry gate; confirmation-flip "
        "fast-exit after 3 consecutive against-days (exempt from min-hold). Two complexes combined into ONE equal-risk inverse-vol "
        "book. Weekly rebalance, 1-day execution lag, ~8bps cost on EVERY leg's turnover each entry/exit/roll.\n"
        "KILL CONDITION (soft, recorded on verdict): the confirmation-gated book MUST show (1) trade count <= ungated depth-only "
        "construction and (2) net Sharpe >= ungated, on identical data; and BOTH soy-only and cattle-only sub-books must be "
        "independently positive. If not, the mutation is falsified.\n"
        "SCOPE broad: 'physically-anchored processing-margin convergence' is a universal claim, so the IDENTICAL frozen signal is "
        "auto-applied to 3 DISJOINT convergence complexes on holdout (energy 3:2:1 refining crack, KC-vs-Chicago wheat, Brent-WTI "
        "grade spread); >=60% OOS-positive required or it is rejected as a soy/cattle-specific artifact. NOTE: energy_crack and "
        "brent_wti share CL (correlated); wheat_convergence is independent; none overlap the soy/cattle search legs."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "no_trend_gate": {"trend_gate": False},     # depth-only head-to-head (single-knob ablation)
        "soy_only": {"complexes": ["soy"]},
        "cattle_only": {"complexes": ["cattle"]},
    },
    scope="broad",
    generalization_universes=GEN_UNIVERSES,
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=6,
    expectations=[
        {"name": "confirmation_raises_sharpe",
         "claim": "confirmation-gated book net Sharpe >= ungated (depth-only) book on the search window",
         "check": _chk_conf_sharpe},
        {"name": "confirmation_cuts_turnover",
         "claim": "confirmation-gated book trade count <= ungated depth-only book (fewer falling-knife entries)",
         "check": _chk_conf_turnover},
        {"name": "both_subbooks_positive",
         "claim": "soy-only AND cattle-only sub-books each independently positive Sharpe (one-complex result = fail)",
         "check": _chk_subbooks},
    ],
)