"""
Physical-margin CONVERGENCE book — sanctioned-data rebuild.

WHY THIS DIFFERS FROM THE PRIOR (FAILED) VERSION
  The prior module called `fut_curve(...)`, an adapter that is NOT in the tested set and whose
  assumed signature was wrong (it crashed inside the adapter: `int64 <= str`). The sanctioned
  adapters expose CONTINUOUS FRONT-MONTH futures (yf_panel) but NOT individual contract months,
  so the term-structure CARRY leg cannot be measured honestly. Rather than fabricate it, this
  frozen version tests the CONVERGENCE premium STANDALONE on board margins built from continuous
  front-month settles. The carry claim is moved out of pre-registration and its soft-check dropped
  (a mechanism I cannot measure is not asserted), per harness discipline.

DESIGN (all faithful to the original convergence design):
  - Board processing margin m_t and gross notional n_t per complex (native units).
  - CONVERGENCE: trailing-252d percentile of m_t; long-margin below 10th pct, short above 90th.
  - HYSTERESIS exit at 50th pct, 10d min-hold, 60d max-hold backstop.
  - VOL GATE blocks entries when 20d spread vol is in its top decile.
  - Inverse-vol equal-risk across whichever of the 3 complexes are live, weekly rebalance.
  - Signal lagged 1 day (our responsibility), ~10bps costs on spread turnover.

No external side effects: data is read via the sanctioned yf_panel adapter only.
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, inv_vol_position
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

SID = "physical_margin_convergence_v2"
START = "2005-01-01"   # RBOB (RB=F) listed ~2005 -> bounds the energy-crack leg

# Three ticker-disjoint production complexes, continuous front-month yfinance futures symbols -------
_LEGS = {
    "soy_crush":    {"ZS": "ZS=F", "ZM": "ZM=F", "ZL": "ZL=F"},  # beans / meal / oil
    "cattle_crush": {"LE": "LE=F", "GF": "GF=F", "ZC": "ZC=F"},  # live cattle / feeders / corn-feed
    "energy_crack": {"CL": "CL=F", "HO": "HO=F", "RB": "RB=F"},  # crude / distillate / gasoline (3:2:1)
}
_SECTOR = {"soy_crush": "grains", "cattle_crush": "livestock", "energy_crack": "energy"}
_YF = sorted({sym for legs in _LEGS.values() for sym in legs.values()})

DEFAULTS = dict(
    entry_low=0.10, entry_high=0.90,
    exit_long=0.50, exit_short=0.50,
    min_hold=10, max_hold=60,
    vol_window=20, vol_gate_pct=0.90,
    lookback=252,
    target_vol=0.10, vol_lb=60, max_pos=2,
    cost_bps=10.0,
    rebalance="W",
    ret_clip=0.10,   # guard against continuous front-month roll spikes
)


# ---------------------------------------------------------------------------------------------------
def _margin(name, panel):
    """Board processing margin m and gross leg notional n (native units) from front-month prices."""
    legs = _LEGS[name]
    if any(sym not in panel.columns for sym in legs.values()):
        return None, None
    P = {root: panel[sym] for root, sym in legs.items()}
    if name == "soy_crush":
        zs, zm, zl = P["ZS"], P["ZM"], P["ZL"]               # ZS c/bu, ZM $/ton, ZL c/lb
        m = 0.022 * zm + 0.11 * zl - zs / 100.0
        n = 0.022 * zm + 0.11 * zl + zs / 100.0
    elif name == "cattle_crush":
        le, gf, zc = P["LE"], P["GF"], P["ZC"]               # LE,GF c/lb, ZC c/bu
        m = 12.0 * le - 7.5 * gf - 0.5 * zc                  # $/head
        n = 12.0 * le + 7.5 * gf + 0.5 * zc
    else:  # energy_crack 3:2:1
        cl, ho, rb = P["CL"], P["HO"], P["RB"]               # CL $/bbl, HO,RB $/gal
        g, d = 42.0 * rb, 42.0 * ho
        m = (2.0 * g + d - 3.0 * cl) / 3.0                   # $/bbl
        n = (2.0 * g + d + 3.0 * cl) / 3.0
    df = pd.concat([m.rename("m"), n.rename("n")], axis=1).dropna()
    if df.empty:
        return None, None
    return df["m"], df["n"]


def _roll_pct(s, window):
    """Trailing percentile rank of the current value within a window ending at t (no look-ahead)."""
    return s.rolling(window, min_periods=window).apply(lambda a: (a <= a[-1]).mean(), raw=True)


def _positions(margin, spread_ret, p):
    """Stateful hysteresis machine + vol gate; trailing info only, the 1-day lag is applied later."""
    pa = _roll_pct(margin, p["lookback"]).values
    vol = spread_ret.rolling(p["vol_window"], min_periods=p["vol_window"]).std()
    va = _roll_pct(vol, p["lookback"]).values
    el, eh, xl, xs = p["entry_low"], p["entry_high"], p["exit_long"], p["exit_short"]
    mnh, mxh, vg = p["min_hold"], p["max_hold"], p["vol_gate_pct"]
    out = np.zeros(len(margin)); state = 0; hold = 0
    for i in range(len(margin)):
        just_exit = False
        if state != 0:
            hold += 1
            if state == 1 and ((hold >= mnh and pa[i] >= xl) or hold >= mxh):
                state, hold, just_exit = 0, 0, True
            elif state == -1 and ((hold >= mnh and pa[i] <= xs) or hold >= mxh):
                state, hold, just_exit = 0, 0, True
        if state == 0 and not just_exit:
            if not (np.isnan(pa[i]) or np.isnan(va[i])) and va[i] <= vg:      # vol gate
                if pa[i] < el:
                    state, hold = 1, 0                                       # long-margin (cheap)
                elif pa[i] > eh:
                    state, hold = -1, 0                                      # short-margin (rich)
        out[i] = state
    return pd.Series(out, index=margin.index)


# ---------------------------------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Continuous front-month settle panel for all legs (sanctioned yf_panel; futures, no surv. bias)."""
    return yf_panel(_YF, START)


def load_gen_data(label) -> pd.DataFrame:
    """scope='local' -> stage-2 battery not run; provided for API completeness."""
    panel = load_data()
    if label in _LEGS:
        syms = [s for s in _LEGS[label].values() if s in panel.columns]
        return panel[syms]
    return panel


def signal(panel, **params):
    p = {**DEFAULTS, **params}
    if not isinstance(panel, pd.DataFrame):
        panel = load_data()

    cols, poss = {}, {}
    for name in _LEGS:
        m, n = _margin(name, panel)
        if m is None or len(m) < p["lookback"] + p["vol_window"] + 5:
            continue
        sr = (m.diff() / n.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sr = sr.clip(-p["ret_clip"], p["ret_clip"])         # mute front-month roll spikes
        cols[name] = sr
        poss[name] = _positions(m, sr, p)

    if not cols:
        return pd.Series(dtype=float, name=SID), []

    all_idx = sorted(set().union(*[c.index for c in cols.values()]))
    spread_rets = pd.DataFrame(cols).reindex(all_idx).fillna(0.0)
    pos_df = pd.DataFrame(poss).reindex(all_idx).fillna(0.0)[spread_rets.columns]

    sig = pos_df.shift(1).fillna(0.0)                       # 1-day lag is OUR responsibility
    W = inv_vol_position(sig, spread_rets, target_vol=p["target_vol"], vol_lb=p["vol_lb"],
                         max_pos=p["max_pos"], rebalance=p["rebalance"])
    W = W.reindex(spread_rets.index).fillna(0.0)

    rets = net_of_cost(W, spread_rets, cost_bps=p["cost_bps"], name=SID)   # W already lag-aligned
    trades = trades_from_weights(W, spread_rets, _SECTOR)  # kit stamps entry_regime (contract)

    nz = W.abs().sum(axis=1)
    if (nz > 0).any():
        rets = rets.loc[nz[nz > 0].index[0]:]
    rets.name = SID
    return rets, trades


# --- soft expectations (machine-checkable mechanism claims) -----------------------------------------
def _search_trades(ctx):
    return [t for t in ctx["trades"] if t["entry_date"] < ctx["holdout_start"]]

def _check_per_spread_nonneg(ctx):
    pnl = {}
    for t in _search_trades(ctx):
        pnl[t["ticker"]] = pnl.get(t["ticker"], 0.0) + t["pnl"]
    if not pnl:
        return {"pass": False, "observed": "no_trades"}
    return {"pass": min(pnl.values()) >= 0.0, "observed": {k: round(v, 4) for k, v in pnl.items()}}

def _check_hysteresis_lowers_turnover(ctx):
    # control: near-instant exits + no min-hold -> hysteresis should cut the trade count
    _, ctrl = ctx["spec"].signal(ctx["panel"], exit_long=0.10, exit_short=0.90, min_hold=0)
    hs = ctx["holdout_start"]
    n_hyst = len(_search_trades(ctx))
    n_ctrl = sum(1 for t in ctrl if t["entry_date"] < hs)
    if n_ctrl == 0:
        return {"pass": False, "observed": "no_control_trades"}
    ratio = n_hyst / n_ctrl
    return {"pass": ratio <= 0.80, "observed": round(ratio, 3)}

def _check_multiweek_holds(ctx):
    tr = _search_trades(ctx)
    if not tr:
        return {"pass": False, "observed": "no_trades"}
    mh = float(np.mean([t["hold_days"] for t in tr]))
    return {"pass": mh >= 10.0, "observed": round(mh, 1)}


SPEC = StrategySpec(
    id=SID,
    family="production_margin_convergence",
    title="Physical-margin CONVERGENCE book (soy crush + cattle crush + energy crack 3:2:1) — "
          "board-margin, hysteresis-banded, vol-gated, inverse-vol weekly",
    markets=["futures"],
    data_desc="yf_panel continuous front-month settles: soy (ZS/ZM/ZL), cattle (LE/GF/ZC), energy "
              "3:2:1 (CL/HO/RB). Board processing margins in native units; per-$-notional spread "
              "returns from margin diffs. (Term-structure carry is NOT testable on sanctioned "
              "front-month-only data and is therefore not part of this frozen spec.)",
    pre_registration=(
        "CLAIM: a CONVERGENCE premium inside physical production spreads — a board processing margin "
        "at a trailing-252d distribution extreme reverts toward processing cost via real-asset "
        "optionality. ENTRY: long-margin below the 10th pct, short-margin above the 90th. HYSTERESIS "
        "exit at the 50th pct with a 10d min-hold and 60d max-hold backstop (Mitchell-2010). VOL GATE "
        "blocks entries when 20d spread vol is in its top decile. Equal-risk (inverse-vol) across "
        "whichever of the 3 complexes are live, weekly rebalance, signal shifted 1d. Costs ~10bps on "
        "spread turnover (~1 tick/leg). PRIMARY metric: net Sharpe of combined spread P&L; "
        "diagnostics: per-complex in-sample PnL (per_spread_nonneg), turnover/fee-drag vs a single-"
        "threshold control (hysteresis_lowers_turnover), mean hold length (multiweek_holds). "
        "DROPPED vs prior draft: the roll-CARRY entry gate and matched 2nd/3rd-month construction "
        "required individual contract months, which are NOT available through the sanctioned data "
        "adapters; rather than assert a mechanism we cannot measure, carry is removed from this frozen "
        "spec and left to a later experiment if a contract-month adapter is validated. "
        "SCOPE = local: sanctioned data supports exactly 3 ticker-disjoint production complexes, all "
        "needed in-book for sector/name spread; generalization guarded by requiring each complex "
        "independently non-negative in-sample and confirmed by forward-validation on the 2022+ holdout. "
        "Standalone first; pair with the trend hedge only in a later experiment. FROZEN: default is the "
        "pre-committed primary; grid is the honest DSR neighbourhood, not a search."
    ),
    load_data=load_data,
    signal=signal,
    default_params=DEFAULTS,
    grid={
        "default": {},
        "shallow_band": {"entry_low": 0.15, "entry_high": 0.85},
        "tighter_exit": {"exit_long": 0.40, "exit_short": 0.60},
        "short_lookback": {"lookback": 126},
        "longer_minhold": {"min_hold": 15},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=2,
    expectations=[
        {"name": "per_spread_nonneg",
         "claim": "each complex (soy/cattle/energy) has non-negative in-sample net PnL",
         "check": _check_per_spread_nonneg},
        {"name": "hysteresis_lowers_turnover",
         "claim": "hysteresis (50th-pct exit + 10d min-hold) yields <=80% of the single-threshold "
                  "control's in-sample trade count",
         "check": _check_hysteresis_lowers_turnover},
        {"name": "multiweek_holds",
         "claim": "mean in-sample hold >= 10 trading days (real convergence holds, not whipsaw)",
         "check": _check_multiweek_holds},
    ],
)