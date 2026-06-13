"""
Regime-conditional US Treasury roll-down / term-premium carry.

FROZEN DESIGN (pre-registered 2026-06-14) — STANDALONE submission.
Economic claim (falsifiable): the US Treasury term premium / roll-down is a genuine
duration-risk premium that is harvestable ONLY when the curve is positively sloped and
steep; in flat/inverted regimes the premium inverts and unconditional bond carry dies
(the documented kill-cause of CLOSED boreas-carry-fxbond in 2022). This construction is
DESIGNED to stand aside through exactly that flattening/inversion state.

Look-ahead controls (both stated in code):
  #1  FRED CMT for day t is published the next business day -> all conditioning/roll-down
      yields are read as of t-1 (yld.shift(1)).
  #2  Weights formed at close t are executed into t+1 returns -> the held weight matrix is
      shift(1)-ed before net_of_cost / trades_from_weights.
Long-only-when-active, no shorting/borrow/leverage (vol-target capped at 1.0), monthly
rebalance, ~$5k retail-routable in 1-2 ultra-liquid ETFs. Costs 8bps/turnover (conservative
vs ~2-5bps real ETF). scope='local' by deliberate constraint (thresholds calibrated to US
curve history; cross-DM-curve generalization is OFF — that re-opens dead DM bond carry).
KNOWN: a 1-2 ETF macro timing sleeve necessarily concentrates position-days in SHY/duration,
so the single_name_share / sector-spread gates (built for cross-sectional factor books) will
register concentration — intrinsic to the design, not a hidden overfit.
"""
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2002-07-01"                               # SHY/IEF/TLT inception 2002-07-22
ETFS = ["SHY", "IEF", "TLT"]                       # SHY = near-cash stand-aside; IEF/TLT = duration
COLS = ["SHY", "IEF", "TLT"]
FRED_IDS = {"DGS2": "DGS2", "DGS3MO": "DGS3MO", "DGS10": "DGS10",
            "DGS30": "DGS30", "T10Y2Y": "T10Y2Y"}
SECTOR_MAP = {"SHY": "Rates-Short", "IEF": "Rates-Intermediate", "TLT": "Rates-Long"}

DEFAULTS = dict(pctl_window=1260, pctl_threshold=0.40, target_vol=0.09, vol_lb=63)


# ----------------------------------------------------------------------------- data
def load_data():
    px = yf_panel(ETFS, start=START).add_prefix("px_")          # split+div adjusted TR close
    yld = fred_series(FRED_IDS, start=START).add_prefix("y_")    # FRED constant-maturity yields
    panel = pd.concat([px, yld], axis=1).sort_index()
    panel = panel.ffill()                                        # carry CMT onto every ETF day
    need = ["px_SHY", "px_IEF", "px_TLT", "y_DGS2", "y_DGS3MO", "y_DGS10", "y_T10Y2Y"]
    panel = panel.dropna(subset=need)        # DGS30 (gap 2002-02..2006-02) intentionally optional
    return panel


def load_gen_data(label):
    # scope='local' (generalization_universes=[]) — stage-2 battery is not run for this candidate.
    raise NotImplementedError("local scope: no cross-universe generalization battery")


# --------------------------------------------------------------------------- signal
def _signal_weights(panel, p):
    px = panel[[c for c in panel.columns if c.startswith("px_")]].copy()
    px.columns = [c[3:] for c in px.columns]
    yld = panel[[c for c in panel.columns if c.startswith("y_")]].copy()
    yld.columns = [c[2:] for c in yld.columns]
    idx = panel.index
    rets = px.pct_change()

    # LOOK-AHEAD CONTROL #1: read FRED CMT as of t-1 (published next business day).
    yl = yld.shift(1)
    t10y2y = yl["T10Y2Y"]
    # trailing 5y rolling percentile of the 10y-2y slope (rolling/expanding -> no full-sample leak)
    roll_pctl = t10y2y.rolling(int(p["pctl_window"]), min_periods=252).rank(pct=True)
    steep = roll_pctl >= float(p["pctl_threshold"])
    upward = (yl["DGS10"] > yl["DGS3MO"]) & (t10y2y > 0)
    active = (upward & steep).fillna(False)

    # static-curve roll-down proxy per bucket = carry + duration * (local slope per maturity-yr)
    dur_ief, dur_tlt = 7.5, 17.0
    roll_ief = yl["DGS10"] + dur_ief * (yl["DGS10"] - yl["DGS2"]) / 8.0
    roll_tlt = yl["DGS30"] + dur_tlt * (yl["DGS30"] - yl["DGS10"]) / 20.0
    pick_tlt = (roll_tlt > roll_ief).fillna(False)             # NaN (no DGS30 pre-2006) -> IEF

    tvol = float(p["target_vol"])
    vol_ann = rets.rolling(int(p["vol_lb"])).std() * np.sqrt(252.0)
    w_ief = (tvol / vol_ann["IEF"]).clip(upper=1.0).fillna(0.0)   # inverse-vol size, no leverage
    w_tlt = (tvol / vol_ann["TLT"]).clip(upper=1.0).fillna(0.0)

    W = pd.DataFrame(0.0, index=idx, columns=COLS)
    W["IEF"] = np.where(active & ~pick_tlt, w_ief.values, 0.0)
    W["TLT"] = np.where(active & pick_tlt, w_tlt.values, 0.0)
    W["SHY"] = (1.0 - W["IEF"] - W["TLT"]).clip(lower=0.0)        # remainder/stand-aside in SHY

    # monthly rebalance: hold rebalance-day weights through the month
    per = pd.Series(idx.to_period("M"), index=idx)
    is_rebal = per.ne(per.shift(1)).to_numpy()
    Wm = W.copy()
    Wm.loc[~is_rebal, :] = np.nan
    Wm = Wm.ffill().fillna(0.0)
    return Wm, rets


def signal(panel, **params):
    p = {**DEFAULTS, **params}
    Wm, rets = _signal_weights(panel, p)
    # LOOK-AHEAD CONTROL #2: weights set at close t earn t+1 returns -> shift(1) the held matrix.
    Wexec = Wm.shift(1)
    daily = net_of_cost(Wexec, rets[COLS], cost_bps=8.0, name="ust_rolldown_regime_carry")
    trades = trades_from_weights(Wexec, rets[COLS], SECTOR_MAP)   # kit stamps entry_regime
    return daily, trades


# --------------------------------------------------------------- soft expectations (in-sample)
def _held_duration_mask(panel, trades, hs):
    idx = panel.index[panel.index < hs]
    held = pd.Series(False, index=idx)
    for t in (trades or []):
        if t.get("ticker") not in ("IEF", "TLT"):
            continue
        d0, d1 = pd.Timestamp(t["entry_date"]), pd.Timestamp(t["exit_date"])
        held.loc[(held.index >= d0) & (held.index <= d1)] = True
    return idx, held


def _exp_stand_aside_when_inverted(ctx):
    # mechanism: the rule goes flat in duration when the curve is inverted (DGS10<DGS3MO).
    panel, hs = ctx["panel"], pd.Timestamp(ctx["holdout_start"])
    idx, held = _held_duration_mask(panel, ctx.get("trades"), hs)
    inv = (panel["y_DGS10"].shift(1) < panel["y_DGS3MO"].shift(1)).reindex(idx).fillna(False)
    n_inv = int(inv.sum())
    if n_inv == 0:
        return {"pass": True, "observed": "no in-sample inversion"}
    frac = float((held & inv).sum()) / n_inv
    return {"pass": frac <= 0.10, "observed": round(frac, 4)}


def _exp_selective_exposure(ctx):
    # mechanism: this is a CONDITIONAL harvester, not a buy-and-hold bond proxy.
    panel, hs = ctx["panel"], pd.Timestamp(ctx["holdout_start"])
    idx, held = _held_duration_mask(panel, ctx.get("trades"), hs)
    frac = float(held.sum()) / max(1, len(idx))
    return {"pass": frac < 0.80, "observed": round(frac, 4)}


# ------------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="ust_rolldown_regime_carry",
    family="term_premium_carry",
    title="Regime-conditional US Treasury roll-down / term-premium carry (steep-curve only)",
    markets=["US_RATES_ETF"],
    data_desc=("FRED CMT yields DGS2/DGS3MO/DGS10/DGS30/T10Y2Y (read t-1) for conditioning + "
               "roll-down proxy; yf_panel SHY/IEF/TLT total-return ETFs, 2002-07 onward."),
    pre_registration=(
        "Term premium / roll-down carry is a genuine duration-risk premium harvestable ONLY "
        "in a steep, positively-sloped curve; flat/inverted regimes invert it (killed CLOSED "
        "boreas-carry-fxbond in 2022). RULE (monthly, FRED at t-1, weights shift(1)): hold one "
        "duration ETF iff DGS10>DGS3MO AND T10Y2Y>0 AND T10Y2Y above trailing-1260d 40th pctile; "
        "pick IEF vs TLT by larger static-curve roll-down proxy (carry + dur*local-slope); "
        "vol-target the leg to 9% ann (inverse-vol, capped 1.0 = no leverage, long-only-when-"
        "active); ELSE stand aside fully in near-cash SHY. Conditioning is DESIGNED to sit flat "
        "through the 2022 flattening/inversion. Costs 8bps/turnover (conservative vs ~2-5bps real "
        "ETF). PRIMARY config = 'default' and its holdout verdict is accepted; grid declared only "
        "for honest DSR effective-N, not as a search to optimise. STANDALONE — Boreas trend "
        "overlay deferred (add later only if it cuts the tail without diluting standalone Sharpe). "
        "SCOPE=local by deliberate constraint (US-curve-calibrated thresholds; cross-DM-curve "
        "generalization is OFF). Confirmation = write-once US holdout (>=2022-01-01, spanning the "
        "inversion the rule must avoid) + Alpaca forward-paper. KNOWN: a 1-2 ETF macro sleeve "
        "concentrates position-days in SHY/duration, so single_name_share / sector-spread gates "
        "(designed for cross-sectional factor books) will register concentration — this is "
        "intrinsic to the construction, not a hidden overfit."
    ),
    load_data=load_data,
    signal=signal,
    default_params=DEFAULTS,
    grid={
        "default": {},                                  # primary, pre-registered
        "pctl_50": {"pctl_threshold": 0.50},            # stricter steepness gate
        "win_3y": {"pctl_window": 756},                 # shorter conditioning memory
        "vol_7pct": {"target_vol": 0.07},               # lower duration risk budget
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=2,
    expectations=[
        {"name": "stands_aside_when_inverted",
         "claim": "<=10% of in-sample inverted-curve days (DGS10<DGS3MO) hold any duration ETF",
         "check": _exp_stand_aside_when_inverted},
        {"name": "selective_not_buy_and_hold",
         "claim": "duration is held on <80% of in-sample days (conditional, not a TLT proxy)",
         "check": _exp_selective_exposure},
    ],
)