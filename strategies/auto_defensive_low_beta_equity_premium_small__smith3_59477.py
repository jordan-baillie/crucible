"""
Defensive low-beta (Betting-Against-Beta) premium — US small-cap, IWM-hedged beta-neutral, monthly.

Frazzini-Pedersen / Baker-Bradley-Wurgler: a leverage-constraint + lottery-preference RISK premium
(you are paid to hold the boring low-beta names that the constrained cannot lever and the
lottery-seeking shun) — tested in the LESS-ARBITRAGED small-cap corner the large-cap low-vol ETFs
(USMV/SPLV) structurally do not transact. Long the lowest-beta names, SHORT a liquid index (IWM) in
the size that neutralizes the book's ex-ante beta — isolating the BAB premium, not equity direction.

NO external side effects. The ONLY novel code is the beta-rank + beta-neutral hedge sizing; every
lookahead-prone step (cost, ledger+regime stamping, universe/sector map) is delegated to the kit.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2001-01-01"
_SECTOR_MAP = {}  # internal module state populated by load_data(); read by trades_from_weights()

DEFAULTS = {
    "hedge": "IWM",        # index hedge instrument (IWM primary; SPY = grid sensitivity)
    "rank_by": "beta",     # ranking variable: 252d regression beta (vol = grid sensitivity)
    "n_long": None,        # None -> bottom-quintile count clipped to [25,40]
    "beta_lb": 252,        # OLS beta window
    "vol_lb": 60,          # trailing vol window for inverse-vol (equal-risk) sizing
    "dvol_lb": 60,         # trailing dollar-volume window for the liquidity floor
    "dvol_floor": 1e6,     # deployability floor (EXCLUDES microcap/Amihud mirage; NOT the signal)
    "gross_long": 1.0,     # gross long exposure (hedge adds ~0.6-0.8 short -> gross <= ~2x)
    "cost_bps": 8.0,       # realistic turnover cost
    "min_names": 80,       # skip a rebalance if the eligible set is degenerate
}


# ----------------------------- helpers (frozen) -----------------------------
def _rolling_beta(R, m, lb):
    """252d OLS market beta = rolling Cov(r_i, r_m) / Var(r_m). Trailing-only, no look-ahead."""
    mp = max(60, int(lb * 0.8))
    m = m.reindex(R.index)
    mean_R = R.rolling(lb, min_periods=mp).mean()
    mean_m = m.rolling(lb, min_periods=mp).mean()
    cov = R.mul(m, axis=0).rolling(lb, min_periods=mp).mean().sub(mean_R.mul(mean_m, axis=0))
    var = (m * m).rolling(lb, min_periods=mp).mean() - mean_m ** 2
    return cov.div(var, axis=0)


def _rebal_dates(idx):
    """Last trading day of each calendar month (EOD monthly rebalance)."""
    s = pd.Series(np.arange(len(idx)), index=idx)
    last = s.groupby(idx.to_period("M")).max().values
    return idx[last]


# ----------------------------- data -----------------------------
def load_data() -> pd.DataFrame:
    # PIT, survivorship-clean small-cap, sector-spread (delisted INCLUDED via the kit universe).
    tickers, sector_map = sector_universe("Small", 40)
    _SECTOR_MAP.clear()
    _SECTOR_MAP.update(sector_map)

    px = sep_panel(tickers, START, field="closeadj")          # split+div adjusted -> returns/beta/vol
    try:
        rc = sep_panel(tickers, START, field="close")         # raw close for TRUE dollar volume
    except Exception:
        rc = px                                               # pre-registered fallback
    vol = sep_panel(tickers, START, field="volume")

    rc = rc.reindex(index=px.index, columns=px.columns)
    vol = vol.reindex(index=px.index, columns=px.columns)
    dvol = rc * vol                                           # $ volume is split-invariant (price*shares)

    # Beta hedge: IWM/SPY ETFs load via yf_panel (the ETF adapter), verified before any backtest.
    ixp = yf_panel(["IWM", "SPY"], START).reindex(px.index).ffill()

    return pd.concat({"px": px, "dvol": dvol, "idx": ixp}, axis=1)


# ----------------------------- signal -----------------------------
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    px = panel["px"].astype(float)
    dvol = panel["dvol"].astype(float)
    ixp = panel["idx"].astype(float)
    hedge = p["hedge"]

    R = px.pct_change()
    m = ixp[hedge].pct_change()
    beta = _rolling_beta(R, m, p["beta_lb"])                                   # neutralize THIS beta
    vol = R.rolling(p["vol_lb"], min_periods=max(20, int(p["vol_lb"] * 0.7))).std()
    liq = dvol.rolling(p["dvol_lb"], min_periods=max(20, int(p["dvol_lb"] * 0.7))).median()
    rankvar = vol if p["rank_by"] == "vol" else beta                          # rank variable (frozen)

    cols = px.columns
    rb = _rebal_dates(px.index)
    tw = pd.DataFrame(index=rb, columns=cols, dtype=float)   # long target weights at rebalances
    th = pd.Series(index=rb, dtype=float)                    # index short weight (= -portfolio beta)

    for d in rb:
        b, v, l, rv = beta.loc[d], vol.loc[d], liq.loc[d], rankvar.loc[d]
        elig = b.notna() & v.notna() & (v > 0) & rv.notna() & (l >= p["dvol_floor"])
        names = rv[elig].sort_values().index                # lowest beta (or vol) first
        if len(names) < p["min_names"]:
            continue
        n = p["n_long"] if p["n_long"] else int(np.clip(len(names) // 5, 25, 40))
        sel = names[:n]
        inv = 1.0 / v[sel]                                   # inverse-vol (equal-risk) weights
        w = inv / inv.sum() * p["gross_long"]
        tw.loc[d] = 0.0
        tw.loc[d, w.index] = w.values
        th.loc[d] = -float((w * b[sel]).sum())              # short index to zero ex-ante beta

    # Hold weights constant between monthly rebalances; skipped (degenerate) months keep prior book.
    W_long = tw.reindex(px.index, method="ffill").fillna(0.0)
    hedge_w = th.reindex(px.index, method="ffill").fillna(0.0)

    Rf = R.fillna(0.0)                                       # delisting drop captured on last live day
    HC = "__HEDGE_IDX__"
    rets_all = Rf.copy()
    rets_all[HC] = m.fillna(0.0)
    W_full = W_long.copy()
    W_full[HC] = hedge_w
    rets_all = rets_all.reindex(columns=W_full.columns)

    # LAG IS OURS: weights are EOD/same-day -> shift(1) before pricing (no look-ahead).
    Wl = W_full.shift(1).fillna(0.0)
    daily = net_of_cost(Wl, rets_all, cost_bps=p["cost_bps"], name="bab_smid")

    active = Wl.abs().sum(axis=1)
    if (active > 0).any():
        daily = daily.loc[active[active > 0].index[0]:]
    daily = daily.fillna(0.0)

    # Ledger = ALPHA book only (the IWM short is a DECLARED hedge sleeve -> judged separately).
    trades = trades_from_weights(W_long.shift(1).fillna(0.0), Rf, _SECTOR_MAP)
    return daily, trades


# ----------------------------- gen data (scope='local' -> unused; provided for safety) -----------
def load_gen_data(label) -> pd.DataFrame:
    return load_data()


# ----------------------------- soft expectations (machine-checked mechanism claims) -----------
def _chk_turnover(ctx):
    tr = ctx.get("trades") or []
    hd = [t["hold_days"] for t in tr if t.get("hold_days") is not None]
    med = float(np.median(hd)) if hd else 0.0
    return {"pass": med >= 18.0, "observed": round(med, 1)}


def _chk_beta_neutral(ctx):
    r = ctx["search"].dropna()
    iwm = ctx["panel"]["idx"]["IWM"].pct_change()
    df = pd.concat([r, iwm.reindex(r.index)], axis=1).dropna()
    if len(df) < 100:
        return {"pass": False, "observed": "insufficient"}
    c = np.cov(df.iloc[:, 0].values, df.iloc[:, 1].values)
    beta = float(c[0, 1] / c[1, 1])
    return {"pass": abs(beta) <= 0.35, "observed": round(beta, 3)}


def _chk_defensive(ctx):
    r = ctx["search"].dropna()
    iwm = ctx["panel"]["idx"]["IWM"].pct_change()
    df = pd.concat([r, iwm.reindex(r.index)], axis=1).dropna()
    if len(df) < 100:
        return {"pass": False, "observed": "insufficient"}
    down = df[df.iloc[:, 1] < 0].iloc[:, 0]
    md = float(down.mean()) if len(down) else 0.0
    return {"pass": md >= 0.0, "observed": round(md, 6)}


# ----------------------------- spec -----------------------------
SPEC = StrategySpec(
    id="bab_smid_beta_neutral",
    family="low_beta_defensive",
    title="Defensive low-beta (BAB) premium — US small-cap, IWM-hedged beta-neutral, monthly",
    markets=["us_equity"],
    data_desc=("Sharadar SEP survivorship-clean daily closeadj + raw close*volume for a PIT small-cap "
               "sector-spread universe (delisted INCLUDED); IWM/SPY ETF closes (yf_panel) for the beta hedge."),
    pre_registration=(
        "FROZEN BEFORE ANY FIT. Hypothesis: the betting-against-beta / low-vol defensive premium "
        "(Frazzini-Pedersen; Baker-Bradley-Wurgler) — a leverage-constraint/lottery-preference RISK "
        "premium, not a price forecast — survives in the LESS-ARBITRAGED US small-cap corner that "
        "large-cap low-vol ETFs (USMV/SPLV) structurally do not transact. SCOPE=LOCAL: the edge is "
        "claimed universe-specific (small-cap), so the cross-universe stage-2 battery is NOT the right "
        "test (it would wrongly demand the crowded large-cap version also work); forward-validation "
        "confirms. CONSTRUCTION (no optimization): PIT survivorship-clean small-cap via "
        "sector_universe('Small',40), delisted INCLUDED (low-vol Sharpe is survivorship-inflated). "
        "Liquidity floor (deployability, NOT signal): trailing 60d median $volume >= $1e6, PIT — this "
        "DELIBERATELY excludes the microcap/Amihud illiquidity mirage so the measured edge is the "
        "beta/vol premium. Beta = 252d OLS of daily stock returns on the IWM index. Long book = 25-40 "
        "lowest-beta eligible names (bottom-quintile count clipped to [25,40]), inverse-vol "
        "(equal-risk) weighted, gross long=1.0. Hedge = SHORT IWM at weight -sum(w_i*beta_i) -> ex-ante "
        "beta~0; declared as a hedge sleeve (hedge_tickers=['IWM'], hedge_cap=0.50) so the deployment "
        "gate judges the long ALPHA book alone; no single-name shorts (small-caps long-only, no borrow). "
        "Monthly rebalance (last trading day), weights shift(1) (lag is ours), ~8bps cost, gross<=~2x. "
        "GATE-0 FALLBACK: if no ETF loads for the hedge, use an equal-weight basket of the universe's "
        "most-liquid large names as market proxy (pre-registered). NULL = beta-neutral zero-Sharpe "
        "(NOT absolute-return). Free choices that must ALL be robust (scored as DSR effective-N via "
        "grid): IWM vs SPY hedge, regression-beta vs realized-vol ranking, basket size, liquidity floor "
        "— fragility to any falsifies. CROWDING FALSIFICATION (heavy/prose, not a cheap same-night "
        "check, so not coded as a soft expectation): a parallel large-cap construction must be WEAKER "
        "in-sample than this small-cap book, or the crowding thesis is falsified and the candidate "
        "rejected. Holdout 2022-01-01+ is the sole arbiter; if passed, a fresh write-once forward-paper "
        "run with a pre-registered ~3-month verdict date gates entry into the carry+trend book as a "
        "modest defensive sleeve (per the 2026-06-08 dilution lesson)."
    ),
    load_data=load_data,
    signal=signal,
    default_params=dict(DEFAULTS),
    grid={
        "default": {},
        "spy_hedge": {"hedge": "SPY"},
        "rank_vol": {"rank_by": "vol"},
        "n40": {"n_long": 40},
        "floor2m": {"dvol_floor": 2e6},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=40,
    hedge_tickers=["IWM"],
    hedge_cap=0.50,  # beta-neutral hedge is ~0.4 of position-days (full neutralization, not a trim); < 0.60 hard cap
    expectations=[
        {"name": "low_turnover",
         "claim": "monthly cadence -> median trade hold_days >= 18 (positions held ~1 month, low fee drag)",
         "check": _chk_turnover},
        {"name": "beta_neutral",
         "claim": "|beta of net returns to IWM| <= 0.35 (the hedge neutralizes equity direction)",
         "check": _chk_beta_neutral},
        {"name": "defensive_downmarket",
         "claim": "mean net return on down-IWM days >= 0 (defensive earn-in-down-markets profile, not a hidden beta bet)",
         "check": _chk_defensive},
    ],
)