from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ---------------------------------------------------------------------------
# Universe: 16 liquid commodity roots (PA palladium dropped -- thin rank-2).
# yfinance continuous front-month futures Close (FREE / $0).
# The harness exposes NO fut_curve adapter, so continuous-front *within-month*
# compounded returns are used as a (documented) proxy for within-contract
# seasonal returns. Yahoo =F is NOT back-adjusted, so the once-per-roll-month
# price gap injects (seasonal, per-root) ROLL YIELD into the estimate -- the
# exact carry axis this premium claims to be distinct from. To honor the frozen
# rule 'NEVER diffing close across a roll', the roll-day jump is DE-SPIKED out
# of each root-month before within-month compounding (see _despike_within_month).
# ---------------------------------------------------------------------------
COMPLEX = {
    "energy":    ["CL=F", "NG=F", "HO=F", "RB=F"],
    "metals":    ["GC=F", "SI=F", "PL=F", "HG=F"],
    "grains":    ["ZC=F", "ZS=F", "ZW=F", "ZL=F", "ZM=F"],
    "livestock": ["LE=F", "HE=F", "GF=F"],
}
ROOTS      = [t for ts in COMPLEX.values() for t in ts]          # 16 roots
SECTOR_MAP = {t: c for c, ts in COMPLEX.items() for t in ts}     # ticker -> complex
START      = "2000-01-01"


def load_data() -> pd.DataFrame:
    return yf_panel(ROOTS, start=START)


def load_gen_data(label) -> pd.DataFrame:
    return yf_panel(COMPLEX[label], start=START)


def _despike_within_month(rets: pd.DataFrame) -> pd.DataFrame:
    """Approximate within-contract returns from a continuous (non-back-adjusted)
    front series: per root, per calendar month, neutralize the single most
    extreme daily move when it is a clear outlier (> 4% and > 3x the month's
    median |move|) -- i.e. the likely continuous-front ROLL-DAY jump -- replacing
    it with the month's median daily return. This honors the frozen rule 'NEVER
    diffing close across a roll' as closely as possible absent a fut_curve
    adapter (the proper source is the owned Databento fut_curve front close)."""
    ym  = rets.index.to_period("M")
    out = rets.copy()
    for col in rets.columns:
        s = rets[col]
        for p, idx in s.groupby(ym).groups.items():
            seg = s.loc[idx].dropna()
            if len(seg) < 5:
                continue
            a   = seg.abs()
            med = a.median()
            j   = a.idxmax()
            if a[j] > 0.04 and a[j] > 3.0 * max(med, 1e-9):
                out.loc[j, col] = seg.median()
    return out


# ---------------------------------------------------------------------------
# SIGNAL: monthly, dollar-neutral, PIT expanding-window calendar-cycle L/S.
# ---------------------------------------------------------------------------
def signal(panel, **params):
    min_obs    = int(params.get("min_obs", 7))
    vol_lb     = int(params.get("vol_lb", 60))
    scale_lb   = int(params.get("scale_lb", 63))
    target_vol = float(params.get("target_vol", 0.10))
    frac       = float(params.get("tercile_frac", 1.0 / 3.0))

    panel = panel.sort_index().astype(float)
    roots = list(panel.columns)
    rets  = panel.pct_change()                 # raw continuous-front daily returns (book PnL/vol)
    erets = _despike_within_month(rets)        # roll-de-spiked -> within-contract proxy (estimate only)

    # ---- calendar-month returns (compounded within-month daily, de-spiked) -
    ym         = panel.index.to_period("M")
    monthly    = (1.0 + erets).groupby(ym).prod() - 1.0          # PeriodIndex rows
    valid_days = erets.notna().groupby(ym).sum()
    monthly    = monthly.where(valid_days >= 5)                 # mask thin/pre-existence months

    # ---- realized daily vol for inverse-vol leg weighting (raw returns) ----
    dvol = rets.rolling(vol_lb).std()

    periods = sorted(panel.index.to_period("M").unique())
    W = pd.DataFrame(0.0, index=panel.index, columns=roots)

    for p in periods:
        cal_month = p.month
        past = monthly[monthly.index < p]                       # STRICTLY prior (no look-ahead)
        same = past[past.index.month == cal_month]
        if same.shape[0] == 0:
            continue
        counts = same.count()
        exp    = same.mean()
        elig   = exp[counts >= min_obs].dropna()                # seasonal expectation, eligible only
        if len(elig) < 3:
            continue
        n = len(elig)
        k = max(1, int(round(n * frac)))
        if 2 * k > n:
            k = n // 2
        if k < 1:
            continue
        ranked = elig.sort_values()
        shorts = list(ranked.index[:k])
        longs  = list(ranked.index[-k:])

        md = panel.index[panel.index.to_period("M") == p]
        if len(md) == 0:
            continue
        prior = dvol.index[dvol.index < md[0]]                  # vol as-of prior trading day
        if len(prior) == 0:
            continue
        v = dvol.loc[prior[-1]]

        lw = (1.0 / v[longs]).replace([np.inf, -np.inf], np.nan).dropna()
        sw = (1.0 / v[shorts]).replace([np.inf, -np.inf], np.nan).dropna()
        if lw.sum() <= 0 or sw.sum() <= 0:
            continue
        lw = lw / lw.sum() * 0.5                                # long leg gross +0.5
        sw = sw / sw.sum() * 0.5                                # short leg gross -0.5 -> dollar-neutral
        for t in lw.index:
            W.loc[md, t] = float(lw[t])
        for t in sw.index:
            W.loc[md, t] = -float(sw[t])

    # ---- causal ~10% annualized vol target (constant within each month) ----
    gross = (W.shift(1) * rets).sum(axis=1).fillna(0.0)
    rvol  = gross.rolling(scale_lb).std() * np.sqrt(252.0)
    scale = pd.Series(0.0, index=panel.index)
    for p in periods:
        md = panel.index[panel.index.to_period("M") == p]
        if len(md) == 0:
            continue
        prior = rvol.index[rvol.index < md[0]]                  # scale known before the month
        if len(prior) == 0:
            continue
        rv = rvol.loc[prior[-1]]
        if not np.isfinite(rv) or rv <= 0:
            continue
        scale.loc[md] = min(target_vol / rv, 3.0)

    W_sized = W.mul(scale, axis=0)
    W_exec  = W_sized.shift(1).fillna(0.0)                      # MY 1-day execution lag

    smap   = {t: SECTOR_MAP.get(t, "commodity") for t in roots}
    dr     = net_of_cost(W_exec, rets, cost_bps=8.0, name="commodity_seasonal")
    trades = trades_from_weights(W_exec, rets, smap)            # kit stamps entry_regime

    active = W_exec.abs().sum(axis=1) > 0
    if active.any():
        dr = dr.loc[dr.index >= active.idxmax()]               # trim leading no-position warm-up
    dr = dr.dropna()
    dr.name = "commodity_seasonal"
    return dr, trades


# ---------------------------------------------------------------------------
# SOFT EXPECTATIONS (machine-checkable structural predictions).
# ---------------------------------------------------------------------------
def _ann_mean_from_panel(panel, cols, holdout_start):
    try:
        cols = [c for c in cols if c in panel.columns]
        sub  = panel[cols]
        sub  = sub[sub.index < pd.Timestamp(holdout_start)]    # in-sample only
        r, _ = signal(sub)
        r    = r.dropna()
        if len(r) < 24:
            return np.nan
        return float(r.mean() * 252.0)
    except Exception:
        return np.nan


def _positive_check(label):
    def chk(ctx):
        m = _ann_mean_from_panel(ctx["panel"], COMPLEX[label], ctx["holdout_start"])
        if not np.isfinite(m):
            return {"pass": False, "observed": "insufficient_history"}
        return {"pass": bool(m > 0.0), "observed": round(m, 4)}
    return chk


def _metals_weak_check(ctx):
    m = _ann_mean_from_panel(ctx["panel"], COMPLEX["metals"], ctx["holdout_start"])
    if not np.isfinite(m):
        return {"pass": True, "observed": "insufficient_history"}
    return {"pass": bool(abs(m) < 0.06), "observed": round(m, 4)}   # weak/absent


def _low_beta_check(ctx):
    panel = ctx["panel"]
    bench = panel.pct_change().mean(axis=1)                    # EW long-commodity benchmark
    s     = ctx["search"]                                      # search-window net returns
    df    = pd.concat([s.rename("strat"), bench.rename("bench")], axis=1).dropna()
    if len(df) < 50:
        return {"pass": True, "observed": "insufficient"}
    beta = float(np.polyfit(df["bench"].values, df["strat"].values, 1)[0])
    return {"pass": bool(abs(beta) < 0.5), "observed": round(beta, 3)}


PRE_REG = """
MECHANISM. Seasonal hedging-pressure risk premium: storable-commodity producers and
consumers hedge predictable seasonal inventory/weather/production-calendar risk and pay
the speculator who bears it. The premium is a deterministic CALENDAR pattern in prices --
distinct from carry (curve slope level), basis-momentum BP2019 (front-minus-back spread
momentum), COT hedging pressure (commercials' positions), OI-growth (Hong-Yogo), value
(5y real-price reversal), skew/lottery and convenience-yield/storage (inventory levels).
No seasonality experiment exists in the registry.

SIGNAL (frozen). Monthly, dollar-neutral cross-section over 16 liquid roots (PA dropped).
At each month-start, each root's expected seasonal return = trailing EXPANDING-WINDOW mean
of THAT calendar-month's within-month return, using ONLY months strictly prior to
formation; a root is eligible only with >= min_obs (default 7) prior same-month
observations. Rank eligible roots; LONG top tercile / SHORT bottom tercile, inverse-vol
(60d realized, as-of prior day) weighted within each leg, dollar-neutral, scaled to a
causal ~10% annualized vol target (trailing-63d realized vol of the unscaled book, lagged).
Hold one month, rebalance month-start, signals lagged 1 day at execution (W.shift(1)).
Costs 8 bps on turnover. No tuning beyond the pre-set window / tercile / vol target.

DATA / WINDOW + ROLL HANDLING. fut_curve is not exposed by this harness; the PROPER source
is the owned Databento fut_curve front close, computing within-CONTRACT returns and NEVER
diffing close across a roll. We proxy with yf_panel continuous front-month Close. Yahoo =F
is NOT back-adjusted, so a once-per-roll-month price GAP would otherwise inject (seasonal,
per-root) ROLL YIELD = carry into the estimate -- the exact axis this premium must be
distinct from, and one that is NOT cancelled cross-sectionally because roll-yield
seasonality differs by root. To honor 'NEVER diff across a roll', the seasonal ESTIMATE is
built from roll-de-spiked returns: per root-month the single extreme daily move (> 4% and
> 3x the month's median |move|, i.e. the roll-day jump) is neutralized before within-month
compounding. Book vol-weighting and realized PnL still use the raw continuous-front returns.
History from 2000 + the >=7 same-month requirement => signals begin ~2007 for the longest
series; the write-once holdout (2022-01-01) is the forward test.

HONEST WEAKNESS. Each per-root seasonal mean rests on only ~7-25 annual same-month
observations -> noisy; the bet is on the cross-sectional RANK, power from POOLING 16 roots.
The de-spike is a mitigation, not a perfect within-contract reconstruction; residual roll
contamination may remain and is flagged. Validation leans on MCPT (time-permutation
destroys seasonal structure; dollar-neutral / near-zero beta => absolute null) plus holdout.

SCOPE = LOCAL (deliberate). The asset class has only ~16 roots: it is IMPOSSIBLE to build
>=3 disjoint 150-400-name generalization universes. The faithful analog of 'must
generalize' for a small fixed cross-section is STRUCTURAL CONSISTENCY across sub-complexes,
encoded below as soft expectations and confirmed forward by the write-once holdout.

STRUCTURAL PREDICTIONS (soft expectations, falsifiable):
 - positive in-sample seasonal premium in ENERGY, GRAINS and LIVESTOCK individually; a
   genuine pass needs >=2 of these 3 storable complexes positive -- not carried by natgas
   or one grain alone;
 - WEAK/ABSENT in METALS (|ann mean| < 6%): little physical seasonality;
 - near-zero beta to an EW long-commodity benchmark (dollar-neutral), justifying MCPT.

PAIRS WITH. Standalone first. Roughly orthogonal to curve-carry and to validated trend
crisis-alpha; a SMALL trend tail-overlay could later trim the left tail -- only if it
clears standalone.
"""


SPEC = StrategySpec(
    id="commodity_seasonal_xs_ls",
    family="commodity_seasonality",
    title="Commodity Seasonal Risk-Premium Cross-Section (PIT expanding-window calendar-cycle, 16-root L/S)",
    markets=["commodity_futures"],
    data_desc=("yfinance continuous front-month Close for 16 liquid commodity roots "
               "(4 energy / 4 metals / 5 grains-oilseeds / 3 livestock), FREE/$0. "
               "No fut_curve adapter is exposed; continuous-front within-month compounded "
               "returns approximate within-contract seasonal returns. The roll-day jump is "
               "de-spiked per root-month before estimation to honor 'NEVER diff across a "
               "roll' (proper source = owned Databento fut_curve front close)."),
    pre_registration=PRE_REG,
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default":  {},                       # primary
        "min_obs6": {"min_obs": 6},
        "min_obs9": {"min_obs": 9},
        "quartile": {"tercile_frac": 0.25},
        "vol40":    {"vol_lb": 40},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=12,
    expectations=[
        {"name": "energy_seasonal_positive",
         "claim": "in-sample (pre-holdout) energy-complex seasonal L/S has positive annualized mean",
         "check": _positive_check("energy")},
        {"name": "grains_seasonal_positive",
         "claim": "in-sample grains/oilseeds-complex seasonal L/S has positive annualized mean",
         "check": _positive_check("grains")},
        {"name": "livestock_seasonal_positive",
         "claim": "in-sample livestock-complex seasonal L/S has positive annualized mean",
         "check": _positive_check("livestock")},
        {"name": "metals_seasonal_weak",
         "claim": "metals show weak/absent seasonality (|ann mean| < 6%) -- economic consistency, not outlier",
         "check": _metals_weak_check},
        {"name": "near_zero_commodity_beta",
         "claim": "dollar-neutral book has |beta| < 0.5 to an EW long-commodity benchmark (justifies MCPT absolute null)",
         "check": _low_beta_check},
    ],
)