"""
VARIANT — Amihud illiquidity premium, self-consistent own-impact cost gate,
DEPLOYABLE CONCENTRATED SHORT LEG.

Keeps everything that survived in the elite Amihud line: survivorship-clean
small-cap search universe (Sharadar SEP, delisted included), trailing-12m Amihud,
size-tercile x sector neutral banding (long the capacity-tradeable 60-80th pct
illiquidity band, gated by each name's OWN Amihud-implied round-trip impact at
pre-registered trade size $T), dollar-neutral, beta-neutral, monthly rebalance
with hysteresis.

THE MUTATION (informed by the FALSIFIED single-ETF-substitution sibling, which
showed beta_to_universe 0.91 / sel-alpha -0.77 — i.e. the multi-name short leg
does real cross-sectional work): the ~100-name short quintile is replaced by a
CONCENTRATED top-N (default 25, grid 15) short book drawn ONLY from the
most-liquid quintile within each size tercile, greedily matched to the long
leg's sector and size-tercile weights. Any residual book beta is patched with a
SMALL beta-scaled IWM short, hard-capped at 20% of short-leg gross — a patch,
NOT a substitute. IWM is a DECLARED hedge sleeve (hedge_tickers) so the
deployment gate judges the alpha book alone.

Lag discipline: all signal inputs at month-end t use data through t only;
weights are shifted ONE DAY (W.shift(1)) before being applied to returns.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights, pit_panel

START = "2000-01-01"
_SEARCH_DEPTH = 70   # per-sector small-cap names in the SEARCH universe (~770 names)
_DEEP_DEPTH = 100    # 'small_deep' gen slice = names 71..100 per sector (ticker-DISJOINT)


# ----------------------------------------------------------------------------- data

def _build_panel(tickers, sector_map, start=START):
    """One panel: adjusted close (returns), raw close+volume ($-volume / Amihud),
    PIT marketcap (size terciles), IWM close (residual-beta patch)."""
    adj = sep_panel(tickers, start, field="closeadj")          # survivorship-clean, adjusted
    cl = sep_panel(tickers, start, field="close")              # unadjusted, for $-volume
    vo = sep_panel(tickers, start, field="volume")             # raw share volume
    cols = list(adj.columns)
    f = sf1(cols, ["marketcap"], dimension="ARQ")
    mc = pit_panel(f, "marketcap", adj.index, cols)            # datekey-based PIT, no lookahead
    iwm = yf_panel(["IWM"], start)["IWM"].reindex(adj.index).ffill()
    panel = pd.concat(
        {
            "adj": adj,
            "close": cl.reindex(columns=cols),
            "volume": vo.reindex(columns=cols),
            "mcap": mc.reindex(columns=cols),
            "hedge": iwm.to_frame("IWM"),
        },
        axis=1,
    )
    panel.attrs["sector_map"] = dict(sector_map)
    return panel


def load_data():
    tk, smap = sector_universe("Small", top_n_per_sector=_SEARCH_DEPTH)
    return _build_panel(tk, smap)


def load_gen_data(label):
    """Three universes DISJOINT from the small-cap search set:
    mid tier, large tier, and the deeper small-cap tranche (names 71..100/sector)."""
    if label == "mid":
        tk, sm = sector_universe("Mid", top_n_per_sector=30)
    elif label == "large":
        tk, sm = sector_universe("Large", top_n_per_sector=25)
    elif label == "small_deep":
        big, sm_big = sector_universe("Small", top_n_per_sector=_DEEP_DEPTH)
        search, _ = sector_universe("Small", top_n_per_sector=_SEARCH_DEPTH)
        srch = set(search)
        tk = [t for t in big if t not in srch]
        sm = {t: sm_big[t] for t in tk}
    else:
        raise ValueError(f"unknown generalization universe: {label}")
    return _build_panel(tk, sm)


# ----------------------------------------------------------------------------- helpers

def _greedy_short(cands, sec_d, ter_d, sec_tgt, ter_tgt, n):
    """Greedy bi-constraint match: fill n short slots from liquidity-sorted
    candidates while respecting per-sector and per-tercile target counts derived
    from the LONG leg's exposures. Sector match is hard; tercile is soft(+1)."""
    sec_cap = {s: int(np.ceil(n * w)) for s, w in sec_tgt.items() if w > 0}
    ter_cap = {k: int(np.ceil(n * w)) + 1 for k, w in ter_tgt.items() if w > 0}
    picked, sc, tc = [], {}, {}
    for t in cands:
        s, k = sec_d.get(t), ter_d.get(t)
        if s not in sec_cap or sc.get(s, 0) >= sec_cap[s]:
            continue
        if tc.get(k, 0) >= ter_cap.get(k, 0):
            continue
        picked.append(t)
        sc[s] = sc.get(s, 0) + 1
        tc[k] = tc.get(k, 0) + 1
        if len(picked) == n:
            return picked
    for t in cands:  # relax tercile constraint, keep sector (+1 slack)
        if t in picked:
            continue
        s = sec_d.get(t)
        if s not in sec_cap or sc.get(s, 0) >= sec_cap[s] + 1:
            continue
        picked.append(t)
        sc[s] = sc.get(s, 0) + 1
        if len(picked) == n:
            break
    return picked


# ----------------------------------------------------------------------------- signal

def signal(panel, band_lo=0.60, band_hi=0.80, short_n=25, trade_size=5000.0,
           max_impact_bps=50.0, hyst=0.05, beta_lb=60, vol_lb=63,
           etf_patch_cap=0.20, cost_bps=8.0, min_adv=2.0e5, min_price=3.0,
           min_bucket=8):
    adj, cl, vo = panel["adj"], panel["close"], panel["volume"]
    mcap = panel["mcap"]
    iwm_px = panel["hedge"]["IWM"]
    smap = dict(panel.attrs.get("sector_map") or {c: "Unknown" for c in adj.columns})
    sec_all = pd.Series({c: smap.get(c, "Unknown") for c in adj.columns})

    rets = adj.pct_change()
    dv = (cl * vo).where(lambda x: x > 0)                       # raw $-volume
    adv = dv.rolling(63, min_periods=40).mean()
    amihud = (rets.abs() / dv).rolling(252, min_periods=160).mean()  # E[|r| per $ traded]
    mkt = rets.mean(axis=1)                                     # universe EW return
    beta = rets.rolling(beta_lb).cov(mkt).div(mkt.rolling(beta_lb).var(), axis=0)
    dvol = rets.rolling(vol_lb, min_periods=40).std()

    month_ends = rets.index.to_series().groupby(rets.index.to_period("M")).max()
    rows, prev_long = {}, set()

    for t in month_ends:
        a, v, m, b, s = amihud.loc[t], adv.loc[t], mcap.loc[t], beta.loc[t], dvol.loc[t]
        px = cl.loc[t]
        ok = (a.notna() & v.notna() & m.notna() & b.notna() & s.notna()
              & (px >= min_price) & (v >= min_adv) & (s > 0) & (a > 0))
        names = a.index[ok]
        if len(names) < 120:
            continue

        ter = pd.qcut(m[names].rank(method="first"), 3, labels=False)   # size terciles
        sec = sec_all.reindex(names).fillna("Unknown")
        df = pd.DataFrame({"a": a[names], "ter": ter.values, "sec": sec.values}, index=names)
        bsize = df.groupby(["ter", "sec"])["a"].transform("size")
        pct = df.groupby(["ter", "sec"])["a"].rank(pct=True)             # illiquidity pct in bucket
        rt_bps = 2.0 * a[names] * trade_size * 1e4                       # own round-trip impact @ $T

        # LONG: mid-illiquid band, own-impact gated; incumbents get hysteresis
        new = (pct >= band_lo) & (pct <= band_hi) & (bsize >= min_bucket) & (rt_bps <= max_impact_bps)
        inc = pd.Index([x for x in prev_long if x in names])
        keep = ((pct.reindex(inc) >= band_lo - hyst) & (pct.reindex(inc) <= band_hi + hyst)
                & (rt_bps.reindex(inc) <= 1.5 * max_impact_bps)).fillna(False)
        longs = pd.Index(sorted(set(new.index[new]) | set(keep.index[keep])))
        if len(longs) < 10:
            continue
        iv = 1.0 / s[longs]
        wL = iv / iv.sum()                                               # long gross = 1.0

        # SHORT: concentrated top-N from the most-LIQUID quintile per tercile,
        # sector/tercile matched to the long leg (borrowable, near-zero impact @ $T)
        liq = df.groupby("ter")["a"].rank(pct=True)
        longset = set(longs)
        pool = [x for x in names if liq[x] <= 0.20 and x not in longset]
        if len(pool) < short_n:
            continue
        sec_tgt = wL.groupby(sec.loc[longs]).sum()
        ter_tgt = wL.groupby(ter.loc[longs]).sum()
        cands = sorted(pool, key=lambda x: -v[x])                        # most liquid first
        picked = _greedy_short(cands, sec.to_dict(), ter.to_dict(),
                               sec_tgt.to_dict(), ter_tgt.to_dict(), short_n)
        if len(picked) < max(10, int(0.6 * short_n)):
            continue
        psec = sec.loc[picked]
        stgt = sec_tgt.reindex(psec.unique()).fillna(0.0)
        if stgt.sum() <= 0:
            wS = pd.Series(1.0 / len(picked), index=picked)
        else:
            stgt = stgt / stgt.sum()
            wS = pd.Series(0.0, index=picked)
            for sname, grp in psec.groupby(psec):
                wS[grp.index] = stgt[sname] / len(grp)
        wS = wS.clip(upper=0.10)                                          # per-name short cap 10%
        wS = wS / wS.sum()                                                # short gross = 1.0

        # RESIDUAL beta patch ONLY (IWM beta ~1 vs this small/mid universe),
        # hard-capped at 20% of short-leg gross — never a substitute short
        beta_net = float((wL * b[longs]).sum() - (wS * b[picked]).sum())
        patch = float(np.clip(beta_net, -etf_patch_cap, etf_patch_cap))

        row = pd.Series(0.0, index=list(adj.columns) + ["IWM"])
        row[longs] = wL.values
        row[picked] = row[picked] - wS.values
        row["IWM"] = -patch
        rows[t] = row
        prev_long = set(longs)

    if not rows:
        empty = pd.Series(dtype=float, name="amihud_conc_short_v3")
        return empty, []

    W = pd.DataFrame(rows).T.sort_index().reindex(rets.index).ffill().fillna(0.0)
    rets_all = rets.copy()
    rets_all["IWM"] = iwm_px.pct_change()

    W_lag = W.shift(1)                                                    # execute next day
    net = net_of_cost(W_lag, rets_all, cost_bps=cost_bps, name="amihud_conc_short_v3")

    # self-consistent NAME-SPECIFIC impact cost on turnover (one-way Amihud impact @ $T)
    imp = (amihud * trade_size).clip(upper=0.02).shift(1)
    turn = W_lag[adj.columns].diff().abs()
    extra = (turn * imp.reindex(columns=adj.columns)).sum(axis=1).fillna(0.0)
    daily = (net - extra.reindex(net.index).fillna(0.0)).rename("amihud_conc_short_v3")

    smap["IWM"] = "ETF Hedge"
    trades = trades_from_weights(W_lag, rets_all, smap)
    return daily, trades


# ----------------------------------------------------------------------------- spec

SPEC = StrategySpec(
    id="amihud_illiq_concentrated_short_v3",
    family="liquidity_premium",
    title="Amihud illiquidity premium — own-impact cost gate + concentrated top-N "
          "exposure-matched borrowable short leg + capped residual IWM beta patch",
    markets=["us_smallcap", "us_midcap"],
    data_desc="Sharadar SEP closeadj/close/volume (survivorship-clean, delisted incl.), "
              "SF1 ARQ marketcap via PIT datekey panel, IWM via yfinance (declared hedge)",
    pre_registration=(
        "Premium: Amihud illiquidity (limits-to-arbitrage risk premium, not a forecast). "
        "Frozen design: small-cap search universe 2000+, trailing-12m Amihud, size-tercile x "
        "sector buckets, LONG the 60-80th pct illiquidity band gated by own Amihud round-trip "
        "impact at $T=5000 <= 50bps; SHORT a concentrated top-N (primary N=25, variant N=15) "
        "book from the most-liquid quintile per tercile, greedily matched to long sector/tercile "
        "weights, per-name short cap 10% of short gross; residual beta patched with IWM capped at "
        "20% of short gross (patch, NOT substitute — the full-ETF-substitution sibling is "
        "FALSIFIED: beta 0.91, sel-alpha -0.77). Dollar-neutral, gross <= ~2x, monthly rebalance "
        "with hysteresis, all returns NET of 8bps turnover cost PLUS name-specific Amihud impact. "
        "IN-SEARCH KILL ARBITERS (pre-registered, mirroring the gate that demoted the sibling): "
        "beta_to_universe < 0.3 AND selection_alpha_sharpe > 0, else dead before holdout. "
        "Generalization: premium must be OOS-positive in >=60% of {mid, large, small_deep}; "
        "N=15 vs N=25 must degrade gracefully (a cliff falsifies the concentrated-short thesis). "
        "Standalone first; trend tail-overlay only after holdout+MCPT pass."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"band_lo": 0.60, "band_hi": 0.80, "short_n": 25,
                    "trade_size": 5000.0, "max_impact_bps": 50.0},
    grid={
        "default": {},
        "n15": {"short_n": 15},
        "t10k": {"trade_size": 10000.0},
        "band55_85": {"band_lo": 0.55, "band_hi": 0.85},
    },
    scope="broad",
    generalization_universes=["mid", "large", "small_deep"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=45,
    hedge_tickers=["IWM"],
    hedge_cap=0.35,
)