"""
Reversal-frequency liquidity-provision premium (Akbas/Boehmer/Jiang/Koch 2022).

FIX vs failed v1: the SEP parquet cache does NOT carry the 'open' field
(pyarrow column-projection failure inside sep_panel on field='open') — the
overnight/intraday decomposition is therefore not computable from owned data.
The mechanism (liquidity providers absorbing transient order-flow noise; high
reversal-frequency names carry a provision premium) is re-implemented on the
CLOSE-TO-CLOSE decomposition, which is fully available:

  - Each month-end, over trailing 21 trading days, compute the FREQUENCY of
    reversal days: sign(ret_t) != sign(ret_{t-1}), counting only days where
    |ret_{t-1}| > 25bps and volume > 0 (excludes flat-noise days).
  - SEARCH UNIVERSE: survivorship-clean SMALL + MID caps (sector-spread),
    tradability-filtered (dollar-ADV > $1M, price > $3).
    LONG top quintile by reversal frequency, equal weight.
  - Single short IWM hedge sized by trailing 60d beta of the long book,
    capped at 1.0x, DECLARED on the spec (hedge_tickers) so the alpha book
    is gated alone.
  - Monthly rebalance, 25bps/side single names, 3bps IWM.
  - scope='broad': mechanism is universal but should ATTENUATE up the cap
    spectrum. Generalization universes are DISJOINT-by-construction tier-2
    liquidity slices of Small and Mid (search names removed) plus untouched
    Large caps.

Lag discipline: weights are built same-day from trailing data and passed as
W.shift(1) to net_of_cost / trades_from_weights — the 1-day execution lag is
applied explicitly below.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2010-01-01"

DEFAULTS = dict(
    lb=21,            # trailing window (days) for reversal frequency
    min_move=0.0025,  # 25bps minimum |prior-day ret| for a day to qualify
    top_frac=0.20,    # long top quintile
    min_price=3.0,
    min_adv=1.0e6,    # $1M trailing dollar-ADV
    beta_lb=60,       # trailing beta window for the IWM hedge
    beta_cap=1.0,     # FROZEN hedge-notional cap (1.0x)
)


# ---------------------------------------------------------------- universes

def _search_universe():
    """Pre-registered search universe: sector-spread SMALL + MID caps,
    survivorship-clean (delisted included by the kit)."""
    tk_s, sm_s = sector_universe(marketcap="Small", top_n_per_sector=40)
    tk_m, sm_m = sector_universe(marketcap="Mid", top_n_per_sector=40)
    tickers, seen = [], set()
    for t in list(tk_s) + list(tk_m):
        if t not in seen:
            tickers.append(t)
            seen.add(t)
    sector_map = {**dict(sm_s), **dict(sm_m)}
    return tickers, sector_map


def _field_with_fallback(tickers, start, field, fallback):
    """Defensive read: SEP cache schema has gaps (e.g. 'open' missing entirely);
    if a secondary field is absent, fall back rather than crash the run."""
    try:
        return sep_panel(tickers, start, field=field)
    except Exception:
        return sep_panel(tickers, start, field=fallback)


def _build_panel(tickers, sector_map, start=START):
    # ONLY cache-safe fields: closeadj (guaranteed default), volume,
    # and close with a closeadj fallback. NO 'open' (absent from the cache).
    fields = {
        "closeadj": sep_panel(tickers, start, field="closeadj"),
        "volume": sep_panel(tickers, start, field="volume"),
    }
    fields["close"] = _field_with_fallback(tickers, start, "close", "closeadj")

    common = fields["closeadj"].columns
    for df in fields.values():
        common = common.intersection(df.columns)
    common = pd.Index(sorted(common))
    panel = pd.concat({f: df[common] for f, df in fields.items()}, axis=1)

    # IWM hedge prices (ETF -> yfinance, not Sharadar SEP)
    iwm = yf_panel(["IWM"], start)
    panel = pd.concat([panel, pd.concat({"hedge": iwm}, axis=1)], axis=1)

    # embed the sector map as a one-row object block so signal() is self-contained
    sec = pd.DataFrame(
        [{t: sector_map.get(t, "Unknown") for t in common}], index=[panel.index[0]]
    )
    panel = pd.concat([panel, pd.concat({"sector": sec}, axis=1)], axis=1)
    return panel


def load_data() -> pd.DataFrame:
    tk, sm = _search_universe()
    return _build_panel(tk, sm)


def _tier2_slice(marketcap, search, deep_n, cap_per_sector):
    """Next liquidity tier within a cap band: pull deeper per-sector list, remove
    every search-universe name, then cap per sector -> DISJOINT by construction."""
    tk, sm = sector_universe(marketcap=marketcap, top_n_per_sector=deep_n)
    keep, per_sec = [], {}
    for t in tk:
        if t in search:
            continue
        s = sm.get(t, "Unknown")
        if per_sec.get(s, 0) < cap_per_sector:
            keep.append(t)
            per_sec[s] = per_sec.get(s, 0) + 1
    return keep, dict(sm)


def load_gen_data(label) -> pd.DataFrame:
    """Generalization universes — all DISJOINT from the Small+Mid search universe."""
    search_tk, _ = _search_universe()
    search = set(search_tk)

    if label == "small_tier2":
        tk, sm = _tier2_slice("Small", search, deep_n=90, cap_per_sector=35)
    elif label == "mid_tier2":
        tk, sm = _tier2_slice("Mid", search, deep_n=90, cap_per_sector=35)
    elif label == "large_cap":
        tk, sm = sector_universe(marketcap="Large", top_n_per_sector=30)
        tk = [t for t in tk if t not in search]
        sm = dict(sm)
    else:
        raise ValueError(f"unknown generalization universe: {label}")

    return _build_panel(tk, sm)


# ------------------------------------------------------------------- signal

def signal(panel, **params):
    p = {**DEFAULTS, **params}

    sector_map = panel["sector"].dropna(how="all").iloc[0].to_dict()
    closeadj = panel["closeadj"].astype(float)
    close = panel["close"].astype(float).replace(0.0, np.nan)
    volume = panel["volume"].astype(float)
    iwm_ret = panel["hedge"]["IWM"].astype(float).pct_change()

    rets = closeadj.pct_change()
    prev = rets.shift(1)

    # reversal-day indicator: today's return flips the sign of yesterday's
    # (yesterday's move big enough, real volume) -> liquidity-provision proxy
    qual = (prev.abs() > p["min_move"]) & (volume > 0)
    rev = ((np.sign(rets) * np.sign(prev)) < 0) & qual
    freq = rev.astype(float).rolling(p["lb"], min_periods=int(p["lb"] * 0.7)).mean()

    # tradability (trailing-only -> no lookahead beyond the explicit shift below)
    adv = (close * volume).rolling(21, min_periods=10).mean()
    eligible = (adv > p["min_adv"]) & (close > p["min_price"]) & freq.notna()

    idx = closeadj.index
    month_ends = pd.DatetimeIndex(
        pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last().values
    )

    # ---- long book: equal-weight top quintile at each month-end, held one month
    rows = {}
    for d in month_ends:
        f = freq.loc[d].where(eligible.loc[d]).dropna()
        if len(f) < 30:
            continue
        n = max(10, int(round(len(f) * p["top_frac"])))
        w = pd.Series(0.0, index=closeadj.columns)
        w[f.nlargest(n).index] = 1.0 / n
        rows[d] = w
    if not rows:
        empty = pd.Series(dtype=float, name="reversal_freq_lp_cc")
        return empty, []
    W_long = pd.DataFrame(rows).T.reindex(idx).ffill().fillna(0.0)

    # ---- IWM beta hedge (trailing 60d beta of the long book, capped 1.0x, monthly reset)
    r_long_pre = (W_long.shift(1) * rets).sum(axis=1)
    beta = (
        r_long_pre.rolling(p["beta_lb"]).cov(iwm_ret)
        / iwm_ret.rolling(p["beta_lb"]).var()
    )
    gross = W_long.sum(axis=1)
    hedge_raw = -beta.clip(lower=0.0, upper=p["beta_cap"]) * gross
    is_reb = pd.Series(idx.isin(month_ends), index=idx)
    h = hedge_raw.where(is_reb).ffill().fillna(0.0)

    # ---- assemble full weight matrices over a shared return panel (incl. IWM)
    rets_all = rets.copy()
    rets_all["IWM"] = iwm_ret
    W_alpha = W_long.reindex(columns=rets_all.columns, fill_value=0.0)
    W_hedge = pd.DataFrame(0.0, index=idx, columns=rets_all.columns)
    W_hedge["IWM"] = h

    # explicit 1-day execution lag, per-leg realistic costs (25bps names / 3bps IWM)
    r_alpha = net_of_cost(W_alpha.shift(1), rets_all, cost_bps=25.0,
                          name="reversal_freq_lp_cc")
    r_hedge = net_of_cost(W_hedge.shift(1), rets_all, cost_bps=3.0,
                          name="iwm_hedge")
    daily = r_alpha.add(r_hedge, fill_value=0.0).dropna()
    daily.name = "reversal_freq_lp_cc"

    sector_map = dict(sector_map)
    sector_map["IWM"] = "ETF-Hedge"
    trades = trades_from_weights((W_alpha + W_hedge).shift(1), rets_all, sector_map)

    return daily, trades


# --------------------------------------------------------------------- spec

SPEC = StrategySpec(
    id="reversal_freq_lp_cc_v1",
    family="liquidity_provision",
    title=("Reversal-frequency liquidity-provision premium (Akbas et al. 2022, "
           "close-to-close variant) — Small+Mid-cap long tilt, IWM beta-hedge"),
    markets=["us_smallmid_equity"],
    data_desc=("Sharadar SEP (survivorship-clean, delisted incl.): closeadj/close/volume "
               "for sector-spread Small+Mid caps (open NOT in the SEP cache -> "
               "close-to-close reversal decomposition); IWM via yfinance for the "
               "declared beta-hedge sleeve."),
    pre_registration=(
        "FROZEN before any run: monthly long-only book of the top reversal-frequency "
        "quintile (21d trailing freq of sign(ret_t)!=sign(ret_{t-1}), |ret_{t-1}|>25bps, "
        "vol>0) in ADV>$1M, price>$3 SMALL+MID caps (the pre-registered search "
        "cross-section); equal weight; single IWM short sized by trailing 60d beta "
        "capped at 1.0x (frozen cap), declared as hedge sleeve; 25bps/side names, "
        "3bps IWM. PREDICTIONS: (1) hedged book OOS-positive in the Small+Mid search "
        "universe and the untouched small_tier2 / mid_tier2 liquidity slices; "
        "(2) ATTENUATION up the cap spectrum — weakest/zero in large_cap (a "
        "large-cap-only pass is mechanism-INCONSISTENT and counts against the thesis); "
        "(3) MCPT must beat permuted panels (benchmark-relative null) to exclude "
        "residual microstructure artifacts. No parameter search beyond the declared "
        "grid (lb 21/42, move 25/15bps, quintile/decile)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "lb42": {"lb": 42},
        "move15bps": {"min_move": 0.0015},
        "decile": {"top_frac": 0.10},
    },
    scope="broad",
    generalization_universes=["small_tier2", "mid_tier2", "large_cap"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=25,
    hedge_tickers=["IWM"],
    hedge_cap=0.50,
)