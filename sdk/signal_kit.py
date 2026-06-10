"""sdk/signal_kit.py — the tested boilerplate every generated strategy needs.

WHY: sampling the generated modules showed the LLM re-implementing the same ~150 lines
per file — cost model (25 files), trade-ledger run-length extractors (9+ variants),
cross-sectional z-score/winsorize (6+ variants with different quantiles), PIT fundamental
panels. Every duplicate is a fresh chance for a lookahead bug inside helpers the harness
can't see. The CONTRACT now mandates THESE; the only novel code per experiment is the
actual signal (the original design intent).

All functions are pure, lookahead-safe by construction, and unit-tested (tests/test_signal_kit.py).
Reference implementation: the deployed full-gate-pass val_mom_trend_smallcap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def xs_zscore(df: pd.DataFrame, winsor: tuple = (0.05, 0.95)) -> pd.DataFrame:
    """Cross-sectional (per-date) z-score with quantile winsorization.
    Winsorize first (kills single-name outliers driving the whole sort), then standardize.
    NaNs stay NaN (names without data never get a fake neutral score)."""
    lo = df.quantile(winsor[0], axis=1)
    hi = df.quantile(winsor[1], axis=1)
    w = df.clip(lower=lo, upper=hi, axis=0)
    mu = w.mean(axis=1)
    sd = w.std(axis=1).replace(0, np.nan)
    return w.sub(mu, axis=0).div(sd, axis=0)


def net_of_cost(W: pd.DataFrame, rets: pd.DataFrame, cost_bps: float = 8.0,
                name: str = "strategy") -> pd.Series:
    """Daily net-of-cost portfolio returns from a (LAGGED) weight matrix.

    W must already be lagged (weights known at yesterday's close, applied to today's
    returns) — pass W.shift(1) if you built same-day weights. Cost = turnover * bps.
    """
    gross = (W * rets).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1)
    net = (gross - turnover * cost_bps * 1e-4).fillna(0.0)
    net.name = name
    return net


def market_regime(rets: pd.DataFrame, trend_lb: int = 126, vol_lb: int = 63) -> pd.Series:
    """Per-date market-regime label from TRAILING data only (shift(1) — the label for day t
    uses information through t-1; a regime stamped with same-day data is lookahead).

    4 labels: bull/bear (sign of trailing equal-weight universe return) × calm/vol
    (trailing realized vol vs its expanding median). Coarse on purpose — the cross-regime
    gates need honest stratification, not a forecasting model.
    """
    mkt = rets.mean(axis=1)
    trend = mkt.rolling(trend_lb, min_periods=trend_lb // 2).sum()
    vol = mkt.rolling(vol_lb, min_periods=vol_lb // 2).std()
    vol_med = vol.expanding(min_periods=vol_lb).median()
    lab = pd.Series("?", index=rets.index)
    known = trend.notna() & vol.notna() & vol_med.notna()
    lab[known] = (np.where(trend[known] >= 0, "bull", "bear")
                  + np.where(vol[known] > vol_med[known], "_vol", "_calm"))
    return lab.shift(1).fillna("?")


def trades_from_weights(W: pd.DataFrame, rets: pd.DataFrame, sector_map: dict,
                        book: float = 1_000_000.0, min_weight: float = 1e-6,
                        regimes: "pd.Series | None" = None) -> list:
    """CONTRACT trade ledger from a weight matrix: one trade per contiguous held run
    per name (run-length extraction). Vectorized inner loop (numpy), matches the
    deployed val_mom implementation's ledger semantics.

    regimes: per-date label Series (use market_regime(rets)) — stamps each trade's
    entry_regime so the cross-regime robustness gates are REAL. Without it every trade
    is regime '?' and the three regime gates pass vacuously (I3).
    """
    if regimes is None:
        regimes = market_regime(rets)
    reg = regimes.reindex(W.index).fillna("?").astype(str).tolist()
    trades = []
    W_arr, R_arr = W.fillna(0.0).values, rets.reindex_like(W).fillna(0.0).values
    dstr = [d.strftime("%Y-%m-%d") for d in W.index]
    for cj, t in enumerate(W.columns):
        col = W_arr[:, cj]
        mask = np.abs(col) > min_weight
        if not mask.any():
            continue
        i, n = 0, len(col)
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                seg_w = col[i:j + 1]
                seg_r = R_arr[i:j + 1, cj]
                trades.append({
                    "ticker": t,
                    "sector": sector_map.get(t, "Unknown"),
                    "entry_date": dstr[i],
                    "exit_date": dstr[j],
                    "hold_days": int(j - i + 1),
                    "position_value": float(np.nanmean(seg_w) * book),
                    "pnl": float(np.nansum(seg_w * seg_r) * book),
                    "entry_regime": reg[i],
                })
                i = j + 1
            else:
                i += 1
    return trades


def pit_panel(sf1_df: pd.DataFrame, field: str, dates: pd.DatetimeIndex,
              tickers: list) -> pd.DataFrame:
    """Point-in-time fundamental panel: each value known only from its FILING DATE
    (datekey, never calendardate — that's lookahead), forward-filled daily.
    sf1_df: long frame from sdk.adapters.sf1 with [ticker, datekey, <field>]."""
    df = sf1_df.reset_index() if sf1_df.index.name else sf1_df.copy()
    cols = {c.lower(): c for c in df.columns}
    tcol, dcol = cols.get("ticker"), cols.get("datekey")
    vcol = cols.get(field.lower())
    if tcol is None or dcol is None or vcol is None:
        return pd.DataFrame(index=dates, columns=tickers, dtype=float)
    wide = (df.pivot_table(index=dcol, columns=tcol, values=vcol, aggfunc="last")
              .sort_index())
    return wide.reindex(index=dates.union(wide.index)).ffill().reindex(index=dates,
                                                                       columns=tickers)
