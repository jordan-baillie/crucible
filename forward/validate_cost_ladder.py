"""forward/validate_cost_ladder.py — validate the FROZEN liquidity cost ladder against real
effective spreads, by dollar-volume decile (Leg B Phase 2; pre-reg §5 'modelable' check).

Why not realized fills: our books execute via opening-auction (OPG) orders, which JOIN the auction
and largely avoid the spread -> realized OPG fill-vs-open ~0, validating 'OPG is cheap', not the
ladder. The ladder models CROSSING cost (spread+impact). The right validation target is the real
effective SPREAD per decile. We estimate it with Corwin-Schultz (2012, JF) from daily high/low —
the academic standard when quotes aren't available (and US market is closed at run time).

Builds a decile-spanning universe from the frozen DV map, pulls ~1y daily OHLC (yfinance, free),
computes the CS proportional effective spread per name, aggregates the median per decile, and
compares to LADDER_CENTRAL / LADDER_CONSERVATIVE. Honest verdict: is the ladder CONSERVATIVE
(>= real spread, the safe direction) per decile, and within ±20%?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sdk import cost_model as cm  # noqa: E402

OUT = ROOT / "forward" / "cost_ladder_validation.json"
NAMES_PER_DECILE = 6
LOOKBACK_DAYS = 365


def decile_universe(dv_map: dict) -> dict:
    """{decile(1..10): [tickers]} — names sampled from each DV decile of the frozen map."""
    dv = pd.Series(dv_map, dtype=float).dropna()
    dv = dv[dv > 0]
    q = dv.rank(pct=True)
    out = {d: [] for d in range(1, 11)}
    # deterministic: within each decile take the median-DV names (avoid the boundary extremes)
    for d in range(1, 11):
        lo, hi = (d - 1) / 10, d / 10
        band = dv[(q > lo) & (q <= hi)]
        if band.empty:
            continue
        band = band.sort_values()
        mid = len(band) // 2
        pick = band.iloc[max(0, mid - NAMES_PER_DECILE): mid + NAMES_PER_DECILE]
        out[d] = list(pick.index)
    return out


def corwin_schultz_bps(ohlc: pd.DataFrame) -> float:
    """Median Corwin-Schultz proportional effective spread (bps, round-trip) over the window.
    ohlc: DataFrame with High, Low columns. Negative 2-day estimates set to 0 (per the paper)."""
    H = ohlc["High"].astype(float).values
    L = ohlc["Low"].astype(float).values
    if len(H) < 3:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(H / L) ** 2                       # single-day log(H/L)^2
        beta = hl[:-1] + hl[1:]                       # two-day sum
        Hm = np.maximum(H[:-1], H[1:])
        Lm = np.minimum(L[:-1], L[1:])
        gamma = np.log(Hm / Lm) ** 2
        k = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        S = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))  # proportional spread
    S = S[np.isfinite(S)]
    if S.size == 0:
        return np.nan
    S = np.where(S < 0, 0.0, S)                       # paper: negatives -> 0
    return float(np.median(S) * 1e4)                  # bps, round-trip


def fetch_ohlc(tickers: list) -> dict:
    import yfinance as yf
    if not tickers:
        return {}
    df = yf.download(tickers, period=f"{LOOKBACK_DAYS}d", interval="1d",
                     auto_adjust=False, progress=False, group_by="ticker", threads=True)
    out = {}
    for t in tickers:
        try:
            sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            sub = sub.dropna(subset=["High", "Low"])
            if len(sub) >= 30:
                out[t] = sub
        except Exception:
            continue
    return out


def main():
    dv = cm.dollar_volume_map()
    uni = decile_universe(dv)
    all_names = [t for names in uni.values() for t in names]
    print(f"[ladder-validate] {len(all_names)} names across 10 deciles; fetching OHLC...", flush=True)
    ohlc = fetch_ohlc(all_names)
    print(f"[ladder-validate] {len(ohlc)} names returned usable OHLC", flush=True)

    rows = []
    per_decile = {}
    for d in range(1, 11):
        spreads = []
        for t in uni[d]:
            if t in ohlc:
                s = corwin_schultz_bps(ohlc[t])
                if np.isfinite(s):
                    spreads.append(s)
                    rows.append({"decile": d, "ticker": t, "cs_spread_bps": round(s, 1),
                                 "dv_usd": round(dv.get(t.upper(), float("nan")), 0)})
        if spreads:
            real = float(np.median(spreads))
            lc, lv = cm.LADDER_CENTRAL[d], cm.LADDER_CONSERVATIVE[d]
            per_decile[d] = {
                "n": len(spreads), "real_cs_bps": round(real, 1),
                "ladder_central": lc, "ladder_conservative": lv,
                "central_over_real": round(lc / real, 2) if real > 0 else None,
                "central_conservative_(ladder>=real)": bool(lc >= real),
                "within_20pct": bool(real > 0 and abs(lc - real) / real <= 0.20),
            }
    # verdict
    deciles_scored = sorted(per_decile)
    conservative = sum(1 for d in deciles_scored if per_decile[d]["central_conservative_(ladder>=real)"])
    within20 = sum(1 for d in deciles_scored if per_decile[d]["within_20pct"])
    summary = {
        "method": "Corwin-Schultz (2012) effective spread vs frozen DV-decile ladder",
        "lookback_days": LOOKBACK_DAYS, "names_total": len(all_names), "names_usable": len(ohlc),
        "deciles_scored": len(deciles_scored),
        "central_ge_real_count": conservative, "within_20pct_count": within20,
        "per_decile": per_decile,
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    print("\n decile | n | real CS bps | ladder central | conservative? | within20%")
    for d in deciles_scored:
        p = per_decile[d]
        print(f"   {d:>2}   | {p['n']:>1} | {p['real_cs_bps']:>10} | {p['ladder_central']:>13} | "
              f"{str(p['central_conservative_(ladder>=real)']):>12} | {p['within_20pct']}")
    print(f"\n ladder central >= real spread in {conservative}/{len(deciles_scored)} deciles (conservative=safe)")
    print(f" ladder central within ±20% of real in {within20}/{len(deciles_scored)} deciles")
    print(f" artifact: {OUT}")


if __name__ == "__main__":
    main()
