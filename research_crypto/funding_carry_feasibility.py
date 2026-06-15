"""research_crypto/funding_carry_feasibility.py — honest Gate-0/Gate-1 feasibility of the
personally-tradable delta-neutral funding carry (long spot / short perp, collect funding),
across the 5 liquid majors, net of realistic crypto costs. EXPLORATORY (not a promotion run).

Operator 2026-06-15: pursue the funding-carry thread for personal use; re-validate under today's
fees; report whether there's a real, personally-tradable edge.

Finding (see the printed table): the carry is REAL but currently DORMANT — funding has compressed
from ~11-20% ann (2019-2021) to ~0-1% (BTC/ETH) / negative (SOL/XRP) in 2026. The conditional
strategy (hold only when trailing-7d ann funding > hurdle) earns ~3-4% net ann in 2024-26 at a high
Sharpe (delta-neutral -> tiny vol), but is FLAT / slightly negative over the recent 180d (funding
below the hurdle). The high Sharpe OVERSTATES risk: pure funding accrual ignores the basis/depeg/
liquidation tail that is the whole danger of this trade. Conclusion: real mechanism, regime-dependent,
currently not paying; 'have ready for the next high-funding regime', not 'earn steadily now'.

NO LOOK-AHEAD: signal from trailing funding through t-1 (shift(1)); delta-neutral so spot/perp price
moves cancel and the modeled return is the funding accrual minus transition costs.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sdk.adapters import funding_rates  # noqa: E402

MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
ONEWAY_BPS = 10.0  # ~Binance taker 4.5bps x 2 legs (spot+perp) per transition; round-trip ~20bps


def analyze(oneway_bps: float = ONEWAY_BPS):
    f = funding_rates(tuple(MAJORS))                       # daily funding accrual
    ann = f * 365
    trail = ann.rolling(7, min_periods=5).mean().shift(1)  # no-lookahead signal

    def run(hurdle_pct):
        sig = (trail > hurdle_pct / 100.0).astype(float)
        gross = sig * f
        cost = sig.diff().abs().fillna(sig.abs()) * (oneway_bps * 1e-4)
        net = gross - cost
        return net.mean(axis=1), gross.mean(axis=1), sig

    def stats(r):
        r = r.dropna()
        if len(r) < 30:
            return (np.nan, np.nan, np.nan)
        ann_r, vol = r.mean() * 365 * 100, r.std() * np.sqrt(365)
        return (ann_r, vol, (r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else np.nan)

    periods = [("FULL 2019-26", slice(None)), ("2022-2026", slice("2022-01-01", None)),
               ("2024-2026", slice("2024-01-01", None))]
    print(f"{'period':14s} {'hurdle':>7s} {'net_ann%':>8s} {'vol':>6s} {'Sharpe':>7s} {'gross%':>7s} {'#held':>6s}")
    for hurdle in (5, 10, 20):
        pn, pg, sig = run(hurdle)
        periods_h = periods + [("RECENT 180d", slice(pn.index[-180], None))]
        for label, sl in periods_h:
            a, v, s = stats(pn.loc[sl]); ga, _, _ = stats(pg.loc[sl])
            print(f"{label:14s} {hurdle:>6d}% {a:>8.1f} {v:>6.2f} {s:>7.2f} {ga:>7.1f} {sig.loc[sl].sum(axis=1).mean():>6.2f}")
        print()


if __name__ == "__main__":
    analyze()
