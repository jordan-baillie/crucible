"""Cross-asset / cross-venue / cross-regime splitters for the Cross-OOS harness (#102).

These provide OOS axes 2–4 from the plan (§2): an edge must survive being held out by
asset, by venue, and across market regimes — not just across time (CPCV handles time).

Index/label-based, deterministic. No I/O, no network, no randomness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupSplit:
    """Leave-one-group-out split over a categorical label axis (asset or venue)."""
    held_out: str
    train_idx: np.ndarray = field(repr=False)
    test_idx: np.ndarray = field(repr=False)


def leave_one_group_out(labels) -> list[GroupSplit]:
    """For each distinct label L: test = rows with label L, train = the rest.

    `labels` is a 1-D array/Series of categorical group labels (e.g. asset or venue per row).
    Splits are disjoint (test_L) and collectively cover every row exactly once as test.
    """
    arr = np.asarray(pd.Series(labels).to_numpy())
    n = arr.size
    if n == 0:
        return []
    idx = np.arange(n)
    out: list[GroupSplit] = []
    for lab in sorted({str(x) for x in arr.tolist()}):
        mask = np.array([str(x) == lab for x in arr.tolist()], dtype=bool)
        out.append(GroupSplit(held_out=lab, train_idx=idx[~mask], test_idx=idx[mask]))
    return out


def leave_one_asset_out(asset_labels) -> list[GroupSplit]:
    """Cross-asset OOS: hold out one asset (group) at a time."""
    return leave_one_group_out(asset_labels)


def leave_one_venue_out(venue_labels) -> list[GroupSplit]:
    """Cross-venue OOS: hold out one venue at a time (calibrate on the rest)."""
    return leave_one_group_out(venue_labels)


def regime_labels(
    prices,
    trend_window: int = 200,
    chop_band: float = 0.05,
) -> pd.Series:
    """Classify each observation into 'bull' / 'bear' / 'chop' from a reference price series.

    Deterministic trend-band classifier on a benchmark (e.g. BTC) price:
      dev = price / SMA(trend_window) - 1
      dev >  +chop_band → 'bull'
      dev <  -chop_band → 'bear'
      otherwise         → 'chop'
    Rows before the SMA is defined are labelled 'unknown' (excluded by regime_stratify).
    """
    s = pd.Series(np.asarray(prices, dtype=float))
    if trend_window < 1:
        raise ValueError("trend_window must be >= 1")
    sma = s.rolling(trend_window, min_periods=trend_window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = s / sma - 1.0
    out = np.full(len(s), "unknown", dtype=object)
    known = sma.notna().to_numpy()
    devv = dev.to_numpy()
    out[known & (devv > chop_band)] = "bull"
    out[known & (devv < -chop_band)] = "bear"
    out[known & (np.abs(devv) <= chop_band)] = "chop"
    res = pd.Series(out)
    if isinstance(prices, pd.Series):
        res.index = prices.index
    return res


def regime_stratify(labels, include_unknown: bool = False) -> dict[str, np.ndarray]:
    """Map each regime label → integer index array of observations in that regime."""
    arr = np.asarray(pd.Series(labels).to_numpy())
    idx = np.arange(arr.size)
    out: dict[str, np.ndarray] = {}
    for lab in sorted({str(x) for x in arr.tolist()}):
        if lab == "unknown" and not include_unknown:
            continue
        out[lab] = idx[np.array([str(x) == lab for x in arr.tolist()], dtype=bool)]
    return out


__all__ = [
    "GroupSplit", "leave_one_group_out", "leave_one_asset_out", "leave_one_venue_out",
    "regime_labels", "regime_stratify",
]
