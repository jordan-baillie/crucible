"""Combinatorial Purged Cross-Validation (CPCV) — Midas #102.

Implements López de Prado's CPCV (Advances in Financial Machine Learning, ch. 7 & 12):
partition the timeline into N contiguous groups, hold out every C(N, k) combination of k
groups as test, and clean the training set with PURGING and EMBARGO to remove leakage from
overlapping label/holding windows and serial correlation.

Index-based and fully deterministic (no return-value dependence, no randomness, no I/O).
Mirrors the frozen-`Split` convention in funding_spot_carry/splits.py.

Definitions
-----------
- group: a contiguous block of observation indices.
- purge: drop training observations whose *label window* (an observation plus its `h`-bar
  holding horizon) overlaps any test observation. Concretely, drop the `h` training rows
  immediately *before* each test block (their label would peek into the test block) as well
  as the holding-horizon overlap *after*.
- embargo: additionally drop `ceil(embargo_pct * n_obs)` training rows immediately *after*
  each test block, to neutralise short-horizon serial correlation that purging alone misses.

Backtest paths: each of the N groups is covered by C(N-1, k-1) combinations, yielding
phi = (k / N) * C(N, k) reconstructed out-of-sample backtest paths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class CPCVSplit:
    """One combinatorial train/test split (integer index arrays into the observation axis)."""
    test_groups: tuple[int, ...]
    train_idx: np.ndarray = field(repr=False)
    test_idx: np.ndarray = field(repr=False)


def group_edges(n_obs: int, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal [start, stop) index ranges covering [0, n_obs)."""
    if n_obs <= 0:
        raise ValueError("n_obs must be > 0")
    if not 2 <= n_groups <= n_obs:
        raise ValueError("n_groups must satisfy 2 <= n_groups <= n_obs")
    edges = [int(round(i * n_obs / n_groups)) for i in range(n_groups + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n_groups)]


def n_backtest_paths(n_groups: int, k_test: int) -> int:
    """phi = (k / N) * C(N, k) — number of reconstructed OOS backtest paths."""
    if not 1 <= k_test < n_groups:
        raise ValueError("k_test must satisfy 1 <= k_test < n_groups")
    return math.comb(n_groups, k_test) * k_test // n_groups


def _purge_embargo(
    train_mask: np.ndarray,
    test_blocks: list[tuple[int, int]],
    n_obs: int,
    purge: int,
    embargo: int,
) -> np.ndarray:
    """Zero out training observations purged/embargoed around each contiguous test block."""
    for (a, b) in test_blocks:  # [a, b) test block
        # Purge the `purge` rows immediately before the block (their label peeks into test)
        if purge > 0:
            train_mask[max(0, a - purge):a] = False
        # Purge holding-horizon overlap + embargo immediately after the block
        tail = max(purge, 0) + max(embargo, 0)
        if tail > 0:
            train_mask[b:min(n_obs, b + tail)] = False
    return train_mask


def _contiguous_blocks(group_idx: tuple[int, ...], edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge selected groups into contiguous [start, stop) test blocks."""
    spans = sorted(edges[g] for g in group_idx)
    blocks: list[tuple[int, int]] = []
    for a, b in spans:
        if blocks and a == blocks[-1][1]:
            blocks[-1] = (blocks[-1][0], b)  # merge adjacent
        else:
            blocks.append((a, b))
    return blocks


def cpcv_splits(
    n_obs: int,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_pct: float = 0.01,
    purge: int = 1,
) -> list[CPCVSplit]:
    """All C(n_groups, k_test) purged+embargoed train/test splits.

    Parameters
    ----------
    n_obs : total number of observations (rows).
    n_groups : number of contiguous groups N (default 6).
    k_test : groups held out as test per split k (default 2).
    embargo_pct : embargo length as a fraction of n_obs (default 1%).
    purge : label/holding-horizon length in bars to purge around test blocks (default 1).
    """
    if not 1 <= k_test < n_groups:
        raise ValueError("k_test must satisfy 1 <= k_test < n_groups")
    edges = group_edges(n_obs, n_groups)
    embargo = int(math.ceil(embargo_pct * n_obs)) if embargo_pct > 0 else 0
    all_idx = np.arange(n_obs)

    splits: list[CPCVSplit] = []
    for combo in combinations(range(n_groups), k_test):
        test_mask = np.zeros(n_obs, dtype=bool)
        for g in combo:
            a, b = edges[g]
            test_mask[a:b] = True
        train_mask = ~test_mask
        blocks = _contiguous_blocks(combo, edges)
        train_mask = _purge_embargo(train_mask, blocks, n_obs, purge, embargo)
        splits.append(CPCVSplit(
            test_groups=tuple(combo),
            train_idx=all_idx[train_mask],
            test_idx=all_idx[test_mask],
        ))
    return splits


def has_leakage(split: CPCVSplit, purge: int = 1, embargo: int = 0) -> bool:
    """True if any train index sits within (purge before, purge+embargo after) a test index.

    Diagnostic used by tests to assert the purge/embargo actually removed adjacency.
    """
    test = set(split.test_idx.tolist())
    if not test:
        return False
    # No train index within `purge` before any test block start, or within
    # purge+embargo after any test block end.
    test_arr = np.array(sorted(test))
    # block boundaries
    starts = [test_arr[0]]
    ends = []
    for i in range(1, len(test_arr)):
        if test_arr[i] != test_arr[i - 1] + 1:
            ends.append(test_arr[i - 1])
            starts.append(test_arr[i])
    ends.append(test_arr[-1])
    train = set(split.train_idx.tolist())
    for a in starts:
        for d in range(1, purge + 1):
            if (a - d) in train:
                return True
    for b in ends:
        for d in range(1, purge + embargo + 1):
            if (b + d) in train:
                return True
    return False


__all__ = [
    "CPCVSplit", "group_edges", "n_backtest_paths", "cpcv_splits", "has_leakage",
]
