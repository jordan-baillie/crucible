"""Multiple-testing burden from the actual research search history.

The Deflated Sharpe Ratio must be deflated by the number of configurations actually
tried while searching for a strategy (and the dispersion of their Sharpes). Using the
on-the-fly validation grid (a handful of perturbations) badly *under*-counts that burden
and is gameable (shrink the grid -> DSR rises). This module sources the real burden from
the per-strategy experiment logs (research/results/<strategy>.tsv), so DSR reflects the
genuine search that produced the config.

Pure/deterministic except for reading the TSV logs. No network.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ATLAS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ATLAS_ROOT / "research" / "results"
TRADING_DAYS = 252


def search_burden(strategies, results_dir: Path | str | None = None,
                  periods_per_year: int = TRADING_DAYS) -> dict | None:
    """Estimate the multiple-testing burden for a set of strategies from their logs.

    Parameters
    ----------
    strategies : iterable of strategy names (config keys, == TSV stem).
    results_dir : directory of <strategy>.tsv experiment logs.

    Returns dict with:
      n_trials        : number of DISTINCT configurations tried (dedup of the per-row
                        ``params_changed`` deltas), summed across the given strategies.
                        This counts distinct hypotheses, not coordinate-descent micro-steps.
      n_experiments   : total experiment rows (incl. near-duplicate steps).
      sr_variance_pp  : variance of trial Sharpes in PER-PERIOD units (var(ann)/periods),
                        the dispersion term the DSR deflation needs.
      sr_variance_ann : same variance in annualized units (for reporting).
      strategies_found: which strategies had a usable log.
      source          : provenance string.
    Returns None if no usable history is found (caller should fall back to the grid).
    """
    rdir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    all_sharpes: list[float] = []
    n_distinct = 0
    n_rows = 0
    found: list[str] = []
    for strat in strategies:
        f = rdir / f"{strat}.tsv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f, sep="\t")
        except Exception:
            continue
        if "sharpe" not in df.columns:
            continue
        s = pd.to_numeric(df["sharpe"], errors="coerce").dropna()
        if s.empty:
            continue
        all_sharpes.extend(s.tolist())
        n_rows += len(df)
        if "params_changed" in df.columns:
            n_distinct += int(df["params_changed"].fillna("").nunique())
        else:
            n_distinct += int(len(s))
        found.append(strat)

    if len(all_sharpes) < 2 or n_distinct < 2:
        return None

    var_ann = float(np.var(np.asarray(all_sharpes, dtype=float), ddof=1))
    return {
        "n_trials": int(max(n_distinct, 2)),
        "n_experiments": int(n_rows),
        "sr_variance_pp": var_ann / periods_per_year,
        "sr_variance_ann": var_ann,
        "strategies_found": found,
        "source": "research/results/*.tsv (distinct params_changed signatures)",
    }


__all__ = ["search_burden", "DEFAULT_RESULTS_DIR", "TRADING_DAYS"]
