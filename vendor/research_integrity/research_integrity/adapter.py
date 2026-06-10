"""Atlas wiring for the strategy-agnostic cross-OOS battery.

Bridges BacktestEngine output (a daily mark-to-market ``equity_curve`` + a list of
closed-trade dicts) to the pure ``cross_oos`` modules. Nothing here imports the engine,
so every function is unit-testable on synthetic inputs.

Axes (López de Prado four-axis, selection-aware battery):
  - cross-TIME    : CPCV path distribution + Deflated Sharpe over the daily return series.
  - cross-CONFIG  : PBO (CSCV) over a grid of config variants' daily returns.
  - cross-TICKER  : leave-one-random-ticker-group-out (5 seeded groups) + concentration.
  - cross-REGIME  : stratify trade PnL by each trade's ``entry_regime``.

Atlas-specific divergences from the Midas crypto gate table:
  - Annualisation uses 252 trading days (equities), not 365.
  - The cross-VENUE axis is dropped (single venue) and replaced by the ticker-group axis.
  - The explicit 10 bps/side cost-stress gate is dropped: Atlas backtests are already run
    net of realistic commissions, so the whole battery runs on net returns. (A doubled-fee
    re-run can be re-added as an optional gate later.)

Pure functions over numpy/pandas; no I/O, no network, no engine import.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cpcv, gates, overfitting, splitters
from . import metrics as cm

TRADING_DAYS = 252  # equities annualisation factor

# --- Equities-tuned hard-gate table. Same engine/semantics as gates.DEFAULT_GATES
#     (missing measurement == FAIL), thresholds and axes adapted for Atlas. ---
ATLAS_DEFAULT_GATES: dict[str, tuple[str, str, float, str]] = {
    "median_cpcv_sharpe": ("median_cpcv_sharpe", ">=", 0.5,
                           "Median CPCV daily-return Sharpe (annualised) >= 0.5"),
    "frac_paths_positive": ("frac_paths_positive", ">=", 0.55,
                            ">=55% of CPCV paths net-positive"),
    "pbo": ("pbo", "<=", 0.50, "Probability of Backtest Overfitting <= 0.50"),
    "dsr": ("dsr", ">=", 0.90, "Deflated Sharpe Ratio significant at 90%"),
    "top_group_frac": ("top_group_frac", "<", 0.50,
                       "Top ticker < 50% of net PnL (concentration)"),
    "loo_group_ok": ("loo_group_ok", "is_true", 1.0,
                     "Leave-one-ticker-group-out: net Sharpe stays positive on all holdouts"),
    "min_regime_sharpe": ("min_regime_sharpe", ">=", -0.5,
                          "No catastrophic regime (min regime Sharpe >= -0.5)"),
    "regime_concentration_ratio": ("regime_concentration_ratio", "<=", 2.0,
                                   "Regime profit-share/time-share ratio <= 2.0 "
                                   "(frequency-weighted; replaces flat PnL-share cap per board)"),
    "per_regime_expectancy_ok": ("per_regime_expectancy_ok", "is_true", 1.0,
                                 "Net-positive expectancy in every regime with >= min trades"),
    "oos_cagr_degradation_ok": ("oos_cagr_degradation_ok", "is_true", 1.0,
                                "OOS CAGR degradation <= 50% vs in-sample"),
    "forward_net": ("forward_net", ">", 0.0, "OOS (forward holdout) net PnL > 0"),
}

# Gates that the battery can always measure from a single full-period run + a config grid.
# Time-split gates (forward_net, oos_cagr_degradation_ok) are added by the runner when it
# has IS/OOS results.
CORE_GATE_KEYS = (
    "median_cpcv_sharpe", "frac_paths_positive", "pbo", "dsr",
    "top_group_frac", "loo_group_ok", "min_regime_sharpe",
    "regime_concentration_ratio", "per_regime_expectancy_ok",
)

# Two-tier Deflated-Sharpe bars. SCREEN = "promising, keep researching / paper";
# PROMOTE = "clears the multiple-testing bar, may authorize a live config promotion".
# Only a PROMOTE pass maps to summary.overall_verdict == "PASS".
SCREEN_DSR = 0.70
PROMOTE_DSR = 0.90  # == ATLAS_DEFAULT_GATES['dsr'] threshold
PROMOTE_DSR_CAP = 0.99  # FDR-aware bar never exceeds this (some idea must be promotable)


def promote_dsr(n_families: int, base: float = PROMOTE_DSR, cap: float = PROMOTE_DSR_CAP) -> float:
    """Rail 2 — FDR-aware PROMOTE bar that escalates with the cumulative count of DISTINCT
    hypothesis families tested (cross-family multiple testing). Pre-registered Sidak-flavored form:

        promote_dsr = min(cap, 1 - (1 - base) / sqrt(max(1, n_families)))

    n=1 -> 0.90 (today's behaviour, regression-safe); n=4 -> 0.95; n=9 -> ~0.967; n>=100 -> cap 0.99.
    The within-family config search is already handled by the effective-N DSR (search_history.py);
    this corrects the ACROSS-family burden at the promotion threshold only (no double-counting).
    """
    import math
    n = max(1, int(n_families))
    return float(min(cap, 1.0 - (1.0 - base) / math.sqrt(n)))


# ---------------------------------------------------------------------------
# Return-series helpers (cross-TIME axis)
# ---------------------------------------------------------------------------
def daily_returns(equity_curve) -> pd.Series:
    """Daily fractional returns from a mark-to-market equity Series (index preserved)."""
    e = pd.Series(equity_curve, dtype=float).dropna()
    if len(e) < 2:
        return pd.Series(dtype=float)
    r = e.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return r


def cpcv_path_sharpes(returns, n_groups: int = 6, k_test: int = 2,
                      periods: int = TRADING_DAYS) -> list[float]:
    """Annualised Sharpe of each CPCV test block over a per-period return series."""
    r = np.asarray(pd.Series(returns, dtype=float).dropna(), dtype=float)
    if r.size < n_groups:
        return []
    out: list[float] = []
    for s in cpcv.cpcv_splits(r.size, n_groups, k_test):
        seg = r[s.test_idx]
        sr = cm.annualized_sharpe(seg, periods)
        if sr == sr:  # drop NaN
            out.append(float(sr))
    return out


def build_pbo_matrix(return_series: dict | list) -> tuple[np.ndarray, list[str]]:
    """Align a set of config-variant daily-return Series on their common dates.

    Accepts a dict {label: Series} or a list of Series. Returns (matrix T x N, labels)
    where N>=2 columns are required by PBO. Rows with any NaN are dropped.
    """
    if isinstance(return_series, dict):
        labels = [str(k) for k in return_series.keys()]
        series = list(return_series.values())
    else:
        labels = [f"cfg{i}" for i in range(len(return_series))]
        series = list(return_series)
    frame = pd.DataFrame({lab: pd.Series(s, dtype=float) for lab, s in zip(labels, series)})
    frame = frame.dropna(how="any")
    return frame.to_numpy(dtype=float), labels


# ---------------------------------------------------------------------------
# Trade-attribution helpers (cross-TICKER + cross-REGIME axes)
# ---------------------------------------------------------------------------
def group_daily_pnl(trades, group_key: str = "ticker") -> pd.DataFrame:
    """Pivot closed trades into a (date x group) net-PnL frame, keyed by exit date."""
    rows = []
    for t in trades or []:
        ed = t.get("exit_date")
        if ed is None:
            continue
        rows.append((pd.Timestamp(ed).normalize(), str(t.get(group_key, "?")),
                     float(t.get("pnl", 0.0) or 0.0)))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "group", "pnl"])
    piv = df.pivot_table(index="date", columns="group", values="pnl", aggfunc="sum")
    return piv.fillna(0.0).sort_index()


def top_group_frac(group_pnl: pd.DataFrame) -> float:
    """Largest single group's share of total absolute net PnL (concentration)."""
    if group_pnl is None or group_pnl.empty:
        return float("nan")
    net = group_pnl.sum(axis=0)
    denom = float(net.abs().sum())
    if denom <= 0:
        return float("nan")
    return float(net.abs().max() / denom)


def leave_one_ticker_group_out(group_pnl: pd.DataFrame, n_groups: int = 5,
                               seed: int = 0, periods: int = TRADING_DAYS,
                               min_sharpe: float = 0.0) -> dict:
    """Partition tickers into ``n_groups`` random groups; for each, drop it and check the
    remaining portfolio's daily-PnL Sharpe stays above ``min_sharpe`` (Midas #110 method).

    Returns {"ok": bool, "sharpes": {group_index: sharpe}, "n_groups": int}.
    """
    if group_pnl is None or group_pnl.empty or group_pnl.shape[1] < 2:
        return {"ok": False, "sharpes": {}, "n_groups": 0}
    cols = list(group_pnl.columns)
    n_groups = min(n_groups, len(cols))
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(cols), n_groups)
    total = group_pnl.sum(axis=1)
    sharpes: dict[str, float] = {}
    ok = True
    for gi, held in enumerate(parts):
        remaining = total - group_pnl[list(held)].sum(axis=1)
        sr = cm.annualized_sharpe(remaining.to_numpy(dtype=float), periods)
        sharpes[str(gi)] = float(sr) if sr == sr else float("nan")
        if not (sr == sr and sr > min_sharpe):
            ok = False
    return {"ok": ok, "sharpes": sharpes, "n_groups": int(n_groups)}


def regime_attribution(trades, periods: int = TRADING_DAYS,
                       min_regime_trades: int = 5) -> dict:
    """Stratify trade PnL by ``entry_regime`` and compute regime-robustness measures.

    Board memo 2026-06-03 (regime-concentration-gate-calibration): the old flat
    ``max_regime_pnl_frac`` cap measured the (bull-heavy) sample mix, not fragility. It is
    replaced by two frequency-aware measures (and kept only as a diagnostic):
      - regime_concentration_ratio = max_r( profit_share_r / time_share_r ), where time_share
        is the regime's share of trades. ~1.0 means profit tracks regime frequency; a regime
        that is a small share of activity but a large share of profit scores high. Gate <= 2.0.
      - per_regime_expectancy_ok = every regime with >= ``min_regime_trades`` trades has
        net-positive PnL (penalises genuine fragility, not mere upside concentration).
    """
    reg_pnl = group_daily_pnl(trades, group_key="entry_regime")
    counts: dict = {}
    for t in (trades or []):
        counts[str(t.get("entry_regime", "?"))] = counts.get(str(t.get("entry_regime", "?")), 0) + 1
    total_ct = sum(counts.values())
    if reg_pnl is None or reg_pnl.empty or total_ct == 0:
        return {"regime_sharpe": {}, "regime_net": {}, "regime_counts": {},
                "regime_timeshare": {}, "min_regime_sharpe": float("nan"),
                "max_regime_pnl_frac": float("nan"),
                "regime_concentration_ratio": float("nan"), "per_regime_expectancy_ok": False}
    regime_sharpe = {str(r): float(cm.annualized_sharpe(reg_pnl[r].to_numpy(dtype=float), periods))
                     for r in reg_pnl.columns}
    regime_net = {str(r): float(reg_pnl[r].sum()) for r in reg_pnl.columns}
    finite_sr = [v for v in regime_sharpe.values() if v == v]
    total_abs = sum(abs(v) for v in regime_net.values()) + 1e-12
    max_frac = max((abs(v) / total_abs for v in regime_net.values()), default=float("nan"))
    timeshare = {r: counts.get(r, 0) / total_ct for r in regime_net}
    total_pos = sum(max(v, 0.0) for v in regime_net.values())
    ratios = []
    for r, net in regime_net.items():
        ts = timeshare.get(r, 0.0)
        ps = (max(net, 0.0) / total_pos) if total_pos > 0 else 0.0
        ratios.append((ps / ts) if ts > 0 else float("inf"))
    concentration_ratio = max(ratios) if ratios else float("nan")
    per_regime_ok = all(net > 0 for r, net in regime_net.items()
                        if counts.get(r, 0) >= min_regime_trades)
    return {
        "regime_sharpe": regime_sharpe,
        "regime_net": regime_net,
        "regime_counts": counts,
        "regime_timeshare": timeshare,
        "min_regime_sharpe": min(finite_sr) if finite_sr else float("nan"),
        "max_regime_pnl_frac": float(max_frac),
        "regime_concentration_ratio": float(concentration_ratio),
        "per_regime_expectancy_ok": bool(per_regime_ok),
    }


# ---------------------------------------------------------------------------
# Battery assembly
# ---------------------------------------------------------------------------
def assemble_bundle(
    primary_returns,
    trades,
    grid_returns: dict | list | None = None,
    *,
    forward_net: float | None = None,
    oos_cagr_degradation_pct: float | None = None,
    search_burden: dict | None = None,
    n_groups: int = 6,
    k_test: int = 2,
    periods: int = TRADING_DAYS,
    seed: int = 0,
) -> dict:
    """Compute the full gate bundle + per-axis diagnostics from engine outputs.

    Parameters
    ----------
    primary_returns : daily fractional return Series of the validated config (full period).
    trades          : list of closed-trade dicts (need ticker, pnl, exit_date, entry_regime).
    grid_returns    : {label: daily-return Series} for the config grid (PBO/DSR). Optional;
                      if <2 columns survive alignment, PBO/DSR are reported as NaN.
    forward_net     : OOS-split net PnL (forward holdout). Optional (added by the runner).
    oos_cagr_degradation_pct : IS->OOS CAGR degradation in pct (negative == worse). Optional.

    Returns {"bundle": <gate inputs>, "diagnostics": <full detail>}.
    """
    pr = pd.Series(primary_returns, dtype=float).dropna()
    paths = cpcv_path_sharpes(pr, n_groups=n_groups, k_test=k_test, periods=periods)

    # PBO + DSR over the config grid
    pbo_val = float("nan")
    dsr_val = float("nan")
    pbo_detail: dict = {"n_combos": 0, "n_configs": 0}
    mat = None
    if grid_returns is not None:
        mat, labels = build_pbo_matrix(grid_returns)
        if mat.ndim == 2 and mat.shape[1] >= 2 and mat.shape[0] >= 8:
            n_splits = min(16, mat.shape[0] - (mat.shape[0] % 2))
            if n_splits >= 2:
                res = overfitting.pbo_cscv(mat, n_splits=n_splits)
                pbo_val = res["pbo"]
                pbo_detail = {"n_combos": res["n_combos"], "n_configs": res["n_configs"],
                              "n_splits": n_splits, "labels": labels}
            cfg_sr = np.array([cm.sharpe(mat[:, j], 1) for j in range(mat.shape[1])])
            cfg_sr = cfg_sr[np.isfinite(cfg_sr)]
            if cfg_sr.size >= 2:
                dsr_val = overfitting.deflated_sharpe_ratio(
                    sr=cm.sharpe(pr.to_numpy(), 1), n_obs=len(pr),
                    n_trials=mat.shape[1], sr_variance=float(np.var(cfg_sr)),
                    skew=cm.skewness(pr.to_numpy()), kurtosis=cm.kurtosis(pr.to_numpy()),
                )

    # Authoritative DSR: deflate by the REAL research search burden when available
    # (n_trials = distinct configs tried; sr_variance from the experiment log). The grid
    # DSR above (dsr_grid) is a local proxy that under-counts the search and is gameable.
    #
    # EFFECTIVE-N (board memo #413): the raw distinct-config count treats correlated
    # coordinate-descent configs as independent trials, over-stating the multiple-testing
    # burden. When the config-grid return matrix is available, estimate the independent
    # fraction via its eigenvalue participation ratio and haircut the raw count:
    #   effective_n = clip(round(raw_n * participation_ratio / n_grid),
    #                      [max(participation_ratio, 5), raw_n]).
    dsr_grid = dsr_val
    dsr_source = "grid"
    dsr_n_raw = None
    dsr_n_effective = None
    grid_participation = None
    if search_burden and search_burden.get("n_trials", 0) >= 2 \
            and search_burden.get("sr_variance_pp", 0) > 0 and len(pr) >= 2:
        raw_n = int(search_burden["n_trials"])
        eff_n = raw_n
        if mat is not None and mat.ndim == 2 and mat.shape[1] >= 2:
            grid_participation = overfitting.effective_num_trials(mat)
            if grid_participation == grid_participation:  # not NaN
                frac = grid_participation / mat.shape[1]
                lo = min(max(int(round(grid_participation)), 5), raw_n)
                eff_n = int(np.clip(round(raw_n * frac), lo, raw_n))
        dsr_n_raw, dsr_n_effective = raw_n, eff_n
        dsr_val = overfitting.deflated_sharpe_ratio(
            sr=cm.sharpe(pr.to_numpy(), 1), n_obs=len(pr),
            n_trials=eff_n, sr_variance=float(search_burden["sr_variance_pp"]),
            skew=cm.skewness(pr.to_numpy()), kurtosis=cm.kurtosis(pr.to_numpy()),
        )
        dsr_source = "search_history_effective_n"

    # Ticker concentration + leave-one-ticker-group-out
    tkr_pnl = group_daily_pnl(trades, group_key="ticker")
    top_frac = top_group_frac(tkr_pnl)
    loo = leave_one_ticker_group_out(tkr_pnl, n_groups=5, seed=seed, periods=periods)

    # Regime stratification
    reg = regime_attribution(trades, periods=periods)

    bundle = {
        "median_cpcv_sharpe": float(np.median(paths)) if paths else float("nan"),
        "frac_paths_positive": float(np.mean([p > 0 for p in paths])) if paths else float("nan"),
        "pbo": pbo_val,
        "dsr": dsr_val,
        "top_group_frac": top_frac,
        "loo_group_ok": bool(loo["ok"]),
        "min_regime_sharpe": reg["min_regime_sharpe"],
        "regime_concentration_ratio": reg["regime_concentration_ratio"],
        "per_regime_expectancy_ok": bool(reg["per_regime_expectancy_ok"]),
    }
    if forward_net is not None:
        bundle["forward_net"] = float(forward_net)
    if oos_cagr_degradation_pct is not None:
        bundle["oos_cagr_degradation_ok"] = bool(oos_cagr_degradation_pct >= -50.0)

    diagnostics = {
        "cpcv": {
            "n_paths": len(paths),
            "median_sharpe": bundle["median_cpcv_sharpe"],
            "frac_positive": bundle["frac_paths_positive"],
            "min": float(np.min(paths)) if paths else float("nan"),
            "max": float(np.max(paths)) if paths else float("nan"),
            "n_groups": n_groups, "k_test": k_test,
        },
        "pbo": {"value": pbo_val, **pbo_detail},
        "dsr": dsr_val,
        "dsr_grid": dsr_grid,
        "dsr_source": dsr_source,
        "dsr_n_trials_raw": dsr_n_raw,
        "dsr_n_trials_effective": dsr_n_effective,
        "grid_participation_ratio": grid_participation,
        "search_burden": search_burden,
        "ticker_concentration": {"top_group_frac": top_frac, "n_tickers": int(tkr_pnl.shape[1]) if not tkr_pnl.empty else 0},
        "leave_one_ticker_group_out": loo,
        "regime": reg,
        "n_obs": int(len(pr)),
    }
    if forward_net is not None:
        diagnostics["forward_net"] = float(forward_net)
    if oos_cagr_degradation_pct is not None:
        diagnostics["oos_cagr_degradation_pct"] = float(oos_cagr_degradation_pct)

    return {"bundle": bundle, "diagnostics": diagnostics}


def evaluate(bundle: dict, thresholds: dict | None = None) -> dict:
    """Evaluate a battery bundle against the Atlas gate table (missing == FAIL)."""
    table = dict(thresholds or ATLAS_DEFAULT_GATES)
    # Only enforce time-split gates when the runner actually measured them.
    if "forward_net" not in bundle:
        table.pop("forward_net", None)
    if "oos_cagr_degradation_ok" not in bundle:
        table.pop("oos_cagr_degradation_ok", None)
    return gates.evaluate_gates(bundle, thresholds=table)


def _gates_with_dsr(dsr_threshold: float) -> dict:
    """Copy of ATLAS_DEFAULT_GATES with the DSR threshold overridden."""
    g = dict(ATLAS_DEFAULT_GATES)
    key, comp, _thr, _desc = g["dsr"]
    g["dsr"] = (key, comp, dsr_threshold,
                f"Deflated Sharpe Ratio >= {dsr_threshold} (search-history deflated)")
    return g


def evaluate_tiers(bundle: dict, *, screen_dsr: float = SCREEN_DSR,
                   promote_dsr: float = PROMOTE_DSR) -> dict:
    """Two-tier evaluation. Every gate is identical between tiers except the DSR bar.

    Returns {tier, promote, screen} where tier is:
      - "PROMOTE": clears the strict DSR bar -> may authorize a live config promotion.
      - "SCREEN":  clears every gate at the looser DSR bar -> promising, keep researching.
      - "FAIL":    fails a non-DSR gate, or DSR below the screen bar.
    Only "PROMOTE" should map to summary.overall_verdict == "PASS".
    """
    promote = evaluate(bundle, _gates_with_dsr(promote_dsr))
    screen = evaluate(bundle, _gates_with_dsr(screen_dsr))
    if promote["overall_pass"]:
        tier = "PROMOTE"
    elif screen["overall_pass"]:
        tier = "SCREEN"
    else:
        tier = "FAIL"
    return {"tier": tier, "promote": promote, "screen": screen,
            "screen_dsr": screen_dsr, "promote_dsr": promote_dsr}


__all__ = [
    "TRADING_DAYS", "ATLAS_DEFAULT_GATES", "CORE_GATE_KEYS",
    "daily_returns", "cpcv_path_sharpes", "build_pbo_matrix",
    "group_daily_pnl", "top_group_frac", "leave_one_ticker_group_out",
    "regime_attribution", "assemble_bundle", "evaluate", "evaluate_tiers",
    "SCREEN_DSR", "PROMOTE_DSR", "PROMOTE_DSR_CAP", "promote_dsr",
]
