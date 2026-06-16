"""Retro-verification of amihud_illiq_tranched_v3's PRE-REGISTERED soft expectations.

The 2026-06-12 PASS verified all HARD gates but never measured the smith's own
pre-registered mechanism claims. This script computes them from the committed module
(sha1 1397c988fa9f) so the formation logic is byte-identical:

  (a) long-leg turnover: tranched must be <= 60% of the 'untranched' grid variant
  (b) tranche-phase dispersion: all 3 phases positive net sel-alpha; min >= 50% of avg
  (d) Amihud-quintile monotonicity within size terciles
  (+) sector breadth >= 4 per leg

(c) short-N graceful degradation is already visible in the recorded grid
(N=10 1.772 / N=15 1.889 / N=25 1.947 - no cliff).

Output: logs/retro_tranched_v3.json + stdout report.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODULE = REPO / "strategies" / "auto_amihud_illiquidity_premium_deployable_sh_smith1_99153.py"
OUT = REPO / "logs" / "retro_tranched_v3.json"

spec = importlib.util.spec_from_file_location(MODULE.stem, MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("[1/5] loading search panel (SEP small+mid, 5 sectors)...", flush=True)
panel = mod.load_data()
close, closeadj, volume = panel["close"], panel["closeadj"], panel["volume"]
rets = closeadj.pct_change()
stocks = [c for c in close.columns if c != mod.HEDGE]
dates = rets.index
print(f"    panel: {len(stocks)} stocks x {len(dates)} days", flush=True)

# ---------------------------------------------------------------- (a) turnover
# Capture the LAGGED long-leg weight matrix from inside signal() by wrapping the
# module's net_of_cost binding (called as net_of_cost(Wl.shift(1), ..., name="long")).
captured: dict = {}
_orig_noc = mod.net_of_cost

def _capture_noc(W, r, cost_bps=8.0, name="strategy"):
    captured[name] = W
    return _orig_noc(W, r, cost_bps=cost_bps, name=name)

results: dict = {"module_sha_expected": "1397c988fa9f"}

turn = {}
for label, params in (("tranched", {}), ("untranched", {"n_tranches": 1})):
    captured.clear()
    mod.net_of_cost = _capture_noc
    try:
        daily, _tr = mod.signal(panel, **params)
    finally:
        mod.net_of_cost = _orig_noc
    Wl = captured["long"]  # already shift(1)-lagged
    ann_turn = float((Wl - Wl.shift(1)).abs().sum(axis=1).mean() * 252)
    sr = float(daily.mean() / daily.std() * np.sqrt(252))
    turn[label] = {"annualized_long_leg_turnover": round(ann_turn, 3),
                   "net_sharpe": round(sr, 3)}
    print(f"[2/5] {label}: long-leg turnover {ann_turn:.2f}x/yr, net Sharpe {sr:.3f}", flush=True)

ratio = turn["tranched"]["annualized_long_leg_turnover"] / turn["untranched"]["annualized_long_leg_turnover"]
results["a_turnover"] = {**turn, "tranched_over_untranched": round(ratio, 3),
                         "claim": "ratio <= 0.60", "pass": bool(ratio <= 0.60)}

# ------------------------------------------------- shared trailing features
p = dict(amihud_lb=63, size_lb=126, n_short=15, long_band=1.5, short_band=1.6,
         cost_long_bps=30.0, cost_short_bps=7.5, borrow_rate=0.005,
         px_lo=10.0, px_hi=500.0, short_name_cap=0.10)
dvol = (close[stocks] * volume[stocks]).replace(0.0, np.nan)
amihud = ((rets[stocks].abs() / dvol) * 1e6).rolling(
    p["amihud_lb"], min_periods=int(p["amihud_lb"] * 0.6)).mean()
size = dvol.rolling(p["size_lb"], min_periods=60).median()
month_ends = pd.Series(dates, index=dates).groupby(dates.to_period("M")).max()
mkt = rets[stocks].mean(axis=1)  # universe EW return (for selection-alpha residual)

# ------------------------------------------- (b) tranche-phase dispersion
# Phase p standalone book: form a cohort via the module's own _form_cohort only at
# month-ends where (y*12+m)%3==p, hold (ffill) until the next formation 3 months
# later. Same per-leg costs + borrow; the IWM trim is excluded (it is a declared
# residual sleeve - sel-alpha is judged on the alpha book, matching the gate).
print("[3/5] tranche-phase decomposition...", flush=True)
phases, breadth_long, breadth_short = {}, [], []
for ph in range(3):
    prev_l, prev_s, rows = set(), set(), {}
    for d in month_ends:
        if (d.year * 12 + d.month) % 3 != ph:
            continue
        w, prev_l2, prev_s2 = mod._form_cohort(
            amihud.loc[d], size.loc[d], close[stocks].loc[d], prev_l, prev_s, p)
        if w is not None:
            rows[d] = w
            prev_l, prev_s = prev_l2, prev_s2
            breadth_long.append(len({mod._SECTORS.get(n, "?") for n in prev_l}))
            breadth_short.append(len({mod._SECTORS.get(n, "?") for n in prev_s}))
    W = (pd.DataFrame(rows).T.reindex(dates).ffill().fillna(0.0)
         .reindex(columns=stocks, fill_value=0.0))
    Wl, Ws = W.clip(lower=0.0), W.clip(upper=0.0)
    r = (_orig_noc(Wl.shift(1), rets, cost_bps=p["cost_long_bps"], name=f"l{ph}")
         + _orig_noc(Ws.shift(1), rets, cost_bps=p["cost_short_bps"], name=f"s{ph}")
         - (Ws.abs().sum(axis=1).shift(1) * p["borrow_rate"] / 252.0).fillna(0.0))
    r = r.dropna()
    m = mkt.reindex(r.index).fillna(0.0)
    beta = float(r.cov(m) / m.var())
    resid = r - beta * m
    phases[f"phase_{ph}"] = {
        "net_sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3),
        "beta_to_universe": round(beta, 3),
        "sel_alpha_sharpe": round(float(resid.mean() / resid.std() * np.sqrt(252)), 3),
    }
    print(f"    phase {ph}: {phases[f'phase_{ph}']}", flush=True)

sel = [v["sel_alpha_sharpe"] for v in phases.values()]
results["b_phase_dispersion"] = {
    **phases,
    "claim": "all sel-alpha > 0 AND min >= 50% of mean",
    "pass": bool(all(s > 0 for s in sel) and min(sel) >= 0.5 * (sum(sel) / 3)),
}

# ------------------------------------------------ (d) quintile monotonicity
# Within each size tercile at each month-end: EW forward 1-month return per Amihud
# quintile. Claim: premium increases with illiquidity (Spearman>0 and Q5>Q1).
print("[4/5] Amihud-quintile monotonicity...", flush=True)
me = list(month_ends)
mono = {}
for t in range(3):
    qrets = {q: [] for q in range(5)}
    for i, d in enumerate(me[:-1]):
        am_row, sz_row, px_row = amihud.loc[d], size.loc[d], close[stocks].loc[d]
        valid = am_row.notna() & sz_row.notna() & px_row.notna() & (px_row > 1.0)
        names = am_row.index[valid]
        if len(names) < 90:
            continue
        terc = pd.Series(pd.qcut(sz_row[names].rank(method="first"), 3, labels=False),
                         index=names)
        members = terc.index[terc == t]
        if len(members) < 25:
            continue
        quint = pd.Series(pd.qcut(am_row[members].rank(method="first"), 5, labels=False),
                          index=members)  # 4 = most illiquid
        fwd = (closeadj.loc[d:me[i + 1], members].iloc[-1]
               / closeadj.loc[d:me[i + 1], members].iloc[0] - 1.0)
        for q in range(5):
            qq = quint.index[quint == q]
            if len(qq):
                qrets[q].append(float(fwd[qq].mean()))
    avg = [float(np.mean(qrets[q])) * 12 for q in range(5)]  # annualized EW
    from scipy.stats import spearmanr
    rho = float(spearmanr(range(5), avg).statistic)
    mono[f"tercile_{t}"] = {"quintile_ann_returns_q1_to_q5": [round(x, 4) for x in avg],
                            "spearman": round(rho, 3),
                            "q5_minus_q1": round(avg[4] - avg[0], 4)}
    print(f"    tercile {t}: {mono[f'tercile_{t}']}", flush=True)

results["d_monotonicity"] = {
    **mono,
    "claim": "per tercile: spearman > 0 AND Q5 > Q1",
    "pass": bool(all(v["spearman"] > 0 and v["q5_minus_q1"] > 0 for v in mono.values())),
}

# ------------------------------------------------------------ sector breadth
results["sector_breadth"] = {
    "avg_sectors_long": round(float(np.mean(breadth_long)), 2),
    "avg_sectors_short": round(float(np.mean(breadth_short)), 2),
    "claim": ">= 4 per leg",
    "pass": bool(np.mean(breadth_long) >= 4 and np.mean(breadth_short) >= 4),
}

results["c_short_n_degradation"] = {
    "from_recorded_grid": {"short_n10": 1.772, "default_n15": 1.889, "short_n25": 1.947},
    "claim": "graceful (no cliff) toward smaller N",
    "pass": True,
    "note": "verified from the recorded verdict grid, not recomputed",
}

results["all_pass"] = bool(all(results[k]["pass"] for k in
                               ("a_turnover", "b_phase_dispersion", "d_monotonicity",
                                "sector_breadth", "c_short_n_degradation")))

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
print(f"[5/5] ALL PASS: {results['all_pass']} -> {OUT}", flush=True)
