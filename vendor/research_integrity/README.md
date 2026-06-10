# research_integrity — shared research-integrity rails

Portfolio-wide methodology so **every** edge-search project (Hermes, Midas, Credibility, Atlas) is
validated to the same standard. This is the discipline that caught Atlas's survivorship mirage and a
candidate that beat **DSR 0.986 + the FDR bar + in-search OOS yet failed the write-once holdout (−1.21)**.

Origin: Atlas `research/cross_oos` (best version; itself ported from the Midas #102 harness). Board
decision 2026-06-06 (`ceo-board/memos/2026-06-06-portfolio-reallocation-board`): promote to shared infra.

## Install (editable, per environment)

```bash
pip install -e /root/shared/research_integrity      # or add to PYTHONPATH
```

## The model

Everything is **pure functions over a per-period return series + a list of closed-trade dicts**, so it
bolts onto ANY backtest/walk-forward — you do NOT replace your backtester. A trade dict needs at least
`{ticker, pnl, exit_date, entry_regime}` (+ `entry_date, hold_days, position_value, sector` for
deployment-sanity).

## The 4 gates (use in this order)

```python
import os; os.environ["RESEARCH_INTEGRITY_DIR"] = "/root/hermes/research"   # per-project state dir
import research_integrity as ri

# 0. QUARANTINE: split your data; search only on data < holdout_start (e.g. 2025-01-01). The holdout
#    is NEVER seen during model/feature/hypothesis selection. (You enforce this in your own runner.)

# 1. SCORE the search-period result (your runner produced daily/per-bet returns + trades + a config grid)
bundle = ri.assemble_bundle(returns, trades, grid_returns=grid, forward_net=oos_net,
                            search_burden=burden)          # burden from your config sweep
n_fam  = ri.distinct_families(extra=ri.family_of("pitcher_k"))   # Rail 2: across-hypothesis count
tiers  = ri.evaluate_tiers(bundle["bundle"], promote_dsr=ri.promote_dsr(n_fam))  # FDR-aware bar

# 2. DEPLOYMENT-SANITY (Rail 3): did it actually deploy a real book? auto-FAIL artifacts.
dep = ri.deployment_sanity(trades, primary_config=cfg,
                           strategy_meta={"max_positions": N, "max_sector_concentration": K})
final_tier = "FAIL" if not dep["passed"] else tiers["tier"]

# 3. WRITE-ONCE HOLDOUT (Rail 1): ONLY for a PROMOTE candidate. Run your frozen model on the
#    quarantined holdout ONCE, then gate. Single-use: a burned config_hash cannot be re-tested.
h = ri.config_hash("pitcher_k", cfg, market="mlb")
if final_tier == "PROMOTE" and ri.ledger_lookup(h) is None:
    holdout_sharpe, deg, dep_h = your_holdout_run(cfg)     # YOUR runner on the holdout period
    passed, reasons = ri.holdout_gate(holdout_sharpe, deg, dep_h["passed"])
    ri.ledger_append({"config_hash": h, "passed": passed, "holdout_sharpe": holdout_sharpe, ...})
    if not passed: final_tier = "FAIL"   # burned

# 4. log the run to the registry (Rail 2 cumulative family count)
ri.append_run({"strategy":"pitcher_k","family":ri.family_of("pitcher_k"),"final_tier":final_tier, ...})
```

## Golden rules (non-negotiable, from Atlas's lessons)

1. **The holdout is the only incorruptible arbiter.** In-search OOS (a split WITHIN the searched period)
   is contaminated by hypothesis/feature selection and can pass while overfit. Never validate without it.
2. **Single-use.** Each `config_hash` may touch the holdout ONCE. Each distinct hypothesis tested vs the
   same holdout erodes its power — be deliberate; after a few nulls the no-edge conclusion firms.
3. **Deployment-sanity before trusting any tier.** A strategy that doesn't actually deploy (too few
   trades/bets, single-name/single-game concentration) auto-FAILs regardless of DSR.
4. **Pre-register, don't tune-to-rescue, accept the verdict.** Economic/literature plausibility is NOT a
   substitute for holdout validation (low-vol+reversal was plausible and still failed).
5. **FDR bar rises with the cumulative family count** (`promote_dsr(n_families)`): 1→0.90, 9→0.967, 100→0.99.

## Per-project adaptation notes

- **Hermes (betting):** map per-bet net-of-vig PnL → a returns series; trades = bets with
  `{ticker=game/pitcher, pnl, exit_date, entry_regime=season-phase}`. Deployment-sanity analog: enough
  bets, spread across games/pitchers (not one team), single-game share ≤ threshold. CLV can be an extra
  forward gate. Holdout = a held-out slice of the season never used for model selection.
- **Midas (crypto):** already has #102; reconcile to import this instead. Regime = market regime; group =
  asset/venue (use `splitters.leave_one_ticker_group_out`).
- **Atlas:** keeps its own `research/cross_oos` (mid-trial; do NOT refactor now). RECONCILE post-2026-08-01:
  make Atlas re-export from this package to remove drift. Until then this is a faithful copy.

## What's project-specific (you write it)
The shared package is SCORING + GATES. You provide the **runner**: produce search-period returns+trades,
quarantine + produce holdout returns+trades. projects write their own runners over the pure gates — they were
ATLAS reference runners (they lazy-import the Atlas engine) — copy their shape, not their guts.
