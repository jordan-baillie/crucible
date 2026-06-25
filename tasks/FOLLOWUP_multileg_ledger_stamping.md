# FOLLOW-UP: multi-leg ledger merges bypass kit regime-stamping (HY-credit 0.003 coverage)

**Opened 2026-06-25** from the retro impact audit
(`research-wiki/methodology/retro-impact-2026-06-25-regime-coverage-warmup.md`). Distinct from the
warmup-coverage amendment: this is a REAL stamping defect the coverage gate is correctly catching, not
a false positive. The conservative fallback keeps the strategy demoted — which is right — but the root
cause is a re-usable footgun that will recur.

## Root cause (verified)

`auto_us_high_yield_credit_carry_duration_hedg_smith4_65532` recorded `regime_coverage = 0.003`
(0.3% of trades carry a real `entry_regime`). It is NOT a warmup artifact — the warmup-exclusion fix
leaves it at 0.003 (panel unrecognised → legacy denominator; and even with a proxy it would stay low).

In `signal()`:
```python
trades = trades_from_weights(W, rets_df, SECTOR_MAP)   # credit leg: kit-stamped (good)
if tw != 0.0:
    trend_leg, tr_trades = _trend_overlay(...)         # _trend_overlay -> trend_returns()
    trades = list(trades) + list(tr_trades)            # <-- appends a SECOND, UNSTAMPED ledger
```
The credit-leg ledger is correctly stamped by `signal_kit.trades_from_weights` (which calls
`market_regime`). But `tr_trades` comes from the Boreas `trend_returns()` helper, which produces its
own ledger WITHOUT a regime stamp (`entry_regime` defaults to `'?'`). The trend leg has far more
trades than the 2-name credit book, so the merged ledger is ~99.7% `'?'`.

The kit’s contract (`agent/codegen.py`): “Never write entry_regime yourself; the kit’s labeller is the
standard … a ledger whose trades lack real entry_regime [is vacuous].” A multi-leg book that
**concatenates** ledgers silently violates this whenever any contributing helper isn’t stamped.

## This is a CLASS, not one strategy

Any strategy that builds its ledger by merging trades from multiple legs/helpers (trend overlays,
hedge sleeves, sub-books) inherits the bug if any leg’s ledger isn’t kit-stamped. Sweep candidates:
grep strategies for `trades = list(... ) + list(...)`, `+ tr_trades`, `+ hedge_trades`, or any append
to a `trades_from_weights` result. (Known: the trend-overlay pattern reused across the
illiquidity-trend and credit-carry families via `_trend_overlay`/`trend_returns()`.)

## Proposed structural fix (footgun-proof the merge, not just patch one caller)

1. **Single stamping entry point.** Make `trend_returns()` (and any helper that returns a ledger)
   return trades via `trades_from_weights(..., regimes=market_regime(<that leg's rets>))` so EVERY
   leg is stamped against its own panel. Prefer this over per-caller fixes.
2. **A merge helper that cannot drop stamps.** Add `signal_kit.merge_ledgers(*ledgers, rets=...)`
   that stamps any `'?'`-only contributor against a supplied/derived regime series before
   concatenation — so `trades = merge_ledgers(credit, trend, rets=combined_rets)` is the only
   blessed way to combine, and an unstamped leg is impossible to merge in silently.
3. **Codegen contract update** (`agent/codegen.py`): forbid bare `list(a)+list(b)` ledger merges;
   require `merge_ledgers`. State that appending a non-kit ledger is the canonical way to break the
   coverage gate.
4. **Keep the gate as the backstop.** The regime-coverage gate (post-warmup-fix) ALREADY catches this
   correctly — leave it; the fix is upstream so honest multi-leg books aren’t falsely demoted.

## Acceptance
- `trend_returns()` / `_trend_overlay` ledgers carry real `entry_regime`; HY-credit-smith4-65532
  re-runs with coverage well above 0.80 on its post-warmup trades (then judged on its actual merits,
  not on a stamping artifact).
- `signal_kit.merge_ledgers` added + unit-tested (an unstamped contributor gets stamped or the merge
  raises; a fully-stamped merge is unchanged).
- Repo sweep: every multi-leg ledger merge routed through `merge_ledgers`; list any strategy whose
  recorded coverage was depressed purely by an unstamped appended leg.
- Re-run the retro audit slice for the affected family; confirm no PROMOTE is demoted by a stamping
  artifact (vs a genuine regime/edge failure).

## Non-goals
- Do NOT stamp `entry_regime` by hand in strategy code (violates the kit-is-the-standard contract).
- Do NOT weaken the coverage gate to tolerate unstamped legs — the gate is correct; fix the producer.
