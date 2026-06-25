# FOLLOW-UP: `_price_matrix` panel-recognition gap (regime/beta/breadth go silently inert)

**Opened 2026-06-25** from the retro impact audit
(`research-wiki/methodology/retro-impact-2026-06-25-regime-coverage-warmup.md`). Sibling of the
just-shipped composite-field fix (crucible@`368aa9d`), which recognised `eq_px`/`etf_px` panels but
**7 of 17** audited panels remain unrecognised → their long-only benchmark + `regime_fragile`,
`beta_confound`, `breadth_overfit` gates stay `not_evaluated`, and the regime-coverage warmup boundary
falls back to the conservative legacy denominator. **Not a regression** (these never resolved) — a
pre-existing footgun: gates that silently do nothing while showing a green/“not_evaluated” badge.

## Evidence (the 7 still-unrecognised panels, two distinct sub-classes)

### Sub-class A — SEMANTIC price-field names that contain no price token
`_price_matrix` / `_price_fields_composite` recognise a field only if it equals or has an
underscore-token in `{"px","close","closeadj","price","prices","adj_close"}`. These pack real PRICE
cross-sections under names that miss that test:
- `auto_crypto_delta_neutral_basis_carry_cross_s_smith3_45792`: `pd.concat([perp, spot])` → fields
  `perp`, `spot` (both are close prices).
- `auto_storage_confirmed_commodity_curve_carry__omdtr1_82730`: `{"fut","carry","inv"}` → `fut` is
  the futures price.
- `auto_delta_neutral_commodity_term_structure_c_omdtr3_83283`: `(root, field)` MultiIndex whose
  price field is a commodity-specific name.
- `auto_treasury_auction_supply_concession_liqui_smith4_65016`: a FLAT ETF price frame with
  `.attrs["auctions"]`; unrecognised likely because it has `< 5` columns (the flat-panel branch
  requires `shape[1] >= 5`).

### Sub-class B — panels with NO price level at all (returns/metrics)
A price cross-section genuinely cannot be extracted; the regime/beta proxy needs a different source:
- `auto_commodity_convenience_yield_dynamics_cro_omdtr3_80629`: `{"carry","ret"}` (returns, not prices).
- `auto_cross_sectional_on_chain_value_premium_i_smith3_22992`: on-chain metric panels (no price).
- `auto_us_high_yield_credit_carry_duration_hedg_smith4_65532`: 2-name credit/rates book — see the
  separate stamping follow-up; its panel is also non-standard (`ctx["panel"]`).

## Proposed structural fix (declare, don't guess — make the class unrepresentable)

Guessing field names by token is inherently fragile and unbounded (every new asset class invents a
name). Replace it with an explicit, single-sourced declaration, heuristic only as legacy fallback:

1. **`StrategySpec.market_proxy` (optional Callable | field spec).** A strategy may declare how to
   build its (dates × assets) market proxy — e.g. `("level", "perp")`, `("level", "fut")`, or a
   `Callable[[panel], pd.DataFrame]`. The regime burner pre-reg already defines the proxy as “the
   equal-weight daily mean return of the strategy’s loaded panel”; this just lets the strategy point
   at the right block when the name isn’t guessable. Hashes into the frozen design like
   `hedge_tickers`.
2. **`_price_matrix(panel, spec=None)`**: if `spec.market_proxy` is set, use it (single source of
   truth); else the current exact + composite heuristic; else `None`. Plumb `spec` through the one
   call site (`GateContext`, line ~1007) and the two internal users.
3. **Sub-class B**: when no price level exists, the proxy for regime/beta should fall back to a
   declared return field (`("ret_field", "ret")`) or the strategy’s own daily return series (the
   pre-reg’s “equal-weight mean return of the panel” reduces to exactly this for a returns panel).
   A pure-metrics panel with no market legitimately stays `not_evaluated` — but that must be a
   DELIBERATE, logged outcome, not an accident of name-matching.
4. **Codegen contract update** (`agent/codegen.py`): instruct generated strategies to either name the
   price field with a recognised token OR set `market_proxy`. This closes the footgun at the source.
5. **Loud-not-silent guard**: when `_price_matrix` returns `None`, the verdict should record an
   explicit `proxy_unavailable` reason on each affected gate (it partly does) AND the digest should
   surface a count of strategies running with un-evaluated regime/beta — so this can’t hide again.

## Acceptance
- Each of the 7 panels above either resolves a proxy (sub-class A + the returns case of B) or records
  a deliberate `not_evaluated: no-market-proxy` (pure-metrics), never silent name-miss.
- New tests in `tests/test_breadth_overfit.py`: `perp/spot`, `fut/carry/inv`, flat `<5`-col ETF, and a
  `market_proxy`-declared strategy.
- Exact-match + composite paths unchanged for already-recognised panels (byte-identical).
- Re-run the retro impact audit; confirm 0 new PROMOTE demotions before any demotion activates.

## Non-goals
- Do NOT hardcode an ever-growing alias list (`perp`,`spot`,`fut`,…) — that just moves the footgun.
- Do NOT touch the frozen regime-burner numeric rule; this is plumbing + an optional spec field.
