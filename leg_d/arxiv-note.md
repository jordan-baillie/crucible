# An Honest Null: Industrialized Validation of ~100 LLM-Generated Trading Strategies, and an Economic Out-of-Sample That Killed the Survivors

*Technical note — DRAFT for SSRN / arXiv (q-fin.TR / q-fin.PM). Numbers are grounded in the project's experiment ledger and the 2026-06-15 cost-aware re-score. **Citation IDs marked `[verify]` must be confirmed against arXiv/SSRN before submission**; the Bailey–López de Prado references are canonical and stable. Edit author/affiliation block before posting.*

## Abstract

Large language models can generate plausible quantitative trading strategies at near-zero marginal cost, producing a volume of candidates that vastly exceeds the rate at which they can be honestly evaluated. The binding constraint has shifted from strategy *generation* to trustworthy strategy *falsification*. We describe an autonomous research system in which LLM agents propose hypotheses and a fixed, non-bypassable statistical gate stack — combinatorial purged cross-validation (CPCV), the Deflated Sharpe Ratio (DSR), the Probability of Backtest Overfitting (PBO), a family-count-indexed false-discovery bar shared across agents, a single-use write-once holdout, and a Monte-Carlo permutation test (MCPT) — adjudicates them. Over ~100 pre-registered experiments spanning six asset/strategy domains, the system produced 56 outright failures, 34 near-misses, one validated *structure* (a complementary two-premium book), and exactly one strategy that cleared the full statistical stack. We then report an *economic* out-of-sample: the single statistical pass failed live execution (92% short-leg order cancellation on stock-borrow constraints; realized trading cost ≈3× the modeled assumption, flipping expectancy negative). We pre-registered and froze a cost-and-borrow-aware deployability gate and re-scored the entire corpus through it. Of the 21 strongest strategies, **none is cleanly deployable**: the strongest results are concentrated in the un-borrowable, illiquid corner — a market-structure constraint, not a fixable miscalibration. We argue the resulting *negative knowledge*, openly published, is the system's primary asset and a costly, hard-to-fake signal of validator honesty.

## 1. Introduction

A growing body of work documents that LLM-generated trading strategies overstate out-of-sample performance through memorization and look-ahead leakage — the "profit mirage" and "alpha illusion" findings, and blindfold/anonymization tests in which performance collapses once identifying tokens are removed `[verify]`. Concurrently, autonomous discovery loops (e.g. RD-Agent(Q), QuantEvolve, AlphaAgent) increasingly couple LLM hypothesis generation to backtesting `[verify]`. Most such loops validate at information-coefficient or single-split backtest level — precisely the regime in which overfitting and false discovery are least controlled (Bailey et al., 2014; Bailey & López de Prado, 2014; López de Prado, 2018).

We take the opposite design stance: hypothesis generation is cheap and untrusted; the value is in an *adversarial, non-bypassable validation stack* that the generator cannot see or influence, and in the honest accumulation of what fails. This note describes that stack, reports the resulting near-complete null over ~100 experiments, and presents an economic out-of-sample step that eliminated even the lone statistical survivor.

## 2. The validation stack

A candidate is reported only if every gate holds; a single failure demotes it.

1. **CPCV + DSR + PBO** over the full configuration grid actually searched, so the Sharpe is deflated by the true number of trials (Bailey & López de Prado, 2014; Bailey et al., 2017).
2. **Family-indexed false-discovery bar**, rising with the count of distinct strategy families ever tested and **shared across all agents** — the central safeguard against N parallel searchers inflating false discovery.
3. **Write-once holdout**: a quarantined slice read exactly once, enforced by a single-use ledger that refuses a second evaluation of the same configuration. A 0.986-DSR candidate cleared all prior gates and failed here.
4. **MCPT**: permute each asset's returns to destroy serial and cross-sectional structure, re-run the *frozen* signal; reject if structureless data matches or exceeds the real Sharpe (isolates construction artifacts — e.g. bid-ask-bounce harvesting, volatility-targeted noise sorting).
5. **Beta-confound and regime gates**: demote long-only books whose "edge" is market exposure; require non-negative performance in both calm and turbulent volatility regimes and across untouched universes.
6. **Cost-and-borrow deployability** (§4): realistic per-name liquidity cost and borrow feasibility, baked in so a statistically valid but un-tradable result cannot be certified.

Pre-registration is enforced: the frozen primary configuration is the result; the grid exists only to compute the search burden for DSR; results are never tuned post hoc.

## 3. Corpus and statistical results

Across ~100 pre-registered experiments in six domains (cross-sectional equity factors, cross-asset risk premia, crypto carry, volatility-risk premium, futures term structure, sports betting): **56 FAIL, 34 near-miss, 1 validated structure, 1 full-gate pass.** Salient negatives, each with a reproducible ledger entry:

- Standalone *prediction* edges in liquid markets are null at retail scale (cross-sectional momentum; value+quality on survivorship-clean fundamentals; ~22 single-name technical strategies; full-game and player-prop sports markets).
- Developed-market carry (FX and bond) is negative over 2010–2026.
- The one validated *structure* is a complementary **carry+trend** book: trend has no standalone premium but mechanically reduces carry's maximum drawdown ~45% at no Sharpe cost (correlation ≈ −0.02). Diversification, not signal quality, is the source.

## 4. Economic out-of-sample: the deployability re-score

The single full-gate pass — an Amihud-illiquidity long/short — was deployed to live paper trading. Two execution facts emerged: (i) the short leg required borrowing micro-cap names that are not borrowable — **92% of live orders cancelled**; (ii) a sibling small-cap book realized trading cost ≈3× the modeled (frictionless) assumption, turning positive modeled expectancy **negative** in live trading. The statistical stack had certified statistical honesty but not *economic* honesty; the frictionless cost assumption had never met a real fill, and its first out-of-sample test missed by ≈69%.

We pre-registered (and froze) a cost-aware deployability gate: (a) a dollar-volume-decile liquidity cost ladder from microstructure priors, and (b) a borrow-feasibility filter that zeroes short exposure in non-borrowable names. We then re-scored the corpus. Result: **of the 21 strongest strategies, zero is cleanly deployable.** Eleven survive only a *lenient* static-borrowability filter (live hard-to-borrow constraints are strictly worse and not observable historically); the remainder are either liquidity-untested custom-cost constructions or, in three cases, non-equity crypto-carry the equity ladder does not model. The conclusion is structural: the retail-equity results live in the un-borrowable/illiquid corner — *where the illiquidity premium tautologically resides* — and are therefore not deployable by construction, independent of statistical quality.

## 5. Discussion

The system's output is a near-complete, openly documented *negative* result. We argue this is the asset. Strategy validation is a market for lemons: a buyer cannot distinguish an honest validator from a dishonest one, because dishonest validators do not publish failures. A validator that publishes (i) its full graveyard and (ii) the instance in which its own machine produced a false positive *and the subsequent fix* emits a costly, hard-to-fake signal of honesty. The negative knowledge is also directly useful: each documented failure is a pre-mapped landmine for other practitioners running the same searches.

## 6. Limitations

- The borrow filter uses a present-day static shortability snapshot applied retroactively; it is a *lower bound* on infeasibility (intraday hard-to-borrow constraints are unobservable historically and strictly tighten the conclusion).
- The liquidity ladder is set from microstructure priors; an out-of-sample validation against clean live fills (slippage measured vs the official auction print, ≥100 fills) is deferred, and the ladder does not model custom-cost or non-equity constructions. The borrow-wall result does not depend on the ladder.
- Several recent LLM-quant references require ID verification before submission (`[verify]`).

## 7. Conclusion

When generation is free, validation is the product, and *honest* validation — demonstrated by an openly published graveyard and a self-caught false positive — is the differentiator. We release the methodology, the ~100-experiment ledger, and the cost-aware re-score in that spirit.

## References (verify recent IDs before submission)
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (CPCV.)
- "Profit Mirage": LLM backtest performance degradation past training cutoff. arXiv `[verify]`.
- "Alpha Illusion" / blindfold-anonymization LLM trading evaluations. arXiv `[verify]`.
- RD-Agent(Q); QuantEvolve; AlphaAgent — autonomous LLM strategy-discovery loops. arXiv `[verify]`.
