# The Graveyard: ~100 honestly-killed trading strategies, and the machine that killed them

*A public artifact. Draft for review before publishing (blog / arXiv-SSRN note / GitHub). Every number below is grounded in the research-wiki experiment ledger and the cost-aware re-score of 2026-06-15 — nothing is invented. Tone: honest, technical, no hype. The honesty IS the pitch.*

---

## The problem nobody wants to publish

Generative models can now write a plausible trading strategy in seconds. The result is a flood of backtests that look brilliant and die on contact with reality. The literature has a name for it now — the **"profit mirage"**: LLM backtest returns that evaporate the moment you step past the training cutoff (memorization, not edge). Anonymize the tickers and the "alpha" vanishes. Walk-forward validation — still the industry default — is mathematically too weak to stop it.

So the real scarce thing isn't strategy *generation*. It's **trustworthy strategy *killing***. And here's the market-for-lemons trap: a buyer can't tell an honest validator from a dishonest one, because **a dishonest validator never shows you its graveyard.** Anyone can publish their one winner. Almost nobody publishes the 95 they killed — or the moment their own machine fooled itself.

We're publishing ours.

## What we built

An autonomous research loop where LLM "smiths" propose strategy hypotheses for ~$0, and a **non-bypassable gate stack** burns away everything that isn't a real, harvestable edge. The smiths can't see or touch the gates; they emit a hypothesis and a signal function, nothing more. The gates, in order:

1. **Combinatorial Purged Cross-Validation (CPCV) + Deflated Sharpe + Probability of Backtest Overfitting (PBO)** — over the *whole grid the search actually tried*, so the Sharpe is deflated by the real number of attempts, not one.
2. **A false-discovery bar that rises with every distinct strategy family ever tested** — *shared across all agents.* N parallel searchers multiply false-discovery risk; one shared, rising bar is the only thing that contains it. (Today: bar 0.988 after 67 families.)
3. **A write-once holdout** — a quarantined slice read *once*, enforced by a single-use ledger that refuses a second look. The only incorruptible arbiter. A 0.986-Deflated-Sharpe candidate cleared everything else and **still** failed here.
4. **Monte-Carlo permutation test (MCPT)** — shuffle away all structure, re-run the *frozen* signal; if structureless data scores as high as the real thing, the "edge" is a construction artifact. (Strongest red flag we found: permuted data *beating* real data.)
5. **Beta-confound gate** — a long-only book that's really just holding the market gets demoted, not promoted.
6. **Cross-universe breadth + volatility-regime split** — a real mechanism survives in markets it was never tuned on, and in both calm and turbulent regimes.
7. **Cost-and-borrow deployability (new, 2026-06-15)** — realistic per-name liquidity cost + stock-borrow feasibility, baked in so that a *statistically* valid but *un-tradable* result can no longer pass. More on this below — it's the centerpiece.

Pre-registration is sacred throughout: the frozen design is the result; you never tune to rescue a number; a negative result gets the same energy as a win.

## The graveyard (the honest ledger)

Across ~6 research domains and **~100 pre-registered experiments**:

- **56 outright FAIL.** 34 **near-miss** (cleared most gates, failed one honestly). **1 validated structure.** **1** strategy ever cleared the entire gate stack.
- Standalone *prediction* edges in liquid markets: **dead** at retail scale (cross-sectional momentum, value+quality on owned fundamentals, 22 single-name technical strategies on survivorship-clean data, full-game and player-prop sports betting — all null).
- Classic developed-market carry (FX + bond): **negative** (FX carry dead since 2008, bond carry crushed in 2022).
- What *did* survive: **risk premia, not predictions** — and specifically the **combination of complementary premia** (a carry+trend book where trend has no standalone return but mechanically cuts carry's drawdown ~45% at no Sharpe cost). Diversification was the edge, never a better signal.

That's a humbling ledger. It's also the most valuable thing we own — because every entry is a landmine someone else is about to step on, mapped and dated.

## The centerpiece: the machine caught its own false positive

Our one full-gate pass was an illiquidity (Amihud) long/short book. It looked real on every statistical axis. Then we put it into live paper trading and watched reality answer:

- the short leg needed to borrow micro-cap stocks that **can't be borrowed** — **92% of its live orders were cancelled**;
- a sibling small-cap book's live execution cost came in at **~3× the modeled assumption**, turning a positive backtest into **negative** live expectancy.

The gate stack had certified *statistical* honesty but never *economic* honesty — the cost model was a flat, frictionless assumption that had never been tested against a real fill. **Its first out-of-sample contact with real economics failed by ~69%.** So we did the disciplined thing: we pre-registered a cost-and-borrow-aware deployability gate, froze it, and **re-scored all ~100 prior results through it.**

The verdict was decisive and unflattering: **of the 21 strongest strategies, zero is a clean, deployable edge.** The strong results live exactly where they can't be traded — the un-borrowable, micro-cap-illiquid corner. That's market structure, not a fixable bug. We wrote it down, published it, and made the un-deployable-pass *structurally impossible to certify going forward.*

A validator that publishes the moment it fooled itself — and the fix — is making a costly, hard-to-fake signal. That's the entire point.

## Where we stand honestly

We do **not** have a profitable live track record. We have the opposite: a rigorously honest **null**, and an engine that produces it cheaply, at scale, and without lying to itself. In a world drowning in confident, unvalidatable, AI-generated strategies, **the kill-machine is the asset** — not any strategy it finds.

## If this is useful to you

We can run your strategy or backtest through the same gates and tell you, with receipts, whether it's real or a mirage — overfitting probability, deflated Sharpe over your true search burden, a write-once holdout verdict, permutation-test isolation, and cost-and-borrow deployability. If you build agentic/LLM quant tooling, run a prop book, or do manager due-diligence, that's an audit you can't easily get anywhere else.

We're gauging interest in a **paid pilot audit**. If you'd find it valuable, get in touch — details in the outreach note.

---
*Sources: research-wiki experiment ledger (100 pages), `methodology/RAILS.md`, `patterns/META-LESSONS.md`, and `methodology/results-cost-aware-deployability-rescore.md` (the 2026-06-15 re-score). The rails are a standalone package (`research_integrity`).*
