# We published the graveyard: ~100 AI-generated trading strategies, and the one time our own machine fooled itself

*~850 words. For a blog / LinkedIn / Quantocracy / Show HN. One hook, one CTA. Edit for your voice before posting.*

---

Generating a trading strategy used to take a quant weeks. Now an LLM does it in seconds — and produces a backtest that looks brilliant and dies the moment it touches reality. There's a name for this in the literature now: the **profit mirage**. LLM backtest returns evaporate the instant you step past the model's training cutoff. Anonymize the tickers and the "alpha" vanishes. Walk-forward validation — still the industry default — is mathematically too weak to catch it.

So the scarce thing isn't strategy *generation* anymore. It's trustworthy strategy ***killing***.

And there's a nasty trap hiding in that. If you hire someone to validate your strategy, how do you know *they're* honest? You can't — because a dishonest validator never shows you its failures. Anyone can publish their one winner. Almost nobody publishes the 95 they buried, or the moment their own system fooled itself.

We're publishing ours.

## The setup

We built an autonomous loop: LLM "smiths" propose strategy hypotheses for roughly $0, and a **non-bypassable gate stack** burns away everything that isn't a real, harvestable edge. The smiths never see the gates — they emit a hypothesis and a signal function, nothing else. The gates do the rest, in order:

- **Combinatorial purged cross-validation + a deflated Sharpe ratio + probability-of-overfitting** — computed over *every variant the search actually tried*, so a Sharpe is discounted for the real number of attempts, not one cherry-picked run.
- **A false-discovery bar that rises with every new strategy family tested, shared across all the agents.** Run N searchers in parallel and you multiply your false-discovery risk; one shared, rising bar is the only thing that contains it.
- **A write-once holdout** — a quarantined slice you're allowed to look at exactly once, enforced by a ledger that refuses a second peek. The only incorruptible judge. One candidate cleared *everything else* and still died here.
- **A permutation test** — shuffle away all the structure, re-run the *frozen* signal; if noise scores as high as the real thing, your "edge" was an artifact of the construction. (Our scariest red flag: noise *beating* the real data.)
- **Cost-and-borrow deployability** — realistic per-name trading cost and stock-borrow reality, so a statistically valid but un-tradable result can't pass.

## The graveyard

Across about six research domains and **~100 pre-registered experiments**: **56 outright failures. 34 near-misses. One validated structure. Exactly one strategy ever cleared the entire stack.**

The pattern was consistent and humbling. Standalone *prediction* edges in liquid markets are dead at retail scale — cross-sectional momentum, value-and-quality on clean fundamentals, dozens of single-name technical strategies on survivorship-corrected data, sports betting markets: all null. Classic developed-market carry: negative. What actually survived was *risk premia, not predictions* — and specifically the **combination** of complementary premia, where diversification, not a cleverer signal, was the edge.

## The part most people would hide

Our one full-gate pass was an illiquidity long/short book. Beautiful on every statistical axis. Then we put it into live paper trading and reality answered:

- the short leg needed to borrow micro-cap stocks that **can't be borrowed** — **92% of its live orders were cancelled**;
- a sibling small-cap book filled at **~3× the trading cost we'd modeled**, flipping a positive backtest to **negative** in live trading.

Our gate stack had proven the strategy was *statistically* honest. It had never proven it was *economically* honest — the cost assumption was frictionless and had never met a real fill. Its first contact with real economics missed by ~69%.

So we did the disciplined thing. We pre-registered a cost-and-borrow-aware gate, froze it, and **re-scored all ~100 prior results.** The verdict was decisive: **of the 21 strongest strategies, zero is cleanly tradable.** The strong results live exactly where you can't trade them — the un-borrowable, illiquid corner. That's market structure, not a bug you can patch. We wrote it down and made an un-tradable "pass" impossible to certify ever again.

A validator that publishes the moment it fooled itself — and ships the fix — is sending a costly, hard-to-fake signal. That's the whole point.

## Where we honestly stand

We do **not** have a profitable live track record. We have the opposite: a rigorously honest *null*, and an engine that produces it cheaply, at scale, without lying to itself. In a market drowning in confident, unvalidatable, AI-generated strategies, the kill-machine is the asset — not any strategy it finds.

**If you build LLM/agentic quant tooling, run a systematic book, or vet managers:** we'll run your strategy through the same gates and tell you if it's real or a mirage — overfitting probability, deflated Sharpe over your true search burden, a write-once holdout verdict, permutation isolation, and cost/borrow deployability. We're piloting this as a paid audit. If the idea of an outside party who'll tell you *no* — with receipts — is useful, get in touch.

*The full graveyard, methodology, and the cost-aware re-score are linked below.*
