# Leg D — Outreach Kit (the demand test)

**Goal (board memo 2026-06-15, falsifiable):** within 4 weeks, **≥3 qualified parties signal willingness to pay** for a strategy-validation / overfit audit (a signed LOI, a paid pilot, or a concrete dollar offer). This is a **demand test, not a product build** — no productization code until demand is proven. Zero serious buyers → D is falsified → fall back to steady-state $0 forge (E).

**The asset you're selling:** an honest validation engine + a published graveyard (`the-graveyard.md`). You are NOT selling a profitable strategy or a track record — be explicit about that; the honesty is the credential. The offer is a **paid overfit-audit pilot ($2–5K): "send us a strategy/backtest, we tell you if it's real, with receipts."**

---

## Step 1 — Publish the artifact (the inbound magnet)
Publish `the-graveyard.md` (lightly edited for voice) in 2–3 places. The graveyard is the costly signal; it does the credibility work so the cold outreach doesn't have to.
- **arXiv / SSRN** short note ("An honest null: ~100 LLM-generated trading strategies and the gate stack that killed them") — gives the academic/quant-desk audience something citable.
- **A blog post / LinkedIn article** — same content, plainer voice, with the "machine caught its own false positive" as the hook.
- **Quant aggregators:** submit to **Quantocracy** (mailbag@quantocracy.com), post to **r/quant** and **r/algotrading**, and **Hacker News** ("Show HN: we published the graveyard of ~100 killed trading strategies").
- Link every outbound message to the published artifact, not an attachment.

## Step 2 — Cold pitch template (≤120 words, no deck)

> **Subject:** We published the graveyard — ~100 killed strategies + the engine that killed them
>
> Hi [name],
>
> Short version: we built an autonomous loop where LLMs propose trading strategies and a non-bypassable gate stack (CPCV + deflated Sharpe + a write-once holdout + permutation tests + cost/borrow deployability) kills the ones that aren't real. Of ~100 hypotheses, one passed — and we then caught *that one* failing on real execution costs, published the correction, and made un-deployable passes impossible to certify. The full graveyard is here: [link].
>
> Most "validation" can't tell honest from overfit. Ours is built to, and shows its kills. If you're [building LLM quant tooling / running a book / doing manager DD], would a paid pilot audit of one of your strategies be useful — we tell you if it's real, with receipts?
>
> Either way, the writeup might save you a landmine. — [you]

Tailor the one bracketed clause per recipient. Keep it that short.

## Step 3 — Target list (≥10 qualified parties, by category)

**A. LLM/agentic-quant platforms (drowning in unvalidatable strategy-slop — the hottest fit):**
1. **Microsoft Research – RD-Agent team** (RD-Agent(Q), the closest published cousin to our loop) — they validate at IC/backtest level only; our gate stack is the missing half. Strong collaboration angle.
2. Authors of **QuantEvolve / AlphaAgent / Chain-of-Alpha** (recent arXiv discovery-loop papers) — email the corresponding authors; offer to benchmark their output through our gates.
3. **Numerai** (crowd-sourced signals; their entire problem is telling real from overfit) — community + BD contacts.
4. Emerging "AI hedge fund / AI-analyst" startups (e.g. Minotaur Capital, Intelligent Alpha, and the wave of YC-stage agentic-trading tools) — they need an external credibility layer.

**B. Quant prop desks / small systematic funds (would pay to not deploy a mirage):**
5–6. Two systematic prop shops or emerging managers in your network (or warm intros) — the audit reframes as cheap insurance against a bad allocation.

**C. Allocators / manager due-diligence:**
7. A fund-of-funds or family office doing systematic-manager DD — the overfit audit is exactly their diligence question, outsourced.
8. An emerging-manager platform / seeder.

**D. Academic & community (distribution + credibility, lower $ but high signal):**
9. Quant finance academics working on backtest overfitting / DSR (the Bailey-López de Prado lineage) — they'll engage with the methodology and amplify it.
10. A quant newsletter / podcast (e.g. the authors behind the "profit mirage"/"alpha illusion" debunking papers) — offer the graveyard as a case study.

> Aim for **15–20 sent** to land **≥3 willing-to-pay** — assume a low single-digit response rate on cold outreach. Warm intros convert far better; prioritize the 2–3 you can reach through someone.

## Step 4 — The pilot offer one-pager (send only after interest)

**Overfit Audit — pilot.** You send: a strategy spec (or returns + trade ledger + the search grid you actually tried). We return, in ~1 week:
- **Deflated Sharpe** over your *true* search burden (the number that deflates for how many variants you tried).
- **Probability of Backtest Overfitting (PBO)** + CPCV path distribution.
- A **write-once holdout** verdict on a slice you quarantine.
- **Permutation-test isolation** (is the edge structure or artifact?).
- **Cost-and-borrow deployability**: does it survive realistic per-name liquidity cost + borrow feasibility?
- A one-page **verdict: real / mirage / un-deployable**, with the failing gate named.

**Price:** $2–5K per strategy for the pilot (anchor; the point is willingness-to-pay, not the number). **What we don't do:** take custody, trade your money, or promise returns. We tell you the truth about a backtest.

## Step 5 — Track it (the falsifiable scoreboard)
Keep a simple log: party · channel · sent date · response · willing-to-pay (Y/N/maybe). The 4-week metric is **≥3 Y**. Review ~2026-07-13 with Leg B's re-score verdict.

---
*Honesty guardrails (First Principle applies to selling too): never claim a profitable track record, never imply the audit guarantees a strategy will make money, never oversell the gate stack as proprietary magic — it's rigorous, public-methodology validation, and that rigor + the published graveyard is exactly the differentiator.*
