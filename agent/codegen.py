"""Signal codegen: the agent writes a complete StrategySpec module from its proposal,
with a bounded fix-retry loop (reads the traceback, repairs the code). LLM via the pi CLI."""
import ast, json, re
from pathlib import Path
from agent.llm import call as _llm_call, extract_json

CONTRACT = '''
You are writing a Python strategy module for the Crucible research harness. Output ONLY one
```python code block: a complete module with NO external side effects (no file writes, no capital,
no config). It MUST define exactly:

  def load_data() -> pd.DataFrame:        # the panel signal() consumes (use the adapters below)
  def signal(panel, **params) -> (pd.Series daily_returns, list trades):
  def load_gen_data(label) -> pd.DataFrame:  # REQUIRED for scope='broad': the panel for ONE
      # generalization universe (same shape as load_data(); label is one of generalization_universes)
  SPEC = StrategySpec(id=..., family=..., title=..., markets=[...], data_desc=..., pre_registration=...,
                      load_data=load_data, signal=signal, default_params={...}, grid={label:params,...},
                      scope='broad'|'local', generalization_universes=[...], load_gen_data=load_gen_data,
                      holdout_start="2022-01-01", deploy_max_positions=N,
                      expectations=[{"name":..., "claim":..., "check": fn}, ...])  # see SOFT EXPECTATIONS below

CONTRACT:
- daily_returns: a pandas Series of daily net-of-cost portfolio returns, DatetimeIndex, name set.
- trades: list of dicts, each {"ticker","sector","entry_date"(YYYY-MM-DD str),"exit_date","hold_days"(int),
  "position_value"(float),"pnl"(float),"entry_regime"(str, set by trades_from_weights)} — used for
  deployment-sanity + cross-regime gates (needs >=50 trades, spread across
  sectors, no single name >40% of position-days). For a factor book, emit one trade per held position run.
- grid: a few pre-declared param variants for the DSR effective-N (honest search burden); "default"={} is primary.
- scope: 'broad' if the edge is a UNIVERSAL mechanism (a factor/premium theory says appears across markets ->
  a stage-1 pass MUST GENERALISE to other untouched universes, or it's an overfit outlier like BAB);
  'local' if it's a defensibly UNIVERSE-SPECIFIC edge (then forward-validation confirms it).
- For 'broad': declare >=3 generalization_universes (DISJOINT from the search universe — different cap tier,
  different sectors, or sub-slices that share NO tickers) AND implement load_gen_data(label) returning each
  one's panel. The harness runs the STAGE-2 battery automatically on a stage-1 pass: same frozen signal +
  default params on each universe's HOLDOUT only; >=60% must be OOS-positive or the candidate is rejected.
  Keep each gen universe SMALL (~150-400 names) — it runs same-night, N+1 extra signal() calls.
- Apply realistic costs (~8bps on turnover). Inverse-vol size. Weekly rebalance. NO look-ahead (lag signals 1 day).

USE ONLY these tested imports (do NOT download raw / reinvent). Full data inventory: research-wiki/DATA_CATALOG.md.
  from sdk.harness import StrategySpec
  from sdk.adapters import sep_panel, us_universe, sf1, yf_panel, fred_series, trend_returns, inv_vol_position
  from sdk.universe import sector_universe
  from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel
  import numpy as np, pandas as pd

MANDATORY KIT (do NOT re-implement these — every hand-rolled copy is a fresh chance for a
lookahead bug the harness can't see; the ONLY novel code in your module is the signal itself):
- sector_universe(marketcap, top_n_per_sector) -> (tickers, sector_map): sector-spread universe +
  the {ticker: sector} map the trade ledger needs. USE THIS instead of looping us_universe per sector.
- xs_zscore(df, winsor=(0.05,0.95)) -> cross-sectional per-date z-score, winsorized, NaN-preserving.
- net_of_cost(W, rets, cost_bps=8.0, name=...) -> daily net returns from a LAGGED weight matrix
  (pass W.shift(1) if you built same-day weights — the lag is YOUR responsibility, state it in the code).
- trades_from_weights(W, rets, sector_map) -> the CONTRACT trade ledger (run-length per held name),
  auto-stamping each trade's entry_regime (bull/bear x calm/vol, trailing-data-only) — the cross-regime
  robustness gates depend on it. Never write entry_regime yourself; the kit's labeller is the standard.
  MANDATORY: regime stamping is part of the contract — a ledger whose trades lack real entry_regime
  labels (<80% coverage) has its regime gates recorded as NOT EVALUATED and (from 2026-06-26) is
  DEMOTED outright (pre-reg 2026-06-12). Building trades any other way will fail the rails.
- market_regime(rets) -> the per-date regime label Series, if you need it for a regime FILTER in the
  signal itself (it is shift(1)-lagged — safe to act on same-day).
- pit_panel(sf1_df, field, dates, tickers) -> point-in-time fundamental panel (datekey-based, ffilled;
  never use calendardate — that's lookahead).
A typical module body is therefore: build universe -> load panels -> compute YOUR signal -> weights
-> W=...shift(1) -> net_of_cost + trades_from_weights -> return. Keep it SHORT.
- sep_panel(tickers, start, field='closeadj') -> SURVIVORSHIP-CLEAN US equity daily panel from OWNED Sharadar SEP
  (delisted incl, split+div adjusted). **PREFER over yf_panel for US stocks** (yfinance has survivorship bias).
- us_universe(sector=, category='Domestic Common Stock', marketcap=, include_delisted=True, top_n=) -> US ticker
  list from OWNED Sharadar TICKERS (delisted INCLUDED -> survivorship-clean). Pass its output to sep_panel.
  **CRITICAL for cross-sectional EQUITY: you MUST bound the universe — pass top_n (e.g. us_universe(..., top_n=1000)
  = 1000 most-liquid names) and/or marketcap='Large'/'Mid'. NEVER run the full ~16k common-stock universe through
  the rails — the CPCV is pathologically slow + OOM'd at 14.5GB. Target ~few-hundred to ~1500 liquid names.** Cross-sectional ANOMALIES (issuance, value, accruals, low-vol…) live in SMALL/ILLIQUID names — test
  them in us_universe(marketcap='Small' or 'Mid', top_n=~1500), NOT the largest liquid names (there they
  are arbitraged away -> false nulls).
- sf1(tickers, fields, dimension='ARQ') -> OWNED Sharadar fundamentals. Use 'datekey' (filing date) as the as-of
  date to avoid look-ahead (NEVER calendardate). Fields e.g. eps, revenue, bvps, marketcap, pe, de, roe.
- yf_panel(tickers, start) -> Close panel (FREE; futures/ETFs/intl indices — NOT US single stocks).
  fred_series({fred_id:col}, start) -> daily rates/yields/credit-spreads (FREE; e.g. BAMLH0A0HYM2=HY OAS).
- trend_returns(**p) -> (returns, trades) the validated 21-market CTA trend hedge leg.
- inv_vol_position(signal_df, rets, target_vol, vol_lb, max_pos, rebalance) -> weekly-held lagged positions.
- For a COMBINATION: build each leg's daily returns, align (pd.concat axis=1 dropna), vol-match, blend.
- BUT test the premium STANDALONE first. Only ADD a hedge (e.g. trend) if it CUTS THE TAIL without
  diluting the standalone Sharpe — size it to MINIMISE drag, NOT a reflexive 50/50. A ~0-Sharpe hedge
  blended 50/50 HALVES the edge (it sank a real +0.27-Sharpe credit-carry premium to ~0). Prefer the
  standalone leg or a SMALL tail-overlay; do not pair with trend just because the wiki pattern says so.
- HEDGE SLEEVE (broad-index-ETF beta trim, e.g. a residual IWM short): declare it on the spec —
  StrategySpec(hedge_tickers=["IWM"], hedge_cap=0.35) — so the deployment gate judges your ALPHA
  book alone and gates the sleeve on whitelist+cap. An UNDECLARED ETF hedge held continuously will
  force-fail single_name_share (it killed two otherwise-clean Amihud near-misses on 2026-06-11).
  Whitelist: SPY/IVV/VOO/IWM/MDY/IJH/IJR/QQQ/VTI/ITOT/EFA/EEM/ACWI/TLT/IEF/SHY/GLD/USO/DBC.
  The cap is position-day share; >0.60 is never allowed. The sleeve is part of the FROZEN design.
- SOFT EXPECTATIONS (MANDATORY when your pre_registration makes a checkable mechanism claim,
  e.g. "turnover drops 50%", "monotonic across quintiles", "works in all sub-periods"): declare
  them MACHINE-CHECKABLE on the spec — StrategySpec(expectations=[{"name": "turnover_halved",
  "claim": "tranched long-leg turnover <= 60% of untranched", "check": my_check_fn}, ...]).
  Each check(ctx) -> {"pass": bool, "observed": <number/str>}; ctx has panel, spec, search
  (search-window net returns Series), trades (search-window ledger), grid (label -> search-window
  returns Series per grid variant), holdout_start. Use ctx["grid"] to compare variants you already
  declared (free); at most ONE extra signal() call inside a check, and slice anything you recompute
  to dates < ctx["holdout_start"]. Soft fails do NOT block the gates — they are recorded on the
  verdict so a wrong mechanism story is falsified instead of shipped (2026-06-12: a PASS shipped
  claiming tranching halved turnover; retro-check showed it RAISED it 39% — the claim was prose,
  so nothing ran). A claim you cannot check cheaply belongs in pre_registration prose ONLY if you
  also say why it is not machine-checkable.
Be economical and correct. OWNED/FREE data only (see DATA_CATALOG.md). The harness runs ALL the rails; you only produce returns+trades.
'''


_pi = _llm_call  # plumbing consolidated in agent.llm


def _public_adapter_names() -> list:
    """Every public sdk.adapters function name, derived from the code (cannot drift)."""
    import inspect
    from sdk import adapters as _ad
    return sorted(n for n, f in vars(_ad).items()
                  if inspect.isfunction(f) and not n.startswith("_")
                  and f.__module__ == _ad.__name__ and n != "list_adapters")


def _build_contract(template: str) -> str:
    """Single-source the adapter whitelist into the CONTRACT from sdk.adapters so it can NEVER
    drift again. The 2026-06 crypto-adapter omission (binance_*, coinmetrics_*, bybit_funding,
    deribit_dvol existed in sdk.adapters but not in the hardcoded whitelist) made Opus REFUSE
    ~33% of crypto codegens ('I can't write this honestly — no crypto data') and
    hallucinate-then-crash the rest (the runtime_error spike). Fail-OPEN to the static template
    if sdk.adapters can't be introspected — never crash a smith over a contract-render error."""
    try:
        from sdk.adapters import list_adapters
        whitelist = "  from sdk.adapters import " + ", ".join(_public_adapter_names())
        contract = template.replace(
            "  from sdk.adapters import sep_panel, us_universe, sf1, yf_panel, fred_series, trend_returns, inv_vol_position",
            whitelist)
        return (contract + "\n\n=== COMPLETE ADAPTER INVENTORY (authoritative, code-derived — "
                "every adapter below EXISTS and is importable; crypto/futures/macro/SEC included) ===\n"
                + list_adapters())
    except Exception:
        return template  # fail-open: prose guidance still lists the core kit


# Render the whitelist from the code ONCE at import. Single source of truth = sdk.adapters.
CONTRACT = _build_contract(CONTRACT)


def _extract_code(text: str) -> str:
    # take the LARGEST fenced block (Opus sometimes emits a skeleton block first, then the real one)
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL) or re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    return (max(blocks, key=len) if blocks else (text or "")).strip()


def validate_module(code: str) -> str | None:
    """Deterministic pre-flight for a generated strategy module. Returns None if it is safe to
    write+run, else a PRECISE, fixable reason (routed to codegen.fix() like any other gate).

    Runs in-process in microseconds and is the single source of truth for "is this code safe to
    write+run". It closes the two malformed-module casualty CLASSES that the old substring
    heuristic ('def signal' in code) let slip through to the EXPENSIVE sandbox subprocess, where
    they surfaced as opaque casualties triage could not diagnose:
      1. SyntaxError — e.g. the generator wrote its chain-of-thought TRANSCRIPT into the .py
         ('I'll start by...' -> unterminated string literal at line 1).
      2. no module-level `SPEC` — the harness entrypoint is run_experiment(m.SPEC); without it the
         import raises AttributeError ('module ... has no attribute SPEC') AFTER a wasted run.
    The old docstring CLAIMED to prevent the m.SPEC AttributeError but never actually checked for
    SPEC — this makes the gate deliver on that contract.
    """
    c = code or ""
    if "def signal" not in c or len(c) < 300:
        return "INCOMPLETE: empty/truncated module (need imports + `def signal` + module-level `SPEC`)"
    try:
        tree = ast.parse(c)
    except SyntaxError as e:
        return (f"SyntaxError: {e.msg} (line {e.lineno}). Output ONLY a complete python MODULE in one "
                f"```python block — never chain-of-thought / prose / tool-call text in the .py.")
    has_spec = any(
        (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPEC" for t in n.targets))
        or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "SPEC")
        for n in tree.body)
    if not has_spec:
        return ("no module-level `SPEC = StrategySpec(...)` — the harness runs run_experiment(m.SPEC); "
                "a SPEC built inside a function or a different name will not load.")
    return None


def looks_complete(code: str) -> bool:
    """Bool view of validate_module() for legacy call sites (run_worker GATE-1 consistency-skip,
    codegen self-checks). True iff the module compiles AND exposes a module-level SPEC."""
    return validate_module(code) is None


_looks_complete = looks_complete  # internal alias (legacy call sites)


# Observability: stats of the LAST generate() call, read by run_worker for the run record.
LAST_GEN = {"attempts": 0, "empty_retries": 0}


INCOMPLETE_LOG = Path(__file__).resolve().parent.parent / "logs" / "codegen_incomplete.jsonl"


def _log_incomplete(proposal: dict, raw: str, code: str, attempts: int) -> None:
    """Persist the RAW model output whenever codegen returns an incomplete module after its
    retries. Previously this output was DISCARDED, so the 2026-06 refusal spike could only be
    diagnosed by live reproduction. Now triage / the morning report can read the actual cause
    (refusal text vs. truncation vs. wrong format) from logs/codegen_incomplete.jsonl. Logging
    must NEVER break codegen — fail silent."""
    try:
        from datetime import datetime
        rec = {"ts": datetime.now().isoformat(timespec="seconds"),
               "title": str(proposal.get("title", "?"))[:120],
               "attempts": attempts, "raw_len": len(raw or ""), "code_len": len(code or ""),
               "raw_head": (raw or "")[:2000]}
        INCOMPLETE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with INCOMPLETE_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def generate(proposal: dict) -> str:
    prompt = (f"{CONTRACT}\n\n=== PROPOSAL TO IMPLEMENT ===\n{json.dumps(proposal, indent=2)}\n\n"
              f"Write the COMPLETE module now as ONE ```python code block with the full implementation "
              f"(imports + def signal + any helpers). Do NOT emit a skeleton, outline, or partial block.")
    code, raw = "", ""
    LAST_GEN["attempts"] = 0
    for _ in range(3):  # retry-on-empty at the SOURCE -> kills the wasted consistency/fix call per run
        LAST_GEN["attempts"] += 1
        raw = _pi(prompt)
        code = _extract_code(raw)
        if looks_complete(code):
            break
    LAST_GEN["empty_retries"] = LAST_GEN["attempts"] - 1
    if not looks_complete(code):
        # capture WHY (refusal? truncation? wrong format?) so the next regression is diagnosable
        # from the log, not a live repro. The run-loop fix() still repairs from here.
        _log_incomplete(proposal, raw, code, LAST_GEN["attempts"])
    return code  # if still incomplete after 3, the run-loop fix() repairs


def error_class(traceback: str) -> str:
    """Coarse error class for the fail->success memory (Stage 4b): exception type + the failing
    sdk/agent module if one appears in the traceback. Shared with agent.triage."""
    tb = traceback or ""
    exc = None
    for m in re.finditer(r"^(\w+(?:Error|Exception|Warning))\b", tb, re.MULTILINE):
        exc = m.group(1)
    exc = exc or ("SANDBOX" if "SANDBOX" in tb else "THESIS_MISMATCH" if "THESIS MISMATCH" in tb else "unknown")
    mods = re.findall(r'File "/root/crucible/(sdk|agent)/([\w]+)\.py"', tb)
    return f"{exc}:{mods[-1][1]}" if mods else exc


def _past_lessons(err_class: str, limit: int = 2) -> str:
    """Stage 4b: query the triage log for past diagnoses of the SAME error class and inject them
    into the fix prompt — the forge stops re-deriving known root causes from scratch."""
    log = Path(__file__).resolve().parent.parent / "logs" / "triage_log.jsonl"
    if not log.exists() or err_class == "unknown":
        return ""
    try:
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    except ValueError:
        return ""
    hits = [r for r in rows if r.get("error_class") == err_class and r.get("root_cause")][-limit:]
    if not hits:
        return ""
    out = ["\n=== PAST DIAGNOSES of this exact error class (from the nightly triage loop) ==="]
    for r in hits:
        out.append(f"- [{r.get('location', '?')}] {r['root_cause']}"
                   + (f" FIX THAT WORKED: {r['fix_summary']}" if r.get("fix_summary") else ""))
    out.append("If the same root cause applies, fix it the same way instead of guessing.")
    return "\n".join(out)


def fix(code: str, traceback: str) -> str:
    lessons = _past_lessons(error_class(traceback))
    prompt = (f"{CONTRACT}\n\nThe following module FAILED. Fix it; output ONLY the corrected ```python module.\n\n"
              f"=== CODE ===\n{code}\n\n=== ERROR ===\n{traceback[-2500:]}{lessons}")
    return _extract_code(_pi(prompt))


# Stage 4a severity ladder (QuantaAlpha regulator pattern): only major+ costs a regeneration.
# 'minor' (window/normalization/cosmetic deviations that preserve the mechanism) is logged, not fixed —
# the consistency-fix tail (median 347s vs 211s clean) was dominated by repairs of immaterial nits.
SEVERITY_FIX = ("major", "critical")


def consistency_check(proposal: dict, code: str) -> tuple:
    """Verify the generated code FAITHFULLY implements the proposal's economic thesis (catches code that
    claims one mechanism but computes another — e.g. 'split-consistent' but uses raw shares). Fail-OPEN
    on a parse error (best-effort guard, not a hard gate).
    Returns (severity: none|minor|major|critical, issues: str, corrected: str|None) — for major+ the
    SAME call returns the corrected module (saves the separate fix() round trip when possible)."""
    claim = {k: proposal.get(k) for k in ("premium", "market", "signal_approach", "why_not_duplicate")}
    prompt = (f"PROPOSAL (the economic thesis the code MUST implement):\n{json.dumps(claim, indent=2)}\n\n"
              f"GENERATED CODE:\n```python\n{code[:30000]}\n```\n\n"  # full module (Fable writes 15-20K; a truncated view causes false 'code is truncated' verdicts -> wasted fix() calls)
              f"Does the code FAITHFULLY implement that thesis + frozen signal construction? Check specifically: "
              f"the actual computation matches the claimed mechanism/direction; point-in-time data (datekey, no "
              f"look-ahead); correct adjustments (splits, dividends, costs); the right universe; the signal sign.\n"
              f"Grade the WORST deviation found:\n"
              f"- none: faithful implementation\n"
              f"- minor: immaterial deviations (slightly different window/normalization/clipping that "
              f"preserve the mechanism, direction, point-in-time discipline and costs) — acceptable, do NOT flag for repair\n"
              f"- major: the mechanism, universe, costs or rebalance cadence deviates from the frozen design\n"
              f"- critical: wrong sign, look-ahead, wrong data field, or the claimed premium is not what is computed\n"
              f'Return ONLY JSON: {{"severity": "none"|"minor"|"major"|"critical", '
              f'"issues": "specific claim-vs-code mismatches, or empty", '
              f'"corrected_code": "<for major/critical ONLY: the FULL corrected python module as one string; else null>"}}')
    d = extract_json(_pi(prompt))
    if d is None:
        return "none", "", None  # fail-OPEN: best-effort guard, not a hard gate
    if "severity" in d:
        sev = str(d["severity"]).lower()
        if sev not in ("none", "minor", "major", "critical"):
            sev = "major"  # graded but unrecognized -> treat as repair-worthy
    else:
        sev = "major" if d.get("consistent") is False else "none"  # tolerate old-shape replies
    corrected = d.get("corrected_code")
    corrected = corrected.strip() if isinstance(corrected, str) and looks_complete(corrected) else None
    return sev, str(d.get("issues", ""))[:500], corrected
