"""Write an experiment verdict into the shared research wiki (the compounding memory)."""
from datetime import date
from pathlib import Path

import os
import tempfile

from crucible_paths import WIKI  # central config
from sdk.locks import FileLock


def _atomic_write(path: Path, text: str) -> None:
    """temp-file + os.replace so a crash mid-write never leaves a torn page."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _fmt_grid(gs) -> str:
    if not gs:
        return "n/a"
    return " · ".join(f"{k}={v}" for k, v in sorted(gs.items(), key=lambda x: -(x[1] if x[1] is not None else -99)))


def _fmt_diag(d) -> str:
    """Render the assemble_bundle diagnostics the reviewer needs, compactly."""
    if not d:
        return "- n/a (run predates O1 or screen-failed before the rails)"
    def _r(v, nd=3):
        try:
            return round(float(v), nd)
        except (TypeError, ValueError):
            return v
    lines = []
    c = d.get("cpcv") or {}
    if c:
        lines.append(f"- CPCV: {c.get('n_paths')} paths, median {_r(c.get('median_sharpe'))}, "
                     f"{(c.get('frac_positive') or 0):.0%} positive, range [{_r(c.get('min'), 2)}, {_r(c.get('max'), 2)}]"
                     if c.get('min') is not None else f"- CPCV: {c}")
    p = d.get("pbo")
    if isinstance(p, dict):
        lines.append(f"- PBO: {_r(p.get('value'))} (n_combos {p.get('n_combos')}, n_configs {p.get('n_configs')})")
    if d.get("dsr_source") is not None:
        lines.append(f"- DSR trials: raw {d.get('dsr_n_trials_raw')} -> effective {d.get('dsr_n_trials_effective')} "
                     f"(participation {_r(d.get('grid_participation_ratio'))}, source {d.get('dsr_source')}, "
                     f"search burden {_r(d.get('search_burden'))})")
    t = d.get("ticker_concentration") or {}
    if t:
        lines.append(f"- concentration: top group {_r(t.get('top_group_frac'))} of pnl across {t.get('n_tickers')} names")
    r = d.get("regime")
    if isinstance(r, dict):  # compact: the full per-regime dicts are noise; keep the gate-relevant numbers
        lines.append(f"- regime: min Sharpe {_r(r.get('min_regime_sharpe'), 2)}, "
                     f"max pnl frac {_r(r.get('max_regime_pnl_frac'), 2)}, "
                     f"concentration {_r(r.get('regime_concentration_ratio'), 2)}, "
                     f"per-regime expectancy ok={r.get('per_regime_expectancy_ok')}")
    if d.get("n_obs"):
        lines.append(f"- n_obs: {d['n_obs']}")
    return "\n".join(lines) or "- n/a"


def write_experiment(spec, verdict: dict):
    status = ("VALIDATED" if verdict["PASSED_ALL_GATES"]
              else "REJECTED-MCPT" if verdict.get("stage1_pass") and verdict.get("mcpt_pass") is False
              else "CANDIDATE" if verdict.get("stage1_pass")   # cleared stage-1, awaiting confirmation
              else "NEAR-MISS" if (verdict.get("dsr") or 0) and verdict["dsr"] >= 0.85
              else "FAIL")
    page = WIKI / "experiments" / f"{spec.id}.md"
    # ID-COLLISION GUARD: two different proposals can pick the same spec.id (happened with
    # amihud_illiquidity_smallcap 2026-06-10: a cost-hardened VARIANT silently overwrote the
    # original PROMOTE-tier page). Never overwrite a page whose title differs — version it.
    if page.exists():
        try:
            old = page.read_text()
            old_title = next((l[2:].strip() for l in old.splitlines() if l.startswith("# ")), "")
            if old_title and old_title != spec.title:
                import hashlib
                suffix = hashlib.sha1(spec.title.encode()).hexdigest()[:6]
                page = WIKI / "experiments" / f"{spec.id}__{date.today().strftime('%Y%m%d')}-{suffix}.md"
                print(f"[wiki] id collision on '{spec.id}' (existing page is a different experiment) "
                      f"-> versioned page {page.name}")
        except Exception:
            pass
    _atomic_write(page, f"""---
id: {spec.id}
status: {status}
project: {spec.project}
date: {date.today()}
family: {spec.family}
markets: {spec.markets}
data: {spec.data_desc}
generated_by: crucible-agent
---
# {spec.title}

## Pre-registration (FROZEN before running)
{spec.pre_registration}

## Verdict: {status}
- tier **{verdict['tier']}** (FDR bar {verdict['promote_bar']}, n_families {verdict['n_families']})
- DSR {verdict['dsr']} | CPCV {verdict['median_cpcv']} | PBO {verdict['pbo']}
- search Sharpe {verdict['search_sharpe']} -> holdout Sharpe {verdict['holdout_sharpe']} | **holdout_gate PASS={verdict['holdout_pass']}** {verdict['holdout_reasons']}
- deployment passed={verdict['deployment_passed']} peak={verdict['deploy_peak']} sectors={verdict['deploy_sectors']} {verdict['deploy_reasons']}
- full Sharpe {verdict['full_sharpe']} | maxDD {verdict['full_maxdd']} | trades {verdict['n_trades']}
- stage-1 pass: {verdict.get('stage1_pass')} | scope: {verdict.get('scope')}
- stage-2 MCPT: pass={verdict.get('mcpt_pass')} {verdict.get('mcpt') or ''}
- stage-2 generalization: {verdict.get('generalization')} — {verdict.get('generalization_note') or 'n/a'}
- needs_confirmation: {verdict.get('needs_confirmation') or 'none'}
- **PASSED ALL GATES: {verdict['PASSED_ALL_GATES']}**

## Reproducibility (O1)
- module: `{verdict.get('module_file') or 'n/a'}` sha1 `{verdict.get('module_sha') or 'n/a'}` | crucible repo `{verdict.get('repo_sha') or 'n/a'}`
- config_hash (write-once holdout key): `{verdict.get('config_hash') or 'n/a'}`
- default_params: `{verdict.get('default_params')}` | holdout_start: {verdict.get('holdout_start')}
- grid Sharpes (search window): {_fmt_grid(verdict.get('grid_sharpes'))}

## Gate diagnostics
{_fmt_diag(verdict.get('diagnostics'))}
""")
    # append to log + index (use the page stem — may be the versioned name on collision).
    # 3 smiths finish concurrently: serialize the shared-file appends.
    stem = page.stem
    with FileLock("wiki-append", ttl=30):
        with open(WIKI / "log.md", "a") as f:
            f.write(f"\n## [{date.today()}] experiment | {stem} -> {status} (tier {verdict['tier']}, holdout {verdict['holdout_pass']})")
        with open(WIKI / "index.md", "a") as f:
            f.write(f"\n- [[experiments/{stem}]] — {status} ({spec.title})")
    return page
