"""Write an experiment verdict into the shared research wiki (the compounding memory)."""
from datetime import date
from pathlib import Path

WIKI = Path("/root/research-wiki")


def write_experiment(spec, verdict: dict):
    status = ("VALIDATED" if verdict["PASSED_ALL_GATES"]
              else "NEAR-MISS" if (verdict.get("dsr") or 0) and verdict["dsr"] >= 0.85
              else "FAIL")
    page = WIKI / "experiments" / f"{spec.id}.md"
    page.write_text(f"""---
id: {spec.id}
status: {status}
project: {spec.project}
date: {date.today()}
family: {spec.family}
markets: {spec.markets}
data: {spec.data_desc}
generated_by: hephaestus-agent
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
- **PASSED ALL GATES: {verdict['PASSED_ALL_GATES']}**
""")
    # append to log + index
    with open(WIKI / "log.md", "a") as f:
        f.write(f"\n## [{date.today()}] experiment | {spec.id} -> {status} (tier {verdict['tier']}, holdout {verdict['holdout_pass']})")
    with open(WIKI / "index.md", "a") as f:
        f.write(f"\n- [[experiments/{spec.id}]] — {status} ({spec.title})")
    return page
