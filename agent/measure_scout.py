"""MEASURE (frontier #54): do WEB-SEEDED hypotheses (scout / Firecrawl research candidates) clear the
gate stack better than SELF-generated ones? Reads agent/run_log.jsonl (which already records arm +
the full proposal + tier + passed_all + a gate-progress arm_reward per run), classifies each run by
(arm, proposal.seeded_by), and compares ran / stage-1 / PASSED_ALL rates + mean arm_reward.

First Principle: report UNRESOLVED until N per class is adequate; full PASSES are rare, so mean
arm_reward (a continuous gate-PROGRESS score) is the primary signal, with stage-1-rate secondary and
full-pass-rate tertiary. CAVEAT: observational, not a randomized A/B — web-seeded ideas may sit in
different premia/markets than self-generated ones, so a difference is SUGGESTIVE, not causal proof.

    python -m agent.measure_scout
"""
import json
from collections import defaultdict
from pathlib import Path

RUNLOG = Path(__file__).resolve().parent / "run_log.jsonl"
_SELF = {"", "self", "none", "n/a", "na", "novel", "null"}


def classify(o: dict) -> str:
    arm = (o.get("arm") or "explore")
    if arm != "explore":
        return "self-evolutionary"                 # refine/orthogonal/crossover: evolved from elites
    prop = o.get("proposal") or {}
    if "seeded_by" not in prop:
        return "explore-pre-instrument"            # logged before the seeded_by field existed
    sb = str(prop.get("seeded_by") or "").strip().lower()
    return "self-novel" if sb in _SELF else "web-seeded"


def _stats(rs: list) -> dict:
    n = len(rs)
    ran = sum(1 for o in rs if o.get("ran"))
    s1 = sum(1 for o in rs if (o.get("verdict") or {}).get("stage1_pass"))
    pa = sum(1 for o in rs if o.get("passed_all"))
    rew = sum(float(o.get("arm_reward") or 0) for o in rs) / n if n else 0.0
    return {"n": n, "ran": ran, "stage1": s1, "passed_all": pa, "reward": rew}


def main():
    rows = [json.loads(l) for l in RUNLOG.read_text().splitlines() if l.strip()] if RUNLOG.exists() else []
    by = defaultdict(list)
    for o in rows:
        by[classify(o)].append(o)

    print(f"run_log: {len(rows)} runs total\n")
    print(f"{'class':24s} {'n':>4} {'ran%':>6} {'stage1%/ran':>12} {'PASS%':>6} {'reward':>7}")
    for cls in ("web-seeded", "self-novel", "self-evolutionary", "explore-pre-instrument"):
        s = _stats(by.get(cls, []))
        if not s["n"]:
            print(f"{cls:24s} {0:>4}")
            continue
        print(f"{cls:24s} {s['n']:>4} {100*s['ran']/s['n']:>5.0f}% "
              f"{100*s['stage1']/max(s['ran'],1):>11.0f}% {100*s['passed_all']/s['n']:>5.1f}% {s['reward']:>7.2f}")

    web = by.get("web-seeded", [])
    self_ = by.get("self-novel", []) + by.get("self-evolutionary", [])
    print()
    if len(web) < 30 or len(self_) < 30:
        print(f"VERDICT: UNRESOLVED \u2014 need >=~30 per class for any signal "
              f"(web-seeded n={len(web)}, self n={len(self_)}). Keep accumulating nightly; re-run in a few weeks. "
              f"(Pre-instrument explore runs lack seeded_by and are excluded.)")
    else:
        w, s = _stats(web), _stats(self_)
        lift = w["reward"] - s["reward"]
        print(f"VERDICT (observational): web-seeded mean gate-reward {w['reward']:.2f} vs self {s['reward']:.2f} "
              f"(\u0394={lift:+.2f}); stage-1 {100*w['stage1']/max(w['ran'],1):.0f}% vs {100*s['stage1']/max(s['ran'],1):.0f}%; "
              f"PASS {100*w['passed_all']/w['n']:.1f}% vs {100*s['passed_all']/s['n']:.1f}%.")
        print("  -> " + ("Web-seeding lifts gate progress: KEEP/expand the research layer." if lift > 0.05
                         else "No material lift: reconsider the research layer's cost/benefit." if lift < -0.05
                         else "Indistinguishable: keep (cheap) but it's not a clear win."))
        print("  CAVEAT: observational, not a randomized A/B \u2014 treat as suggestive.")


if __name__ == "__main__":
    main()
