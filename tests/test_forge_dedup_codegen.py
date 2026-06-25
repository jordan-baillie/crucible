"""Regressions for the two forge failure classes found 2026-06-19:

1. Empty/truncated codegen output was written to disk and run, loading a module with no `SPEC`
   attribute -> opaque AttributeError that triage could not diagnose ("no diagnosis returned"),
   then looping re-writing the same empty file. GATE: an incomplete module never reaches disk.
2. Dedup let the same dead idea recur every night:
   - family_bucket() scattered sibling crypto premia (funding-dispersion / supply-dilution) into a
     different per-rewording catch-all slug each time, silently defeating the 2/family queue cap.
   - _tested_titles() keyed on the `auto_<slug>_<agent>_<id>` page stem, which can never equal a
     normalized proposal title, so title dedup against tested experiments was a no-op.
"""
from agent import codegen, run_worker
from agent.families import family_bucket
from agent import director


# ---------------- 1. empty-module write gate ----------------

def test_looks_complete_rejects_empty_and_truncated():
    assert codegen.looks_complete("") is False
    assert codegen.looks_complete("def signal(panel):\n    pass") is False  # < 300 chars
    assert codegen.looks_complete("# no signal here\n" + "x = 1\n" * 100) is False
    # has `def signal` + length but NO module-level SPEC: the harness runs run_experiment(m.SPEC),
    # so a SPEC-less module is NOT complete (it used to slip through to an opaque AttributeError
    # casualty). The gate now delivers on its own docstring contract.
    assert codegen.looks_complete("def signal(panel):\n    return panel\n" + "# pad\n" * 100) is False
    # non-compiling (e.g. the generator wrote chain-of-thought prose into the .py) -> rejected
    assert codegen.looks_complete("I'll start by writing def signal but I'm not sure\n" + "x\n" * 200) is False
    # complete = compiles AND has `def signal` AND a module-level SPEC
    complete = "def signal(panel):\n    return panel\nSPEC = object()\n" + "# pad\n" * 100
    assert codegen.looks_complete(complete) is True
    # validate_module() returns the precise reason (None when valid) the fix-loop repairs against
    assert codegen.validate_module(complete) is None
    assert "SPEC" in codegen.validate_module("def signal(panel):\n    return panel\n" + "# pad\n" * 100)
    assert "SyntaxError" in codegen.validate_module("def signal(:\n  bad(\n" + "# pad\n" * 100)


def test_empty_codegen_never_writes_or_runs(monkeypatch, tmp_path):
    """generate() and every fix() return empty -> the worker must NOT write a strategy file,
    must NOT call _run_module, and must record fail_reason='codegen_empty' (not the misleading
    'sandbox_rejected') so triage and the morning report see the real cause."""
    ran = {"called": False}
    monkeypatch.setattr(run_worker.queue, "claim_next",
                        lambda agent: {"id": "q1", "proposal": {"title": "Empty Strat Test"},
                                       "arm": "explore", "parent_ids": []})
    monkeypatch.setattr(run_worker.queue, "complete", lambda *a, **k: None)
    monkeypatch.setattr(run_worker.codegen, "generate", lambda prop: "")
    monkeypatch.setattr(run_worker.codegen, "fix", lambda code, tb: "")
    monkeypatch.setattr(run_worker, "_run_module",
                        lambda stem: ran.__setitem__("called", True) or (None, ""))
    monkeypatch.setattr(run_worker, "RUNLOG", tmp_path / "run_log.jsonl")
    import agent.elite as elite
    monkeypatch.setattr(elite, "record", lambda outcome: None)

    before = set((run_worker.ROOT / "strategies").glob("auto_empty_strat_test*.py"))
    out = run_worker.run_one_from_queue()
    after = set((run_worker.ROOT / "strategies").glob("auto_empty_strat_test*.py"))

    assert ran["called"] is False, "an empty module must never be executed"
    assert after == before, "an incomplete module must never be written to disk"
    assert out["ran"] is False
    assert out["fail_reason"] == "codegen_empty"
    assert out["stages"]["codegen_empty"] >= 1


# ---------------- 2a. family_bucket collapses siblings ----------------

def test_recurring_crypto_premia_collapse_to_one_bucket_each():
    funding = [
        "Cross-Exchange Funding-Dispersion Delta-Neutral Carry (Bybit-Binance)",
        "Cross-Exchange Funding DISPERSION as a Crypto Leverage-Crowd",
        "Cross-Exchange Funding-Spread Harvest — Bybit-Binance perp",
    ]
    dilution = [
        "Crypto supply-emission DILUTION premium — cross-sectional L/S",
        "Crypto token-emission (supply-dilution) premium",
        "Crypto token issuance supply dilution cross-sectional",
        "Crypto supply-inflation dilution premium",
    ]
    assert {family_bucket(t) for t in funding} == {"funding_dispersion"}
    assert {family_bucket(t) for t in dilution} == {"issuance"}
    # the 2/family queue cap now actually caps these clusters (was 7 distinct buckets -> uncapped)
    assert len({family_bucket(t) for t in funding + dilution}) == 2


# ---------------- 2b. tested-experiment dedup keys on the title, not the stem ----------------

def test_strip_auto_affixes():
    assert director._strip_auto_affixes(
        "auto_crypto_supply_inflation_dilution_premium_smith3_33634"
    ) == "crypto_supply_inflation_dilution_premium"
    assert director._strip_auto_affixes("auto-foo-bar-omdtx1-72241") == "foo-bar"


def test_tested_dedup_matches_proposal_title_via_h1(monkeypatch, tmp_path):
    exp = tmp_path / "experiments"
    exp.mkdir()
    (exp / "auto_crypto_xyz_smith3_3803.md").write_text(
        "---\nid: auto_crypto_xyz_smith3_3803\nstatus: FAIL\n---\n"
        "# Crypto supply-emission DILUTION premium — cross-sectional L/S\n\nbody\n",
        encoding="utf-8")
    monkeypatch.setattr(director, "WIKI", tmp_path)

    tested = director._tested_titles()
    # an exact re-proposal of the already-tested (failed) idea is now recognized...
    assert director._norm("Crypto supply-emission DILUTION premium — cross-sectional L/S") in tested
    # ...whereas the OLD stem-based key (auto_..._smith3_3803) never could be
    assert director._norm("auto_crypto_xyz_smith3_3803") not in tested
