"""Stage 4 codegen hygiene: consistency severity ladder (4a) + error-class memory (4b)
+ contract<->adapters drift guard (2026-06-23)."""
import inspect
import json

import pytest

from agent import codegen


# ---------- contract <-> adapters drift guard (2026-06-23) ----------

def test_contract_lists_every_public_adapter():
    """The CONTRACT's adapter whitelist is single-sourced from sdk.adapters so it can never again
    omit an adapter that EXISTS. Regression that motivated this: the crypto adapters (binance_klines,
    binance_universe, coinmetrics_metrics, bybit_funding, deribit_dvol) lived in sdk.adapters but were
    missing from the hardcoded whitelist, so Opus refused ~33% of crypto codegens ('I can't write
    this honestly — no crypto data') and hallucinated-then-crashed the rest. This test fails the
    moment the rendered contract drifts from the code."""
    from sdk import adapters
    names = [n for n, f in vars(adapters).items()
             if inspect.isfunction(f) and not n.startswith("_")
             and f.__module__ == adapters.__name__ and n != "list_adapters"]
    missing = [n for n in names if n not in codegen.CONTRACT]
    assert not missing, f"CONTRACT omits adapters that exist in sdk.adapters: {missing}"
    # the exact adapters whose omission caused the 2026-06 refusal/runtime_error spike
    for crypto in ("binance_klines", "binance_universe", "coinmetrics_metrics",
                   "bybit_funding", "deribit_dvol", "funding_rates"):
        assert crypto in codegen.CONTRACT, f"{crypto} missing from rendered CONTRACT"


# ---------- 4a: severity ladder ----------

def test_severity_parsing(monkeypatch):
    replies = {}

    def fake_pi(prompt):
        return replies["next"]

    monkeypatch.setattr(codegen, "_pi", fake_pi)
    # a genuinely complete module compiles, has `def signal` AND a module-level SPEC (looks_complete
    # gates the corrected-code acceptance on exactly this).
    good_code = "def signal(panel):\n    pass\nSPEC = object()\n" + "# pad\n" * 100

    # minor -> no repair flag, no corrected code
    replies["next"] = json.dumps({"severity": "minor", "issues": "slightly different vol window",
                                  "corrected_code": None})
    sev, issues, corrected = codegen.consistency_check({}, "code")
    assert sev == "minor" and corrected is None
    assert sev not in codegen.SEVERITY_FIX

    # critical with full corrected module -> returned
    replies["next"] = json.dumps({"severity": "critical", "issues": "sign flipped",
                                  "corrected_code": good_code})
    sev, issues, corrected = codegen.consistency_check({}, "code")
    assert sev == "critical" and sev in codegen.SEVERITY_FIX
    assert corrected and "def signal" in corrected

    # major with a SKELETON corrected_code (incomplete) -> rejected, fix() path must run instead
    replies["next"] = json.dumps({"severity": "major", "issues": "wrong universe",
                                  "corrected_code": "def signal(): ..."})
    sev, issues, corrected = codegen.consistency_check({}, "code")
    assert sev == "major" and corrected is None

    # unparseable reply -> fail-OPEN as 'none'
    replies["next"] = "not json at all"
    sev, issues, corrected = codegen.consistency_check({}, "code")
    assert sev == "none" and corrected is None

    # legacy-shape reply ({"consistent": false}) tolerated as major
    replies["next"] = json.dumps({"consistent": False, "issues": "x"})
    sev, _, _ = codegen.consistency_check({}, "code")
    assert sev == "major"


# ---------- 4b: error class + lessons injection ----------

def test_error_class():
    tb = ('Traceback (most recent call last):\n'
          '  File "/root/crucible/sdk/adapters.py", line 10, in sep_panel\n'
          '    raise KeyError("closeadj")\n'
          'KeyError: closeadj')
    assert codegen.error_class(tb) == "KeyError:adapters"
    assert codegen.error_class("ValueError: bad shape") == "ValueError"
    assert codegen.error_class("SANDBOX VIOLATION: open()") == "SANDBOX"
    assert codegen.error_class("") == "unknown"


def test_past_lessons_injection(tmp_path, monkeypatch):
    log = tmp_path / "logs" / "triage_log.jsonl"
    log.parent.mkdir()
    rows = [
        {"error_class": "KeyError:adapters", "location": "data",
         "root_cause": "SEP cache v1 lacks closeadj for post-2025 symbols.",
         "fix_summary": "use sep_panel cache v2 (versioned filename)"},
        {"error_class": "TypeError", "location": "strategy", "root_cause": "unrelated"},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(codegen, "Path", lambda *a, **k: _FakePath(tmp_path))

    out = codegen._past_lessons("KeyError:adapters")
    assert "SEP cache v1" in out and "cache v2" in out
    assert "unrelated" not in out
    assert codegen._past_lessons("unknown") == ""
    assert codegen._past_lessons("NoSuchClass") == ""


class _FakePath:
    """Make codegen._past_lessons resolve its log under tmp_path."""
    def __init__(self, base):
        self.base = base

    def resolve(self):
        return self

    @property
    def parent(self):
        return self

    def __truediv__(self, other):
        return self.base / "logs" / "triage_log.jsonl" if other == "triage_log.jsonl" else self
