"""Operator-directed SEARCH FOCUS dispatch (agent.propose._focus, env CRUCIBLE_FOCUS).

Generator-only steer injected into every arm's prompt; the rails/gate stack are untouched. These
tests lock the dispatch contract: known focuses return their block, aliases coincide, and anything
unknown/empty returns NO steer (the general retail-deployable bias) — so a typo can never silently
mangle the prompt, and the commodities focus always names the full conditioning TRIO of adapters.
"""
import os

import pytest

from agent import propose as P


def _focus_with(value):
    """Evaluate _focus() with CRUCIBLE_FOCUS set to `value` (None = unset), restoring the env after."""
    prev = os.environ.get("CRUCIBLE_FOCUS")
    try:
        if value is None:
            os.environ.pop("CRUCIBLE_FOCUS", None)
        else:
            os.environ["CRUCIBLE_FOCUS"] = value
        return P._focus()
    finally:
        if prev is None:
            os.environ.pop("CRUCIBLE_FOCUS", None)
        else:
            os.environ["CRUCIBLE_FOCUS"] = prev


@pytest.mark.parametrize("value", [None, "", "equity", "stonks", "  ", "cryptos"])
def test_unknown_or_empty_focus_gives_no_steer(value):
    assert _focus_with(value) == ""


def test_crypto_focus_block():
    out = _focus_with("crypto")
    assert "SEARCH FOCUS: CRYPTO" in out
    assert "binance_universe" in out  # the broad cross-section the crypto block must name


def test_commodities_focus_names_the_full_trio():
    out = _focus_with("commodities")
    assert "SEARCH FOCUS: COMMODITY FUTURES" in out
    # the trio must all be present — this is the whole point of the focus
    for adapter in ("fut_curve", "cot_positioning", "eia_series", "usda_nass"):
        assert adapter in out, f"commodities focus missing {adapter}"
    # PIT discipline the smith must obey must be spelled out
    assert "NOT roll-adjusted" in out          # fut_curve within-contract returns
    assert "RELEASE date" in out               # COT Friday + EIA/USDA report release


def test_commodity_and_futures_are_aliases_of_commodities():
    base = _focus_with("commodities")
    assert _focus_with("commodity") == base
    assert _focus_with("futures") == base


def test_focus_is_case_and_whitespace_insensitive():
    assert _focus_with("  CRYPTO ") == _focus_with("crypto")
    assert _focus_with("Commodities") == _focus_with("commodities")
