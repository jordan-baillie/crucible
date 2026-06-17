"""Regression: deployment_sanity position-day SHARES must use |position_value| (abs).

Bug (found 2026-06-17): single_name_share / max_sector_share / hedge_share divided the
biggest name (or sector) by SIGNED dollar-position-days summed over the book. For a
market-neutral long/short book the longs and shorts cancel/flip that denominator
(total_pos_days -> ~0 or negative), so every "share" becomes a degenerate value. A
genuinely diversified 412-name L/S book (true top-name share 0.015) was force-failed with
single_name_share computed as 1.00 (signed total = -47.7e9 vs abs +133.5e9), and a parent
book reported max_sector_share = 2.517 (a "share" > 1 is impossible). This silently killed
the ENTIRE class of market-neutral L/S strategies (deployment-sanity gates the holdout).

Fix: shares use abs(position_value) — concentration is a magnitude concept (a short is as
concentrated as a long). The 0.40 cap is unchanged; long-only books are byte-identical.
These tests freeze: (1) diversified L/S is NOT false-failed, (2) real concentration still
fails, (3) every share stays in [0, 1], (4) long-only behaviour is unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_integrity import deployment_sanity  # noqa: E402

_SECTORS = [f"S{i}" for i in range(11)]
_META = {"max_positions": 20}


def _t(tk, sector, pv, hold=200, entry="2015-01-05", exit_="2015-07-24"):
    return {"ticker": tk, "sector": sector, "entry_date": entry, "exit_date": exit_,
            "hold_days": hold, "position_value": pv, "pnl": 0.0}


def _market_neutral_ls(n_per_side=50, notional=10_000.0):
    """A balanced long/short book whose SIGNED position-days sum to ~0 (the degenerate case)."""
    book = [_t(f"L{i}", _SECTORS[i % 11], +notional) for i in range(n_per_side)]
    book += [_t(f"S{i}", _SECTORS[i % 11], -notional) for i in range(n_per_side)]
    return book


def test_market_neutral_ls_not_false_failed():
    """The exact bug: a diversified market-neutral L/S book must pass, with its TRUE
    (small) single-name share — not the degenerate ~1.0 the signed denominator produced."""
    r = deployment_sanity(_market_neutral_ls(), strategy_meta=_META)
    # signed denominator for a balanced book is ~0 -> proves we're in the degenerate regime
    signed_total = sum((t["hold_days"] + 1e-9) * t["position_value"] for t in _market_neutral_ls())
    assert abs(signed_total) < 1.0, "fixture must be balanced (signed total ~0) to exercise the bug"
    assert r["single_name_share"] < 0.05, r           # true magnitude ~1/100, NOT ~1.0
    assert r["passed"] is True, r["forced_fail_reasons"]
    assert "single_name_share" not in " ".join(r["forced_fail_reasons"])


def test_shares_are_bounded_in_unit_interval():
    """Any 'share' is a fraction of gross exposure: must lie in [0, 1]. The signed bug
    produced max_sector_share = 2.517 (>1) — freeze that this can never recur."""
    r = deployment_sanity(_market_neutral_ls(), strategy_meta=_META)
    assert 0.0 <= r["single_name_share"] <= 1.0, r
    assert 0.0 <= r["max_sector_share"] <= 1.0, r


def test_real_concentration_still_fails():
    """Teeth intact: one name carrying ~all gross exposure must still force-fail, even
    though the book is long/short (so the OLD signed code might have hidden it)."""
    conc = [_t("BIG", _SECTORS[0], +1_000_000, hold=2000)]
    conc += [_t(f"x{i}", _SECTORS[i % 11], (+100 if i % 2 else -100), hold=50) for i in range(60)]
    r = deployment_sanity(conc, strategy_meta=_META)
    assert r["single_name_share"] > 0.40, r
    assert r["passed"] is False
    assert any("single_name_share" in s for s in r["forced_fail_reasons"]) 


def test_long_only_unchanged():
    """Long-only books have pv >= 0 so abs == signed: behaviour must be identical."""
    diversified = [_t(f"N{i}", _SECTORS[i % 11], +10_000) for i in range(60)]
    r = deployment_sanity(diversified, strategy_meta=_META)
    assert r["single_name_share"] < 0.40 and r["passed"] is True, r
    concentrated = [_t("DOM", _SECTORS[0], +1_000_000, hold=2000)]
    concentrated += [_t(f"y{i}", _SECTORS[i % 11], +500, hold=20) for i in range(60)]
    r2 = deployment_sanity(concentrated, strategy_meta=_META)
    assert r2["single_name_share"] > 0.40 and r2["passed"] is False, r2
