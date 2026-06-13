"""fut_curve adapter regression tests (Databento GLBX contract-month substrate).
Skips cleanly if the one-time data pull isn't present (CI / fresh checkout)."""
import os
import pytest

from crucible_paths import DATA

SRC = str(DATA / "databento" / "CL_ohlcv1d.parquet")
pytestmark = pytest.mark.skipif(not os.path.exists(SRC), reason="Databento pull not present")


@pytest.fixture(scope="module")
def cl():
    from sdk.adapters import fut_curve
    return fut_curve("CL")


def test_shape_and_coverage(cl):
    assert cl.index.min().year == 2010 and cl.index.max().year >= 2026
    # both nearest contracts present essentially every traded day
    assert (cl["close_1"].notna() & cl["close_2"].notna()).mean() > 0.99


def test_no_spreads_or_butterflies(cl):
    import re
    pat = re.compile(r"^CL[FGHJKMNQUVXZ]\d{1,2}$")
    for col in ("symbol_1", "symbol_2"):
        bad = cl[col].dropna()[~cl[col].dropna().str.fullmatch(pat.pattern)]
        assert bad.empty, f"non-outright symbols leaked into {col}: {bad.unique()[:5]}"


def test_front_expires_before_second(cl):
    """Rank-1 must always be the earlier expiry: on roll days the front symbol CHANGES to the
    old second symbol — verify the chain advances monotonically through month codes."""
    rolls = cl[cl["symbol_1"] != cl["symbol_1"].shift()].dropna(subset=["symbol_1"])
    # after a roll, the new front is almost always yesterday's second contract
    prev_second = cl["symbol_2"].shift()
    roll_days = cl["symbol_1"] != cl["symbol_1"].shift()
    match = (cl.loc[roll_days, "symbol_1"] == prev_second[roll_days]).mean()
    assert match > 0.85, f"only {match:.0%} of rolls promote the prior second contract"
    assert len(rolls) > 100  # ~monthly rolls over 16y


def test_decade_wrap_disambiguation(cl):
    """CLX0 = Nov 2010 AND Nov 2020 (single-digit year codes recycle each decade)."""
    assert cl.loc["2010-10-15", "symbol_1"] == "CLX0"
    assert cl.loc["2020-10-15", "symbol_1"] == "CLX0"
    # and in between, X-contracts of other years appear (i.e. it's not stuck)
    mid = cl.loc["2015-10-15", "symbol_1"]
    assert mid == "CLX5"


def test_days_to_roll_counts_down(cl):
    within = cl.groupby("symbol_1")["days_to_roll_1"]
    assert (within.last() == 0).mean() > 0.95  # last day on the front == roll day


def test_ret_field_roll_and_negative_safe(cl):
    """field='ret' (added 2026-06-14): roll-safe, recycled-symbol-safe, negative-base-safe.
    Guards the three futures-returns footguns that crashed/contaminated hand-rolled returns."""
    from sdk.adapters import fut_curve
    ret = fut_curve("CL", field="ret")
    assert ret.name == "CL_ret" and len(ret) == len(cl)
    # 1. roll-safe: return is NaN at every front-contract change (no diff across rolls)
    sym = cl["symbol_1"]
    rolls = sym.ne(sym.shift()) & sym.shift().notna()
    assert ret[rolls].isna().all(), "returns bridged a roll boundary"
    # 2. recycled-symbol-safe + negative-base-safe: annualized vol must be sane (~0.3-0.5 for CL).
    #    A symbol-string groupby would bridge CLZ0 2010<->2020 (vol >1.0); a raw pct_change off
    #    the -$2.67 2020-04-20 settle would explode to +439%.
    assert ret.std() * (252 ** 0.5) < 0.7, "vol too high — roll/recycle/negative contamination"
    assert ret.loc["2020-04-21"] != ret.loc["2020-04-21"] or abs(ret.loc["2020-04-21"]) < 1, \
        "negative-base pct_change not masked (2020-04-21 off the -$2.67 base)"
    # 3. list form -> wide panel; bare-string footgun-safe
    panel = fut_curve(["CL", "GC"], field="ret")
    assert list(panel.columns) == ["CL", "GC"]
    # 4. invalid field rejected loudly
    with pytest.raises(ValueError):
        fut_curve("CL", field="close")
