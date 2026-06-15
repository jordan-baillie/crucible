"""Tests for the MACRO-NEUTRALIZATION gate (pre-reg: prereg-macro-neutralization-gate.md).

Annotate-only ship: _macro_decomp must COMPUTE the right metrics and never raise; it does NOT demote
(no demotion branch exists yet). All synthetic — NO network. Plus the fred_series provenance guard.
"""
import numpy as np
import pandas as pd
import pytest

from sdk.harness import _macro_decomp, MACRO_MIN_OBS
from sdk import adapters as ad

_COLS = ["dur", "slope", "breakeven", "credit", "usd", "oil", "gold", "vol"]


def _synth_factors(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    return pd.DataFrame(rng.normal(0, 0.01, size=(n, len(_COLS))), index=idx, columns=_COLS)


# ---------------------------------------------------------------- core decomposition behaviour
def test_orthogonal_edge_survives():
    """An edge uncorrelated with macro -> low R², neutral Sharpe ≈ gross Sharpe (survives)."""
    f = _synth_factors(seed=1)
    y = pd.Series(np.random.default_rng(99).normal(0.0005, 0.01, size=len(f)), index=f.index)
    r = _macro_decomp(y, f)
    assert r["evaluated"] is True
    assert r["n_obs"] >= MACRO_MIN_OBS and r["n_factors"] == 8
    assert r["macro_r2"] < 0.1                                    # macro explains ~k/n only
    assert abs(r["macro_residual_sharpe"] - r["gross_sharpe"]) < 0.5


def test_confounded_edge_collapses():
    """A disguised macro bet (returns ≈ the credit factor) -> high R², neutral Sharpe collapses."""
    f = _synth_factors(seed=2)
    rng = np.random.default_rng(7)
    f["credit"] = rng.normal(0.0015, 0.004, size=len(f))         # credit carries a robust positive premium
    y = 3.0 * f["credit"] + pd.Series(rng.normal(0, 0.0003, size=len(f)), index=f.index)
    r = _macro_decomp(y, f)  # deterministic (seed 7): r2=0.999 gross=4.67 resid=-0.44 p=0.0
    assert r["evaluated"] is True
    assert r["macro_r2"] > 0.9                                    # macro explains nearly all variance
    assert r["gross_sharpe"] > 2.0                               # gross LOOKS like a stellar edge
    assert abs(r["macro_residual_sharpe"]) < 1.0                  # ...but collapses once macro is hedged
    assert r["gross_sharpe"] - abs(r["macro_residual_sharpe"]) > 2.0
    assert r["macro_block_pvalue"] is not None and r["macro_block_pvalue"] < 0.05
    assert abs(r["macro_betas"]["credit"]) > 2.0                  # credit beta dominates the diagnostic
    assert abs(r["macro_betas"]["credit"]) > max(abs(v) for k, v in r["macro_betas"].items() if k != "credit")


def test_not_evaluated_too_few_obs():
    f = _synth_factors(n=200, seed=3)
    y = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=len(f)), index=f.index)
    r = _macro_decomp(y, f)
    assert r["evaluated"] is False and "overlapping obs" in r["note"]


def test_not_evaluated_thin_coverage():
    f = _synth_factors(seed=4)
    f.loc[f.index[: int(len(f) * 0.5)], "gold"] = np.nan          # 50% coverage < 80% floor
    y = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=len(f)), index=f.index)
    r = _macro_decomp(y, f)
    assert r["evaluated"] is False and "coverage" in r["note"] and "gold" in r["note"]


def test_macro_block_unavailable_is_not_evaluated():
    assert _macro_decomp(pd.Series([1.0, 2.0, 3.0]), None)["evaluated"] is False
    assert _macro_decomp(pd.Series([1.0, 2.0, 3.0]), pd.DataFrame())["evaluated"] is False


# ---------------------------------------------------------------- fred_series provenance guard (pure)
def test_guard_strict_raises_on_revised_release():
    with pytest.raises(ValueError):
        ad._check_fred_ids(["GDPC1"], allow_revised=False)


def test_guard_strict_raises_on_unknown_id():
    with pytest.raises(ValueError):
        ad._check_fred_ids(["TOTALLY_UNKNOWN_ID"], allow_revised=False)


def test_guard_strict_allows_the_frozen_macro_block():
    ad._check_fred_ids(["DGS10", "T10Y2Y", "T10YIE", "DBAA", "DAAA", "DTWEXBGS", "DCOILWTICO", "VIXCLS"],
                       allow_revised=False)  # must not raise


def test_guard_default_warns_on_revised(capsys):
    ad._check_fred_ids(["CPIAUCSL"], allow_revised=True)          # must not raise
    assert "LOOK-AHEAD" in capsys.readouterr().err


def test_allowlist_and_denylist_cover_the_frozen_sets():
    for sid in ["DGS10", "T10Y2Y", "T10YIE", "DBAA", "DAAA", "DTWEXBGS", "DCOILWTICO", "VIXCLS"]:
        assert sid in ad.MARKET_OBSERVED_FRED
    for sid in ["GDP", "CPIAUCSL", "UNRATE", "PAYEMS"]:
        assert sid in ad.REVISED_FRED_RELEASES
