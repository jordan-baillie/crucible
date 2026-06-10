"""Golden tests for the statistical rails — the math the whole system's honesty rests on."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import research_integrity as ri  # noqa: E402
from research_integrity.cpcv import cpcv_splits, has_leakage  # noqa: E402
from research_integrity.overfitting import pbo_cscv, deflated_sharpe_ratio  # noqa: E402


def test_promote_dsr_schedule():
    """The FDR bar must rise with families tested; spot-check the schedule."""
    assert ri.promote_dsr(1) == pytest.approx(0.90, abs=0.01)
    bars = [ri.promote_dsr(n) for n in (1, 4, 10, 28, 100)]
    assert bars == sorted(bars), "bar must be monotonically non-decreasing in n_families"
    assert bars[-1] < 1.0, "bar must stay a valid probability"


def test_cpcv_no_leakage():
    """Purge/embargo correctness: has_leakage (the in-package oracle) must be clean
    on every generated split, and train/test must never intersect."""
    splits = cpcv_splits(1500, n_groups=8, k_test=2, purge=1)
    assert len(splits) == 28  # C(8,2)
    for sp in splits:
        assert not has_leakage(sp, purge=1, embargo=5), "purged CPCV split leaked"
        assert len(set(sp.train_idx) & set(sp.test_idx)) == 0


def test_pbo_on_known_matrices():
    """PBO ~ high (≈0.5) when in-sample ranking is luck; low when skill is consistent."""
    rng = np.random.default_rng(0)
    n_days, n_cfg = 1200, 12
    skill = rng.normal(0, 0.01, (n_days, n_cfg)) + np.linspace(0, 0.002, n_cfg)
    pbo_skill = pbo_cscv(skill)["pbo"]
    noise = rng.normal(0, 0.01, (n_days, n_cfg))
    pbo_noise = pbo_cscv(noise)["pbo"]
    assert pbo_skill < 0.2, f"consistent skill should yield low PBO, got {pbo_skill}"
    assert 0.25 < pbo_noise < 0.75, f"pure noise should yield ~0.5 PBO, got {pbo_noise}"


def test_deflated_sharpe_penalizes_search():
    """Same observed (daily) Sharpe, more trials -> strictly lower DSR."""
    sr = 1.0 / np.sqrt(252)  # 1.0 annualized, expressed per-day as the package expects
    var = (1 + 0.5 * sr * sr) / 1500
    d1 = deflated_sharpe_ratio(sr, 1500, 1, var)
    d20 = deflated_sharpe_ratio(sr, 1500, 20, var)
    d200 = deflated_sharpe_ratio(sr, 1500, 200, var)
    assert d1 > d20 > d200, "more trials must deflate confidence"
    assert d1 == pytest.approx(0.9926, abs=0.01)
    assert d20 == pytest.approx(0.7043, abs=0.02)
    assert 0.0 <= d200 <= 1.0
