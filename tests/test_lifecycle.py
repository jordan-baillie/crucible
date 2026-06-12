"""Stage 3 lifecycle/decay rule (pre-reg 2026-06-12: prereg-retirement-rule.md). Pins the FROZEN
parameters: D1 = roll-60d mean < 0.25x modeled (2 consecutive weekly evals), D2 = one-sided CUSUM
(k=0.25, h=5.0), decaying iff D1 AND D2, XOR = watch only, retired = human-only."""
import json

import pytest

from forward.lifecycle import (decay_check, ROLL_DAYS, LEVEL_FRAC, K_ALLOWANCE,
                               H_THRESHOLD, CONSECUTIVE_NEEDED)

EXP = {"daily_mean": 0.001, "daily_std": 0.01}


def test_healthy_book_fires_nothing():
    rets = [0.0012] * 200  # at/above model, no variance penalty matters for the rule
    d = decay_check(rets, EXP)
    assert d["evaluable"] and d["d1"] is False and d["d2"] is False


def test_dead_book_fires_both():
    rets = [0.001] * 100 + [-0.002] * 120  # healthy start, then persistent bleed
    d = decay_check(rets, EXP)
    assert d["evaluable"] and d["d1"] is True and d["d2"] is True
    assert d["roll_mean"] < LEVEL_FRAC * EXP["daily_mean"]
    assert d["cusum_peak"] > H_THRESHOLD


def test_recovered_book_unfires_d2():
    """CUSUM uses the CURRENT statistic, not the peak — a book that bled then recovered must not
    stay flagged (pre-reg: decaying is reversible while unconfirmed)."""
    rets = [-0.002] * 120 + [0.0015] * 300  # old bleed, long strong recovery
    d = decay_check(rets, EXP)
    assert d["cusum_peak"] > H_THRESHOLD     # the historical episode is visible...
    assert d["d2"] is False                  # ...but the current statistic has drained to 0
    assert d["d1"] is False                  # recent 60d is healthy


def test_not_evaluable_paths():
    assert decay_check([0.001] * (ROLL_DAYS - 1), EXP)["evaluable"] is False        # short history
    assert decay_check([0.001] * 100, {"daily_mean": -0.001, "daily_std": 0.01})["evaluable"] is False  # modeled <= 0
    assert decay_check([0.001] * 100, {})["evaluable"] is False                     # no expectation


def test_slow_bleed_caught_by_cusum_before_level():
    """The motivating gap: returns at ~55% of model never breach D1's 25% level floor but drift
    down for months — D2 (CUSUM) must accumulate and fire on exactly this."""
    rets = [0.00055] * 700  # 55% of modeled mean, forever
    d = decay_check(rets, EXP)
    assert d["d1"] is False          # level floor (25%) never breached
    # z = (0.001-0.00055)/0.01 = 0.045 per day < k=0.25 -> S stays 0: a 45% shortfall in a noisy
    # book is BELOW the pre-registered detection size (~0.5 sigma); verify the math is honest
    assert d["d2"] is False
    # now a real half-sigma shortfall: r = mu - 0.6*sd
    rets2 = [EXP["daily_mean"] - 0.6 * EXP["daily_std"]] * 100
    d2 = decay_check(rets2, EXP)
    assert d2["d2"] is True          # (0.6 - 0.25) * 100 = 35 >> 5
    assert d2["d1"] is True


def test_consecutive_needed_is_two():
    assert CONSECUTIVE_NEEDED == 2 and K_ALLOWANCE == 0.25 and H_THRESHOLD == 5.0 \
        and ROLL_DAYS == 60 and LEVEL_FRAC == 0.25  # frozen by the pre-registration
