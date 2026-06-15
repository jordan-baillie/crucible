"""Tests for the breadth / Fundamental-Law overfit gate (prereg-breadth-overfit-gate.md).

Worked-example assertions = primary-source verification (Grinold; Clarke-de Silva-Thorley; Buckle).
The Check is annotate-only (date-gated 2026-06-29) so it must be inert today.
"""
import numpy as np
import pandas as pd
import pytest

from sdk.stats import effective_breadth, implied_ic, break_even_cost_bps


# ---------------------------------------------------------------- primary-source worked examples
def test_effective_breadth_worked_examples():
    assert effective_breadth(100, 0.2) == pytest.approx(4.808, abs=1e-2)   # saturates toward 1/rho=5
    assert effective_breadth(100, 0.0) == 100.0                            # rho=0 -> nominal
    assert effective_breadth(100, 1.0) == pytest.approx(1.0, abs=1e-2)     # rho clamped to 0.999 -> ~1 bet
    assert effective_breadth(100, -0.5) == 100.0                           # negative rho clamped to 0 (conservative)


def test_implied_ic_and_break_even_worked_examples():
    assert implied_ic(1.0, 100) == pytest.approx(0.10, abs=1e-9)           # IR/sqrt(BR)
    assert break_even_cost_bps(0.12, 6) == pytest.approx(200.0, abs=1e-6)  # 12% / 6x -> 200 bps
    assert break_even_cost_bps(0.10, 0) == float("inf")                    # no turnover -> infinite headroom


def test_implied_ic_flags_narrow_high_ir_book():
    """Discrimination: a broad daily book is plausible; a narrow correlated monthly book is not."""
    broad = implied_ic(1.5, effective_breadth(100, 0.1) * 252)
    narrow = implied_ic(1.5, effective_breadth(5, 0.3) * 12)
    assert broad < 0.10 and narrow > 0.20


# ---------------------------------------------------------------- the Check (annotate-only today)
def _ctx(n_names, rho, n_days=1200, rebal="ME", drift=0.0006, seed=0):
    from sdk.gates import GateContext
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-06-01", periods=n_days)
    common = rng.normal(0, 0.01, n_days)
    cols = {}
    for i in range(n_names):
        idio = rng.normal(0, 0.01, n_days)
        x = np.sqrt(rho) * common + np.sqrt(max(1e-9, 1 - rho)) * idio    # target pairwise corr ~rho
        cols[f"N{i}"] = 100 * np.exp(np.cumsum(x))
    px = pd.DataFrame(cols, index=idx)
    rib = px.pct_change().mean(axis=1) + drift                            # equal-weight book + drift
    search = rib[rib.index < "2024-01-01"]
    rebal_dates = px.resample(rebal).last().index
    trades = [{"ticker": c, "entry_date": str(d.date())}
              for d in rebal_dates if d < pd.Timestamp("2024-01-01") for c in cols]
    return GateContext(spec=None, panel=None, price_matrix=px, search=search,
                       search_trades=trades, holdout_pass=True, deploy_candidate=True)


def test_breadth_check_active_and_spares_broad_book():
    """Gate is ACTIVE (active_from amended to 2026-06-15). A broad, diversified book (many names,
    low correlation) has a plausible implied IC -> NOT demoted."""
    from sdk.harness import _gc_breadth_overfit
    res = _gc_breadth_overfit(_ctx(n_names=100, rho=0.03, rebal="W", drift=0.0003, seed=1))
    assert res.name == "breadth_overfit" and res.failure_mode == 1
    assert res.active is True                              # LIVE now
    assert res.evaluated is True
    assert res.metrics["implied_ic"] < 0.20 and res.passed is True and res.demotes is False


def test_breadth_check_demotes_narrow_high_ir_book():
    """Concentrated, highly-correlated, high-IR book (~few effective bets) -> implausible implied IC
    -> DEMOTE (the live overfit flag). This is the conservative-by-design false-positive mode too."""
    from sdk.harness import _gc_breadth_overfit
    res = _gc_breadth_overfit(_ctx(n_names=3, rho=0.95, rebal="ME", drift=0.0009, seed=5))
    assert res.evaluated is True and res.metrics["implied_ic"] > 0.20
    assert res.passed is False and res.active is True and res.demotes is True


def test_price_matrix_handles_both_multiindex_orientations():
    """#61 fix: _price_matrix must read BOTH (field,asset) [px at level 0] and (asset,field) [binance
    klines: symbol at level 0, close at level 1], and leave flat equity panels untouched."""
    from sdk.harness import _price_matrix
    idx = pd.bdate_range(end="2026-06-01", periods=300)
    flat = pd.DataFrame(100 + np.random.default_rng(0).normal(0, 1, (300, 6)).cumsum(0),
                        index=idx, columns=[f"A{i}" for i in range(6)])
    assert _price_matrix(flat) is flat                                  # equity flat panel -> unchanged
    fa = pd.concat({"close": flat, "volume": flat}, axis=1)             # (field, asset)
    assert list(_price_matrix(fa).columns) == list(flat.columns)        # close cross-section at level 0
    af = pd.concat({s: flat[[s]].rename(columns={s: "close"}).assign(volume=1.0) for s in flat.columns}, axis=1)
    af.columns.names = ["symbol", "field"]                              # (asset, field) = klines layout
    pm = _price_matrix(af)
    assert pm is not None and set(pm.columns) == set(flat.columns)      # close cross-section at level 1 -> symbols


def test_breadth_evaluates_a_crypto_klines_strategy():
    """#61: the breadth gate previously returned not_evaluated for ALL crypto (klines MultiIndex panel
    -> price_matrix None -> <2 mappable). After the _price_matrix fix it must EVALUATE."""
    from sdk.harness import _price_matrix, _gc_breadth_overfit
    from sdk.gates import GateContext
    rng = np.random.default_rng(5)
    idx = pd.bdate_range(end="2026-06-01", periods=800)
    syms = [f"C{i}USDT" for i in range(15)]
    common = rng.normal(0, 0.01, 800)
    cols = {}
    for s in syms:
        px = 100 * np.exp(np.cumsum(0.4 * common + np.sqrt(0.84) * rng.normal(0, 0.01, 800)))
        for f in ("open", "high", "low", "close", "volume"):
            cols[(s, f)] = px if f == "close" else px * 1.001
    panel = pd.DataFrame(cols, index=idx); panel.columns.names = ["symbol", "field"]
    pm = _price_matrix(panel)
    search = (pm.pct_change().mean(axis=1).dropna() + 0.0005)
    search = search[search.index < "2024-01-01"]
    rebal = panel.resample("ME").last().index
    trades = [{"ticker": s, "entry_date": str(d.date())}
              for d in rebal if d < pd.Timestamp("2024-01-01") for s in syms]
    res = _gc_breadth_overfit(GateContext(spec=None, panel=panel, price_matrix=pm, search=search,
                                          search_trades=trades, holdout_pass=True, deploy_candidate=True))
    assert res.evaluated is True and res.metrics["n_names"] >= 10
    assert res.metrics["effective_breadth"] is not None and res.metrics["implied_ic"] is not None


def test_breadth_check_not_evaluated_on_losing_or_thin():
    from sdk.harness import _gc_breadth_overfit
    res = _gc_breadth_overfit(_ctx(n_names=40, rho=0.15, drift=-0.001, seed=2))  # negative drift -> IR<=0
    assert res.evaluated is False and res.passed is None    # no positive edge -> nothing to flag
