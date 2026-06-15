"""Regression tests for the free-source adapters (COT / CBOE / funding / auctions).
Network-dependent — marked so CI can deselect with `-m "not network"`."""
import pandas as pd
import pytest

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def cot():
    from sdk.adapters import cot_positioning
    return cot_positioning(roots=["CL", "GC", "ZC"], start_year=2020)


def test_cot_shape_and_pit(cot):
    assert {"CL_comm_net", "CL_noncomm_net", "CL_oi", "GC_comm_net", "ZC_oi"} <= set(cot.columns)
    assert len(cot) > 250  # ~52 weeks * 6 years
    # release-date index: COT publishes Fridays (as-of Tue + 3d). Holiday weeks shift —
    # but the bulk must be Fridays, and NOTHING may sit on the as-of Tuesday itself.
    assert (cot.index.dayofweek == 4).mean() > 0.95
    assert (cot.index.dayofweek == 1).sum() == 0
    # commercials are structurally net SHORT crude (hedging producers)
    assert (cot["CL_comm_net"] < 0).mean() > 0.7


def test_cboe_depth_and_grid():
    from sdk.adapters import cboe_index
    p = cboe_index()
    assert {"VIX3M", "VVIX", "SKEW", "PUT"} <= set(p.columns)
    assert p["SKEW"].dropna().index.min().year <= 1991
    assert p["VIX3M"].dropna().index.min().year <= 2010
    assert p.index.max() > pd.Timestamp.today() - pd.Timedelta(days=7)
    v = p["VIX3M"].dropna()
    assert (v > 5).all() and (v < 150).all()  # sane vol-index range


def test_bare_string_arg_not_iterated():
    """Footgun class that crashed the 2026-06-13 hedging-pressure run: a smith passing a
    bare string ('VVIX') instead of a list got it iterated to ['V','V','I','X']. Every
    name/symbol/type adapter must accept a single string. Network-light (cboe only)."""
    from sdk.adapters import cboe_index
    p = cboe_index("VVIX")
    assert list(p.columns) == ["VVIX"], f"bare string iterated: got {list(p.columns)}"


def test_funding_rates_daily():
    from sdk.adapters import funding_rates
    f = funding_rates(symbols=("BTCUSDT",))
    s = f["BTCUSDT"].dropna()
    assert s.index.min().year <= 2020 and len(s) > 1500
    # daily sum of 8h prints: |daily| rarely exceeds 1% even in mania regimes
    assert s.abs().quantile(0.99) < 0.01
    # default-rate eras exist (0.01%/8h = 0.0003/day) — sanity that scale is right
    assert 0.00001 < s.median() < 0.001


def test_treasury_auctions():
    from sdk.adapters import treasury_auctions
    a = treasury_auctions()
    assert len(a) > 1200  # 1261 Note+Bond auctions 2010→mid-2026 (verified); Bills excluded by default
    assert a["auction_date"].min().year <= 2011
    assert a["auction_date"].is_monotonic_increasing
    # announcement precedes auction (the point-in-time conditioning variable)
    both = a.dropna(subset=["announcement_date"])
    assert (both["announcement_date"] <= both["auction_date"]).mean() > 0.99


@pytest.mark.network
def test_binance_klines_perp_and_basis():
    from sdk.adapters import binance_klines, CRYPTO_MAJORS
    assert len(CRYPTO_MAJORS) >= 10
    perp = binance_klines(("BTCUSDT",), market="perp")
    assert not perp.empty
    fields = set(perp["BTCUSDT"].columns)
    assert {"open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"} <= fields
    assert perp.index.min().year <= 2019  # deep history
    spot = binance_klines(("BTCUSDT",), market="spot")
    basis = (perp[("BTCUSDT", "close")] / spot[("BTCUSDT", "close")] - 1).dropna()
    assert len(basis) > 1000  # multi-year basis series computable


def test_binance_klines_bare_string_guard():
    # footgun class: a bare string must NOT iterate per-character (would request 'B','T','C'...)
    import inspect
    from sdk import adapters
    src = inspect.getsource(adapters.binance_klines)
    assert "isinstance(symbols, str)" in src  # the guard is present


@pytest.mark.network
def test_coinmetrics_community_metrics():
    from sdk.adapters import coinmetrics_metrics, CM_COMMUNITY_MAJORS
    assert "btc" in CM_COMMUNITY_MAJORS
    p = coinmetrics_metrics(("btc",), ("PriceUSD", "AdrActCnt", "CapMVRVCur"))
    assert not p.empty
    free = {m for (_, m) in p.columns}
    assert {"PriceUSD", "AdrActCnt"} <= free   # known free community metrics
    assert p.index.min().year <= 2011          # deep daily history
    assert p[("btc", "AdrActCnt")].dropna().iloc[-1] > 0


def test_coinmetrics_bare_string_guard():
    import inspect
    from sdk import adapters
    src = inspect.getsource(adapters.coinmetrics_metrics)
    assert "isinstance(assets, str)" in src and "isinstance(metrics, str)" in src
