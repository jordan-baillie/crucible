"""Tests for the data-coverage-expansion adapters (#55-60). Live-API smokes are @network (CI
deselects with -m "not network"); list_adapters() + the EIA key-pending guard are pure (always run)."""
import pandas as pd
import pytest

from sdk import adapters as A


# ---------------------------------------------------------------- pure (no network) — always run
def test_list_adapters_indexes_the_new_sources():
    idx = A.list_adapters()
    for name in ("binance_universe", "fred_vintage", "deribit_dvol", "french_factors",
                 "sec_insider", "sep_panel", "fred_series"):
        assert name in idx, f"list_adapters() missing {name}"
    assert "from sdk.adapters import" in idx


def test_eia_is_key_pending_and_fails_loudly_without_a_key(monkeypatch):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    # no network call should happen — it must raise on the missing key first
    monkeypatch.setattr(A, "_http_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call network")))
    with pytest.raises(RuntimeError, match="EIA key"):
        A.eia_series("PET.WCESTUS1.W")


# ---------------------------------------------------------------- live API smokes (@network)
@pytest.mark.network
def test_binance_universe_broad_and_clean():
    u = A.binance_universe(top_n=40, market="perp")
    assert isinstance(u, list) and len(u) >= 20
    assert all(s.endswith("USDT") for s in u)
    assert not any(x in s for s in u for x in ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"))
    assert "BTCUSDT" in u  # the top pair is always present


@pytest.mark.network
def test_fred_vintage_is_point_in_time():
    df = A.fred_vintage({"GDPC1": "rgdp"}, vintage_date="2020-06-30")
    assert not df.empty and "rgdp" in df.columns
    # a vintage as-of 2020-06-30 cannot contain observations dated AFTER that vintage
    assert df.index.max() <= pd.Timestamp("2020-06-30")


@pytest.mark.network
def test_deribit_dvol_shape_and_range():
    d = A.deribit_dvol("BTC")
    assert {"open", "high", "low", "close"} <= set(d.columns) and len(d) > 500
    assert d["close"].between(5, 250).mean() > 0.95  # annualised vol points, plausible band


@pytest.mark.network
def test_french_factors_ff5_decimal_and_complete():
    ff = A.french_factors("ff5_daily")
    assert {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"} <= set(ff.columns) and len(ff) > 10_000
    assert ff["Mkt-RF"].abs().max() < 0.5  # DECIMAL returns, not percent


@pytest.mark.network
def test_sec_cik_and_insider():
    cik = A.sec_cik("AAPL")
    assert cik and len(cik) == 10 and cik.isdigit()
    ins = A.sec_insider("AAPL", limit=8)
    assert not ins.empty and set(ins["code"].unique()) <= {"A", "D"}
    assert {"date", "shares", "price", "net_usd"} <= set(ins.columns)
