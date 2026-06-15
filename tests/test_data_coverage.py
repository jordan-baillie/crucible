"""Tests for the data-coverage-expansion adapters (#55-60). Live-API smokes are @network (CI
deselects with -m "not network"); list_adapters() + the key-guard contracts are pure (always run)."""
import pandas as pd
import pytest

from sdk import adapters as A


# ---------------------------------------------------------------- pure (no network) — always run
def test_list_adapters_indexes_the_new_sources():
    idx = A.list_adapters()
    for name in ("binance_universe", "fred_vintage", "deribit_dvol", "french_factors",
                 "sec_insider", "sep_panel", "fred_series", "eia_series", "usda_nass"):
        assert name in idx, f"list_adapters() missing {name}"
    assert "from sdk.adapters import" in idx


def _no_key_anywhere(monkeypatch, tmp_path, env_var):
    """Strip the env var AND point SECRETS at an empty JSON, so the key-guard is tested in
    isolation — robust to a real key now living in the operator's secrets file. Also makes any
    network call an immediate failure so the guard must raise BEFORE touching the wire."""
    import crucible_paths
    empty = tmp_path / "empty_secrets.json"
    empty.write_text("{}")
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(crucible_paths, "SECRETS", empty)
    monkeypatch.setattr(A, "_http_get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network must not be called")))


def test_eia_fails_loudly_when_no_key_anywhere(monkeypatch, tmp_path):
    _no_key_anywhere(monkeypatch, tmp_path, "EIA_API_KEY")
    with pytest.raises(RuntimeError, match="EIA key"):
        A.eia_series("PET.WCESTUS1.W")


def test_usda_nass_fails_loudly_when_no_key_anywhere(monkeypatch, tmp_path):
    _no_key_anywhere(monkeypatch, tmp_path, "USDA_NASS_KEY")
    with pytest.raises(RuntimeError, match="USDA NASS key"):
        A.usda_nass("CORN")


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


@pytest.mark.network
def test_eia_series_live_crude_stocks():
    # PET.WCESTUS1.W = US ending crude stocks, weekly (thousand barrels) — strictly positive
    s = A.eia_series("PET.WCESTUS1.W", start="2022-01-01")
    assert isinstance(s, pd.Series) and len(s) > 100
    assert s.index.is_monotonic_increasing
    assert (s > 0).all()
    assert s.index.min() >= pd.Timestamp("2022-01-01")  # start filter honoured


@pytest.mark.network
def test_usda_nass_live_corn_stocks():
    df = A.usda_nass("CORN", statisticcat_desc="STOCKS")
    assert not df.empty and "Value" in df.columns
    assert (df["commodity_desc"] == "CORN").all()        # query filter respected
    assert (df["statisticcat_desc"] == "STOCKS").all()
    assert {"year", "reference_period_desc", "unit_desc"} <= set(df.columns)
