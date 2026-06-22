"""
Crypto Variance Risk Premium (VRP) harvest — DVOL-minus-realized vol-timed BTC/ETH beta.

Mechanism: the 30d implied-vol index (Deribit DVOL) persistently sits ABOVE trailing
realized vol -> a positive VRP is the price paid for variance/crash insurance
(Bollerslev-Tauchen-Zhou; Carr-Wu). Perps can't sell options, so we harvest the
risk-aversion channel DIRECTIONALLY: hold long beta while VRP is rich (pro-cyclical),
go flat when it inverts (a low/negative VRP flags un-priced stress). VRP is a UNIVERSAL
premium (equities + crypto) -> scope='broad', generalised to index VRP-timing on untouched
universes (S&P/Nasdaq/Russell/Gold/Crude IV indices vs their ETFs' realized vol).

Only novel code here is the VRP signal + hysteresis state machine; sizing/lag/weekly-hold
come from the validated kit (inv_vol_position), costs/ledger/regime-stamping from the kit.
No external side effects.
"""

from sdk.harness import StrategySpec
from sdk.adapters import (yf_panel, fred_series, inv_vol_position,
                          deribit_dvol, binance_klines, funding_rates)
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
_START = "2019-01-01"
_HOLDOUT = "2024-01-01"   # Deribit DVOL starts ~2021-03 -> a 2022 holdout leaves ~0
                          # search data; 2024 split = ~26mo search / ~30mo holdout (thin
                          # but the most balanced the crypto sample allows; flagged below).

_DEFAULTS = dict(z_enter=0.10, z_exit=-0.10, vol_target=0.20, rv_lb=21,
                 z_win=180, min_hold=7, max_lev=2.0, cost_bps=10.0)  # 10bps/turn ~= 20bps RT

# disjoint broad-generalisation universes: (FRED implied-vol id, ETF for realized vol+beta)
_GEN = {
    "spx_vrp":  ("VIXCLS", "SPY"),   # S&P 500 VIX
    "ndx_vrp":  ("VXNCLS", "QQQ"),   # Nasdaq-100 VXN
    "rut_vrp":  ("RVXCLS", "IWM"),   # Russell 2000 RVX
    "gold_vrp": ("GVZCLS", "GLD"),   # Gold ETF VIX
    "oil_vrp":  ("OVXCLS", "USO"),   # Crude ETF OVX
}
_SECTOR = {"BTCUSDT": "Crypto", "ETHUSDT": "Crypto", "SPY": "US_LargeCap",
           "QQQ": "US_Tech", "IWM": "US_SmallCap", "GLD": "Gold", "USO": "Crude"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_close(df, tickers):
    """Robustly reduce an OHLCV/panel adapter return to a close-price DataFrame."""
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0] if len(tickers) == 1 else df.name)
    if isinstance(df.columns, pd.MultiIndex):
        for fld in ("closeadj", "close", "Close", "adj_close", "Adj Close"):
            for lvl in (0, 1):
                if fld in set(df.columns.get_level_values(lvl)):
                    return df.xs(fld, axis=1, level=lvl)
    return df


def _ann(idx):
    """Realized-vol annualisation factor from the trading calendar (no look-ahead:
    it's a structural property of the asset class). Crypto trades weekends -> 365."""
    return 365.0 if (pd.DatetimeIndex(idx).dayofweek >= 5).mean() > 0.05 else 252.0


def _assemble(px, iv, fund):
    """Build the generic panel signal() consumes: columns = (field, ticker),
    field in {price, iv, funding}. iv ffilled (past-only), funding fillna 0."""
    px = _to_close(px, tuple(px.columns) if hasattr(px, "columns") else ())
    px.index = pd.to_datetime(px.index); px = px.sort_index()
    px = px[~px.index.duplicated(keep="last")].astype(float)
    cols = list(px.columns)

    iv = iv.copy(); iv.index = pd.to_datetime(iv.index); iv = iv.sort_index()
    iv = iv[~iv.index.duplicated(keep="last")].reindex(px.index).ffill().reindex(columns=cols)

    fund = fund.copy(); fund.index = pd.to_datetime(fund.index); fund = fund.sort_index()
    fund = fund[~fund.index.duplicated(keep="last")].reindex(px.index).reindex(columns=cols).fillna(0.0)

    panel = pd.concat({"price": px, "iv": iv, "funding": fund}, axis=1)
    keep = panel["price"].notna().all(axis=1) & panel["iv"].notna().all(axis=1)
    return panel.loc[keep]


def _hysteresis(z, z_enter, z_exit, min_hold):
    """Long(1)/flat(0) state machine with a hysteresis band + min-hold for turnover."""
    out = np.zeros(len(z)); pos = 0; held = 0
    zv = z.values
    for i in range(len(zv)):
        zi = zv[i]
        if pos == 1:
            if held >= min_hold and (np.isnan(zi) or zi <= z_exit):
                pos, held = 0, 0
            else:
                held += 1
        else:
            if (not np.isnan(zi)) and zi >= z_enter:
                pos, held = 1, 0
        out[i] = pos
    return pd.Series(out, index=z.index)


def _vrp_book(panel, z_enter, z_exit, vol_target, rv_lb, z_win, min_hold, max_lev, cost_bps):
    """Core (shared by signal + expectation checks): VRP-z hysteresis -> kit-sized,
    weekly-held, 1-day-LAGGED positions W. Returns (W_lagged, asset_rets, funding)."""
    px = panel["price"].astype(float)
    iv = panel["iv"].astype(float) / 100.0          # DVOL/VIX are annualised vol *points*
    fund = panel["funding"].astype(float).fillna(0.0)

    rets = px.pct_change().fillna(0.0)
    rv = np.log(px).diff().rolling(rv_lb).std() * np.sqrt(_ann(px.index))  # same units as iv
    vrp = iv - rv
    z = (vrp - vrp.rolling(z_win).mean()) / vrp.rolling(z_win).std()

    sig = pd.DataFrame({c: _hysteresis(z[c], z_enter, z_exit, min_hold) for c in px.columns})
    # kit owns inverse-vol sizing (cap = max_lev), weekly rebalance, AND the 1-day lag:
    W = inv_vol_position(sig, rets, target_vol=vol_target, vol_lb=rv_lb, max_pos=max_lev)
    W = W.reindex_like(rets).fillna(0.0)
    return W, rets, fund


def _net(W, rets, fund, cost_bps, name):
    """Daily net return: kit turnover-cost + funding drag on the (lagged) long."""
    base = net_of_cost(W, rets, cost_bps=cost_bps, name=name)
    drag = -(W * fund.reindex_like(W).fillna(0.0)).sum(axis=1)
    return base.add(drag, fill_value=0.0).rename(name)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data():
    """Primary search universe: BTC + ETH perps (DVOL native to these two only)."""
    tickers = ("BTCUSDT", "ETHUSDT")
    px = _to_close(binance_klines(tickers, market="perp"), tickers).reindex(columns=list(tickers))
    iv = pd.concat({"BTCUSDT": deribit_dvol("BTC"), "ETHUSDT": deribit_dvol("ETH")}, axis=1)
    fund = _to_close(funding_rates(tickers), tickers).reindex(columns=list(tickers))
    return _assemble(px, iv, fund)


def load_gen_data(label):
    """One broad-generalisation universe (untouched market): index IV vs its ETF.
    Same panel shape as load_data(); funding=0 (cash ETF, not a perp)."""
    iv_id, etf = _GEN[label]
    iv = fred_series({iv_id: etf}, start=_START)            # column named == ETF ticker
    px = _to_close(yf_panel([etf], start=_START), (etf,)).reindex(columns=[etf])
    fund = pd.DataFrame(0.0, index=px.index, columns=[etf])
    return _assemble(px, iv, fund)


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def signal(panel, **params):
    p = {**_DEFAULTS, **{k: v for k, v in params.items() if k in _DEFAULTS}}
    W, rets, fund = _vrp_book(panel, **p)
    daily = _net(W, rets, fund, p["cost_bps"], "vrp_timed_beta")
    daily = daily.iloc[p["z_win"] + p["rv_lb"]:]           # drop pure warm-up (no position possible)
    trades = trades_from_weights(W, rets, {c: _SECTOR.get(c, "Other") for c in W.columns})
    return daily, trades


# --------------------------------------------------------------------------- #
# Soft expectations (machine-checkable mechanism claims)
# --------------------------------------------------------------------------- #
def _search_panel(ctx):
    panel = ctx["panel"]
    return panel.loc[panel.index < pd.Timestamp(ctx["holdout_start"])]


def _check_vrp_positive(ctx):
    """Mechanism: DVOL persistently > realized vol (VRP > 0 most days)."""
    panel = _search_panel(ctx)
    iv = panel["iv"] / 100.0
    rv = np.log(panel["price"]).diff().rolling(_DEFAULTS["rv_lb"]).std() * np.sqrt(_ann(panel.index))
    frac = float(((iv - rv).stack() > 0).mean())
    return {"pass": frac > 0.5, "observed": round(frac, 4)}


def _check_both_legs_positive(ctx):
    """Generalisation requirement: BOTH BTC and ETH must contribute (not one lucky asset)."""
    W, rets, fund = _vrp_book(_search_panel(ctx), **_DEFAULTS)
    means = {t: float(_net(W[[t]], rets[[t]], fund[[t]], _DEFAULTS["cost_bps"], t).mean())
             for t in W.columns}
    ok = len(means) >= 2 and all(v > 0 for v in means.values())
    return {"pass": bool(ok), "observed": ", ".join(f"{k}={v:.5f}" for k, v in means.items())}


def _check_min_hold(ctx):
    """Turnover claim: the 7-day min-hold + weekly rebalance keeps mean hold >= 7 days."""
    tr = ctx.get("trades") or []
    if not tr:
        return {"pass": False, "observed": 0.0}
    m = float(np.mean([t.get("hold_days", 0) for t in tr]))
    return {"pass": m >= _DEFAULTS["min_hold"], "observed": round(m, 2)}


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
SPEC = StrategySpec(
    id="crypto_vrp_dvol_timed_beta",
    family="variance_risk_premium",   # shares the FDR family ratchet with prior VRP attempts (honest)
    title="Crypto VRP — DVOL-minus-realized vol-timed BTC/ETH beta",
    markets=["crypto"],
    data_desc=("Deribit DVOL 30d IV (BTC,ETH) + Binance perp daily close + funding (OWNED/free). "
               "Generalisation: FRED IV indices (VIX/VXN/RVX/GVZ/OVX) vs ETF (SPY/QQQ/IWM/GLD/USO) "
               "realized vol (free)."),
    pre_registration=(
        "PREMIUM: variance risk premium — compensation for bearing variance/crash risk; DVOL_t sits "
        "persistently above realized vol, a pro-cyclical premium harvested DIRECTIONALLY (perps cannot "
        "sell options).\n"
        "RULE (frozen, no in-sample tuning beyond the declared grid): for each asset VRP_t = DVOL_t/100 "
        "- RV_t, RV_t = trailing-21d log-return vol annualised on the asset's own calendar (crypto 365, "
        "equities 252 — detected, not look-ahead). z-score VRP_t vs its trailing 180d mean/std; hold a "
        "LONG perp leg when z>=+0.10, go FLAT when z<=-0.10 (hysteresis band; 7-day min-hold). Each held "
        "leg inverse-vol sized to ~20% per-leg target vol, capped at 2x notional (kit inv_vol_position, "
        "which also applies the WEEKLY rebalance and the mandatory 1-day lag — see code). BTC & ETH are "
        "equal-risk by construction (same vol target). Net of ~20bps round-trip taker (cost_bps=10 on "
        "one-way turnover) PLUS funding paid/received on the long.\n"
        "PRIMARY = this standalone 2-asset vol-managed VRP-timed book.\n"
        "SECONDARY (pre-registered, NOT folded into the primary — per the no-reflexive-50/50 lesson): a "
        "25%-risk canonical TS-trend tail-overlay (validated Boreas / 21-mkt CTA) as crisis-alpha for the "
        "pro-cyclical VRP leg; evaluated by portfolio-level pairing AFTER the standalone passes, not here.\n"
        "SCOPE broad: VRP is universal (equities+crypto). Stage-2 confirms on DISJOINT untouched markets "
        "(index IV-vs-ETF VRP-timing, same frozen rule). A crypto-only pass that fails the equity/"
        "commodity replication is a 2-asset fluke. Soft expectation also requires BOTH BTC & ETH legs +ve.\n"
        "CAVEATS (honest, gate0): (1) SHORT SAMPLE — DVOL starts ~2021-03 so the crypto book spans only "
        "~5yr / ~3-4 macro regimes; holdout_start moved to 2024-01-01 to leave a usable search window; "
        "DSR effective-N is thin — flagged. (2) FUNDING is assumed daily-aggregated by the adapter; if it "
        "returns per-8h-interval rates the long-leg drag is ~3x larger and the net edge thinner. (3) "
        "CONCENTRATION — DVOL exists only for BTC/ETH, so this is an irreducibly 2-asset macro book: "
        "single_name_share ~50% and sector-spread is limited by construction (deploy_max_positions=2). "
        "Not gamed — recorded as an inherent property, like a small-market trend book. (4) 30d IV vs 21d "
        "RV is a horizon approximation (grid tests rv_lb=30). Forward-validate live in perp paper before "
        "any conviction. No new data purchase."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default":    {},
        "wider_band": {"z_enter": 0.25, "z_exit": -0.25},
        "short_z":    {"z_win": 90},
        "rv_match30": {"rv_lb": 30},
        "hi_target":  {"vol_target": 0.30},
    },
    scope="broad",
    generalization_universes=list(_GEN.keys()),
    load_gen_data=load_gen_data,
    holdout_start=_HOLDOUT,
    deploy_max_positions=2,
    expectations=[
        {"name": "vrp_positive_on_average",
         "claim": "DVOL exceeds realized vol on >50% of search-window days (positive VRP exists)",
         "check": _check_vrp_positive},
        {"name": "both_crypto_legs_positive",
         "claim": "both BTC and ETH legs have positive mean net return in-sample (not one lucky asset)",
         "check": _check_both_legs_positive},
        {"name": "min_hold_respected",
         "claim": "mean trade hold >= 7 days (hysteresis + weekly rebalance control turnover)",
         "check": _check_min_hold},
    ],
)