"""
G10 FX VALUE — PPP/REER mean-reversion, carry-orthogonalized.

Premium: the currency VALUE premium (compensation for bearing real-exchange-rate
mean-reversion / currency-crash risk). It is explicitly residualized against the
1-month rate differential (carry) each month, so the book is carry-orthogonal BY
CONSTRUCTION — this is NOT the (dead) DM-FX carry harvest and NOT a price forecast.

Universe (search): 9 G10 currencies vs USD. Monthly. Long top-3 undervalued /
short bottom-3 overvalued residual-value currencies, dollar-neutral, inverse-vol
within legs, ~10% vol target (<=2x notional), hysteresis to cap turnover.

NO external side effects: only OWNED/FREE data (yfinance FX — FX is not
survivorship-biased; FRED OECD CPI + 3M interbank rates). The harness runs all rails.

Lookahead control (stated explicitly):
  * Nominal FX spot is real-time at month-end m (known at m).
  * CPI is monthly, shifted +2 months (publication delay): at month-end m we use the
    reference-(m-2) CPI, which is released ~mid-(m-1) -> definitely known at m.
  * Short rates shifted +1 month (monthly-average publication lag).
  * Signal computed AT month-end m from data <= m; weights applied with a 1-day lag
    (W = monthly target ffilled to daily, passed to net_of_cost / trades_from_weights
    as W.shift(1) — the lag is ours and is explicit here).
  * Vol scaler uses ONLY trailing returns (rets.loc[:m]).
The only hand-rolled code is the signal; returns/costs/trades come from the kit.

Mechanism claims (machine-checked in expectations):
  * carry_orthogonal: the value book's returns correlate <0.40 with the pure carry book.
  * residual_cuts_carry: residualizing value on carry LOWERS |corr| to the carry book
    versus the un-residualized value book.
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ------------------------------------------------------------------ constants
US_CPI  = "CPIAUCSL"            # US CPI, all urban consumers (monthly)
US_RATE = "IR3TIB01USM156N"    # US 3-month interbank rate (OECD, monthly, %)
START   = "2003-01-01"         # yfinance FX history practical start

# G10 vs USD. yf ticker + whether to invert to a uniform "USD per 1 foreign unit".
G10_META = {
    "EUR": dict(yf="EURUSD=X", invert=False, cpi="CP0000EZ19M086NEST", rate="IR3TIB01EZM156N", sec="Europe_Core"),
    "JPY": dict(yf="USDJPY=X", invert=True,  cpi="JPNCPIALLMINMEI",     rate="IR3TIB01JPM156N", sec="Asia"),
    "GBP": dict(yf="GBPUSD=X", invert=False, cpi="GBRCPIALLMINMEI",     rate="IR3TIB01GBM156N", sec="Europe_Other"),
    "AUD": dict(yf="AUDUSD=X", invert=False, cpi="AUSCPIALLQINMEI",     rate="IR3TIB01AUM156N", sec="Commodity"),
    "CAD": dict(yf="USDCAD=X", invert=True,  cpi="CANCPIALLMINMEI",     rate="IR3TIB01CAM156N", sec="Commodity"),
    "CHF": dict(yf="USDCHF=X", invert=True,  cpi="CHECPIALLMINMEI",     rate="IR3TIB01CHM156N", sec="Europe_Core"),
    "NZD": dict(yf="NZDUSD=X", invert=False, cpi="NZLCPIALLQINMEI",     rate="IR3TIB01NZM156N", sec="Commodity"),
    "SEK": dict(yf="USDSEK=X", invert=True,  cpi="SWECPIALLMINMEI",     rate="IR3TIB01SEM156N", sec="Europe_Other"),
    "NOK": dict(yf="USDNOK=X", invert=True,  cpi="NORCPIALLMINMEI",     rate="IR3TIB01NOM156N", sec="Commodity"),
}

# Generalization universes — DISJOINT currency codes from G10 (different markets),
# disjoint from each other; chosen for high FRED-CPI reliability. The mechanism is
# universal, so a stage-1 pass must survive on these untouched EM cross-sections.
EM_LATAM = {
    "MXN": dict(yf="USDMXN=X", invert=True, cpi="MEXCPIALLMINMEI", rate="IR3TIB01MXM156N", sec="LatAm"),
    "BRL": dict(yf="USDBRL=X", invert=True, cpi="BRACPIALLMINMEI", rate="IRSTCI01BRM156N", sec="LatAm"),
    "CLP": dict(yf="USDCLP=X", invert=True, cpi="CHLCPIALLMINMEI", rate="IR3TIB01CLM156N", sec="LatAm"),
    "COP": dict(yf="USDCOP=X", invert=True, cpi="COLCPIALLMINMEI", rate="IR3TIB01COM156N", sec="LatAm"),
}
EM_CEE = {
    "PLN": dict(yf="USDPLN=X", invert=True, cpi="POLCPIALLMINMEI", rate="IR3TIB01PLM156N", sec="CEE"),
    "HUF": dict(yf="USDHUF=X", invert=True, cpi="HUNCPIALLMINMEI", rate="IR3TIB01HUM156N", sec="CEE"),
    "CZK": dict(yf="USDCZK=X", invert=True, cpi="CZECPIALLMINMEI", rate="IR3TIB01CZM156N", sec="CEE"),
    "TRY": dict(yf="USDTRY=X", invert=True, cpi="TURCPIALLMINMEI", rate="IR3TIB01TRM156N", sec="CEE"),
}
EM_ASIA = {
    "KRW": dict(yf="USDKRW=X", invert=True, cpi="KORCPIALLMINMEI", rate="IR3TIB01KRM156N", sec="Asia"),
    "INR": dict(yf="USDINR=X", invert=True, cpi="INDCPIALLMINMEI", rate="IRSTCI01INM156N", sec="Asia"),
    "IDR": dict(yf="USDIDR=X", invert=True, cpi="IDNCPIALLMINMEI", rate="IRSTCI01IDM156N", sec="Asia"),
    "ZAR": dict(yf="USDZAR=X", invert=True, cpi="ZAFCPIALLMINMEI", rate="IR3TIB01ZAM156N", sec="EMEA"),
    "ILS": dict(yf="USDILS=X", invert=True, cpi="ISRCPIALLMINMEI", rate="IR3TIB01ILM156N", sec="EMEA"),
}
GEN_META = {"em_latam": EM_LATAM, "em_cee": EM_CEE, "em_asia": EM_ASIA}

SECTOR_MAP = {}
for _m in (G10_META, EM_LATAM, EM_CEE, EM_ASIA):
    for _c, _d in _m.items():
        SECTOR_MAP[_c] = _d["sec"]

DEFAULTS = dict(
    mode="value",        # 'value' (residual PPP) or 'carry' (pure rate-differential book)
    residualize=True,    # residualize value against carry cross-sectionally each month
    band=2,              # hysteresis: extra slots kept beyond top/bottom-n (capped by slack)
    n_leg=3,             # target names per leg (auto-shrinks on small universes)
    z_window=60, z_min=36,   # trailing window (months) for the own-mean RER z-score
    cpi_lag=2, rate_lag=1,   # publication delays (months) — point-in-time
    vol_lb=63, vol_min=20,   # trailing daily vol for inverse-vol sizing
    vol_floor=0.02,          # annualized vol floor (stops a near-pegged ccy dominating)
    book_vol_lb=126, target_vol=0.10, max_scaler=2.0,  # vol target, <=2x notional cap
    cost_bps=8.0, name="fx_value_ppp",
)


# ------------------------------------------------------------------ data build
def _fetch_fred(id_to_name, start):
    """Fetch each FRED series individually so a missing/renamed id is skipped, not fatal."""
    cols = {}
    for fid, name in id_to_name.items():
        try:
            df = fred_series({fid: name}, start)
            if df is not None and name in df.columns and df[name].notna().any():
                cols[name] = df[name]
        except Exception:
            continue
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _build_panel(meta, start):
    """Panel with MultiIndex columns (field, ccy): field in {spot, cpi, rate}.
    spot = USD per 1 foreign unit (uniform direction). cpi/rate include 'USD'."""
    ccys = list(meta.keys())
    yf_to_ccy = {meta[c]["yf"]: c for c in ccys}
    px = yf_panel(list(yf_to_ccy.keys()), start)
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    spot = pd.DataFrame(index=px.index)
    for tk, c in yf_to_ccy.items():
        if tk not in px.columns:
            continue
        s = pd.to_numeric(px[tk], errors="coerce")
        if meta[c]["invert"]:                 # USDxxx (foreign per USD) -> invert
            s = 1.0 / s.where(s > 0)
        spot[c] = s
    spot = spot.dropna(how="all")
    if spot.empty:
        return pd.concat({"spot": spot, "cpi": pd.DataFrame(), "rate": pd.DataFrame()}, axis=1)
    didx = spot.index

    cpi_ids = {meta[c]["cpi"]: c for c in ccys if meta[c].get("cpi")}
    cpi_ids[US_CPI] = "USD"
    rate_ids = {meta[c]["rate"]: c for c in ccys if meta[c].get("rate")}
    rate_ids[US_RATE] = "USD"

    cpi = _fetch_fred(cpi_ids, start).reindex(didx, method="ffill")
    rate = _fetch_fred(rate_ids, start).reindex(didx, method="ffill")
    return pd.concat({"spot": spot, "cpi": cpi, "rate": rate}, axis=1).sort_index()


def load_data():
    return _build_panel(G10_META, START)


def load_gen_data(label):
    return _build_panel(GEN_META[label], START)


# ------------------------------------------------------------------ signal bits
def _empty(index, p):
    return pd.Series(0.0, index=pd.DatetimeIndex(index), name=p["name"]), []


def _residualize(value, carry):
    """Per-month cross-sectional OLS residual of value on carry (carry-orthogonal score)."""
    cols = list(value.columns)
    out = pd.DataFrame(np.nan, index=value.index, columns=cols)
    cvals = carry.reindex(columns=cols)
    for dt in value.index:
        v = value.loc[dt].values.astype(float)
        c = cvals.loc[dt].values.astype(float)
        vm = np.isfinite(v)
        if vm.sum() < 4:
            continue
        finite_c = c[vm][np.isfinite(c[vm])]
        cm = finite_c.mean() if finite_c.size else np.nan
        if not np.isfinite(cm):                       # no carry data -> just demeaned value
            row = np.full(len(cols), np.nan); row[vm] = v[vm] - v[vm].mean()
            out.loc[dt] = row; continue
        cc = np.where(np.isfinite(c), c, cm)
        x, y = cc[vm], v[vm]
        if np.std(x) < 1e-12:
            res = y - y.mean()
        else:
            sl, ic = np.polyfit(x, y, 1); res = y - (sl * x + ic)
        row = np.full(len(cols), np.nan); row[vm] = res
        out.loc[dt] = row
    return out


def _inv_vol_w(names, vol_series, floor):
    v = vol_series.reindex(names).astype(float).clip(lower=floor)
    inv = 1.0 / v
    if not np.isfinite(inv.values).any():
        return pd.Series(1.0 / len(names), index=names)
    inv = inv.fillna(np.nanmean(inv.values))
    return inv / inv.sum()


def _vol_scaler(w, rets, dt, p):
    hist = rets.loc[:dt].iloc[-p["book_vol_lb"]:]              # trailing only -> no lookahead
    if len(hist) < max(20, p["vol_min"]):
        return 1.0
    book = (hist[w.index] * w.values).sum(axis=1)
    bv = book.std() * np.sqrt(252.0)
    if not np.isfinite(bv) or bv < 1e-9:
        return 1.0
    return float(np.clip(p["target_vol"] / bv, 0.0, p["max_scaler"]))


def _select_and_size(score, rets, m_index, ccys, p):
    dvol = rets.rolling(p["vol_lb"], min_periods=p["vol_min"]).std() * np.sqrt(252.0)
    vol_m = dvol.reindex(m_index, method="ffill")
    Wm = pd.DataFrame(0.0, index=m_index, columns=ccys)
    prev_long, prev_short = [], []
    for dt in m_index:
        s = score.loc[dt].dropna() if dt in score.index else pd.Series(dtype=float)
        s = s.loc[[c for c in s.index if c in ccys]]
        if len(s) < 4:
            prev_long, prev_short = [], []; continue
        n = min(p["n_leg"], max(1, len(s) // 2))
        extra = min(int(p["band"]), max(0, len(s) - 2 * n))   # disable hysteresis if no slack
        band = n + extra
        order = list(s.sort_values(ascending=False).index)
        entry_long, keep_long = set(order[:n]), set(order[:band])
        entry_short, keep_short = set(order[-n:]), set(order[-band:])

        new_long = [c for c in prev_long if c in keep_long]
        new_short = [c for c in prev_short if c in keep_short]
        for c in order:                                       # fill longs from the top
            if len(new_long) >= n: break
            if c in new_long or c in new_short: continue
            if c in entry_long: new_long.append(c)
        for c in reversed(order):                             # fill shorts from the bottom
            if len(new_short) >= n: break
            if c in new_short or c in new_long: continue
            if c in entry_short: new_short.append(c)
        new_short = [c for c in new_short if c not in new_long]
        if not new_long or not new_short:
            prev_long, prev_short = new_long, new_short; continue

        v = vol_m.loc[dt] if dt in vol_m.index else pd.Series(dtype=float)
        wl = _inv_vol_w(new_long, v, p["vol_floor"]) * 0.5    # leg sums to +0.5
        ws = _inv_vol_w(new_short, v, p["vol_floor"]) * -0.5  # leg sums to -0.5 -> dollar-neutral
        w = pd.Series(0.0, index=ccys)
        w.loc[new_long] = wl.reindex(new_long).values
        w.loc[new_short] = ws.reindex(new_short).values
        Wm.loc[dt] = (w * _vol_scaler(w, rets, dt, p)).values
        prev_long, prev_short = new_long, new_short
    return Wm


def signal(panel, **params):
    p = dict(DEFAULTS); p.update(params)
    try:
        spot = panel["spot"].astype(float).sort_index()
        cpi = panel["cpi"].astype(float).sort_index()
        rate = panel["rate"].astype(float).sort_index()
    except Exception:
        return _empty(getattr(panel, "index", pd.DatetimeIndex([])), p)

    ccys = [c for c in spot.columns if c != "USD" and c in cpi.columns]
    if "USD" not in cpi.columns or len(ccys) < 4:
        return _empty(spot.index, p)
    for c in ccys + ["USD"]:
        if c not in rate.columns:
            rate[c] = np.nan
    rate = rate[ccys + ["USD"]]; cpi = cpi[ccys + ["USD"]]; spot = spot[ccys]

    # daily asset returns: FX spot return + local cash accrual (USD funding cancels in a
    # dollar-neutral book, so it is omitted). Used via the lagged weight matrix.
    rets = (spot.pct_change() + (rate[ccys] / 100.0 / 252.0).fillna(0.0)).reindex(columns=ccys)

    # ---- monthly point-in-time inputs ----
    me_spot = spot.resample("ME").last()
    cpi_m = cpi.resample("ME").last().shift(p["cpi_lag"])      # +2m publication lag
    rate_m = rate.resample("ME").last().shift(p["rate_lag"])   # +1m publication lag

    # real exchange rate Q = S(USD/foreign) * CPI_foreign / CPI_US. Level (CPI base year)
    # is irrelevant because it is z-scored vs its own trailing mean.
    rer = me_spot[ccys].mul(cpi_m[ccys]).div(cpi_m["USD"], axis=0)
    mu = rer.rolling(p["z_window"], min_periods=p["z_min"]).mean()
    sd = rer.rolling(p["z_window"], min_periods=p["z_min"]).std()
    z = ((rer - mu) / sd).replace([np.inf, -np.inf], np.nan)   # TIME-SERIES own-mean z
    value = (-z).sub((-z).mean(axis=1), axis=0)                # undervalued>0; xs-demeaned

    carry = rate_m[ccys].sub(rate_m["USD"], axis=0)            # foreign - US short rate (%)
    carry = carry.sub(carry.mean(axis=1), axis=0)

    if p["mode"] == "carry":
        score = carry
    else:
        score = _residualize(value, carry) if p["residualize"] else value

    Wm = _select_and_size(score, rets, me_spot.index, ccys, p)
    W = Wm.reindex(rets.index, method="ffill").fillna(0.0)
    Wlag = W.shift(1).fillna(0.0)                              # 1-day lag (ours, explicit)

    rfill = rets.fillna(0.0)
    dr = net_of_cost(Wlag, rfill, cost_bps=p["cost_bps"], name=p["name"])
    trades = trades_from_weights(Wlag, rfill, SECTOR_MAP)      # kit stamps entry_regime
    return dr, trades


# ------------------------------------------------------------------ expectations
def _corr(a, b):
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 20:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def exp_carry_orthogonal(ctx):
    """Value book returns should correlate <0.40 with the pure carry book (carry-orthogonal)."""
    g = ctx.get("grid", {}); d, c = g.get("default"), g.get("carry_book")
    if d is None or c is None:
        return {"pass": False, "observed": "grid_missing"}
    r = _corr(d, c)
    return {"pass": bool(np.isfinite(r) and abs(r) < 0.40),
            "observed": None if not np.isfinite(r) else round(r, 3)}


def exp_residual_cuts_carry(ctx):
    """Residualizing value on carry should LOWER |corr| to the carry book vs un-residualized value."""
    g = ctx.get("grid", {})
    d, nr, c = g.get("default"), g.get("no_residual"), g.get("carry_book")
    if d is None or nr is None or c is None:
        return {"pass": False, "observed": "grid_missing"}
    rd, rn = _corr(d, c), _corr(nr, c)
    if not (np.isfinite(rd) and np.isfinite(rn)):
        return {"pass": False, "observed": "nan"}
    return {"pass": bool(abs(rd) <= abs(rn) + 1e-9),
            "observed": f"resid|corr|={round(abs(rd),3)} vs noresid|corr|={round(abs(rn),3)}"}


# ------------------------------------------------------------------ spec
SPEC = StrategySpec(
    id="auto_g10_fx_value_ppp_reer_mean_reversion_car_smith1_58999",
    family="fx_value",
    title="G10 FX Value (PPP/REER mean-reversion, carry-orthogonalized)",
    markets=["FX"],
    data_desc=("G10 spot FX (yfinance; FX is not survivorship-biased) + OECD monthly CPI and "
               "3M interbank rates (FRED), all point-in-time (CPI +2m, rate +1m publication lag)."),
    pre_registration=(
        "The currency VALUE premium: real-exchange-rate (PPP/REER) mean-reversion. Compute each "
        "currency's RER vs USD (spot * CPI_foreign / CPI_US), z-score vs its OWN trailing 60m mean; "
        "undervalued = +score. Cross-sectionally residualize value against the 1m carry differential "
        "each month so the book is carry-orthogonal BY CONSTRUCTION (this is NOT the DM carry harvest). "
        "Long top-3 undervalued / short bottom-3 overvalued residual-value names, dollar-neutral, "
        "inverse-vol within legs, ~10% vol target (<=2x notional), monthly with hysteresis. Costs 8bps "
        "on turnover; signals lagged 1 day. CLAIMS (machine-checked): (1) value-book returns correlate "
        "<0.40 with the pure carry book; (2) residualization lowers |corr| to the carry book vs the "
        "un-residualized value book. Scope=broad: the value premium is a universal mechanism, so a "
        "stage-1 pass must generalize OOS to disjoint EM cross-sections (LatAm/CEE/Asia), >=60% positive."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "no_residual": {"residualize": False},
        "carry_book": {"mode": "carry"},
        "n_leg_2": {"n_leg": 2},
        "z_window_48": {"z_window": 48, "z_min": 30},
    },
    scope="broad",
    generalization_universes=["em_latam", "em_cee", "em_asia"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=6,
    expectations=[
        {"name": "carry_orthogonal",
         "claim": "value-book daily returns correlate <0.40 with the pure carry book",
         "check": exp_carry_orthogonal},
        {"name": "residual_cuts_carry",
         "claim": "residualizing value on carry lowers |corr| to the carry book vs un-residualized value",
         "check": exp_residual_cuts_carry},
    ],
)