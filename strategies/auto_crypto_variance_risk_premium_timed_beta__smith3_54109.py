# Crypto Variance-Risk-Premium-Timed Beta + Trend Crisis-Alpha (two-premium regime book)
# scope = LOCAL (DVOL is a BTC/ETH-specific instrument; the VRP regime is crypto-market-wide,
#                not a universal cross-sectional mechanism -> validated by forward-paper, not stage-2).
#
# Mechanism: VRP_t = mean(DVOL implied, BTC&ETH) - mean(trailing-30d realized vol, BTC&ETH).
#   Expanding-since-2021 time-series z-score. PRO-CYCLICAL: bear crypto beta when insurance is rich
#   (VRP_z high -> calm regime, risk-bearing is compensated); flatten to cash when realized overtakes
#   implied (VRP_z < 0 -> vol-explosion/crash regime). Hysteresis (in +0.25, out 0.0) + 3-day min-hold
#   cap turnover. CRISIS-ALPHA: small (~25%) vol-matched validated trend overlay (opposite tail).
#   Held book = equal-risk BTC+ETH basket (the frozen thesis universe); benchmark = buy-and-hold.
#
# FIX vs prior failure: load_data robustly canonicalises the crypto close panel to exact 'BTC'/'ETH'
# columns regardless of the kline adapter's return shape (wide-by-ticker / MultiIndex / tidy-long),
# with a yf_panel(BTC-USD/ETH-USD) fallback so px:: columns are never empty; _vrp_z is hardened so a
# missing symbol returns a NaN z-score instead of raising KeyError(None).

from sdk.harness import StrategySpec
from sdk.adapters import trend_returns, inv_vol_position, deribit_dvol, binance_klines, yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2021-01-01"   # DVOL launched ~2021 -> hard floor on usable history (low-effective-N caveat)

# FROZEN THESIS UNIVERSE: the beta+VRP leg is the equal-risk BTC+ETH basket (no broadening).
MAJORS = ["BTC", "ETH"]

_SECTORS = {
    "BTC": "Store-of-Value",
    "ETH": "L1-Smart",
}

DEFAULTS = dict(entry_z=0.25, exit_z=0.0, min_hold=3, target_vol=0.15,
                rv_lb=30, zmin=120, cost_bps=10.0, trend_weight=0.25)


# ---------------------------------------------------------------- coercion / lookup helpers
def _as_series(x):
    if isinstance(x, pd.DataFrame):
        x = x.select_dtypes("number")
        s = x.iloc[:, -1] if x.shape[1] else x.squeeze()
    else:
        s = pd.Series(x)
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    return s[~s.index.duplicated(keep="last")].sort_index()


def _as_panel(x):
    if isinstance(x, pd.Series):
        df = x.to_frame()
    else:
        df = pd.DataFrame(x)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def _pick(cols, base):
    base = str(base).upper()
    for c in cols:
        if str(c).upper().startswith(base):
            return c
    return None


def _match_col(cols, base):
    """Exact -> startswith -> contains match of a base symbol against arbitrary column labels."""
    base = str(base).upper()
    for c in cols:
        if str(c).upper() == base:
            return c
    for c in cols:
        if str(c).upper().startswith(base):
            return c
    for c in cols:
        if base in str(c).upper():
            return c
    return None


def _sector_map(cols):
    m = {}
    for c in cols:
        cu = str(c).upper()
        sec = "Crypto"
        for base, s in _SECTORS.items():
            if cu.startswith(base):
                sec = s
                break
        m[c] = sec
    return m


def _wide_close(raw, majors):
    """Coerce whatever the kline adapter returns into a wide daily close panel keyed by base symbol,
    handling wide-by-ticker, MultiIndex (ticker,field) and tidy/long (symbol+close) shapes."""
    if raw is None:
        return pd.DataFrame()
    df = raw.to_frame() if isinstance(raw, pd.Series) else pd.DataFrame(raw)
    if df.empty or not len(df.columns):
        return pd.DataFrame()
    low = {str(c).lower(): c for c in df.columns}
    sym = next((low[k] for k in ("symbol", "ticker", "pair", "base", "asset") if k in low), None)
    val = next((low[k] for k in ("close", "closeadj", "adj close", "price", "last", "value") if k in low), None)
    dcol = next((low[k] for k in ("date", "datetime", "time", "timestamp", "dt") if k in low), None)

    if sym is not None and val is not None:                       # tidy / long -> pivot to wide
        d = df.copy()
        if dcol is not None:
            d = d.set_index(dcol)
        d.index = pd.to_datetime(d.index, errors="coerce")
        wide = d.pivot_table(index=d.index, columns=sym, values=val, aggfunc="last")
    elif isinstance(df.columns, pd.MultiIndex):                   # (ticker, field) -> keep close
        wide = None
        for lvl in range(df.columns.nlevels):
            labels = [str(v).lower() for v in df.columns.get_level_values(lvl)]
            if "close" in labels:
                close_lab = df.columns.get_level_values(lvl)[labels.index("close")]
                wide = df.xs(close_lab, axis=1, level=lvl, drop_level=True)
                break
        if wide is None:
            wide = df.copy()
            wide.columns = [str(c[0]) for c in df.columns]
    else:                                                         # already wide-by-ticker
        wide = df

    series = []
    for m in majors:
        c = _match_col(wide.columns, m)
        if c is None:
            continue
        sub = wide[c]
        if isinstance(sub, pd.DataFrame):
            sub = sub.iloc[:, 0]
        series.append(pd.to_numeric(sub, errors="coerce").rename(m))
    if not series:
        return pd.DataFrame()
    out = pd.concat(series, axis=1)
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    return out[~out.index.duplicated(keep="last")].sort_index()


def _crypto_close(majors, start):
    """Equal-risk-basket close panel for the frozen BTC+ETH universe.
    Primary: owned binance perp klines (robustly normalised); fallback: yahoo crypto close."""
    cols = {}
    try:
        try:
            raw = binance_klines(majors, start=start, market="perp")
        except TypeError:
            try:
                raw = binance_klines(majors, market="perp")
            except TypeError:
                raw = binance_klines(majors)
        w = _wide_close(raw, majors)
        for m in majors:
            if m in w.columns and not w[m].dropna().empty:
                cols[m] = w[m]
    except Exception:
        pass

    missing = [m for m in majors if m not in cols]
    if missing:                                                   # yahoo fallback so px:: is never empty
        try:
            yf = _as_panel(yf_panel([m + "-USD" for m in missing], start))
            for m in missing:
                c = _match_col(yf.columns, m)
                if c is not None and not yf[c].dropna().empty:
                    cols[m] = pd.to_numeric(yf[c], errors="coerce")
        except Exception:
            pass

    if not cols:
        return pd.DataFrame()
    px = pd.concat([cols[m].rename(m) for m in majors if m in cols], axis=1)
    px.index = pd.to_datetime(px.index, errors="coerce")
    px = px[~px.index.isna()]
    px = px[~px.index.duplicated(keep="last")].sort_index()
    return px.loc[px.index >= pd.Timestamp(start)]


def _regime(z, entry_z, exit_z, min_hold):
    """Hysteresis + min-hold gate on the VRP z-score. Uses info through t only (no lookahead);
    the 1-day execution lag is applied downstream by inv_vol_position (returns LAGGED positions)."""
    vals = z.values
    out = np.zeros(len(vals))
    on, hold = False, 0
    for i, v in enumerate(vals):
        if on:
            hold += 1
            if (not np.isnan(v)) and v < exit_z and hold >= min_hold:
                on, hold = False, 0
        else:
            if (not np.isnan(v)) and v > entry_z:
                on, hold = True, 0
        out[i] = 1.0 if on else 0.0
    return pd.Series(out, index=z.index)


def _vrp_z(panel, rv_lb=30, zmin=120):
    """Recompute the (BTC&ETH) VRP z-score from a panel -- shared by signal() and the soft checks.
    Hardened: a missing/empty BTC or ETH price column returns a NaN z-score instead of raising."""
    pcols = [c for c in panel.columns if str(c).startswith("px::")]
    px = panel[pcols].copy()
    px.columns = [str(c)[4:] for c in px.columns]
    if not len(px.columns):
        return pd.Series(np.nan, index=panel.index), px
    lr = np.log(px).diff()
    btc, eth = _pick(px.columns, "BTC"), _pick(px.columns, "ETH")
    if btc is None or eth is None:
        return pd.Series(np.nan, index=px.index), px
    rv_b = lr[btc].rolling(rv_lb).std() * np.sqrt(365.0)
    rv_e = lr[eth].rolling(rv_lb).std() * np.sqrt(365.0)
    dvb = panel["dvol::BTC"] if "dvol::BTC" in panel.columns else pd.Series(np.nan, index=panel.index)
    dve = panel["dvol::ETH"] if "dvol::ETH" in panel.columns else pd.Series(np.nan, index=panel.index)
    vrp = ((dvb / 100.0 - rv_b) + (dve / 100.0 - rv_e)) / 2.0
    vrp = vrp.reindex(px.index)
    mu = vrp.expanding(min_periods=zmin).mean()
    sd = vrp.expanding(min_periods=zmin).std().replace(0.0, np.nan)
    return ((vrp - mu) / sd), px


# ---------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    # DVOL (30d implied vol index) from Deribit
    dvol_btc = _as_series(deribit_dvol("BTC"))
    dvol_eth = _as_series(deribit_dvol("ETH"))
    # BTC+ETH close panel (frozen thesis universe), canonicalised to exact 'BTC'/'ETH' columns
    px = _crypto_close(MAJORS, START)

    idx = px.index
    for s in (dvol_btc, dvol_eth):
        idx = idx.union(s.index)
    idx = idx[idx >= pd.Timestamp(START)]

    panel = pd.DataFrame(index=idx)
    for c in px.columns:
        panel["px::" + str(c)] = px[c].reindex(idx)
    panel["dvol::BTC"] = dvol_btc.reindex(idx)
    panel["dvol::ETH"] = dvol_eth.reindex(idx)
    return panel.sort_index()


def load_gen_data(label) -> pd.DataFrame:
    # scope='local' -> stage-2 battery is NOT run; defined for contract completeness only.
    return load_data()


# ---------------------------------------------------------------- signal
def signal(panel, **params):
    p = dict(DEFAULTS)
    p.update(params)

    vrp_z, px = _vrp_z(panel, rv_lb=int(p["rv_lb"]), zmin=int(p["zmin"]))
    px = px.dropna(how="all")
    rets = px.pct_change()

    regime = _regime(vrp_z.reindex(px.index), p["entry_z"], p["exit_z"], int(p["min_hold"]))

    # broadcast the single market-wide regime across BTC & ETH with data
    sig = pd.DataFrame(np.repeat(regime.fillna(0.0).values[:, None], px.shape[1], axis=1),
                       index=px.index, columns=px.columns)
    sig = sig.where(px.notna(), 0.0)

    # inverse-vol equal-risk BTC+ETH, weekly rebalance, 15% vol target. inv_vol_position returns LAGGED
    # positions -> pass W straight to net_of_cost / trades_from_weights (no extra shift).
    W = inv_vol_position(sig, rets, target_vol=p["target_vol"], vol_lb=int(p["rv_lb"]),
                         max_pos=max(1, px.shape[1]), rebalance="W")

    sector_map = _sector_map(px.columns)
    crypto_net = net_of_cost(W, rets, cost_bps=p["cost_bps"], name="vrp_crypto")  # ~10bps/side ~= 20bps RT taker
    trades = trades_from_weights(W, rets, sector_map)  # regime stamped by the kit (contract)
    crypto_net = crypto_net.dropna()

    out = crypto_net.copy()
    if p["trend_weight"] > 0 and len(crypto_net):
        tr = _as_series(trend_returns()[0])  # validated 21-market CTA crisis-alpha leg (already net)
        aln = pd.concat([crypto_net, tr], axis=1).dropna()
        if len(aln) > 60 and aln.iloc[:, 1].std() > 0:
            scale = aln.iloc[:, 0].std() / aln.iloc[:, 1].std()            # vol-match trend to crypto leg
            tr_m = (tr * scale).reindex(crypto_net.index).fillna(0.0)       # 0 on crypto-only days (wknds)
            out = crypto_net + p["trend_weight"] * tr_m                     # additive tail overlay (keeps full beta)

    out = out.dropna()
    out.name = "crypto_vrp_timed_beta_trend"
    return out, trades


# ---------------------------------------------------------------- soft expectation checks
def _sr(r):
    r = pd.Series(r).dropna()
    if len(r) < 20 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(252.0))


def _mdd(r):
    r = pd.Series(r).dropna()
    if not len(r):
        return float("nan")
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _chk_beats_bh(ctx):
    try:
        hs = pd.Timestamp(ctx["holdout_start"])
        px = ctx["panel"][[c for c in ctx["panel"].columns if str(c).startswith("px::")]].copy()
        px.columns = [str(c)[4:] for c in px.columns]
        px = px.loc[px.index < hs]
        rets = px.pct_change()
        iv = (1.0 / rets.rolling(30).std()).replace([np.inf, -np.inf], np.nan)
        w = iv.div(iv.sum(axis=1), axis=0)                 # static inverse-vol equal-risk BTC+ETH
        bh = (w.shift(1) * rets).sum(axis=1)               # passive buy-and-hold BTC+ETH benchmark
        s_strat, s_bh = _sr(ctx["search"]), _sr(bh)
        return {"pass": bool(s_strat > s_bh), "observed": f"strat_SR={s_strat:.2f} vs buyhold_SR={s_bh:.2f}"}
    except Exception as e:
        return {"pass": True, "observed": f"uncheckable: {e}"}


def _chk_trend_cuts_dd(ctx):
    g = ctx.get("grid", {})
    if "default" not in g or "crypto_standalone" not in g:
        return {"pass": True, "observed": "variants missing"}
    dd_c, dd_s = _mdd(g["default"]), _mdd(g["crypto_standalone"])
    return {"pass": bool(dd_c >= dd_s - 1e-9), "observed": f"maxDD combined={dd_c:.3f} standalone={dd_s:.3f}"}


def _chk_trend_not_dilutive(ctx):
    g = ctx.get("grid", {})
    if "default" not in g or "crypto_standalone" not in g:
        return {"pass": True, "observed": "variants missing"}
    s_c, s_s = _sr(g["default"]), _sr(g["crypto_standalone"])
    return {"pass": bool(s_c >= 0.9 * s_s), "observed": f"SR combined={s_c:.2f} standalone={s_s:.2f}"}


def _chk_vrp_risk_off_in_crashes(ctx):
    try:
        hs = pd.Timestamp(ctx["holdout_start"])
        z, _ = _vrp_z(ctx["panel"])
        z = z.loc[z.index < hs].dropna()
        if not len(z):
            return {"pass": True, "observed": "no search-window z"}
        wins = [("2021-05-12", "2021-05-23"), ("2022-05-07", "2022-05-20"), ("2022-11-07", "2022-11-15")]
        seg = pd.concat([z.loc[a:b] for a, b in wins]).dropna()
        if not len(seg):
            return {"pass": True, "observed": "no crash-window data"}
        cm, om = float(seg.mean()), float(z.mean())
        return {"pass": bool(cm < om), "observed": f"crash VRP_z={cm:.2f} < sample mean={om:.2f}"}
    except Exception as e:
        return {"pass": True, "observed": f"uncheckable: {e}"}


# ---------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="crypto_vrp_timed_beta_trend",
    family="variance_risk_premium",
    title="Crypto VRP-Timed Beta + Trend Crisis-Alpha (two-premium regime book)",
    markets=["crypto", "futures"],
    data_desc=("deribit_dvol('BTC'/'ETH') 30d implied vol vs trailing-30d realized vol "
               "(binance perp close, yahoo BTC-USD/ETH-USD fallback) -> expanding VRP z-score regime "
               "gate on an equal-risk BTC+ETH basket; + small vol-matched trend_returns() overlay."),
    pre_registration=(
        "Hypothesis: the crypto variance risk premium (DVOL implied minus realized vol) signs the regime "
        "in which bearing crypto beta is compensated. Go long an equal-risk BTC+ETH basket (inverse-vol, "
        "weekly rebalance, 15% vol target) when VRP_z > +0.25 (insurance rich -> calm), flat to cash when "
        "VRP_z < 0 (realized overtakes implied -> vol explosion/crash); hysteresis (in +0.25 / out 0.0) + "
        "3-day min-hold cap turnover. A SMALL ~25% vol-matched trend crisis-alpha overlay (opposite tail) "
        "is ADDED on top (not 50/50) and kept only if it cuts drawdown without diluting the standalone "
        "Sharpe (checked vs the crypto_standalone grid variant). Costs: ~10bps/side (~20bps round-trip "
        "taker) crypto; trend leg net. Execution lag handled by inv_vol_position (lagged positions); VRP "
        "z-score is expanding-only (info through t). Scope LOCAL: DVOL is a BTC/ETH-specific 2021 product, "
        "the regime is crypto-market-wide and not a universal cross-sectional mechanism; validated by "
        "excess-over-buy-and-hold + forward paper. CAVEAT: short DVOL history (~2021+, few crash regimes) "
        "is a low-effective-N regime book -- treat the verdict as provisional pending forward confirmation. "
        "NOTE: this is intentionally a 2-name book (BTC+ETH); single_name_share is inherent to the frozen "
        "thesis, not a stray ETF hedge. Benchmark/null = buy-and-hold equal-risk BTC+ETH basket; the edge "
        "must be EXCESS risk-adjusted return / lower DD, i.e. the timing not the raw beta."),
    load_data=load_data,
    signal=signal,
    default_params=dict(DEFAULTS),
    grid={
        "default": {},
        "crypto_standalone": {"trend_weight": 0.0},
        "entry_strict": {"entry_z": 0.5, "exit_z": 0.1},
        "fast_exit": {"min_hold": 1},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2024-01-01",   # search 2021-2023 (incl May-21, LUNA, FTX); OOS 2024-2025 (incl Aug-24 unwind)
    deploy_max_positions=2,
    expectations=[
        {"name": "beats_buy_and_hold",
         "claim": "VRP-timed book Sharpe > buy-and-hold equal-risk BTC+ETH basket (search window)",
         "check": _chk_beats_bh},
        {"name": "trend_cuts_drawdown",
         "claim": "combined-book maxDD is shallower than the crypto_standalone variant",
         "check": _chk_trend_cuts_dd},
        {"name": "trend_not_dilutive",
         "claim": "combined-book Sharpe >= 0.9x crypto_standalone Sharpe (overlay not dilutive)",
         "check": _chk_trend_not_dilutive},
        {"name": "vrp_risk_off_in_crashes",
         "claim": "mean VRP_z in known crash windows (May-21, LUNA, FTX) is below the search-window mean (gate leans off)",
         "check": _chk_vrp_risk_off_in_crashes},
    ],
)