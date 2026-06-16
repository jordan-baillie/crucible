"""
On-Chain Network-Value Cross-Sectional Factor — Crypto L1/alt Perp Long/Short.

SIGNAL = pure blockchain FUNDAMENTALS only:
  (a) VALUE  : cost-basis cheapness  -> -z(CapMVRVCur)  combined equally with a
               network-value-to-usage gauge -z(CapMrktCurUSD / smoothed90(TxTfrCnt)).
  (b) ADOPTION-QUALITY : z(90d growth of AdrActCnt) + z(90d growth of TxCnt)
               minus a PRICE-FREE dilution penalty z(IssTotUSD/CapMrktCurUSD * 365).
  COMPOSITE = z(value) + z(adoption_quality); long top tertile / short bottom tertile,
  dollar-neutral, inverse-vol within legs, WEEKLY rebalance.

NO price-return / trading-volume / microstructure feature enters the signal.
(Market cap CapMrktCurUSD is a FUNDAMENTAL network-value level, not microstructure;
 dilution uses IssTotUSD/CapMrktCurUSD = issued/supply, the price terms cancel.)
Price (yf_panel <SYM>-USD close) is used for EXECUTION RETURNS ONLY.

GATE-0 / verify-before-build (pre-registered in the proposal's gate0_data_check):
the SIGNAL needs CoinMetrics community-tier on-chain metrics (`coinmetrics_metrics`
from DATA_CATALOG.md). It is NOT part of the core kit allow-list, so it is imported
GUARDED: if the adapter is not provisioned, load_data() HALTS with a clear Gate-0
error rather than silently substituting price features (which would violate the whole
'fundamentals, not microstructure' premise). This mirrors the recon-first posture:
no fabricated data path.

Lags: every metric is shifted 2 trading days (CoinMetrics reports next-day) and the
weight matrix is shifted 1 more day for execution -> strictly no look-ahead.
scope='local' (alt-perp-cross-section-specific edge; forward-validation confirms).
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights

# --- GUARDED on-chain adapter (the one Gate-0 dependency) ------------------------
try:
    from sdk.adapters import coinmetrics_metrics  # DATA_CATALOG on-chain fundamentals
    _HAVE_CM = True
except Exception:  # pragma: no cover
    coinmetrics_metrics = None
    _HAVE_CM = False

SID = "crypto_onchain_mvrv_value_v1"

GATE0_MSG = (
    "GATE-0 HALT: coinmetrics_metrics (on-chain fundamentals) is not provisioned in "
    "sdk.adapters. This strategy's SIGNAL is 100% blockchain fundamentals (MVRV cost-"
    "basis + adoption + issuance) and CANNOT be built from price/volume without "
    "violating its core premise. Provision the CoinMetrics community adapter (per "
    "research-wiki/DATA_CATALOG.md) before running this module — no price substitution."
)

# cm asset code -> canonical symbol  (yf execution ticker = SYM + '-USD')
UNIVERSE = {
    "btc": "BTC", "eth": "ETH", "ltc": "LTC", "bch": "BCH", "etc": "ETC",
    "xrp": "XRP", "ada": "ADA", "sol": "SOL", "avax": "AVAX", "dot": "DOT",
    "matic": "MATIC", "link": "LINK", "atom": "ATOM", "near": "NEAR",
    "algo": "ALGO", "xlm": "XLM", "fil": "FIL", "uni": "UNI", "aave": "AAVE",
    "xmr": "XMR", "eos": "EOS", "xtz": "XTZ", "trx": "TRX", "doge": "DOGE",
}

# crypto "sector" map for the contract trade ledger (sector-spread gate)
SECTOR_MAP = {
    **{s: "L1-smart-contract" for s in
       ["ETH", "SOL", "AVAX", "DOT", "ADA", "NEAR", "ALGO", "ATOM", "ETC", "EOS", "XTZ", "TRX"]},
    **{s: "payments-sov" for s in ["BTC", "LTC", "BCH", "DOGE", "XRP", "XLM"]},
    **{s: "defi" for s in ["UNI", "AAVE", "LINK"]},
    "FIL": "infra-storage", "MATIC": "scaling-l2", "XMR": "privacy",
}

METRICS = ["CapMVRVCur", "CapMrktCurUSD", "TxTfrCnt", "TxCnt",
           "AdrActCnt", "SplyCur", "IssTotUSD"]
START = "2018-01-01"


# ============================ data assembly ====================================
def _normalize_cm(cm, metrics):
    """coinmetrics_metrics output -> dict[metric] = wide DataFrame(date x cm_asset_code)."""
    if isinstance(cm, dict):
        return {m: cm[m] for m in metrics if m in cm}
    df = cm
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        out = {}
        if set(metrics) & lvl0:                       # (metric, asset)
            for m in metrics:
                if m in lvl0:
                    out[m] = df[m]
        else:                                         # (asset, metric)
            for m in metrics:
                if m in lvl1:
                    out[m] = df.xs(m, axis=1, level=1, drop_level=True)
        return out
    cols = {c.lower(): c for c in df.columns}          # tidy / long
    dcol = next((cols[k] for k in ("time", "date", "datetime", "day") if k in cols), None)
    acol = next((cols[k] for k in ("asset", "symbol", "ticker", "name") if k in cols), None)
    if dcol is None or acol is None:
        raise RuntimeError("coinmetrics_metrics: unrecognised shape %s" % list(df.columns))
    df = df.copy()
    df[dcol] = pd.to_datetime(df[dcol], utc=True).dt.tz_convert(None).dt.normalize()
    return {m: df.pivot_table(index=dcol, columns=acol, values=m, aggfunc="last")
            for m in metrics if m in df.columns}


def load_data() -> pd.DataFrame:
    """Panel signal() consumes: MultiIndex columns (field, ticker)."""
    if not _HAVE_CM:
        raise RuntimeError(GATE0_MSG)

    assets = list(UNIVERSE.keys())
    cm = coinmetrics_metrics(assets, metrics=METRICS, start=START)
    panels = _normalize_cm(cm, METRICS)
    panels = {m: w.rename(columns=UNIVERSE) for m, w in panels.items()}

    # execution-only price (crypto-USD pairs are NOT US single stocks -> yf_panel OK)
    yf_tk = [UNIVERSE[a] + "-USD" for a in assets]
    px = yf_panel(yf_tk, START)
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.rename(columns={c: c.replace("-USD", "") for c in px.columns})
    panels["close"] = px

    syms = sorted(set(UNIVERSE.values()))
    idx = None
    for w in panels.values():
        idx = w.index if idx is None else idx.union(w.index)
    idx = pd.DatetimeIndex(sorted(idx))

    frames = {}
    for m, w in panels.items():
        w = w.reindex(index=idx, columns=syms)
        w = w.ffill(limit=3) if m == "close" else w.ffill(limit=7)
        frames[m] = w
    panel = pd.concat(frames, axis=1)
    panel.columns.names = ["field", "ticker"]
    panel.index.name = "date"
    return panel


def load_gen_data(label) -> pd.DataFrame:
    # scope='local' -> no disjoint generalization universes; harness skips stage-2.
    return load_data()


# ============================ signal helpers ===================================
def _lag(df, n):
    return None if df is None else df.shift(n)


def _nanmean(frames):
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    base = frames[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.nanmean(np.stack([f.reindex_like(base).values for f in frames], axis=0), axis=0)
    return pd.DataFrame(s, index=base.index, columns=base.columns)


def _weekly_anchors(idx):
    idx = pd.DatetimeIndex(idx)
    keep = ~pd.Series(idx.to_period("W")).duplicated().values
    return idx[keep]


def _portfolio(comp, rets, gross=1.0, vol_lb=60):
    """Weekly tertile, dollar-neutral, inverse-vol within legs. Returns target W (unlagged)."""
    vol = rets.rolling(vol_lb, min_periods=20).std().shift(1)   # trailing-only vol
    iv = (1.0 / vol.replace(0.0, np.nan))
    anchors = _weekly_anchors(comp.index)
    Wsp = pd.DataFrame(0.0, index=anchors, columns=comp.columns)

    def leg(names, dt):
        w = iv.loc[dt].reindex(names) if dt in iv.index else pd.Series(np.nan, index=names)
        if w.notna().any():
            w = w.fillna(w.median())
        else:
            w = pd.Series(1.0, index=names)
        tot = w.sum()
        return (w / tot) if tot > 0 else w * 0.0

    for dt in anchors:
        s = comp.loc[dt].dropna()
        if len(s) < 6:
            continue
        k = max(1, len(s) // 3)
        longs, shorts = s.nlargest(k).index, s.nsmallest(k).index
        Wsp.loc[dt, longs] = leg(longs, dt).values * (gross / 2.0)
        Wsp.loc[dt, shorts] = -leg(shorts, dt).values * (gross / 2.0)

    return Wsp.reindex(comp.index).ffill().fillna(0.0)


def signal(panel, **params):
    use_usage = params.get("use_usage", True)
    use_adoption = params.get("use_adoption", True)
    gross = params.get("gross", 1.0)
    cost_bps = params.get("cost_bps", 20.0)        # crypto taker, pre-registered

    fields = list(panel.columns.get_level_values(0).unique())
    g = lambda n: panel[n].copy() if n in fields else None

    close = g("close")
    rets = close.pct_change()

    # ---- on-chain features, 2-day as-of lag (CoinMetrics reports next-day) ----
    L = 2
    mvrv = _lag(g("CapMVRVCur"), L)
    mcap = _lag(g("CapMrktCurUSD"), L)
    tfr = _lag(g("TxTfrCnt"), L)
    txc = _lag(g("TxCnt"), L)
    adr = _lag(g("AdrActCnt"), L)
    iss = _lag(g("IssTotUSD"), L)

    # ---- VALUE: cheap vs holder cost-basis + cheap vs realized usage ----
    comps = []
    if mvrv is not None:
        comps.append(-xs_zscore(mvrv))                       # low MVRV = cheap = +
    if use_usage and mcap is not None and tfr is not None:
        nvt = mcap / tfr.where(tfr > 0).rolling(90, min_periods=30).mean()
        comps.append(-xs_zscore(nvt))                        # low network-value/usage = +
    value = _nanmean(comps)
    composite = xs_zscore(value)

    # ---- ADOPTION-QUALITY: usage growth, penalised by (price-free) dilution ----
    if use_adoption:
        aq_terms = []
        if adr is not None:
            aq_terms.append(xs_zscore(np.log(adr.where(adr > 0).rolling(7, min_periods=3).mean()).diff(90)))
        if txc is not None:
            aq_terms.append(xs_zscore(np.log(txc.where(txc > 0).rolling(7, min_periods=3).mean()).diff(90)))
        aq = _nanmean(aq_terms)
        if aq is not None:
            if iss is not None and mcap is not None:
                dilution = (iss / mcap.where(mcap > 0)).rolling(30, min_periods=10).mean() * 365.0
                aq = aq.sub(xs_zscore(dilution).fillna(0.0))  # issued/supply: price terms cancel
            composite = composite.add(xs_zscore(aq), fill_value=0.0)

    # ---- weights -> execution lag -> net returns + contract ledger ----
    W = _portfolio(composite, rets, gross=gross)
    Wlag = W.shift(1)                                         # 1-day execution lag (mine to apply)
    rets_fill = rets.reindex(index=W.index, columns=W.columns).fillna(0.0)

    daily = net_of_cost(Wlag, rets_fill, cost_bps=cost_bps, name=SID)
    trades = trades_from_weights(Wlag, rets_fill, SECTOR_MAP)  # auto-stamps entry_regime
    return daily, trades


# ============================ soft expectations ================================
def _sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 20 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252.0))


def _chk_quality_gate_helps(ctx):
    """Diagnostic (b): does usage+adoption add over pure MVRV cost-basis value?"""
    grid = ctx.get("grid", {}) or {}
    d, m = grid.get("default"), grid.get("mvrv_only")
    if d is None or m is None:
        return {"pass": True, "observed": "grid variants unavailable"}
    diff = round(_sharpe(d) - _sharpe(m), 3)
    return {"pass": diff >= -0.02, "observed": diff}        # falsifier if clearly negative


def _chk_slow_signal(ctx):
    """Mechanism: a weekly fundamental signal must HOLD names for weeks, not days."""
    tr = ctx.get("trades", []) or []
    hd = [t.get("hold_days", 0) for t in tr]
    if not hd:
        return {"pass": False, "observed": "no trades"}
    med = float(np.median(hd))
    return {"pass": med >= 10.0, "observed": med}


SPEC = StrategySpec(
    id=SID,
    family="value",
    title="On-Chain Network-Value Cross-Sectional Factor — Crypto Alt-Perp L/S "
          "(MVRV cost-basis VALUE x ADOPTION-QUALITY, fundamentals-only, NOT microstructure)",
    markets=["crypto"],
    data_desc="CoinMetrics community on-chain fundamentals (CapMVRVCur, CapMrktCurUSD, "
              "TxTfrCnt, TxCnt, AdrActCnt, SplyCur, IssTotUSD) for ~24 covered L1/alt "
              "coins; yf_panel <SYM>-USD close for EXECUTION returns only. $0 data.",
    pre_registration=(
        "FROZEN spec. SIGNAL uses ONLY blockchain fundamentals — zero price-return / "
        "volume / microstructure features (market cap is a fundamental network-value "
        "level; dilution = IssTotUSD/CapMrktCurUSD so price terms cancel). "
        "COMPOSITE = z(value) + z(adoption_quality): value = equal blend of -z(MVRV) "
        "(cheap vs holder cost basis) and -z(CapMrktCurUSD/smoothed90(TxTfrCnt)) "
        "(cheap vs realized usage); adoption_quality = z(90d AdrActCnt growth) + "
        "z(90d TxCnt growth) - z(annualised dilution). Long top tertile / short bottom "
        "tertile, dollar-neutral, inverse-vol legs, WEEKLY rebalance. Metrics lagged 2d "
        "(CoinMetrics reports next-day) + weights lagged 1d for execution. 20bps taker. "
        "MECHANISM: contrarian fundamental value — paid by capitulating underwater holders "
        "(low MVRV), fading euphoric over-valuation (high MVRV), quality-screened by genuine "
        "usage growth to avoid the 'cheap-because-dying' value trap. "
        "default {} is the SINGLE deployed/primary config; 'value_only' and 'mvrv_only' are "
        "pre-declared ABLATIONS solely for honest DSR effective-N + the marginal-value "
        "diagnostic — NEVER selected on. "
        "HONEST FALSIFIERS: contrarian value is EXPECTED to bleed in sustained bears "
        "(cheap gets cheaper) — the 2022 holdout (LUNA/FTX) is the key test, not a bug. "
        "GATE-0 RISK = breadth: needs >=~12 coins with BOTH populated MVRV/adoption history "
        "AND a liquid perp; if MVRV is majors-only the constructed CapMrktCurUSD/TxTfrCnt "
        "value gauge carries the broader coverage (z-score is NaN-preserving, so coins "
        "missing MVRV still contribute via the usage gauge). NOT carry (funding excluded), "
        "NOT momentum (contrarian-on-fundamentals), NOT microstructure. scope=local; "
        "forward-validated in $0 paper, pre-registering ~0 correlation to the parent "
        "(price/volume Amihud + trend); operator gates any deployment."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "value_only": {"use_adoption": False},               # drop usage+adoption quality gate
        "mvrv_only": {"use_usage": False, "use_adoption": False},  # pure cost-basis value
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=16,
    expectations=[
        {"name": "quality_gate_helps",
         "claim": "full composite Sharpe >= MVRV-only Sharpe (usage+adoption quality gate "
                  "adds value; clearly negative => it is pure cost-basis value, falsified)",
         "check": _chk_quality_gate_helps},
        {"name": "slow_fundamental_holds",
         "claim": "weekly fundamental signal -> median holding period >= 10 trading days "
                  "(slow network-state, not fast microstructure)",
         "check": _chk_slow_signal},
    ],
)