"""
Illiquidity-conditioned crowding-reversal in crypto perps (Aether-class hybrid).

Mechanism (single liquidity-provision / immediacy premium, harvested at the
WHERE x WHEN interaction):
  WHERE  -> Amihud illiquidity (|ret| / dollar-volume): thin books pay most for
            immediacy, so capital is concentrated in the illiquid alts where the
            premium is structurally compensated (suppressed in BTC/ETH-tier majors).
  WHEN   -> taker-flow imbalance (taker_buy_quote / total_quote_volume), demeaned:
            short coins with crowded one-sided aggressive BUYING, long aggressive
            SELLING/capitulation; the side mean-reverts as taker inventory clears.
  SIDE = flow-reversal rank ; SIZE = inverse-vol x Amihud rank ; dollar-neutral
  top/bottom-quantile L/S, gross<=2x, EOD daily with EWM turnover-control.
  Regime gate steps gross down when crypto macro-liquidity is in trending stress
  (stablecoin USDT+USDC supply contraction AND one-sided aggregate funding).

This module ships the CORE STANDALONE (proven on holdout first). The small (<=25%)
crypto trend tail-overlay described in pre-registration is deferred and added only
if the core clears the holdout (per the over-blend anti-pattern).

NOTE ON IMPORTS: the WHEN/WHERE terms need taker_buy_quote + dollar volume +
trade-count, which only the OWNED crypto adapters expose (per DATA_CATALOG.md);
yf_panel returns Close only and cannot build this signal. We therefore use the
owned crypto adapters (binance_klines / binance_universe / coinmetrics_metrics /
funding_rates) rather than reinventing any data path. No raw downloads.
"""
from sdk.harness import StrategySpec
from sdk.adapters import binance_klines, binance_universe, coinmetrics_metrics, funding_rates
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ---------------------------------------------------------------- data assembly
_FIELD_ALIASES = {
    "close": ["close", "closeadj", "Close"],
    "dvol":  ["quote_volume", "quote_asset_volume", "quotevolume", "qvol", "dollar_volume"],
    "tbq":   ["taker_buy_quote", "taker_buy_quote_volume",
              "taker_buy_quote_asset_volume", "tbq"],
    "ntrades": ["trades", "num_trades", "n_trades", "count", "number_of_trades"],
}

# generalization universes: DISJOINT later-rank perp slices (share no tickers
# with the binance_universe(75) search set) -> different liquidity tiers.
_GEN = {"rank_76_150": (75, 150), "rank_151_225": (150, 225), "rank_226_300": (225, 300)}


def _pick(kl, key):
    """Extract one field's (dates x tickers) frame from a klines panel, robust to
    whether the field lives on column level-0 or level-1."""
    lv0 = set(kl.columns.get_level_values(0))
    lv1 = set(kl.columns.get_level_values(-1))
    for n in _FIELD_ALIASES[key]:
        if n in lv0:
            return kl.xs(n, axis=1, level=0).apply(pd.to_numeric, errors="coerce")
        if n in lv1:
            return kl.xs(n, axis=1, level=-1).apply(pd.to_numeric, errors="coerce")
    raise KeyError(f"binance_klines missing field {key} (tried {_FIELD_ALIASES[key]})")


def _macro(idx):
    """Macro-liquidity regime inputs (stablecoin supply + aggregate funding).
    Graceful fallback to NaN -> regime gate defaults to full gross (no-op)."""
    try:
        cm = coinmetrics_metrics(("usdt", "usdc"), ("SplyCur",))
        if isinstance(cm.columns, pd.MultiIndex):
            sup = cm.xs("SplyCur", axis=1, level=-1).apply(pd.to_numeric, errors="coerce").sum(axis=1)
        else:
            sup = cm.filter(like="SplyCur").apply(pd.to_numeric, errors="coerce").sum(axis=1)
    except Exception:
        sup = pd.Series(index=idx, dtype=float)
    try:
        fr = funding_rates()
        fund = fr.mean(axis=1) if isinstance(fr, pd.DataFrame) else pd.Series(fr)
    except Exception:
        fund = pd.Series(index=idx, dtype=float)
    m = pd.concat({"supply": sup.reindex(idx).ffill(),
                   "funding": fund.reindex(idx).ffill()}, axis=1)
    return m


def _assemble(tickers):
    kl = binance_klines(list(tickers), market="perp")
    close, dvol = _pick(kl, "close"), _pick(kl, "dvol")
    tbq, nt = _pick(kl, "tbq"), _pick(kl, "ntrades")
    panel = pd.concat({"close": close, "dvol": dvol, "tbq": tbq, "ntrades": nt}, axis=1)
    panel.columns.names = ["field", "ticker"]
    panel = panel.sort_index()
    mac = _macro(panel.index)
    mac.columns = pd.MultiIndex.from_product([["macro"], mac.columns], names=["field", "ticker"])
    return pd.concat([panel, mac], axis=1)


def load_data() -> pd.DataFrame:
    return _assemble(binance_universe(75))


def load_gen_data(label) -> pd.DataFrame:
    lo, hi = _GEN[label]
    allt = list(binance_universe(hi + 30))
    return _assemble(allt[lo:hi])


# ---------------------------------------------------------------- helpers
def _sector_map(panel):
    """Liquidity-tier 'sector' map (T1..T5) for the trade ledger's diversification
    gate. Tier is the relevant cohort for a microstructure book; used for LABELS
    only (never as a signal)."""
    med = panel["dvol"].median(axis=0).dropna()
    if med.empty:
        return {}
    try:
        tier = pd.qcut(med.rank(method="first"), 5,
                       labels=[f"T{i}" for i in range(1, 6)], duplicates="drop")
    except ValueError:
        tier = pd.Series("T3", index=med.index)
    return {t: str(s) for t, s in tier.astype(str).items()}


def _build_weights(panel, params):
    flow_lb = int(params.get("flow_lb", 5))
    amihud_lb = int(params.get("amihud_lb", 21))
    vol_lb = int(params.get("vol_lb", 30))
    gross = float(params.get("gross", 2.0))
    q = float(params.get("quantile", 0.20))
    amihud_off = bool(params.get("amihud_off", False))
    mns = int(params.get("min_names_side", 3))

    close = panel["close"].astype(float)
    dvol = panel["dvol"].astype(float).replace(0.0, np.nan)
    tbq = panel["tbq"].astype(float)
    nt = panel["ntrades"].astype(float)
    ret = close.pct_change()

    # pre-registered trailing liquidity floor (drops dead/new listings)
    dvol_med = dvol.rolling(30, min_periods=10).median()
    nt_med = nt.rolling(30, min_periods=10).median()
    floor_ok = (dvol_med >= 1e6) & (nt_med >= 500) & ret.notna()

    # WHERE: Amihud illiquidity -> cross-sectional rank (high = illiquid = paid)
    amihud = (ret.abs() / dvol).replace([np.inf, -np.inf], np.nan)
    amihud = amihud.rolling(amihud_lb, min_periods=10).mean()
    amw = pd.DataFrame(1.0, index=ret.index, columns=ret.columns) if amihud_off \
        else amihud.rank(axis=1, pct=True)

    # WHEN: demeaned taker-flow imbalance; short crowded buying, long capitulation
    taker_imb = (tbq / dvol).clip(0.0, 1.0)
    imb_dm = taker_imb - taker_imb.rolling(flow_lb, min_periods=2).mean()
    flow_sig = -xs_zscore(imb_dm)

    # SIZE: inverse-vol x Amihud weight, restricted to floor-passing names
    inv_v = (1.0 / ret.rolling(vol_lb, min_periods=10).std()).replace([np.inf, -np.inf], np.nan)
    raw = (flow_sig * amw * inv_v).where(floor_ok)

    # top/bottom-quantile L/S
    rnk = raw.rank(axis=1, pct=True)
    sel = raw.where((rnk >= 1 - q) | (rnk <= q))

    # turnover-control (hysteresis proxy): light EWM on past selections only
    sel = sel.ewm(span=3, min_periods=1).mean().where(ret.notna())

    # dollar-neutral, gross-normalized
    long_w, short_w = sel.where(sel > 0), sel.where(sel < 0)
    W = (long_w.div(long_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
         + short_w.div(short_w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)) * (gross / 2.0)

    # breadth: flat unless both legs have >= min_names_side
    ok = ((sel > 0).sum(axis=1) >= mns) & ((sel < 0).sum(axis=1) >= mns)
    W = W.mul(ok.astype(float), axis=0)

    # REGIME GATE: cut gross in trending macro-liquidity stress
    if "macro" in panel.columns.get_level_values(0):
        mac = panel["macro"]
        sup_g = mac["supply"].pct_change(30)
        f = mac["funding"].abs()
        one_sided = f > f.rolling(180, min_periods=30).quantile(0.80)
        stress = ((sup_g < 0) & one_sided).reindex(W.index).fillna(False)
        gate = pd.Series(1.0, index=W.index)
        gate[stress.values] = 0.25
        W = W.mul(gate, axis=0)

    return W, ret


# ---------------------------------------------------------------- signal
def signal(panel, **params):
    W, ret = _build_weights(panel, params)
    # weights are built same-day -> lag 1 day (our responsibility) before P&L/ledger
    Wl = W.shift(1).fillna(0.0)
    daily = net_of_cost(Wl, ret, cost_bps=10.0,  # 10bps one-way ~ 20bps round-trip taker
                        name="illiq_crowd_reversal")
    trades = trades_from_weights(Wl, ret, _sector_map(panel))
    return daily, trades


# ---------------------------------------------------------------- expectations
def _sharpe(s):
    s = s.dropna()
    return float(s.mean() / s.std() * np.sqrt(252)) if len(s) > 20 and s.std() > 0 else float("nan")


def _chk_collinear(ctx):
    """WHERE and WHEN must be ~orthogonal, else the interaction is not real."""
    p = ctx["panel"]
    ret = p["close"].astype(float).pct_change()
    dvol = p["dvol"].astype(float).replace(0.0, np.nan)
    ar = (ret.abs() / dvol).replace([np.inf, -np.inf], np.nan).rolling(21, min_periods=10).mean().rank(axis=1, pct=True)
    ti = (p["tbq"].astype(float) / dvol).clip(0, 1)
    m = ar.index < pd.Timestamp(ctx["holdout_start"])
    ar, ti = ar[m], ti[m]
    cors = []
    for d in ar.index:
        v = pd.concat([ar.loc[d], ti.loc[d]], axis=1).dropna()
        if len(v) > 5:
            cors.append(v.iloc[:, 0].corr(v.iloc[:, 1]))
    c = float(np.nanmean(cors)) if cors else float("nan")
    return {"pass": (c == c) and abs(c) < 0.6, "observed": round(c, 3)}


def _chk_interaction(ctx):
    """Amihud-weighted flow (default) must out-earn the flat unweighted flow
    control (Parent-2 baseline) on the search window -> the interaction adds info."""
    g = ctx["grid"]
    sd, sf = _sharpe(g.get("default", pd.Series(dtype=float))), \
             _sharpe(g.get("flat_flow", pd.Series(dtype=float)))
    return {"pass": (sd == sd and sf == sf and sd > sf),
            "observed": f"default_SR={round(sd, 2)} flat_SR={round(sf, 2)}"}


# ---------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="aether_illiq_crowd_reversal_perps",
    family="liquidity_provision",
    title="Illiquidity-conditioned crowding-reversal (crypto perps, dollar-neutral L/S)",
    markets=["crypto"],
    data_desc="Binance USDT perps daily OHLCV + quote-volume + trades + taker_buy_quote "
              "(Amihud WHERE + taker-flow WHEN); coinmetrics USDT/USDC SplyCur + funding "
              "for the macro-liquidity regime gate. OWNED/FREE.",
    pre_registration=(
        "ONE liquidity-provision/immediacy premium at the WHERE x WHEN interaction. "
        "WHERE=Amihud illiquidity rank (thin books pay most), WHEN=demeaned taker-flow "
        "imbalance (short crowded aggressive buying, long capitulation). Taker-flow sets "
        "the SIDE; inverse-vol x Amihud-rank sets the SIZE, concentrating provision in "
        "illiquid crowded names and suppressing it in arbitraged majors. Dollar-neutral "
        "top/bottom-20% L/S, gross<=2x, EOD daily, EWM turnover-control, ~20bps round-trip "
        "taker cost, signals lagged 1 day. Regime gate cuts gross to 25% in trending macro "
        "stress (stablecoin supply contraction + one-sided funding). "
        "CHECKABLE CLAIMS: (1) WHERE and WHEN are ~orthogonal (|xs-corr|<0.6) so the "
        "interaction is real [expectation: amihud_flow_orthogonal]; (2) the Amihud-weighted "
        "signal out-earns the flat unweighted flow control on the search window "
        "[expectation: interaction_beats_flat]. PROSE-ONLY (not cheaply machine-checkable "
        "here, validated by the harness battery instead): monotonicity of risk-adjusted "
        "return across Amihud tiers, MCPT vs flow-shuffle AND amihud-shuffle nulls, and the "
        "deferred <=25% crypto-trend tail-overlay (added only if the core clears holdout)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"flow_lb": 5, "amihud_lb": 21, "vol_lb": 30, "gross": 2.0,
                    "quantile": 0.20, "min_names_side": 3},
    grid={
        "default": {},
        "flat_flow": {"amihud_off": True},   # Parent-2 control (no Amihud weighting)
        "flow_lb3": {"flow_lb": 3},
        "flow_lb7": {"flow_lb": 7},
    },
    scope="broad",
    generalization_universes=list(_GEN),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {"name": "amihud_flow_orthogonal",
         "claim": "Amihud rank vs taker-imbalance mean xs-corr |c|<0.6 (interaction is real)",
         "check": _chk_collinear},
        {"name": "interaction_beats_flat",
         "claim": "Amihud-weighted flow Sharpe > flat unweighted flow Sharpe (search window)",
         "check": _chk_interaction},
    ],
)