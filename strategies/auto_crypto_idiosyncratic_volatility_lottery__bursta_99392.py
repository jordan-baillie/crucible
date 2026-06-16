"""
Crypto idiosyncratic-volatility / lottery-preference premium.
Long low-IVOL majors / short high-IVOL alts, beta-neutral via a BTC offset, perp-deployable.

MECHANISM: in the young, retail-dominated crypto perp market, lottery-like high-IVOL alts are
systematically OVERPAID for (retail buys upside-skew). Harvested cross-sectionally (not a
directional forecast): long the calm (low residual-vol) majors, short the lottery (high
residual-vol) alts, hedge the residual BTC-beta to ~0. NOT trend, NOT short-term reversal,
NOT funding/basis carry (no funding is accrued).

NO LOOK-AHEAD: every weight is decided from data through date t; the whole weight matrix is
then .shift(1) before net_of_cost / trades_from_weights -- the lag is explicit and ours.

CONSERVATIVE BIAS (stated): yfinance crypto EXCLUDES dead coins (LUNA, FTT, ...). The high-IVOL
SHORT leg would have PROFITED from those collapses, so this survivor-only test UNDERSTATES the
short leg => results are biased conservative.

COST: 20bps charged per unit ONE-WAY turnover (net_of_cost) -> deliberately heavier than the
proposal's ~20bps round-trip framing; conservative for deep-major perps (Binance/Bybit taker+slip).
"""
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2021-01-01"
HOLDOUT_START = "2024-01-01"   # search 2021-2023 spans the 2022 bear + 2023 recovery; OOS 2024-2026

BTC = "BTC-USD"
ALTS = ["ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
        "LINK-USD", "DOT-USD", "TRX-USD", "LTC-USD", "BCH-USD", "MATIC-USD", "ATOM-USD",
        "NEAR-USD", "APT-USD", "ARB-USD"]
ALL_TICKERS = [BTC] + ALTS

# loose "sectors" so the trade-ledger sector-spread gate is meaningful for a crypto book
SECTOR_MAP = {
    "BTC-USD": "store-of-value", "ETH-USD": "smart-contract-L1", "SOL-USD": "smart-contract-L1",
    "BNB-USD": "exchange", "XRP-USD": "payment", "ADA-USD": "smart-contract-L1",
    "DOGE-USD": "meme", "AVAX-USD": "smart-contract-L1", "LINK-USD": "oracle-defi",
    "DOT-USD": "interop", "TRX-USD": "smart-contract-L1", "LTC-USD": "payment",
    "BCH-USD": "payment", "MATIC-USD": "scaling-L2", "ATOM-USD": "interop",
    "NEAR-USD": "smart-contract-L1", "APT-USD": "smart-contract-L1", "ARB-USD": "scaling-L2",
}

NAME = "crypto_ivol_lottery"


def load_data() -> pd.DataFrame:
    px = yf_panel(ALL_TICKERS, START)               # daily spot closes (FREE; crypto, not US stocks)
    px = px.reindex(columns=ALL_TICKERS).sort_index()
    return px.dropna(how="all")


def _build_weights(px, beta_lb, ivol_lb, hyst, min_names):
    """Weekly long-low-IVOL / short-high-IVOL terciles, BTC-beta hedged. Returns SAME-DAY weights."""
    rets = px.pct_change()
    btc = rets[BTC]
    alts = [c for c in ALTS if c in rets.columns]
    aret = rets[alts]

    # 60d residual of each alt vs BTC -> 30d IVOL = std of those residuals
    cov = aret.rolling(beta_lb).cov(btc)
    var = btc.rolling(beta_lb).var()
    beta = cov.divide(var, axis=0)
    resid = aret.subtract(beta.multiply(btc, axis=0))
    ivol = resid.rolling(ivol_lb).std()

    rb_dates = [d for d in px.index if d.weekday() == 4]   # Fridays => weekly == 7-day min hold
    state, rb_w = {}, {}
    for d in rb_dates:
        iv = ivol.loc[d, alts].dropna()
        bt = beta.loc[d, alts]
        iv = iv[bt.reindex(iv.index).notna()]
        if len(iv) < min_names:                            # drop weeks with too few sortable coins
            continue
        q = (iv.rank(method="first") - 1.0) / (len(iv) - 1.0)   # cross-sectional percentile in [0,1]
        new = {}
        for t in iv.index:
            pv, prev = q[t], state.get(t)
            if prev == "L":                                # hysteresis: sticky membership
                new[t] = "L" if pv <= 1/3 + hyst else ("S" if pv >= 2/3 else None)
            elif prev == "S":
                new[t] = "S" if pv >= 2/3 - hyst else ("L" if pv <= 1/3 else None)
            else:
                new[t] = "L" if pv <= 1/3 else ("S" if pv >= 2/3 else None)
        state = {k: v for k, v in new.items() if v is not None}
        longs = [t for t, v in new.items() if v == "L"]
        shorts = [t for t, v in new.items() if v == "S"]
        if not longs or not shorts:
            continue
        w = pd.Series(0.0, index=ALL_TICKERS)
        w[longs] = 1.0 / len(longs)
        w[shorts] = -1.0 / len(shorts)
        net_beta = float(sum(w[t] * float(bt.get(t, 0.0)) for t in longs + shorts))
        w[BTC] += -net_beta                                # BTC offset -> net book beta ~ 0
        rb_w[d] = w

    if not rb_w:
        return None, rets
    rbw = pd.DataFrame(rb_w).T.reindex(columns=ALL_TICKERS).fillna(0.0)
    return rbw.reindex(px.index, method="ffill").fillna(0.0), rets


def signal(panel, **params):
    cost_bps = params.get("cost_bps", 20.0)
    target_vol = params.get("target_vol", 0.10)
    gross_cap = params.get("gross_cap", 2.0)
    beta_lb = params.get("beta_lb", 60)
    ivol_lb = params.get("ivol_lb", 30)
    hyst = params.get("hyst", 0.10)
    min_names = params.get("min_names", 15)

    px = panel.copy()
    W_raw, rets = _build_weights(px, beta_lb, ivol_lb, hyst, min_names)
    if W_raw is None:
        return pd.Series(dtype=float, name=NAME), []

    # vol-target to target_vol off trailing 30d book vol, refreshed only at rebalance; gross<=gross_cap
    raw_ret = (W_raw.shift(1) * rets).sum(axis=1)
    ann_vol = raw_ret.rolling(30, min_periods=20).std() * np.sqrt(365.0)
    raw_scale = target_vol / ann_vol
    cap = (gross_cap / W_raw.abs().sum(axis=1)).replace([np.inf, -np.inf], np.nan)
    scaled = np.minimum(raw_scale, cap)
    rb_dates = [d for d in px.index if d.weekday() == 4]
    scale = scaled.reindex(rb_dates).reindex(px.index, method="ffill")
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    W = W_raw.multiply(scale, axis=0)

    Wl = W.shift(1).fillna(0.0)                 # decision at t-1 traded at t (lag is ours)
    daily = net_of_cost(Wl, rets, cost_bps=cost_bps, name=NAME)
    trades = trades_from_weights(Wl, rets, SECTOR_MAP)   # kit stamps entry_regime (contract)

    nz = Wl.abs().sum(axis=1)
    live = nz[nz > 0].index
    if len(live):
        daily = daily.loc[live.min():]
    return daily.dropna(), trades


def load_gen_data(label) -> pd.DataFrame:
    # scope='local' -> stage-2 generalization battery is not run; provided for interface completeness.
    return load_data()


# ---------------- soft expectations (machine-checkable mechanism claims) ----------------
def _check_beta_neutral(ctx):
    r = ctx["search"].dropna()
    btc = ctx["panel"][BTC].pct_change().reindex(r.index)
    df = pd.concat([r, btc], axis=1).dropna()
    if len(df) < 60:
        return {"pass": False, "observed": "insufficient"}
    x, y = df.iloc[:, 1].to_numpy(), df.iloc[:, 0].to_numpy()
    beta = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    return {"pass": abs(beta) < 0.15, "observed": round(beta, 4)}


def _check_subperiods(ctx):
    r = ctx["search"]
    bear = float(r.loc["2022-01-01":"2022-12-31"].sum())
    rec = float(r.loc["2023-01-01":"2023-12-31"].sum())
    return {"pass": bool(bear > 0 and rec > 0),
            "observed": f"bear2022={bear:.3f} recovery2023={rec:.3f}"}


def _tercile_means(panel, end):
    px = panel.loc[:end]
    rets = px.pct_change()
    btc = rets[BTC]
    alts = [c for c in ALTS if c in rets.columns]
    aret = rets[alts]
    beta = aret.rolling(60).cov(btc).divide(btc.rolling(60).var(), axis=0)
    ivol = aret.subtract(beta.multiply(btc, axis=0)).rolling(30).std()
    rb = [d for d in px.index if d.weekday() == 4]
    lo, mi, hi = [], [], []
    for i in range(len(rb) - 1):
        d, nxt = rb[i], rb[i + 1]
        iv = ivol.loc[d, alts].dropna()
        if len(iv) < 12:
            continue
        wk = (px.loc[nxt, iv.index] / px.loc[d, iv.index] - 1.0)   # next-week realized return
        q = (iv.rank(method="first") - 1.0) / (len(iv) - 1.0)
        lo.append(wk[q <= 1/3].mean())
        hi.append(wk[q >= 2/3].mean())
        mi.append(wk[(q > 1/3) & (q < 2/3)].mean())
    return float(np.nanmean(lo)), float(np.nanmean(mi)), float(np.nanmean(hi))


def _check_monotonic(ctx):
    end = pd.Timestamp(ctx["holdout_start"]) - pd.Timedelta(days=1)   # search-only
    lm, mm, hm = _tercile_means(ctx["panel"], end)
    return {"pass": bool(lm > mm and mm > hm),
            "observed": f"low={lm:.4f} mid={mm:.4f} high={hm:.4f}"}


SPEC = StrategySpec(
    id="crypto_ivol_lottery",
    family="low_vol_lottery",
    title="Crypto idiosyncratic-vol / lottery premium (long low-IVOL / short high-IVOL, beta-neutral, perp-deployable)",
    markets=["crypto"],
    data_desc="yfinance daily spot closes: BTC + 17 deepest USDT-perp alt-majors (2021->), BTC as the single beta factor",
    pre_registration=(
        "PRIMARY (single pre-registered config, NO grid -> DSR effective-N=1): weekly (Fri close, "
        "strict t-1 lag) compute each alt's 30d IDIOSYNCRATIC vol = std of daily-return residuals after "
        "regressing the coin on BTC over a trailing 60d window. Universe = the 17 currently-deepest "
        "USDT-perp alt-majors (BTC EXCLUDED from the sort -> used only as the beta factor/hedge). LONG "
        "bottom IVOL tercile / SHORT top IVOL tercile, equal-weight within leg (>=5 names/leg; skip weeks "
        "with <15 sortable coins). Add a BTC offset = -(net book beta) so net BTC-beta ~ 0. Vol-target the "
        "book to 10% annualized off trailing 30d book vol, refreshed only at rebalance; cap gross at 2x; "
        "10% hysteresis on tercile membership + weekly (>=7d) min hold to cap turnover; 20bps one-way taker "
        "charged on turnover (conservative vs the 20bps round-trip framing). "
        "MECHANISM: retail overpays for lottery-like high-IVOL alts in the young, under-arbitraged crypto "
        "perp corner -> low-IVOL earns, high-IVOL bleeds (a defensive low-vol/lottery premium; NOT trend, "
        "NOT short-term reversal, NOT funding/basis carry -> no funding accrued). "
        "SCOPE=local by limits-to-arbitrage: the IVOL/lottery premium is universal but ARBITRAGED AWAY in "
        "mature equities (our own BAB cross-market FAIL) and survives where shorting memecoins was "
        "historically hard and the marginal trader is retail. "
        "CONSERVATIVE BIAS (explicit): yfinance crypto excludes dead coins (LUNA/FTT/...); the short leg "
        "would have PROFITED from those collapses, so this survivor-only test UNDERSTATES the short leg. "
        "CHECKABLE CLAIMS (soft expectations): (1) net BTC-beta ~0 in-search; (2) MONOTONIC tercile gradient "
        "low>mid>high; (3) positive in BOTH the 2022 bear and 2023 recovery sub-periods. "
        "Holdout 2024-01-01. Forward-paper validate >=40 weekly rebalances / >=2 regimes before any "
        "deployment conviction."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={"default": {}},
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT_START,
    deploy_max_positions=14,
    expectations=[
        {"name": "beta_neutral",
         "claim": "net BTC-beta of the book is ~0 (|beta|<0.15) in-search",
         "check": _check_beta_neutral},
        {"name": "monotonic_ivol_gradient",
         "claim": "low-IVOL tercile > mid > high-IVOL tercile (monotonic lottery gradient) in-search",
         "check": _check_monotonic},
        {"name": "both_subperiods_positive",
         "claim": "net returns positive in BOTH the 2022 bear and 2023 recovery sub-periods in-search",
         "check": _check_subperiods},
    ],
)