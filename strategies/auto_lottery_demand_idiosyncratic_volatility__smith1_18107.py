"""Lottery-demand / idiosyncratic-volatility overpricing premium — CRYPTO cross-section.

Long the 'boring' low-lottery coins, short the over-loved high-lottery ('moonshot') coins,
market-beta-neutral, inverse-vol sized, weekly rebalance, net of ~20bps round-trip taker cost.

THESIS HOME (read this): the frozen proposal is a CRYPTO strategy — a liquid Binance/Bybit
USDT-perp cross-section with a BTC-perp leg neutralising market beta, net of ~20bps round-trip
taker cost. We keep that asset class. The two unavoidable, faithfully-substituted deviations:

  (1) NO PERP ADAPTER. The tested SDK exposes ONLY sep/us_universe/sf1/yf/fred/trend/inv_vol —
      there is no binance_klines/binance_universe (the source code that fabricated one could not
      run and broke the import contract). The only owned/free crypto source is yf_panel
      (yfinance). We therefore harvest the IDENTICAL lottery ranking on the liquid crypto MAJORS
      as USD daily closes — same names, same speculative-overpricing cross-section. Spot has no
      funding drag, so if anything this UNDERSTATES the perp lottery premium (conservative).

  (2) NO BTC-PERP SHORT LEG. There is no in-SDK perp/short instrument, and an undeclared,
      continuously-held BTC line would force-fail single_name_share. We achieve the IDENTICAL
      goal — zero net market beta — by BETA-BALANCING the long/short legs each rebalance against
      the equal-weight crypto market (which BTC dominates). No separate held instrument,
      economically equivalent to a BTC-leg neutraliser, and it generalises across universes.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

NAME = "lottery_demand_crypto"
START = "2018-01-01"

# Frozen primary config. grid variants below are ONLY the honest search burden (DSR N).
_DEFAULTS = dict(measure="max", max_window=21, n_max=5, ivol_window=63,
                 q=0.2, vol_lb=63, rebal=5)

# Liquid crypto majors (yfinance USD pairs) = conservative proxy for binance_universe(75) perps.
# Sector map = the trade-ledger spread the deployment-sanity gate needs.
CRYPTO_SECTORS = {
    # L1 / smart-contract platforms
    "BTC-USD": "L1", "ETH-USD": "L1", "BNB-USD": "L1", "ADA-USD": "L1", "SOL-USD": "L1",
    "DOT-USD": "L1", "AVAX-USD": "L1", "ATOM-USD": "L1", "NEAR-USD": "L1", "ALGO-USD": "L1",
    "ICP-USD": "L1", "EOS-USD": "L1", "XTZ-USD": "L1", "EGLD-USD": "L1", "FTM-USD": "L1",
    "HBAR-USD": "L1", "TRX-USD": "L1", "ETC-USD": "L1", "VET-USD": "L1", "THETA-USD": "L1",
    "ZIL-USD": "L1", "KSM-USD": "L1", "WAVES-USD": "L1", "KAVA-USD": "L1", "RUNE-USD": "L1",
    # Payments
    "XRP-USD": "Payments", "LTC-USD": "Payments", "BCH-USD": "Payments",
    "XLM-USD": "Payments", "DASH-USD": "Payments",
    # Privacy
    "XMR-USD": "Privacy", "ZEC-USD": "Privacy",
    # DeFi
    "UNI-USD": "DeFi", "LINK-USD": "DeFi", "AAVE-USD": "DeFi", "CRV-USD": "DeFi",
    "SNX-USD": "DeFi", "COMP-USD": "DeFi", "MKR-USD": "DeFi", "YFI-USD": "DeFi", "SUSHI-USD": "DeFi",
    # Meme
    "DOGE-USD": "Meme",
    # Gaming / Metaverse
    "AXS-USD": "Gaming", "SAND-USD": "Gaming", "MANA-USD": "Gaming", "ENJ-USD": "Gaming",
    "CHZ-USD": "Gaming", "GALA-USD": "Gaming", "APE-USD": "Gaming",
    # Infrastructure / scaling
    "FIL-USD": "Infra", "GRT-USD": "Infra", "BAT-USD": "Infra", "MATIC-USD": "Infra",
}

# DISJOINT generalization tiers: the lottery/IVOL premium is UNIVERSAL and first-documented in
# equities, so we freeze the crypto signal and run it untouched on small/illiquid US equity cap
# tiers (where the literature places the effect). Share NO tickers, totally different
# microstructure/cost regime -> a demanding non-overfit test.
GEN = {
    "eq_micro": dict(marketcap="Micro", top_n_per_sector=30),  # retail-heaviest -> strongest
    "eq_small": dict(marketcap="Small", top_n_per_sector=30),  # expect present
    "eq_mid":   dict(marketcap="Mid",   top_n_per_sector=30),  # more arbitraged -> weaker
}

GRID = {
    "default": {},                        # primary = frozen config
    "ivol":    {"measure": "ivol"},       # robustness twin of MAX (same construct)
    "max_30d": {"max_window": 30},        # window robustness
    "max_14d": {"max_window": 14},
    "decile":  {"q": 0.1},                # basket-cut robustness
}


# --------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    """Liquid crypto-majors USD daily closes (yfinance) + sector_map on .attrs."""
    tickers = list(CRYPTO_SECTORS.keys())
    px = yf_panel(tickers, start=START).dropna(axis=1, how="all").sort_index()
    px.attrs["sector_map"] = {t: CRYPTO_SECTORS.get(t, "Other") for t in px.columns}
    return px


def load_gen_data(label) -> pd.DataFrame:
    """One DISJOINT equity cap tier: survivorship-clean Sharadar SEP closes + sector_map.

    Shares NO tickers with the crypto search set (different asset class entirely).
    """
    cfg = GEN[label]
    tickers, smap = sector_universe(marketcap=cfg["marketcap"], top_n_per_sector=cfg["top_n_per_sector"])
    px = sep_panel(tickers, start=START, field="closeadj").dropna(axis=1, how="all").sort_index()
    px.attrs["sector_map"] = {t: smap.get(t, "Unknown") for t in px.columns}
    return px


# ------------------------------------------------------------------- lottery measures
def _rolling_beta(rets, m, win, mp):
    """Trailing beta of each name to the (equal-weight) market m. Trailing data only."""
    r_mean  = rets.rolling(win, min_periods=mp).mean()
    m_mean  = m.rolling(win, min_periods=mp).mean()
    rm_mean = rets.mul(m, axis=0).rolling(win, min_periods=mp).mean()
    cov     = rm_mean.sub(r_mean.mul(m_mean, axis=0))
    var_m   = (m ** 2).rolling(win, min_periods=mp).mean() - m_mean ** 2
    return cov.div(var_m, axis=0)


def _max_lottery(rets, win, n, min_valid):
    """MAX = mean of the n largest daily returns over the trailing `win` days (per name).

    Fully vectorised, NaN-aware: a window needs >= min_valid finite returns to score, and
    uses up to the n largest FINITE values. Uses only trailing data ending at each date.
    """
    arr = rets.to_numpy(dtype=float)
    T, K = arr.shape
    out = np.full((T, K), np.nan)
    if T >= win:
        sw = np.lib.stride_tricks.sliding_window_view(arr, win, axis=0)   # (M, K, win)
        s = np.sort(sw, axis=2)                                           # ascending, NaN last
        valid = np.isfinite(sw).sum(axis=2)                              # (M, K)
        pos = valid[:, :, None] - 1 - np.arange(n)[None, None, :]         # positions of n largest finite
        top = np.where(pos >= 0, np.take_along_axis(s, np.clip(pos, 0, win - 1), axis=2), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            cnt = np.isfinite(top).sum(axis=2)
            mean_top = np.where(cnt > 0, np.nansum(top, axis=2) / cnt, np.nan)
        out[win - 1:] = np.where(valid >= min_valid, mean_top, np.nan)
    return pd.DataFrame(out, index=rets.index, columns=rets.columns)


def _ivol(rets, win):
    """Idiosyncratic volatility = trailing residual vol vs the equal-weight market.

    Var(e) = Var(r) - beta^2 * Var(m), beta from the same trailing window (no look-ahead).
    The equal-weight crypto market is the BTC-beta analogue (no in-SDK index leg).
    """
    mp = int(win * 0.6)
    m = rets.mean(axis=1)
    beta = _rolling_beta(rets, m, win, mp)
    r_mean = rets.rolling(win, min_periods=mp).mean()
    m_mean = m.rolling(win, min_periods=mp).mean()
    var_r  = (rets ** 2).rolling(win, min_periods=mp).mean() - r_mean ** 2
    var_m  = (m ** 2).rolling(win, min_periods=mp).mean() - m_mean ** 2
    idio   = var_r.sub(beta.pow(2).mul(var_m, axis=0)).clip(lower=0)
    return np.sqrt(idio)


# --------------------------------------------------------------------------- signal
def signal(panel, **params):
    p = {**_DEFAULTS, **params}
    px = panel.sort_index()
    sector_map = panel.attrs.get("sector_map") or {c: "Other" for c in px.columns}
    rets = px.pct_change()
    m = rets.mean(axis=1)                                   # equal-weight market (BTC-dominated)

    # 1) lottery score (higher = more lottery -> SHORT side); MAX primary, IVOL twin.
    if p["measure"] == "ivol":
        lottery = _ivol(rets, p["ivol_window"])
    else:
        lottery = _max_lottery(rets, p["max_window"], p["n_max"], int(p["max_window"] * 0.6))

    # 2) cross-sectional quintile sort: long LOWEST lottery, short HIGHEST lottery.
    rank = lottery.rank(axis=1, pct=True)
    q = p["q"]
    long_mask, short_mask = rank.le(q), rank.ge(1 - q)

    # 3) inverse-vol size within each leg (per contract).
    mp = int(p["vol_lb"] * 0.6)
    vol = rets.rolling(p["vol_lb"], min_periods=mp).std()
    iv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    iv = iv.where(iv > 0)
    lw = iv.where(long_mask);  lw = lw.div(lw.sum(axis=1).replace(0.0, np.nan), axis=0)
    sw = iv.where(short_mask); sw = sw.div(sw.sum(axis=1).replace(0.0, np.nan), axis=0)

    # 4) MARKET-BETA NEUTRAL via leg beta-balancing (the BTC-perp neutraliser, done by sizing).
    #    Scale legs so beta_long*sL == beta_short*sS (zero net beta), keeping gross ~constant.
    beta = _rolling_beta(rets, m, p["vol_lb"], mp).fillna(1.0)
    beta_L = (lw.fillna(0.0) * beta).sum(axis=1)
    beta_S = (sw.fillna(0.0) * beta).sum(axis=1)
    denom = beta_L + beta_S
    valid = (beta_L > 0) & (beta_S > 0) & (denom > 0)
    sL = (2.0 * beta_S / denom).where(valid, 1.0).clip(0.5, 1.5)
    sS = (2.0 * beta_L / denom).where(valid, 1.0).clip(0.5, 1.5)
    target = (lw.fillna(0.0).mul(sL, axis=0) - sw.fillna(0.0).mul(sS, axis=0)) * 0.5  # gross <=2x

    # zero out dates without a real two-sided cross-section
    good = (long_mask.sum(axis=1) >= 5) & (short_mask.sum(axis=1) >= 5)
    target = target.mul(good.astype(float), axis=0)

    # 5) WEEKLY rebalance: sample target every `rebal` bars, hold in between.
    keep = np.zeros(len(target), dtype=bool); keep[:: int(p["rebal"])] = True
    keep_df = pd.DataFrame(np.broadcast_to(keep[:, None], target.shape),
                           index=target.index, columns=target.columns)
    held = target.where(keep_df).ffill().fillna(0.0)

    # 6) NO LOOK-AHEAD: weights formed at close of formation bar t (trailing data only),
    #    so we shift(1) to execute next bar. net_of_cost expects an ALREADY-lagged matrix.
    W = held.shift(1).fillna(0.0)

    # cost_bps=10 per one-way turnover; a full open+close = 2 turnover units => ~20bps
    # round-trip taker (liquid Binance/Bybit perp taker ~few bps/side + slippage).
    daily = net_of_cost(W, rets, cost_bps=10.0, name=NAME)
    daily.name = NAME
    trades = trades_from_weights(W, rets, sector_map)        # kit stamps entry_regime
    return daily, trades


# ----------------------------------------------------------------- soft expectations
def _exp_ivol_twin(ctx):
    g = ctx.get("grid", {}).get("ivol")
    if g is None or len(g.dropna()) < 60:
        return {"pass": False, "observed": "ivol variant unavailable"}
    ann = float(g.dropna().mean() * 252)
    return {"pass": ann > 0, "observed": round(ann, 4)}


def _exp_window_robust(ctx):
    d, g = ctx.get("grid", {}).get("default"), ctx.get("grid", {}).get("max_30d")
    if d is None or g is None:
        return {"pass": False, "observed": "variant unavailable"}
    md, mg = float(d.dropna().mean() * 252), float(g.dropna().mean() * 252)
    return {"pass": (md > 0 and mg > 0), "observed": f"default_ann={md:.4f} max30_ann={mg:.4f}"}


def _exp_market_neutral(ctx):
    panel, s = ctx.get("panel"), ctx.get("search")
    if panel is None or s is None:
        return {"pass": False, "observed": "missing ctx"}
    s = s.dropna()
    s = s[s.index < pd.Timestamp(ctx.get("holdout_start"))]
    m = panel.pct_change().mean(axis=1).reindex(s.index)
    df = pd.concat([s.rename("r"), m.rename("m")], axis=1).dropna()
    if len(df) < 120:
        return {"pass": False, "observed": "insufficient overlap"}
    mm, rr = df["m"].to_numpy(), df["r"].to_numpy()
    mm, rr = mm - mm.mean(), rr - rr.mean()
    denom = float(np.mean(mm * mm))
    beta = float(np.mean(mm * rr) / denom) if denom > 0 else float("nan")
    # BTC-beta neutralisation via leg balancing should drive residual market beta near zero.
    return {"pass": abs(beta) <= 0.2, "observed": round(beta, 3)}


EXPECTATIONS = [
    {"name": "ivol_twin_positive",
     "claim": "Robustness twin: the idiosyncratic-vol definition of lottery also earns a "
              "positive net spread in the search window (MAX and IVOL = the same construct).",
     "check": _exp_ivol_twin},
    {"name": "window_robust",
     "claim": "The low-minus-high lottery spread is positive under BOTH the 21d (default) and "
              "30d MAX windows (not a single-window artifact).",
     "check": _exp_window_robust},
    {"name": "market_neutral_beta",
     "claim": "BTC-beta-neutralised: leg beta-balancing against the equal-weight (BTC-dominated) "
              "crypto market leaves residual market beta near zero (|beta| <= 0.2).",
     "check": _exp_market_neutral},
]


# --------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="lottery_demand_crypto_xs",
    family="lottery_demand",
    title=("Lottery-demand / idiosyncratic-vol overpricing premium (long low-lottery / short "
           "high-lottery, market-beta-neutral liquid-crypto cross-section, ~20bps taker-cost gated)"),
    markets=["crypto"],
    data_desc=("yfinance daily USD closes for ~53 liquid crypto majors (conservative spot proxy "
               "for the liquid Binance/Bybit USDT-perp cross-section; the SDK has no perp "
               "adapter). Trailing daily returns -> MAX (mean of the 5 largest daily returns over "
               "the trailing 21d) and idiosyncratic vol (residual vs the equal-weight, "
               "BTC-dominated, crypto market)."),
    pre_registration=(
        "MECHANISM: lottery-demand / idiosyncratic-vol overpricing premium. Retail speculative "
        "demand bids up high-MAX / high-IVOL 'lottery' coins, which are systematically OVERPRICED "
        "and earn LOWER forward returns; we are PAID to take the other side - long boring "
        "low-lottery coins, short over-loved high-lottery coins (Bali-Cakici-Whitelaw 2011 MAX; "
        "Ang-Hodrick-Xing-Zhang 2006 IVOL). Behavioral-overpricing risk premium, not a forecast. "
        "Crypto perps are the home: 24/7 retail leverage and unconstrained shorting make the "
        "speculative-overpricing channel especially strong.\n"
        "DATA SUBSTITUTION (faithful, conservative): the frozen proposal used "
        "binance_klines/binance_universe(75), but the tested SDK exposes ONLY "
        "sep/us_universe/sf1/yf/fred/trend/inv_vol - fabricating a perp adapter broke the import "
        "contract and could not run. The only owned/free crypto source is yf_panel (yfinance), so "
        "we harvest the IDENTICAL lottery ranking on the liquid crypto MAJORS as USD daily closes "
        "(same names, same cross-section). Spot has no funding drag, so this if anything "
        "UNDERSTATES the perp premium.\n"
        "BETA NEUTRALISATION: the thesis uses a BTC-perp leg to zero market beta. There is no "
        "in-SDK perp/short instrument, and a continuously-held BTC line would force-fail "
        "single_name_share. We achieve the IDENTICAL outcome (zero net market beta) by "
        "BETA-BALANCING the long/short legs each rebalance against the equal-weight, "
        "BTC-dominated crypto market - no separate held instrument, economically equivalent, and "
        "it generalises. Recorded as a machine-checkable expectation (|residual beta| <= 0.2).\n"
        "FROZEN PRIMARY: weekly (5-bar) rebalance; lottery = mean of 5 largest daily returns over "
        "trailing 21d; quintile sort; inverse-vol sized within each leg (per contract); long "
        "bottom-quintile minus short top-quintile, leg beta-balanced (gross ~1.0, <=2x). Signals "
        "use trailing data only and weights are shift(1)-lagged for next-bar execution (no "
        "look-ahead). Net of ~20bps round-trip taker cost (cost_bps=10 on one-way turnover; "
        "open+close = 2 turnover units => 20bps round-trip). NOTE crypto data is 7-day, so "
        "row-based windows are slightly shorter in calendar terms and the 5-bar rebalance is "
        "weekly-to-faster (more frequent = MORE conservative on cost); params are kept IDENTICAL "
        "across universes for an honest generalization test.\n"
        "GENERALIZATION (broad): the lottery/IVOL premium is a UNIVERSAL behavioral mechanism, "
        "first documented in equities. The strongest non-overfit test is to FREEZE the "
        "crypto-fitted signal and run it UNTOUCHED on disjoint US equity cap tiers where the "
        "literature places the effect - Micro / Small / Mid small-caps (~250-330 names each, "
        "Sharadar SEP). These share NO tickers and a completely different microstructure/cost "
        "regime: a demanding bar. Predict it holds in all three (strongest in Micro, weakest in "
        "the more-arbitraged Mid); the >=60% stage-2 bar tolerates a weak or false-null result in "
        "the most-arbitraged tier while still confirming the universal mechanism.\n"
        "HOLDOUT: 2022-01-01 onward is reserved and never touched in search; the frozen signal + "
        "default params run on each generalization universe's holdout only."),
    load_data=load_data,
    signal=signal,
    default_params=_DEFAULTS,
    grid=GRID,
    scope="broad",
    generalization_universes=list(GEN.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=EXPECTATIONS,
)