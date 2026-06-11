"""
crypto_xs_momentum — Crypto Cross-Sectional Momentum (market-neutral relative-strength premium)
================================================================================================
Liu–Tsyvinski–Wu style crypto momentum factor on a DYNAMIC, POINT-IN-TIME universe drawn from
a broad ~60-name candidate pool that includes dead/collapsed coins (LUNA1 = Terra classic,
FTT = FTX token, WAVES, OMG, ...). Membership at each date is fully MECHANICAL and trailing:
a coin is tradable iff it has a complete trailing price history (momentum window + vol window
of actual closes). Coins enter only once they have been listed long enough (no hindsight
selection of SOL/AVAX/SHIB at inception) and fall out mechanically when they die and prices
stop printing. NOTE: the free yfinance adapter provides CLOSES ONLY (no volume field), so the
membership rule is data-availability-based rather than dollar-volume-ranked — stated openly,
not hidden; the pool itself is the broadest free set including the losers.

Within the point-in-time universe: rank on past 28-day cumulative return SKIPPING the most
recent 7 days, long the top tercile / short the bottom tercile, inverse-vol weighted within
each leg, scaled DOLLAR-NEUTRAL (long gross = short gross = 0.5 -> total gross 1.0x).
Dollar-neutrality is the explicit defense against the long-only crypto-beta confound.

Data: yfinance daily closes (the FREE crypto source in DATA_CATALOG; yf_panel is the approved
free adapter for non-US-single-stock assets). Spot closes proxy perp prices (stated).

Costs: net_of_cost at 10 bps on turnover (perp taker round-turn ~7–10 bps). Funding is NOT
in the free data; on a dollar-neutral book the long leg PAYS and the short leg RECEIVES
funding of the same sign in expectation, so the omission roughly nets out (stated).

Scope: LOCAL by design — equity XS momentum is dead for us (arbitraged); the defensible claim
is crypto-specific underreaction sustained by limits-to-arbitrage. The gen universes below are
within-corner DISJOINT pool sub-slices (no shared tickers) used as a no-cherry-pick
consistency check, NOT a universal-mechanism claim; confirmation comes from forward-paper.

NO look-ahead: universe membership, signal and vol all use trailing data only; weights are
shift(1)-lagged before net_of_cost / trades_from_weights (the lag is taken HERE, explicitly).
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2018-01-01"

# Broad candidate POOL (~60 names) with crypto sub-sector buckets for the trade-ledger spread
# gate. Includes dead/collapsed/faded coins (LUNA1, FTT, WAVES, OMG, XEM, BTG, ...) so the
# point-in-time availability filter — not survivorship — decides who is in the book each date.
CRYPTO_SECTORS = {
    "BTC-USD": "major",    "ETH-USD": "major",
    "BNB-USD": "exchange", "FTT-USD": "exchange", "CRO-USD": "exchange",
    "OKB-USD": "exchange", "LEO-USD": "exchange",
    "SOL-USD": "layer1",   "ADA-USD": "layer1",   "AVAX-USD": "layer1",
    "DOT-USD": "layer1",   "ATOM-USD": "layer1",  "NEAR-USD": "layer1",
    "ALGO-USD": "layer1",  "EOS-USD": "layer1",   "TRX-USD": "layer1",
    "ETC-USD": "layer1",   "XTZ-USD": "layer1",   "LUNA1-USD": "layer1",
    "FTM-USD": "layer1",   "EGLD-USD": "layer1",  "ICP-USD": "layer1",
    "HBAR-USD": "layer1",  "WAVES-USD": "layer1", "NEO-USD": "layer1",
    "QTUM-USD": "layer1",  "ICX-USD": "layer1",   "ONT-USD": "layer1",
    "ZIL-USD": "layer1",   "KSM-USD": "layer1",
    "MATIC-USD": "layer2", "LRC-USD": "layer2",   "OMG-USD": "layer2",
    "XRP-USD": "payments", "LTC-USD": "payments", "BCH-USD": "payments",
    "XLM-USD": "payments", "DASH-USD": "payments","ZEC-USD": "payments",
    "XMR-USD": "payments", "XEM-USD": "payments", "BTG-USD": "payments",
    "DCR-USD": "payments", "RVN-USD": "payments",
    "UNI-USD": "defi",     "AAVE-USD": "defi",    "MKR-USD": "defi",
    "COMP-USD": "defi",    "SNX-USD": "defi",     "CRV-USD": "defi",
    "SUSHI-USD": "defi",   "YFI-USD": "defi",     "1INCH-USD": "defi",
    "CAKE-USD": "defi",    "RUNE-USD": "defi",    "ZRX-USD": "defi",
    "LINK-USD": "infra",   "FIL-USD": "infra",    "VET-USD": "infra",
    "THETA-USD": "infra",  "GRT-USD": "infra",    "BAT-USD": "infra",
    "STORJ-USD": "infra",  "ANKR-USD": "infra",
    "MANA-USD": "metaverse", "SAND-USD": "metaverse", "AXS-USD": "metaverse",
    "ENJ-USD": "metaverse",  "GALA-USD": "metaverse", "CHZ-USD": "metaverse",
    "DOGE-USD": "meme",    "SHIB-USD": "meme",
}

_ALL = sorted(CRYPTO_SECTORS)

# Within-corner consistency slices: pairwise-DISJOINT thirds of the candidate POOL (no shared
# tickers across slices). LOCAL scope -> robustness checks, not a broad claim. The same
# point-in-time availability filter runs inside each slice.
GEN_SLICES = {
    "crypto_slice_a": _ALL[0::3],
    "crypto_slice_b": _ALL[1::3],
    "crypto_slice_c": _ALL[2::3],
}


def load_data() -> pd.DataFrame:
    """Daily close panel for the full candidate pool (free yfinance; losers included)."""
    return yf_panel(_ALL, start=START)


def load_gen_data(label: str) -> pd.DataFrame:
    """Close panel for one disjoint within-corner pool sub-slice (same shape as load_data())."""
    return yf_panel(GEN_SLICES[label], start=START)


def signal(panel, lookback=28, skip=7, frac=1.0 / 3.0, vol_lb=30,
           min_coins=10, cost_bps=10.0, rebalance="W-FRI"):
    """
    Point-in-time universe: a coin is tradable on date t iff its full trailing momentum
    window (lookback+skip closes) AND trailing vol window (vol_lb returns) actually exist
    (trailing data only — newly-listed coins enter only after seasoning; dead coins exit
    when prices stop). Within it: rank on (P[t-skip]/P[t-skip-lookback] - 1), long top
    `frac` / short bottom `frac`, inverse-vol within leg, dollar-neutral, weekly hold,
    10bps costs. All inputs trailing; final weights shift(1)-lagged below.
    """
    close = panel.sort_index()
    rets = close.pct_change(fill_method=None)

    # --- the signal: 4-week momentum skipping the most recent week (all trailing data) ---
    mom = close.shift(skip).pct_change(lookback, fill_method=None)

    # Trailing vol for inverse-vol sizing (and as a data-sufficiency screen).
    vol = rets.rolling(vol_lb, min_periods=vol_lb).std()

    # POINT-IN-TIME membership: complete trailing data, mechanically determined per date.
    valid = mom.notna() & vol.notna() & (vol > 0) & close.notna()
    mom = mom.where(valid)

    # Cross-sectional percentile rank per date WITHIN the point-in-time universe; only trade
    # dates with enough breadth.
    n = valid.sum(axis=1)
    enough = n >= min_coins
    r = mom.rank(axis=1, pct=True)

    long_mask = (r >= 1.0 - frac) & valid
    short_mask = (r <= frac) & valid

    iv = (1.0 / vol).where(valid, 0.0)

    wl = iv.where(long_mask, 0.0)
    wl = wl.div(wl.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) * 0.5

    ws = iv.where(short_mask, 0.0)
    ws = ws.div(ws.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) * -0.5

    w_daily = (wl + ws).where(enough, other=0.0).fillna(0.0)

    # Weekly rebalance: snapshot weights (and thus the point-in-time universe) at each
    # rebalance date, hold through the week.
    w_reb = w_daily.resample(rebalance).last()
    W = w_reb.reindex(rets.index, method="ffill").fillna(0.0)

    # --- THE LAG: weights built from data through t are traded from t+1 ---
    W = W.shift(1).fillna(0.0)

    daily = net_of_cost(W, rets, cost_bps=cost_bps, name="crypto_xs_momentum")
    trades = trades_from_weights(W, rets, CRYPTO_SECTORS)
    return daily, trades


SPEC = StrategySpec(
    id="crypto_xs_momentum",
    family="momentum",
    title="Crypto Cross-Sectional Momentum — market-neutral relative-strength premium",
    markets=["crypto"],
    data_desc=(
        "yfinance daily closes over a ~60-coin candidate pool incl. dead/collapsed names "
        "(LUNA1, FTT, WAVES, OMG, ...); DYNAMIC point-in-time universe = coins with complete "
        "trailing momentum + vol windows on each date, rebuilt mechanically at each weekly "
        "rebalance (survivorship-aware: trailing data availability, not hindsight, selects "
        "membership; yfinance free adapter provides closes only, so no volume ranking — "
        "stated); spot closes proxy perp prices; 10bps turnover cost (perp taker); funding "
        "omitted (nets across dollar-neutral legs in expectation, stated in module)."
    ),
    pre_registration=(
        "FROZEN single primary config: point-in-time universe = pool coins with a complete "
        "trailing data window on each date (mechanical, trailing only, from a broad pool "
        "incl. losers); within it, rank on past 28d cumulative return skipping the most "
        "recent 7d (orthogonal to the tested-and-failed short-term XS reversal); long top "
        "tercile / short bottom tercile; inverse-vol within leg; dollar-neutral (0.5/0.5 "
        "gross, 1.0x total) to kill the crypto-beta confound; weekly rebalance; 10bps on "
        "turnover; weights lagged 1 day. Hypothesis: behavioral underreaction momentum "
        "survives in crypto (young, retail-dominated, limits-to-arbitrage) even though it is "
        "arbitraged away in equities — hence scope=LOCAL; gen slices are disjoint "
        "within-pool consistency checks only; confirmation is forward-paper, not "
        "cross-asset. Grid holds only pre-declared lookback (21/56), quartile-cut and "
        "breadth-floor (min 15) variants for the honest DSR effective-N; 'default' is the "
        "verdict config — no cherry-picking."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "lb21": {"lookback": 21},
        "lb56": {"lookback": 56},
        "quartile": {"frac": 0.25},
        "minc15": {"min_coins": 15},
    },
    scope="local",
    generalization_universes=list(GEN_SLICES),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
)