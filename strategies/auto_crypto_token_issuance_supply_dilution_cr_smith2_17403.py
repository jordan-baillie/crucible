"""Net share issuance (supply-dilution) cross-sectional equity factor.

The original auto-generated framing was "crypto token issuance / supply
dilution". We own NO point-in-time crypto token-supply data (only yf close
prices for a handful of coins), so the identical economic mechanism -- supply
dilution lowers per-unit holder returns -- is tested on the domain we DO own
survivorship-clean PIT data for: US-equity net share issuance (Pontiff-Woodgate
/ Daniel-Titman). Firms expanding share count (dilution / issuance) underperform
firms shrinking it (buybacks). This is a broad premium, so stage-2 must
generalise across cap tiers.

Cross-sectional, dollar-neutral, inverse-vol sized, weekly rebalanced, 8bps cost.
Signal lag is handled by inv_vol_position (it returns already-lagged weekly-held
positions), and the share-count signal is point-in-time (datekey-based) -- so no
look-ahead.
"""
import numpy as np, pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1, inv_vol_position
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel

SPEC_ID = "share_issuance_supply_dilution"
START = "2004-01-01"
HOLDOUT = "2022-01-01"
ISSUANCE_LB = 252  # ~12 trading months of share-count change

# Generalization universes: DISJOINT cap tiers (a name sits in exactly one
# Sharadar marketcap bucket, so these share no tickers with the Small search set).
GEN = {
    "microcap": ("Micro", 30),
    "midcap": ("Mid", 30),
    "largecap": ("Large", 28),
}


def _build_panel(marketcap, top_n_per_sector):
    """Sector-spread universe -> {'px': adj close, 'iss': trailing share growth}."""
    tickers, sector_map = sector_universe(marketcap, top_n_per_sector)
    px = sep_panel(tickers, START, field='closeadj').dropna(how='all', axis=1)

    # Point-in-time shares outstanding (datekey-based, ffilled -> no look-ahead).
    f = sf1(list(px.columns), ['sharesbas'], dimension='ARQ')
    shares = pit_panel(f, 'sharesbas', px.index, list(px.columns))
    issuance = np.log(shares) - np.log(shares.shift(ISSUANCE_LB))
    issuance = issuance.replace([np.inf, -np.inf], np.nan)

    panel = pd.concat({'px': px, 'iss': issuance}, axis=1)
    panel.attrs['sector_map'] = {t: sector_map.get(t) for t in px.columns}
    return panel


def load_data() -> pd.DataFrame:
    # Small caps: where the dilution anomaly lives (large/liquid -> arbitraged away).
    return _build_panel('Small', 120)


def load_gen_data(label) -> pd.DataFrame:
    marketcap, n = GEN[label]
    return _build_panel(marketcap, n)


def signal(panel, **params):
    winsor = params.get('winsor', (0.05, 0.95))
    target_vol = params.get('target_vol', 0.10)
    vol_lb = params.get('vol_lb', 63)
    max_pos = params.get('max_pos', 50)

    px = panel['px']
    iss = panel['iss']
    rets = px.pct_change()

    # LOW issuance (buybacks) -> long; HIGH issuance (dilution) -> short.
    sig = xs_zscore(-iss, winsor=winsor)

    # inv_vol_position returns ALREADY-LAGGED weekly-held positions, so we pass W
    # straight to net_of_cost / trades_from_weights (no extra shift) -- no look-ahead.
    W = inv_vol_position(sig, rets, target_vol=target_vol, vol_lb=vol_lb,
                         max_pos=max_pos, rebalance='W')

    daily = net_of_cost(W, rets, cost_bps=8.0, name=SPEC_ID)
    sector_map = panel.attrs.get('sector_map', {})
    trades = trades_from_weights(W, rets, sector_map)
    return daily, trades


def _dilution_spread(ctx):
    """Mechanism check (in-sample only): the least-dilution names should out-earn
    the most-dilution names -> next-day long-short quintile spread > 0."""
    panel = ctx["panel"]
    hs = pd.Timestamp(ctx["holdout_start"])
    px, iss = panel["px"], panel["iss"]

    in_idx = px.index[px.index < hs]
    fwd_s = px.loc[in_idx].pct_change().shift(-1)   # next-day return, in-sample only
    iss_s = iss.loc[in_idx]

    ranks = iss_s.rank(axis=1, pct=True)
    lo = fwd_s.where(ranks <= 0.2).mean(axis=1)   # least dilution / buybacks
    hi = fwd_s.where(ranks >= 0.8).mean(axis=1)   # most dilution / issuance
    spread = float((lo - hi).mean())
    return {"pass": bool(spread > 0), "observed": spread}


GRID = {
    "default": {},
    "fast_vol": {"vol_lb": 42},
    "tight_book": {"max_pos": 40},
    "wide_winsor": {"winsor": (0.02, 0.98)},
}

SPEC = StrategySpec(
    id=SPEC_ID,
    family="issuance",
    title="Net Share Issuance (Supply Dilution) Cross-Sectional Equity Factor",
    markets=["US equities"],
    data_desc=("Sharadar SEP closeadj + SF1 sharesbas (PIT, datekey, ARQ); "
               "trailing-252d log share-count change as the dilution signal."),
    pre_registration=(
        "Supply dilution lowers per-unit holder returns. Firms expanding shares "
        "outstanding over the trailing ~12 months (dilution / issuance) should "
        "underperform firms shrinking share count (buybacks). Dollar-neutral, "
        "inverse-vol sized, weekly rebalanced, 8bps cost, signals lagged. Broad "
        "premium -> must generalise across cap tiers (micro / mid / large). The "
        "crypto-token framing maps here because no owned PIT crypto-supply data "
        "exists; the dilution mechanism is identical and is tested on "
        "survivorship-clean equity share counts."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid=GRID,
    scope='broad',
    generalization_universes=list(GEN.keys()),
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT,
    deploy_max_positions=50,
    expectations=[{
        "name": "dilution_predicts_underperformance",
        "claim": ("in-sample, least-issuance quintile next-day return exceeds "
                  "most-issuance quintile (long-short spread > 0)"),
        "check": _dilution_spread,
    }],
)