"""cross_sectional_taker_flow_crowding_reversal

PRE-REGISTRATION / DESIGN NOTE
------------------------------
Original concept ("taker-flow crowding reversal") relied on aggressive order-flow
imbalance (crypto taker buy/sell volume). That data is NOT in OWNED/FREE inventory
(no binance_klines/deribit/orderflow adapter exists; the failed module hallucinated a
`field=` kwarg on a non-existent adapter). Rather than invent unverifiable data, this
implements the UNIVERSAL mechanism that taker-flow crowding proxies: short-horizon
*crowding reversal* driven by the liquidity-provision premium (Lehmann 1990; Jegadeesh
1990). Crowded short-term buying pressure reverses; we go long prior-week losers / short
prior-week winners, cross-sectional, dollar-neutral, inverse-vol sized, weekly rebalance.

Because liquidity-provision reversal is a market-universal effect, scope='broad': a
stage-1 pass must GENERALISE to untouched cap tiers (micro/mid/large) on holdout.

Falsifiable mechanism claim (machine-checked below): reversal is a SHORT-horizon effect,
so a 5-day formation must out-Sharpe a 21-day formation (the latter drifts toward
momentum). Search universe = small caps, where the effect is least arbitraged.
"""
from sdk.harness import StrategySpec
from sdk.adapters import sep_panel
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

SEARCH_START = "2010-01-01"
_SECTOR_MAP: dict = {}

# search universe (small caps) + the disjoint generalization tiers
_GEN_SPECS = {"micro": ("Micro", 35), "mid": ("Mid", 35), "large": ("Large", 35)}


def _build(marketcap: str, top_n_per_sector: int) -> pd.DataFrame:
    tickers, smap = sector_universe(marketcap=marketcap, top_n_per_sector=top_n_per_sector)
    _SECTOR_MAP.update(smap)
    return sep_panel(tickers, SEARCH_START, field="closeadj")


def load_data() -> pd.DataFrame:
    # ~1000 most-sector-spread small caps (reversal lives in less-liquid names)
    return _build("Small", 100)


def load_gen_data(label: str) -> pd.DataFrame:
    # disjoint cap tiers (no ticker overlap with Small) — each ~350 names, runs same-night
    mc, tn = _GEN_SPECS[label]
    return _build(mc, tn)


def _sector_map_for(cols) -> dict:
    return {c: _SECTOR_MAP.get(c, "Unknown") for c in cols}


def signal(panel, lookback=5, vol_lb=63, **params):
    panel = panel.sort_index()
    # fill_method=None: never forward-fill gaps (that would manufacture lookahead returns)
    rets = panel.pct_change(fill_method=None)
    formation = panel.pct_change(lookback, fill_method=None)

    # crowding reversal: long prior-window LOSERS, short prior-window WINNERS
    raw = -xs_zscore(formation)

    # inverse-vol sizing (trailing realised vol)
    iv = 1.0 / rets.rolling(vol_lb).std().replace(0.0, np.nan)
    sig = raw * iv

    # dollar-neutral, gross-1 cross-sectional weights (NaN-preserving)
    sig = sig.sub(sig.mean(axis=1), axis=0)
    W = sig.div(sig.abs().sum(axis=1).replace(0.0, np.nan), axis=0)

    # weekly rebalance: take weights on every 5th trading day, hold (ffill) until next.
    # Each rebalance uses data only up to that day's close -> no within-week lookahead.
    W_held = W.iloc[::5].reindex(W.index, method="ffill")

    # weights are same-day (built from close t); lag 1 day for execution -> no lookahead.
    W_lag = W_held.shift(1)

    daily = net_of_cost(W_lag, rets, cost_bps=8.0, name="cs_crowding_reversal")
    trades = trades_from_weights(W_lag, rets, _sector_map_for(panel.columns))
    return daily, trades


# ----- soft expectation: reversal is a SHORT-horizon effect (free, uses declared grid) -----
def _sharpe(r) -> float:
    if r is None:
        return 0.0
    r = pd.Series(r).dropna()
    if len(r) < 20 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _check_short_horizon(ctx) -> dict:
    g = ctx["grid"]
    s_fast = _sharpe(g.get("default"))  # 5d formation
    s_slow = _sharpe(g.get("vslow"))    # 21d formation (drifts to momentum)
    return {"pass": bool(s_fast > s_slow),
            "observed": f"sharpe_5d={s_fast:.2f} vs sharpe_21d={s_slow:.2f}"}


SPEC = StrategySpec(
    id="cross_sectional_taker_flow_crowding_reversal",
    family="reversal",
    title="Cross-sectional short-horizon crowding reversal (liquidity-provision premium)",
    markets=["US equities"],
    data_desc="Sharadar SEP closeadj (survivorship-clean, split/div adj); sector-spread "
              "small-cap universe via sector_universe; gen tiers micro/mid/large.",
    pre_registration=__doc__,
    load_data=load_data,
    signal=signal,
    default_params={"lookback": 5, "vol_lb": 63},
    grid={
        "default": {},                # 5d reversal (primary)
        "fast": {"lookback": 3},
        "slow": {"lookback": 10},
        "vslow": {"lookback": 21},    # used by the short-horizon expectation
    },
    scope="broad",
    generalization_universes=["micro", "mid", "large"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=40,
    expectations=[
        {
            "name": "reversal_is_short_horizon",
            "claim": "5-day formation reversal out-Sharpes 21-day formation in-sample "
                     "(the effect is short-horizon liquidity provision, not momentum).",
            "check": _check_short_horizon,
        },
    ],
)