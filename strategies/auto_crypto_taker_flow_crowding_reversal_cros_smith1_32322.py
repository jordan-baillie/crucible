"""
Crypto taker-flow crowding short-horizon reversal (cross-sectional).

Mechanism (pre-registered): aggressive taker BUYING crowds price above fair value
over short horizons and mean-reverts.  We have no owned taker/order-flow feed, so we
PROXY crowding with the trailing N-day cross-sectional return (recent winners == names
the taker flow has crowded).  Cross-sectionally LONG recent losers / SHORT recent
winners, weekly rebalance, inverse-vol sized, 15bps round-trip costs (crypto-realistic).

scope='local' on purpose: crypto microstructure (24/7, retail-taker-dominated, no NBBO)
is distinct, so we do NOT assert the equity short-term-reversal literature transfers 1:1.
Forward-validation on the 2022+ holdout is the binding test, not a cross-universe battery.

CAVEAT (in pre_reg): the only crypto price source is yfinance (FREE) which has
survivorship bias (failed coins drop out).  The pre-2022 search window is the least
affected; treat OOS Sharpe as the real test.
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, inv_vol_position
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

ID = "auto_crypto_taker_flow_crowding_reversal_cros_smith1_32322"
START = "2018-01-01"   # by 2018 ~20 liquid pairs have history -> no single-name dominance

# Liquid USD crypto pairs + a coarse "sector" map (the trade ledger needs sectors for
# the cross-sector spread gate).  Names with shorter history just carry NaN early.
CRYPTO_SECTORS = {
    "BTC-USD": "L1",  "ETH-USD": "L1",  "ADA-USD": "L1",  "SOL-USD": "L1",
    "AVAX-USD": "L1", "ALGO-USD": "L1", "NEO-USD": "L1",  "EOS-USD": "L1",
    "ATOM-USD": "L1", "TRX-USD": "L1",  "XTZ-USD": "L1",  "VET-USD": "L1",
    "QTUM-USD": "L1", "ICX-USD": "L1",
    "DOT-USD": "Interop", "LINK-USD": "Oracle",
    "XRP-USD": "Payment", "XLM-USD": "Payment", "LTC-USD": "Payment",
    "BCH-USD": "Payment", "DASH-USD": "Payment", "DGB-USD": "Payment",
    "XMR-USD": "Privacy", "ZEC-USD": "Privacy",
    "MKR-USD": "DeFi", "ZRX-USD": "DeFi", "BAT-USD": "DeFi",
    "OMG-USD": "DeFi", "WAVES-USD": "DeFi", "KNC-USD": "DeFi",
    "DOGE-USD": "Meme",
    "SC-USD": "Storage", "RVN-USD": "Storage", "STORJ-USD": "Storage",
}
TICKERS = list(CRYPTO_SECTORS.keys())

PRE_REG = (
    "Taker-flow crowding reversal: aggressive taker buying crowds price above fair value "
    "over short horizons and mean-reverts. No owned taker/order-flow data -> PROXY crowding "
    "with the trailing 7-day cross-sectional return (winners == recently crowded). "
    "Cross-sectionally LONG recent losers / SHORT recent winners, weekly rebalance, "
    "inverse-vol sized, 15bps round-trip (crypto taker fee + slippage). scope=local: crypto "
    "microstructure is distinct; we forward-validate on the 2022+ holdout rather than claim "
    "cross-universe generalisation. Data = yfinance (FREE), which has crypto survivorship "
    "bias (dead coins drop out); the pre-2022 search window is least affected, OOS Sharpe is "
    "the binding test. Expect: short holding (<=~2wk) and the reversal sign beating the "
    "momentum sign in-sample."
)


def load_data() -> pd.DataFrame:
    # FREE daily close panel for the crypto universe (yfinance; not US single stocks).
    return yf_panel(TICKERS, start=START)


def signal(panel, **params):
    lookback   = int(params.get("lookback", 7))
    vol_lb     = int(params.get("vol_lb", 30))
    target_vol = float(params.get("target_vol", 0.10))
    max_pos    = int(params.get("max_pos", 16))
    momentum   = bool(params.get("momentum", False))   # flag used by the soft expectation only

    rets = panel.pct_change()
    raw  = panel.pct_change(lookback)        # trailing-return crowding proxy (no future data)
    z    = xs_zscore(raw)                     # cross-sectional, winsorized, NaN-preserving
    score = z if momentum else -z             # reversal (default): short crowded winners

    # inv_vol_position returns weekly-held, ALREADY-LAGGED positions -> no extra shift here
    # (the 1-day lag is handled by the kit; net_of_cost consumes the lagged matrix directly).
    W = inv_vol_position(score, rets, target_vol=target_vol, vol_lb=vol_lb,
                         max_pos=max_pos, rebalance="W")

    daily = net_of_cost(W, rets, cost_bps=15.0, name=ID)
    trades = trades_from_weights(W, rets, CRYPTO_SECTORS)
    daily.name = ID
    return daily, trades


def load_gen_data(label) -> pd.DataFrame:
    # scope='local': no cross-universe stage-2 battery. Defined for API completeness.
    return load_data()


# ---- soft expectations (machine-checkable mechanism claims) -------------------------
def _check_short_holding(ctx):
    trades = ctx.get("trades") or []
    if not trades:
        return {"pass": False, "observed": "no_trades"}
    med = float(pd.Series([t["hold_days"] for t in trades]).median())
    return {"pass": med <= 14.0, "observed": med}


def _check_reversal_beats_momentum(ctx):
    # ONE extra signal() call (momentum sign); slice recompute to pre-holdout only.
    h = pd.Timestamp(ctx["holdout_start"])
    mom_ret, _ = signal(ctx["panel"], momentum=True)
    mom_ret = mom_ret[mom_ret.index < h]
    rev = ctx["search"]
    rmean = float(rev.mean()) if len(rev) else 0.0
    mmean = float(mom_ret.mean()) if len(mom_ret) else 0.0
    return {"pass": rmean >= mmean, "observed": f"rev_mean={rmean:.6f}, mom_mean={mmean:.6f}"}


SPEC = StrategySpec(
    id=ID,
    family="crypto_xs_reversal",
    title="Crypto taker-flow crowding short-horizon reversal (cross-sectional)",
    markets=["crypto"],
    data_desc="Daily close for ~34 liquid USD crypto pairs (yfinance, FREE).",
    pre_registration=PRE_REG,
    load_data=load_data,
    signal=signal,
    default_params={"lookback": 7, "vol_lb": 30, "target_vol": 0.10, "max_pos": 16},
    grid={
        "default": {},
        "lb5":  {"lookback": 5},
        "lb10": {"lookback": 10},
        "tv15": {"target_vol": 0.15},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=16,
    expectations=[
        {"name": "short_holding",
         "claim": "median hold_days <= 14 (short-horizon reversal turns over within ~2 weeks)",
         "check": _check_short_holding},
        {"name": "reversal_not_momentum",
         "claim": "in-sample mean return of the reversal sign >= the momentum sign",
         "check": _check_reversal_beats_momentum},
    ],
)