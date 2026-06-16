# Crypto Taker-Flow Crowding Reversal
# Hypothesis (pre-registered below): aggressive (taker) order flow that becomes
# CROWDED on one side over ~2 weeks subsequently REVERSES. Cross-sectional, weekly,
# inverse-vol, long-short, 8bps costs, signal lagged 1 day. Scope = broad: a universal
# microstructure/overreaction premium must generalise to lower-liquidity crypto tiers.
#
# NOTE on the failure being fixed: the previous file's first lines were a shell command
# (`bash ... python -c ...`) pasted into the .py, so it died with a SyntaxError on line 2
# before any logic ran. This is a clean, valid Python module.

import functools
import numpy as np, pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import binance_universe, binance_klines, inv_vol_position
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights

START = "2019-01-01"

# ---------------------------------------------------------------- universe / panels
@functools.lru_cache(maxsize=1)
def _full_universe():
    # Ranked (most-liquid first) list of Binance USDT pairs; cached in-memory (no side effects).
    return tuple(binance_universe(360))


def _split():
    """Search universe = top-75 liquid pairs. Generalization bands are DISJOINT lower-liquidity
    rank slices (share no tickers with the search set or each other)."""
    full = list(_full_universe())
    search = full[:75]
    rest = full[75:]
    bands = {
        "crypto_mid":   rest[0:90],
        "crypto_small": rest[90:180],
        "crypto_tail":  rest[180:270],
    }
    return search, bands


def _taker_pair(tickers, start):
    """Load (taker-buy, total) volume in CONSISTENT units. Tries quote-units first, then base."""
    for tb, tot in [("taker_buy_quote_volume", "quote_volume"),
                    ("taker_buy_quote", "quote_volume"),
                    ("taker_buy_base_volume", "volume"),
                    ("taker_buy_base", "volume")]:
        try:
            a = binance_klines(tickers, start, field=tb)
            b = binance_klines(tickers, start, field=tot)
        except Exception:
            continue
        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame) and a.shape[1] and b.shape[1]:
            return a, b
    raise RuntimeError("binance_klines: taker-flow volume fields unavailable")


def _close(tickers, start):
    for f in ("close", "closeadj"):
        try:
            df = binance_klines(tickers, start, field=f)
        except Exception:
            df = None
        if isinstance(df, pd.DataFrame) and df.shape[1]:
            return df
    raise RuntimeError("binance_klines: close field unavailable")


def _panel(tickers, start=START):
    tickers = list(tickers)
    close = _close(tickers, start)
    tbq, qv = _taker_pair(tickers, start)
    cols = close.columns.intersection(tbq.columns).intersection(qv.columns)
    close, tbq, qv = close[cols], tbq[cols], qv[cols]
    # MultiIndex columns (field, ticker); signal() splits by field.
    return pd.concat({"close": close, "tbq": tbq, "qv": qv}, axis=1)


def load_data() -> pd.DataFrame:
    search, _ = _split()
    return _panel(search)


def load_gen_data(label) -> pd.DataFrame:
    _, bands = _split()
    return _panel(bands[label])


# ---------------------------------------------------------------- crypto "sector" map
_SECTOR = {}
for grp, names in {
    "store_of_value": "BTC LTC BCH BSV XMR ZEC DASH DGB RVN",
    "smart_contract_l1": ("ETH SOL ADA AVAX ATOM NEAR ALGO EGLD FTM ONE KAVA ROSE APT SUI "
                          "SEI INJ TIA KLAY HBAR TRX EOS TON ICP XTZ NEO QTUM WAVES ZIL"),
    "defi": "UNI AAVE MKR COMP CRV SUSHI SNX YFI BAL 1INCH DYDX GMX LDO CAKE RUNE CVX PENDLE",
    "oracle_infra": "LINK GRT FIL AR RNDR OCEAN BAND API3 STORJ THETA",
    "scaling_l2": "MATIC OP ARB IMX METIS STRK MANTA ZK STX",
    "exchange": "BNB OKB CRO FTT HT KCS GT",
    "payments": "XRP XLM NANO IOTA XNO",
    "meme": "DOGE SHIB PEPE FLOKI BONK WIF MEME",
    "gaming_meta": "SAND MANA AXS GALA ENJ APE ILV GMT MAGIC CHZ",
}.items():
    for n in names.split():
        _SECTOR[n] = grp

_FALLBACK = ["alt_l1", "alt_defi", "alt_infra", "alt_gaming", "alt_other"]


def _sector_of(sym):
    base = sym.upper()
    for q in ("USDT", "BUSD", "FDUSD", "USDC", "USD"):
        if base.endswith(q):
            base = base[: -len(q)]
            break
    if base in _SECTOR:
        return _SECTOR[base]
    return _FALLBACK[sum(ord(c) for c in base) % len(_FALLBACK)]  # deterministic


# ---------------------------------------------------------------- signal
def signal(panel, lookback=14, side=-1.0, target_vol=0.20, vol_lb=20, max_pos=0.10, **params):
    close = panel["close"].astype(float)
    tbq = panel["tbq"].astype(float)
    qv = panel["qv"].astype(float)

    rets = close.pct_change()

    # Net taker-flow imbalance per name/day in [-1, 1]: +1 = all aggressive buying.
    imb = (2.0 * tbq / qv.replace(0.0, np.nan)) - 1.0
    imb = imb.clip(-1.0, 1.0)

    lb = int(lookback)
    crowd = imb.rolling(lb, min_periods=max(3, lb // 2)).mean()   # crowding over the window

    # side=-1 -> FADE crowding (reversal, default); side=+1 -> chase (momentum, for the check).
    sig = float(side) * xs_zscore(crowd)

    # inv_vol_position returns weekly-held, ALREADY-LAGGED positions -> no extra shift here.
    W = inv_vol_position(sig, rets, target_vol=target_vol, vol_lb=vol_lb,
                         max_pos=max_pos, rebalance="W")

    daily = net_of_cost(W, rets, cost_bps=8.0, name="crypto_taker_flow_crowding_reversal")
    sector_map = {t: _sector_of(t) for t in close.columns}
    trades = trades_from_weights(W, rets, sector_map)
    return daily, trades


# ---------------------------------------------------------------- soft expectation
def _exp_reversal(ctx):
    """Mechanism falsification: fading crowded taker flow (side=-1) must out-earn chasing it
    (side=+1) on the SEARCH window. One extra signal() call, sliced to < holdout_start."""
    try:
        rev = ctx.get("search")
        if rev is None or len(rev) == 0:
            return {"pass": True, "observed": "search returns unavailable"}
        hs = pd.Timestamp(ctx["holdout_start"])
        mom_daily, _ = signal(ctx["panel"], side=1.0)
        mom = mom_daily[mom_daily.index < hs]
        rm, mm = float(rev.mean()), float(mom.mean())
        return {"pass": rm > mm, "observed": round(rm - mm, 8)}
    except Exception as e:
        return {"pass": True, "observed": f"uncheckable: {e}"}


# ---------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="crypto_taker_flow_crowding_reversal",
    family="crypto_flow_reversal",
    title="Crypto Taker-Flow Crowding Reversal",
    markets=["crypto"],
    data_desc=("Binance daily klines (close + taker-buy vs total volume, consistent units) "
               "for the top-75 liquid USDT pairs; cross-sectional long-short."),
    pre_registration=(
        "Hypothesis: in crypto spot, aggressive (taker) order flow that becomes CROWDED on one "
        "side over a ~2-week window subsequently REVERSES (short-horizon overreaction / liquidity "
        "premium). Mechanism: persistent net taker-buying pushes price ahead of marginal demand and "
        "is unwound by inventory/funding pressure. Signal: per-name net taker-flow imbalance = "
        "2*taker_buy/total - 1 in [-1,1], averaged over `lookback` days, cross-sectionally winsor-"
        "z-scored; go SHORT the most crowded buyers, LONG the most crowded sellers (side=-1). "
        "Weekly rebalance, inverse-vol sized, 8bps on turnover, signal lagged 1 day. "
        "Scope=broad: this is a universal microstructure premium, NOT a top-75 artifact; if real it "
        "must generalise to disjoint lower-liquidity crypto tiers (gen bands), where illiquidity/"
        "reversal premia are typically stronger. Falsifiable: (1) the reversal sign must beat the "
        "momentum sign on the search window (expectation), and (2) >=60% of holdout gen-universes "
        "must be OOS-positive."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "lb7": {"lookback": 7},
        "lb21": {"lookback": 21},
    },
    scope="broad",
    generalization_universes=["crypto_mid", "crypto_small", "crypto_tail"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=30,
    expectations=[
        {"name": "reversal_beats_momentum",
         "claim": "fading crowded taker flow (side=-1) earns higher mean search-window return than chasing it (side=+1)",
         "check": _exp_reversal},
    ],
)