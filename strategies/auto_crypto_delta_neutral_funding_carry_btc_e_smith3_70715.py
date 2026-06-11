"""
crypto_dn_basis_carry — Delta-neutral BTC+ETH cash-and-carry (spot-hedged CME futures basis),
STANDALONE.

Premium: payment for supplying the short side of listed crypto leverage demand, fully
delta-hedged (long 1x spot + short 1x front-month CME future per coin => zero delta by
construction; gross 2x per coin slice, within the cap). NOT a price prediction. Tested
standalone per the 2026-06-08 dilution lesson — no trend blend in this experiment.

DATA NOTE (sandbox fix): the prior version raw-pulled Binance perp funding via urllib —
a SANDBOX VIOLATION (the harness owns ALL I/O; adapters only). There is no funding-rate
adapter in the kit, so this version tests the SAME no-arbitrage carry through its listed
analogue: the CME futures basis (BTC=F / ETH=F vs spot), which IS the funding premium in
term-market form — both are the price of levered long crypto exposure. Everything comes
through the tested yf_panel adapter (futures + crypto spot are exactly its sanctioned use;
no US single stocks involved).

FROZEN pre-registered rule (no tuning beyond the declared grid):
  Per coin independently: hold the cash-and-carry package ONLY while the trailing 7-day mean
  front-month basis, annualized (x12 on the ~1-month contract premium — a frozen monotone
  proxy), exceeds a 5% ann. hurdle (must out-earn ~10bps/leg round-trip costs); otherwise
  flat (stablecoin). Equal-risk split across the two coins (inverse trailing-30d vol of each
  coin's HEDGED carry return, refreshed weekly). Daily regime check on futures trading days.
  P&L = spot_ret - futures_ret (basis decay accrues to the package; the hedge is in the P&L,
  not assumed free) - turnover costs.

NO LOOK-AHEAD: signals are built from data through day t and weights are shift(1)-lagged
before net_of_cost / trades_from_weights — holding through day t+1 earns day t+1's realized
basis decay, decided on data known at t.
"""

import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights

# ----------------------------------------------------------------------------- universe
FUT = {"BTC": "BTC=F", "ETH": "ETH=F"}                # CME front-month continuous (yfinance)
SPOT = {"BTC": "BTC-USD", "ETH": "ETH-USD"}           # yfinance spot closes
SECTOR_MAP = {"BTC": "crypto-L1-store-of-value", "ETH": "crypto-L1-smart-contract"}
START = "2019-01-01"                                  # BTC=F lists 2017-12; ETH=F lists 2021-02


# ----------------------------------------------------------------------------- panels
def load_data() -> pd.DataFrame:
    """Daily panel, MultiIndex columns (coin, field) with field in {spot, fut}.
    All data via the tested yf_panel adapter — futures/crypto are its sanctioned domain."""
    spot = yf_panel(list(SPOT.values()), start=START)
    fut = yf_panel(list(FUT.values()), start=START)

    cols = {}
    for coin in FUT:
        cols[(coin, "spot")] = spot[SPOT[coin]]
        cols[(coin, "fut")] = fut[FUT[coin]]

    panel = pd.DataFrame(cols).sort_index()
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    # gate-0 depth check: full history, NOT a recent window (this exact trap bit us before)
    depth_ok = {"BTC": pd.Timestamp("2019-12-31"),    # BTC=F must cover the pre-holdout years
                "ETH": pd.Timestamp("2021-06-30")}    # ETH=F lists 2021-02 — later start is real
    for coin in FUT:
        f = panel[(coin, "fut")].dropna()
        assert len(f) > 0 and f.index.min() <= depth_ok[coin], (
            f"{FUT[coin]} futures history starts "
            f"{f.index.min().date() if len(f) else 'EMPTY'} — truncated window, "
            f"refusing to backtest on it."
        )
        # gate-0 alignment check: spot must overlap futures days so basis is MEASURED, not assumed
        both = panel[(coin, "spot")].notna() & panel[(coin, "fut")].notna()
        assert both.sum() > 0.90 * panel[(coin, "fut")].notna().sum(), (
            f"{coin}: spot/futures daily closes misaligned — basis unmeasurable."
        )
    return panel


def load_gen_data(label: str) -> pd.DataFrame:
    """scope='local': the crypto leverage-demand carry premium has no analogue universe to
    generalize into (DM futures carry is separately CLOSED-dead). Validation is the per-coin
    robustness + regime stress + fresh forward-paper run in the pre-registration."""
    raise ValueError(f"local-scope strategy has no generalization universes (asked for {label!r})")


# ----------------------------------------------------------------------------- signal
def signal(panel: pd.DataFrame, hurdle_ann: float = 0.05, fund_lb: int = 7,
           vol_lb: int = 30, cost_bps: float = 20.0):
    """cost_bps=20 on package turnover = two legs (spot + futures) x 10bps each, charged by
    net_of_cost on |dW|; a full round trip therefore costs 40bps — the frozen hurdle exists
    precisely so the basis must clear this."""
    coins = sorted({c for c, _ in panel.columns})

    carry_rets, on = {}, {}
    for c in coins:
        valid = panel[(c, "spot")].notna() & panel[(c, "fut")].notna()
        s = panel.loc[valid, (c, "spot")]
        f = panel.loc[valid, (c, "fut")]
        # long spot 1x, short future 1x: basis decay IS the P&L, hedge is not assumed free
        carry_rets[c] = s.pct_change() - f.pct_change()
        prem = f / s - 1.0                             # front-month (~1m) premium over spot
        ann = prem.rolling(fund_lb).mean() * 12.0      # trailing data only; frozen x12 proxy
        on[c] = (ann > hurdle_ann).astype(float)

    R = pd.DataFrame(carry_rets).dropna(how="all")     # union index, auto-aligned per coin
    R = R.fillna(0.0)                                  # pre-listing / non-overlap rows; W=0 there
    ON = pd.DataFrame(on).reindex(R.index).fillna(0.0)

    # equal-risk split across coins: inverse trailing vol of the HEDGED carry return,
    # refreshed weekly to limit sizing turnover; daily on/off regime switch applied on top.
    vol = R.rolling(vol_lb).std().clip(lower=1e-4)
    iv = 1.0 / vol
    w_risk = iv.div(iv.sum(axis=1), axis=0)
    w_risk = w_risk.resample("W-FRI").last().reindex(R.index).ffill()

    W = (ON * w_risk).fillna(0.0)                      # sum(W) <= 1 package notional
    # gross exposure = 2 x W (spot leg + futures leg) -> gross <= 2x, exactly the frozen cap.

    W_lag = W.shift(1).fillna(0.0)                     # decisions act NEXT day — no look-ahead
    daily = net_of_cost(W_lag, R, cost_bps=cost_bps, name="crypto_dn_basis_carry")
    trades = trades_from_weights(W_lag, R, SECTOR_MAP)  # kit stamps entry_regime — never by hand
    return daily, trades


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="crypto_dn_basis_carry_v1",
    family="carry",
    title="Crypto delta-neutral basis-carry (BTC+ETH spot-hedged CME futures, standalone)",
    markets=["crypto"],
    data_desc=(
        "CME front-month crypto futures (BTC=F from 2017-12, ETH=F from 2021-02) and spot "
        "daily closes, ALL via the tested yf_panel adapter ($0, sanctioned for futures/crypto; "
        "no raw I/O). Basis measured from futures-vs-spot closes on futures trading days; "
        "history depth asserted, not assumed."
    ),
    pre_registration=(
        "FROZEN: per coin (BTC, ETH) hold long-spot/short-front-future 1x/1x ONLY while the "
        "trailing 7d mean front-month premium, annualized x12 (frozen monotone proxy for the "
        "funding/carry rate), > 5% ann. hurdle; else flat. Equal-risk (inv 30d vol of hedged "
        "carry return, weekly refresh) across coins; daily check; 10bps/leg costs (20bps on "
        "package turnover); gross <= 2x. STANDALONE — no trend blend in this experiment. "
        "Validation in lieu of breadth (local scope): (a) full stage-1 gates + MCPT under the "
        "ABSOLUTE Sharpe null (book is market-neutral by construction); (b) BTC-only and "
        "ETH-only sub-books must both be positive — a one-coin result is a FAIL (ETH leg only "
        "exists from 2021-02, its listing date); (c) stress on known carry breaks (May-2021, "
        "2022 bear, FTX Nov-2022 — the latter two sit in the holdout); (d) on a pass, a FRESH "
        "write-once forward-paper run (~3 months, pre-registered verdict date) is REQUIRED by "
        "the 2026-06-10 Midas closure before this leg may re-enter the carry+trend book."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"hurdle_ann": 0.05, "fund_lb": 7, "vol_lb": 30, "cost_bps": 20.0},
    grid={
        "default": {},
        "hurdle_3pct": {"hurdle_ann": 0.03},
        "hurdle_8pct": {"hurdle_ann": 0.08},
        "fund_lb_14d": {"fund_lb": 14},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=2,
)