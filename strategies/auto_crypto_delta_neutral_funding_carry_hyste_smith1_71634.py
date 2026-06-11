"""
Crypto delta-neutral basis-carry — HYSTERESIS variant (v1, adapter-clean).

Premium: paid carry for supplying the short side of crypto leverage demand,
fully delta-hedged (long 1x spot + short 1x CME front-month future per asset).
NOT a price prediction — an insurance/liquidity-provision premium.

SANDBOX FIX vs the failed draft: the Binance urllib fetches are GONE. The
harness owns ALL I/O, so the carry signal is rebuilt entirely from the
approved adapters: yf_panel supplies spot (BTC-USD / ETH-USD) and CME
front-month futures (BTC=F / ETH=F). The funding rate is replaced by its
exchange-traded twin — the futures/spot BASIS — which is the same premium
(perps arbitrage their funding against the dated basis). The realized carry
of the long-spot/short-future book is spot_ret - fut_ret; the regime signal
is the trailing annualized basis level.

Single mutation vs the parent binary 5%-hurdle switch:
  ENTER  when trailing 7d mean annualized basis > +10%  (2x the cost hurdle)
  EXIT   when trailing 7d mean annualized basis <   0%  (carry actually flipped)
  HOLD   in the 0%..10% band (exiting+re-entering costs more than marginal carry)
  MIN-HOLD 7 days after entry (kills same-week churn).
Grid includes the parent binary construction ("parent_binary") so the harness
runs the head-to-head on identical data: the mutation must beat it net-of-cost
AND cut round-trips, or it has failed its own thesis.

Annualization heuristic: BTC=F/ETH=F are front-month continuous series with
~0.5-month average time-to-expiry through the roll cycle; the basis level is
annualized x12 as a regime SIGNAL only (thresholds are what's gated, and the
band grid below tests sensitivity to exactly this scaling).

Lag discipline: the regime/weight decision at day t uses data through day t-1
only — implemented as one W.shift(1) on the full weight matrix before
net_of_cost / trades_from_weights (the lag is OURS, stated here).

Costs: 10 bps per leg; a position flip touches 2 legs (spot + future), so
net_of_cost is charged 20 bps on |dW| — entry and exit each pay 2 legs.

scope='local': crypto basis/funding carry has no disjoint generalization
universe (the nearest analogue, DM futures carry, is CLOSED-dead). Forward-
paper validation confirms, per the 2026-06-10 Midas closure requirement.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights

ASSETS = ["BTC", "ETH"]
SPOT_TKR = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
FUT_TKR = {"BTC": "BTC=F", "ETH": "ETH=F"}  # CME front-month (ETH=F live Feb-2021; NaN-handled)
START = "2018-01-01"  # CME BTC futures launched Dec-2017
# Two delta-neutral books; sector labels reflect the standard L1 taxonomy so the
# ledger's sector field is meaningful (kit stamps entry_regime — never hand-set).
SECTOR_MAP = {"BTC": "crypto-store-of-value", "ETH": "crypto-smart-contract"}

_CACHE = {}  # in-memory only — avoids refetching across grid evaluations


# ----------------------------------------------------------------- data layer
def load_data() -> pd.DataFrame:
    """Panel: {BTC,ETH} x {spot close, front-month future close}, via yf_panel
    ONLY (crypto spot + CME futures are exactly its sanctioned use-case)."""
    if "panel" in _CACHE:
        return _CACHE["panel"]
    tickers = [SPOT_TKR[a] for a in ASSETS] + [FUT_TKR[a] for a in ASSETS]
    px = yf_panel(tickers, start=START)
    cols = {}
    for a in ASSETS:
        cols[f"{a}_spot"] = px[SPOT_TKR[a]]
        cols[f"{a}_fut"] = px[FUT_TKR[a]]
    panel = pd.DataFrame(cols).sort_index()
    _CACHE["panel"] = panel
    return panel


def load_gen_data(label) -> pd.DataFrame:
    """scope='local' — crypto carry has no disjoint generalization universe
    (the premium exists only in crypto; DM futures carry is CLOSED-dead)."""
    raise ValueError(f"local-scope strategy: no generalization universe '{label}'")


# --------------------------------------------------------------------- signal
def signal(panel, entry_ann=0.10, exit_ann=0.00, min_hold=7,
           basis_lb=7, vol_lb=30, ann_factor=12.0, cost_bps_per_leg=10.0):
    spot = panel[[f"{a}_spot" for a in ASSETS]].copy()
    fut = panel[[f"{a}_fut" for a in ASSETS]].copy()
    spot.columns = ASSETS
    fut.columns = ASSETS

    spot_ret = spot.pct_change()
    fut_ret = fut.pct_change()
    # Daily book return per asset while ON: long-spot/short-future carry,
    # earned through basis convergence (roll noise included — measured, not
    # assumed zero).
    carry = spot_ret - fut_ret

    # Trailing annualized basis (front-month premium over spot, ~x12 to annual);
    # full window required -> state is never decided on a partial window.
    basis = fut / spot - 1.0
    ann_basis = basis.rolling(basis_lb, min_periods=basis_lb).mean() * ann_factor

    # Hysteresis state machine per asset (independent books, per spec).
    state = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    for a in ASSETS:
        ab = ann_basis[a].values
        st = np.zeros(len(ab))
        on, days = False, 0
        for i in range(len(ab)):
            b = ab[i]
            if not np.isfinite(b):          # pre-launch / no window yet: flat
                on, days = False, 0
            elif on:
                days += 1
                if b < exit_ann and days >= min_hold:
                    on, days = False, 0     # carry actually flipped -> exit
            elif b > entry_ann:
                on, days = True, 0          # carry clearly out-earns frictions
            st[i] = 1.0 if on else 0.0
        state[a] = st

    # Equal-risk sizing across active books: inverse spot-vol, normalized so
    # spot gross <= 1x (total gross <= 2x with the futures legs = the cap).
    iv = 1.0 / spot_ret.rolling(vol_lb, min_periods=vol_lb).std()
    raw = (state * iv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = raw.sum(axis=1)
    weights = raw.div(gross.where(gross > 0, 1.0), axis=0)

    # THE lag: decision from trailing data through t-1 -> applies to day t.
    W = weights.shift(1).fillna(0.0)

    carry = carry.fillna(0.0)  # W is 0 wherever carry is undefined (pre-launch)
    # 2 legs per flip (spot + future) x 10 bps/leg = 20 bps on turnover.
    daily = net_of_cost(W, carry, cost_bps=2.0 * cost_bps_per_leg,
                        name="crypto_basis_carry_hysteresis")
    trades = trades_from_weights(W, carry, SECTOR_MAP)
    return daily, trades


# ----------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="crypto_basis_carry_hysteresis_v1",
    family="carry",
    title=("Crypto delta-neutral basis carry (BTC+ETH spot-hedged CME front-"
           "month), hysteresis bands 10%-in / 0%-out + 7d min-hold — turnover-"
           "hardened evolution of the parent binary 5% switch"),
    markets=["crypto"],
    data_desc=("yf_panel ONLY (sanctioned for futures + crypto): BTC-USD/ETH-USD "
               "spot and BTC=F/ETH=F CME front-month closes, daily; carry = "
               "spot_ret - fut_ret so basis tracking/roll error is measured, "
               "not assumed zero; signal = trailing 7d annualized basis"),
    pre_registration=(
        "Premium: delta-neutral basis carry (exchange-traded twin of perp "
        "funding), tested STANDALONE (no trend blend — 2026-06-08 dilution "
        "lesson). Mutation under test: asymmetric enter>10%/exit<0% hysteresis "
        "+ 7d min-hold vs parent symmetric 5% switch. PASS requires: (a) "
        "stage-1 gates + MCPT under absolute-Sharpe null (book is market-"
        "neutral by construction); (b) BTC-only AND ETH-only sub-books "
        "independently positive — one-coin result is a FAIL; (c) exit band "
        "fires within days at carry-regime breaks (May-2021, FTX Nov-2022 — "
        "in holdout); (d) head-to-head vs grid 'parent_binary': net Sharpe >= "
        "parent AND materially fewer round-trips (>40% target), else revert "
        "to parent spec; (e) flat result across the declared band grid — a "
        "sharp peak is a curve-fit FAIL; (f) on pass, fresh write-once "
        "forward-paper run (~3-month pre-registered verdict) before re-arming "
        "the carry book."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"entry_ann": 0.10, "exit_ann": 0.00, "min_hold": 7},
    grid={
        "default": {},
        "entry8": {"entry_ann": 0.08},
        "entry12": {"entry_ann": 0.12},
        "exit_m2": {"exit_ann": -0.02},
        "exit_p2": {"exit_ann": 0.02},
        # parent construction on identical data — the pre-registered head-to-head
        "parent_binary": {"entry_ann": 0.05, "exit_ann": 0.05, "min_hold": 0},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=2,
)