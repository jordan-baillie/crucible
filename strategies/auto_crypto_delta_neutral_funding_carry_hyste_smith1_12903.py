"""
crypto_basis_carry_hysteresis_cme
=================================
EVOLUTION of the BTC+ETH hysteresis funding-carry elite — SANDBOX-COMPLIANT
REWRITE. The parent's Binance perp funding feed required raw HTTP (banned:
the harness owns ALL I/O; sdk.adapters only). There is no funding adapter,
so the pre-registered mutation is re-cast onto the funding premium's
no-arbitrage twin that IS observable in free data: the CME front-month
cash-and-carry basis (yfinance BTC=F / ETH=F vs spot). In equilibrium perp
funding == annualized futures basis (both price the same leverage demand);
the carry is harvested identically: long spot 1x / short future 1x.

Premium: carry / liquidity-provision — paid the basis for supplying the short
side of crypto leverage demand, fully delta-hedged. NOT a price prediction.
Tested STANDALONE (no trend blend — 2026-06-08 dilution lesson).

The 5-asset breadth mutation is BLOCKED by data (only BTC and ETH have free
listed futures); breadth is banked as untestable and the tested mutation is
funding->basis signal substitution on the SAME 2-asset book, same hysteresis
machine, byte-identical bands/min-hold/costs.

Lookahead hygiene: hysteresis state computed from trailing basis only, and the
weight matrix is shift(1)-lagged BEFORE net_of_cost / trades_from_weights —
day-t basis/state can only drive day-t+1 positions.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights

# ----------------------------------------------------------------------------
# Universe (frozen): the only crypto assets with FREE listed futures + spot
# ----------------------------------------------------------------------------
PAIRS = {"BTC": ("BTC-USD", "BTC=F"), "ETH": ("ETH-USD", "ETH=F")}
SECTOR_MAP = {"BTC": "crypto-store-of-value", "ETH": "crypto-smart-contract"}
START = "2018-01-01"   # BTC=F lists 2017-12; ETH=F 2021-02 (>=3y by now)
_CACHE = {}

DEFAULTS = dict(
    symbols=list(PAIRS),     # grid variant 'btc_only' = single-asset head-to-head
    entry_band=0.10,         # ENTER when trailing 7d mean annualized basis > 10%
    exit_band=0.00,          # EXIT only when < 0% (asymmetric hysteresis, inherited)
    min_hold=7,              # 7 trading days min-hold after entry (inherited)
    sizing="ivol",           # equal-risk across IN assets; 'equal' variant in grid
    cost_bps_per_leg=10.0,   # 10 bps per leg, 2 legs (spot+future) per unit turnover
    vol_lb=60,
)


def load_data():
    """Panel with MultiIndex columns (field, sym); field in {spot, fut}.

    All data via sdk.adapters.yf_panel — NO raw I/O. Gate-0 in code: each
    asset needs >=3y of joint spot+future history or it is dropped; if none
    survive the carry thesis is untestable -> hard RuntimeError. Rows are
    restricted to futures trading days so spot/future close-to-close returns
    are measured over identical intervals (weekend spot drift folds into the
    Fri->Mon bar on BOTH legs).
    """
    if "panel" in _CACHE:
        return _CACHE["panel"]

    spot_raw = yf_panel([v[0] for v in PAIRS.values()], start=START)
    fut_raw = yf_panel([v[1] for v in PAIRS.values()], start=START)
    spot_raw.index = pd.to_datetime(spot_raw.index).normalize()
    fut_raw.index = pd.to_datetime(fut_raw.index).normalize()

    cols, kept = {}, []
    for sym, (spot_tkr, fut_tkr) in PAIRS.items():
        if spot_tkr not in spot_raw.columns or fut_tkr not in fut_raw.columns:
            continue
        s = spot_raw[spot_tkr].dropna()
        f = fut_raw[fut_tkr].dropna()
        joint = s.index.intersection(f.index)
        if len(joint) == 0 or (joint[-1] - joint[0]).days < 3 * 365:
            continue  # Gate-0 drop: <3y of joint spot+future history
        kept.append(sym)
        cols[("spot", sym)] = s
        cols[("fut", sym)] = f
    if not kept:
        raise RuntimeError(
            "Gate-0 FAIL: no asset has >=3y joint spot+futures history; "
            "basis-carry thesis untestable.")

    panel = pd.DataFrame(cols).sort_index()
    panel.columns = pd.MultiIndex.from_tuples(panel.columns,
                                              names=["field", "sym"])
    # keep only futures trading days (at least one future priced)
    panel = panel.loc[panel["fut"].notna().any(axis=1)]
    _CACHE["panel"] = panel
    return panel


def load_gen_data(label):
    # scope='local': listed crypto basis exists only for BTC/ETH in free data.
    # The 'btc_only' grid variant gives the within-domain head-to-head.
    raise ValueError(f"scope='local': no generalization universe '{label}'")


# ----------------------------------------------------------------------------
# Signal: per-asset asymmetric hysteresis on trailing 7d annualized basis
# ----------------------------------------------------------------------------
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    spot = panel["spot"]
    fut = panel["fut"]
    syms = [s for s in p["symbols"] if s in spot.columns]
    spot, fut = spot[syms], fut[syms]
    idx = panel.index

    both = spot.notna() & fut.notna()

    # carry signal: front-month premium F/S - 1, annualized with the average
    # ~1-month tenor of the continuous front contract (x12). Trailing 7-day
    # mean smooths roll-date jumps; trailing-only -> no lookahead.
    basis = (fut / spot - 1.0).where(both)
    ann = basis.rolling(7, min_periods=5).mean() * 12.0

    # delta-neutral daily return per unit leg notional: long spot, short
    # future. Within-contract convergence accrues the carry; the roll jump
    # (paying the new premium) is the genuine cost of re-establishing carry.
    dn = (spot.pct_change() - fut.pct_change()).where(both)

    # hysteresis state machine: enter > entry_band, exit < exit_band,
    # hold in between, min_hold trading days after entry before any exit
    sig = pd.DataFrame(0.0, index=idx, columns=syms)
    for s in syms:
        a = ann[s].reindex(idx).values
        out = np.zeros(len(idx))
        in_pos, entry_i = False, 0
        for i, v in enumerate(a):
            if not in_pos:
                if v == v and v > p["entry_band"]:
                    in_pos, entry_i = True, i
            else:
                if (i - entry_i) >= p["min_hold"] and v == v and v < p["exit_band"]:
                    in_pos = False
            out[i] = 1.0 if in_pos else 0.0
        sig[s] = out

    # equal-risk sizing across IN assets, renormalized so sum(w)<=1 per side
    # => gross (spot long + future short) never exceeds 2x total
    if p["sizing"] == "equal":
        risk = sig.copy()
    else:
        vol = dn.rolling(p["vol_lb"], min_periods=20).std()
        risk = (sig / vol.replace(0.0, np.nan)).fillna(0.0).where(sig > 0, 0.0)
    W = risk.div(risk.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    # membership changes act daily (the hysteresis IS the signal); SIZING is
    # only refreshed weekly to limit turnover (weekly rebalance per contract)
    Wq = W.copy()
    memb = sig.values
    for i in range(1, len(idx)):
        if idx[i].weekday() != 0 and np.array_equal(memb[i], memb[i - 1]):
            Wq.iloc[i] = Wq.iloc[i - 1]

    # LAG: day-t state trades day t+1 — the shift(1) is the lookahead guard
    W_lag = Wq.shift(1).fillna(0.0)

    rets = dn.fillna(0.0)
    daily = net_of_cost(W_lag, rets,
                        cost_bps=2.0 * p["cost_bps_per_leg"],  # 2 legs/unit turnover
                        name="crypto_basis_carry_hyst_cme")
    trades = trades_from_weights(W_lag, rets, SECTOR_MAP)
    return daily.dropna(), trades


# ----------------------------------------------------------------------------
SPEC = StrategySpec(
    id="crypto_basis_carry_hysteresis_cme",
    family="crypto-funding-carry",
    title=("Crypto delta-neutral basis-carry, hysteresis: long spot / short "
           "CME front future on BTC+ETH (sandbox-compliant evolution of the "
           "perp funding-carry hysteresis elite — basis is funding's "
           "no-arbitrage twin)"),
    markets=["crypto"],
    data_desc=("yf_panel only (FREE, sandbox-clean): BTC-USD/ETH-USD spot and "
               "BTC=F/ETH=F CME continuous front futures, daily closes, "
               "restricted to futures trading days; >=3y joint history "
               "enforced at Gate-0 in load_data. $0 owned/free, no raw I/O."),
    pre_registration=(
        "FROZEN single mutation from the 2-asset perp-funding hysteresis "
        "elite: substitute the (sandbox-unavailable) perp funding signal with "
        "its arbitrage twin, the annualized CME front-month basis, on the "
        "SAME 2-asset book. Bands (enter>10% ann., exit<0%), 7-trading-day "
        "min-hold, daily regime check, equal-risk sizing renormalized to "
        "gross<=2x, 10bps/leg costs — inherited byte-identical; only the "
        "carry observable changes. The 5-asset breadth mutation is BANKED AS "
        "UNTESTABLE (no free listed futures beyond BTC/ETH). FALSIFICATION "
        "(pre-registered): (a) either single-asset sub-book negative net of "
        "costs over its full history -> basis-carry thesis FALSIFIED for that "
        "asset; (b) band sweep (entry 8/10/12%, exit -2/0/+2%) must be FLAT — "
        "a sharp peak is a curve-fit FAIL; (c) conclusion must not depend on "
        "equal-weight vs equal-risk sizing; (d) May-2021 / FTX Nov-2022 / "
        "2022-bear: the exit-below-0% band must fire within days on both "
        "assets; (e) if the basis signal's sign disagrees with the parent's "
        "funding-era positions so much that round-trips per unit gross rise "
        ">2x vs the parent, the twin-substitution is FALSIFIED. Premium "
        "tested STANDALONE — no trend blend (2026-06-08 dilution lesson). A "
        "pass requires a fresh write-once forward-paper run before re-"
        "entering the carry+trend book."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "entry8": {"entry_band": 0.08},
        "entry12": {"entry_band": 0.12},
        "exit_m2": {"exit_band": -0.02},
        "exit_p2": {"exit_band": 0.02},
        "equal_weight": {"sizing": "equal"},
        "btc_only": {"symbols": ["BTC"]},  # single-asset head-to-head
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=2,
)