"""
russell2000_recon_deletion_flow — Russell 2000 reconstitution flow premium.

MECHANISM (risk premium, not a forecast): index funds tracking the Russell 2000 are
FORCED sellers of June-reconstitution deletions regardless of price. Providing the
other side of that mechanical flow earns a documented price-pressure reversal
(Cai & Cai 2008; Madhavan 2003; Burnham-Gakidis-Wurgler 2017). The deletions-down
side is the structurally under-arbitraged corner (mandates/career risk prevent
institutions from buying names being thrown out of the index).

FROZEN DESIGN (pre-registered, no tunable forecasting):
  * Rank date: last trading day of May for years <=2003, last trading day of April
    thereafter (published historical Russell rule calendar, fixed ex-ante).
  * Approximate Russell 2000 band = point-in-time marketcap ranks 1001-3000 within
    the 3500 most-liquid US common stocks (survivorship-clean, delisted included).
  * DELETIONS-DOWN basket: in band last year, rank > 3000 (or dropped from coverage)
    this year. Filters at entry use ONLY the prior trading day's data: raw close
    >= $2 and 21d median dollar volume >= $250K.
  * Enter long equal-weight on the first trading day AFTER the last Friday of June
    (post-rebalance — after the forced selling prints); exit last trading day of
    September. Flat October-June (~75% of the year out of market).
  * Declared hedge sleeve: short IWM sized to trailing-60d beta of the basket
    (data strictly before entry), capped at 0.35 gross, constant over the window.
  * Costs: 30bps on single-name turnover, 3bps on IWM.
  * Grid includes a pre-registered PLACEBO (same basket entered the following
    January — should show nothing) and a no-hedge variant.
All look-ahead discipline: PIT fundamentals via datekey (pit_panel), all entry
filters and the hedge beta computed from data strictly before the entry date, and
weights are shift(1)-lagged before net_of_cost / trades_from_weights.
"""
import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights, pit_panel

START = "1998-01-01"     # SEP/SF1 coverage start; first tradable event ~1999
RANK_TOP_N = 3500        # bounded universe (rails: never the full ~16k panel)

_SECTOR_MAP: dict = {}


def _sector_map() -> dict:
    """Sector map for the trade ledger (kit-built; merged across cap tiers)."""
    if _SECTOR_MAP:
        return _SECTOR_MAP
    for cap in (None, "Mid", "Small", "Micro"):
        try:
            _, m = sector_universe(marketcap=cap, top_n_per_sector=300)
            for k, v in m.items():
                _SECTOR_MAP.setdefault(k, v)
        except Exception:
            continue
    _SECTOR_MAP["IWM"] = "ETF (declared hedge)"
    return _SECTOR_MAP


def load_data() -> pd.DataFrame:
    """MultiIndex-column panel: px (closeadj), cls (raw close), dv (21d median $vol),
    mcap (PIT via datekey), etf (IWM close). float32 to bound memory."""
    tickers = us_universe(category="Domestic Common Stock",
                          include_delisted=True, top_n=RANK_TOP_N)
    px = sep_panel(tickers, START, field="closeadj")
    cls = sep_panel(tickers, START, field="close")
    vol = sep_panel(tickers, START, field="volume")
    dv = (cls * vol).rolling(21, min_periods=10).median()
    fund = sf1(tickers, fields=["marketcap"], dimension="ARQ")
    mc = pit_panel(fund, "marketcap", px.index, list(px.columns))  # datekey as-of: PIT
    iwm = yf_panel(["IWM"], START).reindex(px.index)               # hedge leg only
    _sector_map()  # populate ledger sectors once
    return pd.concat(
        {"px": px.astype("float32"), "cls": cls.astype("float32"),
         "dv": dv.astype("float32"), "mcap": mc.astype("float32"),
         "etf": iwm.astype("float32")},
        axis=1,
    )


def load_gen_data(label) -> pd.DataFrame:
    # scope='local' by construction: the premium exists ONLY at an index boundary
    # with large passive AUM tracking it (Russell 2000 is the canonical case).
    raise ValueError(f"local-scope strategy; no generalization universe: {label}")


def signal(panel, band_lo=1001, band_hi=3000, dv_min=250_000.0, px_min=2.0,
           hedge_cap=0.35, beta_lb=60, exit_month=9,
           cost_bps=30.0, hedge_cost_bps=3.0, placebo_month=None):
    px = panel["px"]
    cls = panel["cls"]
    dv = panel["dv"]
    mc = panel["mcap"]
    idx = px.index
    rets = px.pct_change()

    iwm_ret = None
    if "etf" in panel.columns.get_level_values(0) and "IWM" in panel["etf"].columns:
        iwm_px = panel["etf"]["IWM"]
        if iwm_px.notna().sum() > beta_lb:
            iwm_ret = iwm_px.pct_change()

    # --- 1. PIT membership band at each year's frozen rank date ------------------
    members = {}
    for y in sorted(set(idx.year)):
        month = 5 if y <= 2003 else 4  # frozen historical Russell rule calendar
        d = idx[(idx.year == y) & (idx.month == month)]
        if len(d) == 0:
            continue
        m = mc.loc[d[-1]].dropna()      # pit_panel is datekey-based: known by rank date
        if len(m) < band_hi // 2:       # inadequate coverage year -> skip
            continue
        rank = m.rank(ascending=False, method="first")
        members[y] = {"band": set(rank[(rank >= band_lo) & (rank <= band_hi)].index),
                      "rank": rank}

    # --- 2. annual deletion baskets + IWM hedge sleeve ---------------------------
    W = pd.DataFrame(0.0, index=idx, columns=px.columns, dtype="float32")
    Wh = pd.Series(0.0, index=idx, name="IWM")
    for y in sorted(members):
        if (y - 1) not in members:
            continue
        rank_now = members[y]["rank"]
        dels = [t for t in members[y - 1]["band"]
                if (t not in rank_now.index) or (rank_now[t] > band_hi)]
        if not dels:
            continue

        if placebo_month is not None:  # pre-registered placebo: same basket, January
            ed = idx[(idx.year == y + 1) & (idx.month == placebo_month)]
            xd = idx[(idx.year == y + 1) & (idx.month == placebo_month + 2)]
            if len(ed) == 0 or len(xd) == 0:
                continue
            entry, exit_ = ed[0], xd[-1]
        else:
            lf = pd.Timestamp(year=y, month=6, day=30)
            while lf.weekday() != 4:   # last Friday of June (reconstitution effective)
                lf -= pd.Timedelta(days=1)
            after = idx[idx > lf]
            xd = idx[(idx.year == y) & (idx.month == exit_month)]
            if len(after) == 0 or len(xd) == 0:
                continue
            entry, exit_ = after[0], xd[-1]
        if exit_ <= entry:
            continue

        pe = idx.get_loc(entry)
        if pe == 0:
            continue
        ref = idx[pe - 1]  # ALL entry filters/beta use data strictly before entry
        c, lv = cls.loc[ref], dv.loc[ref]
        names = [t for t in dels
                 if np.isfinite(c.get(t, np.nan)) and c[t] >= px_min
                 and np.isfinite(lv.get(t, np.nan)) and lv[t] >= dv_min]
        if len(names) < 5:
            continue
        W.loc[entry:exit_, names] = 1.0 / len(names)

        if iwm_ret is not None and hedge_cap > 0:
            b_pre = rets[names].loc[:ref].tail(beta_lb).mean(axis=1)
            i_pre = iwm_ret.loc[:ref].tail(beta_lb)
            both = pd.concat([b_pre, i_pre], axis=1).dropna()
            if len(both) >= 40 and float(both.iloc[:, 1].var()) > 0:
                beta = float(both.iloc[:, 0].cov(both.iloc[:, 1])
                             / both.iloc[:, 1].var())
                Wh.loc[entry:exit_] = -min(max(beta, 0.0), float(hedge_cap))

    # --- 3. net returns (weights shift(1)-lagged: the lag is OURS) ----------------
    out = net_of_cost(W.shift(1), rets, cost_bps=cost_bps,
                      name="recon_alpha").reindex(idx).fillna(0.0)
    Whf = Wh.to_frame()
    if iwm_ret is not None:
        rh = net_of_cost(Whf.shift(1), iwm_ret.to_frame(),
                         cost_bps=hedge_cost_bps, name="recon_hedge")
        out = out.add(rh.reindex(idx).fillna(0.0), fill_value=0.0)
    out.name = "russell2000_recon_deletion_flow"

    # --- 4. contract trade ledger (kit labels entry_regime) ----------------------
    if iwm_ret is not None:
        W_all = pd.concat([W, Whf], axis=1)
        r_all = pd.concat([rets, iwm_ret.to_frame()], axis=1)
    else:
        W_all, r_all = W, rets
    smap = _sector_map()
    sector_map = {t: smap.get(t, "Unknown") for t in W_all.columns}
    trades = trades_from_weights(W_all.shift(1), r_all, sector_map)
    return out, trades


GRID = {
    "default": {},
    "hold_8w": {"exit_month": 8},            # shorter reversal window
    "no_hedge": {"hedge_cap": 0.0},          # standalone premium first
    "liquid_only": {"dv_min": 1_000_000.0},  # cost-robust sub-basket
    "placebo_jan": {"placebo_month": 1},     # falsification: should show ~nothing
}

SPEC = StrategySpec(
    id="russell2000_recon_deletion_flow_v1",
    family="index_flow",
    title="Russell 2000 reconstitution flow premium — long the forced-deletion basket, IWM-hedged",
    markets=["US small/micro-cap equities (Russell 2000 boundary)"],
    data_desc=("Sharadar SEP closeadj/close/volume (survivorship-clean, delisted incl.) "
               "+ SF1 marketcap PIT via datekey (pit_panel); IWM via yfinance for the "
               "declared hedge sleeve only"),
    pre_registration=(
        "FROZEN ex-ante: rank all US common stocks by PIT marketcap on the historical "
        "Russell rank date (last trading day of May <=2003, of April thereafter); band "
        "= ranks 1001-3000. Deletions-down = in band year y-1, rank>3000 or dropped "
        "year y. Long equal-weight from first trading day after the last Friday of "
        "June, exit end of September; entry filters (close>=$2, 21d median $vol>=$250K) "
        "and the trailing-60d-beta IWM hedge (capped 0.35 gross) use only prior-day "
        "data. Flat Oct-Jun. Costs 30bps single-name / 3bps IWM. Falsifiers: January "
        "placebo entry shows nothing; effect concentrated in weeks post-rebalance."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid=GRID,
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=50,
    hedge_tickers=["IWM"],
    hedge_cap=0.35,
)