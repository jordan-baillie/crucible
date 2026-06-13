# Soybean board-crush convergence — processing-margin (relative-value) risk premium.
# DATA FIX: the original assumed a Databento `fut_curve(root, start=...)` curve adapter for
# matched individual contract months. That adapter is NOT in the sanctioned import set (and the
# importable symbol rejects `start`), so the matched-month/roll machinery is unbuildable through
# the tested rails. Rebuilt on the APPROVED `yf_panel` continuous FRONT-MONTH futures: same GPM
# convergence mechanism, percentile bands, 60d horizon, vol gate, fixed 1:1:1 board ratio.
# Continuous front-month already embeds the roll in the price series, so leg returns = pct_change
# and turnover costs (entry/exit) are charged via net_of_cost; no separate per-roll cost is asserted
# (it cannot be located honestly without individual-contract data).

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# GPM recipes ($/unit). soy: 0.022*ZM($/ton) + 0.11*ZL(c/lb) - ZS(c/bu)/100  -> physical yields.
RECIPES = [
    {"name": "soy_crush",    "out": {"ZM": 0.022, "ZL": 0.11},  "inp": {"ZS": 0.01}},
    {"name": "cattle_crush", "out": {"LE": 12.5},               "inp": {"GF": 7.5, "ZC": 0.5}},
    {"name": "crack_321",    "out": {"RB": 28.0, "HO": 14.0},   "inp": {"CL": 1.0}},
    {"name": "hog_feed",     "out": {"HE": 2.8},                "inp": {"ZC": 0.1}},
]

SECTORS = {
    "ZS": "Oilseeds", "ZM": "Soybean Meal", "ZL": "Soybean Oil",
    "LE": "Live Cattle", "GF": "Feeder Cattle", "ZC": "Grains",
    "CL": "Crude Oil", "RB": "Gasoline", "HO": "Distillates", "HE": "Lean Hogs",
}

# yfinance continuous front-month futures symbols (FREE; futures/ETFs only — never US single stocks).
YF = {
    "ZS": "ZS=F", "ZM": "ZM=F", "ZL": "ZL=F",
    "LE": "LE=F", "GF": "GF=F", "ZC": "ZC=F",
    "CL": "CL=F", "RB": "RB=F", "HO": "HO=F", "HE": "HE=F",
}

SOY_LEGS = ["ZS", "ZM", "ZL"]
GEN_LEGS = {
    "cattle_crush": ["LE", "GF", "ZC"],
    "crack_321":    ["CL", "RB", "HO"],
    "hog_feed":     ["HE", "ZC"],
}


def _find_recipe(cols):
    cset = set(map(str, cols))
    for r in RECIPES:
        if (set(r["out"]) | set(r["inp"])).issubset(cset):
            return r
    return None


def _panel(legs, start="2010-01-01"):
    """Continuous front-month settlement panel for the requested complex (columns = leg roots)."""
    tickers = [YF[l] for l in legs]
    px = yf_panel(tickers, start=start)
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.rename(columns={YF[l]: l for l in legs})
    keep = [l for l in legs if l in px.columns]
    px = px[keep].astype(float).sort_index()
    px = px.ffill(limit=3).dropna()
    return px


def _gpm_and_gross(px, rec):
    legs = list(rec["out"]) + list(rec["inp"])
    raw = pd.DataFrame(0.0, index=px.index, columns=legs)
    for tk, c in rec["out"].items():
        raw[tk] = c * px[tk]
    for tk, c in rec["inp"].items():
        raw[tk] = -c * px[tk]
    gpm = raw.sum(axis=1)
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return gpm, gross, raw, legs


def _spread_series(panel):
    rec = _find_recipe(panel.columns)
    if rec is None:
        return None
    legs = list(rec["out"]) + list(rec["inp"])
    px = panel[legs].astype(float)
    gpm, gross, raw, _ = _gpm_and_gross(px, rec)
    unit_w = raw.div(gross, axis=0)
    leg_ret = px.pct_change().fillna(0.0)
    sret = (unit_w.shift(1) * leg_ret).sum(axis=1)
    return gpm, sret


def load_data() -> pd.DataFrame:
    return _panel(SOY_LEGS)


def load_gen_data(label) -> pd.DataFrame:
    return _panel(GEN_LEGS[label])


def signal(panel, entry_lo=0.20, entry_hi=0.80, exit_q=0.50, lookback=252,
           max_hold=60, vol_lb=20, vol_block_q=0.90, cost_bps=8.0, **_):
    """FROZEN convergence rules on the continuous front-month board crush:
      - GPM percentile bands on trailing `lookback` days, .shift(1) -> strictly past.
      - ENTER long crush < entry_lo pctile; ENTER short crush > entry_hi pctile.
      - EXIT at trailing median OR pre-registered `max_hold` (60d).
      - VOL GATE: no new entry when 20d realised spread-vol is top-decile.
      - SIZE: FIXED 1:1:1 board ratio (gross-1 coefficient/notional weights), held -> turnover
        only on entry/exit.
      - COSTS: cost_bps on leg turnover (net_of_cost). Continuous front-month embeds the roll P&L
        in the price series, so no separate per-roll cost is asserted.
      - LAG: weights set at close t, held from t+1 (W.shift(1))."""
    p = panel.sort_index()
    rec = _find_recipe(p.columns)
    if rec is None or len(p) < 300:
        return pd.Series(0.0, index=p.index, name="crush"), []
    legs = list(rec["out"]) + list(rec["inp"])
    px = p[legs].astype(float)

    gpm, gross, raw, _ = _gpm_and_gross(px, rec)
    unit_w = raw.div(gross, axis=0)                      # gross-1 board-ratio leg weights

    mp = max(60, lookback // 2)
    rq = gpm.rolling(lookback, min_periods=mp)
    p_lo = rq.quantile(entry_lo).shift(1)
    p_hi = rq.quantile(entry_hi).shift(1)
    p_md = rq.quantile(exit_q).shift(1)

    leg_ret = px.pct_change().fillna(0.0)                # continuous front-month leg returns
    sret = (unit_w.shift(1) * leg_ret).sum(axis=1)
    vol = sret.rolling(vol_lb).std()
    vthr = vol.rolling(lookback, min_periods=mp).quantile(vol_block_q).shift(1)

    g = gpm.values; plo = p_lo.values; phi = p_hi.values; pmd = p_md.values
    vv = vol.values; vt = vthr.values; uw = unit_w.values
    n, m = len(px), len(legs)
    W = np.zeros((n, m))
    state, days, cur = 0, 0, None
    for i in range(n):
        if state == 0:
            ready = (not np.isnan(plo[i]) and not np.isnan(phi[i]) and not np.isnan(pmd[i])
                     and not np.isnan(vt[i]) and not np.isnan(uw[i]).any())
            blocked = (not np.isnan(vv[i])) and (vv[i] > vt[i])
            if ready and not blocked:
                if g[i] < plo[i]:
                    state, days, cur = 1, 0, uw[i]
                elif g[i] > phi[i]:
                    state, days, cur = -1, 0, -uw[i]
        else:
            days += 1
            if (state == 1 and not np.isnan(pmd[i]) and g[i] >= pmd[i]) or \
               (state == -1 and not np.isnan(pmd[i]) and g[i] <= pmd[i]) or \
               (days >= max_hold):
                state, cur = 0, None
        if state != 0:
            W[i] = cur
    W = pd.DataFrame(W, index=px.index, columns=legs)
    W_lag = W.shift(1).fillna(0.0)                       # weights set at close t, executed t+1
    name = "%s_convergence" % rec["name"]

    daily = net_of_cost(W_lag, leg_ret, cost_bps=cost_bps, name=name)

    sect = {tk: SECTORS.get(tk, "Commodity") for tk in legs}
    trades = trades_from_weights(W_lag, leg_ret, sect)
    return daily, trades


def _check_max_hold(ctx):
    hs = pd.Timestamp(ctx["holdout_start"])
    trs = [t for t in ctx["trades"] if pd.Timestamp(t["entry_date"]) < hs]
    if not trs:
        return {"pass": False, "observed": "no trades"}
    mx = max(int(t["hold_days"]) for t in trs)
    return {"pass": mx <= 63, "observed": mx}


def _check_vol_gate(ctx):
    hs = pd.Timestamp(ctx["holdout_start"])
    ss = _spread_series(ctx["panel"])
    if ss is None:
        return {"pass": True, "observed": "n/a"}
    gpm, sret = ss
    vol = sret.rolling(20).std()
    vthr = vol.rolling(252, min_periods=126).quantile(0.90).shift(1)
    top = (vol > vthr)
    top = top[top.index < hs]
    ents = [pd.Timestamp(t["entry_date"]) for t in ctx["trades"] if pd.Timestamp(t["entry_date"]) < hs]
    if not ents or int(top.sum()) == 0:
        return {"pass": True, "observed": "n/a"}
    frac = float(np.mean(top.reindex(ents).fillna(False).values))
    return {"pass": frac < 0.10, "observed": round(frac, 3)}


SPEC = StrategySpec(
    id="soy_crush_convergence_v1",
    family="spread_convergence",
    title="Soybean board-crush convergence (processing-margin risk premium, continuous front-month, "
          "percentile-banded, pre-registered 60d horizon)",
    markets=["futures"],
    data_desc="CME soybean complex continuous FRONT-MONTH futures (ZS/ZM/ZL) via yf_panel; board "
              "GPM=0.022*ZM($/ton)+0.11*ZL(c/lb)-ZS(c/bu)/100 ($/bu); leg returns = front-month "
              "pct_change (roll P&L embedded in the continuous series). Generalises to cattle-crush "
              "(LE/GF/ZC), 3-2-1 crack (CL/RB/HO), hog-feed (HE/ZC).",
    pre_registration=(
        "PREMIUM: processing-margin convergence / relative-value premium with a physical-arbitrage "
        "anchor. When the board crush GPM trades far below its trailing distribution the long-crush "
        "position is PAID to bear the risk that processors curtail capacity slowly; crusher on/off "
        "optionality pulls the margin back toward processing cost. Compensation for absorbing "
        "commercial hedging flow inside one physical complex — NOT a directional outright bet.\n"
        "FROZEN SPEC: continuous front-month ZS/ZM/ZL (yf_panel); GPM=0.022*ZM($/ton)+0.11*ZL(c/lb)"
        "-ZS(c/bu)/100; leg returns = front-month pct_change (roll P&L embedded in the continuous "
        "series — no separate per-roll cost is asserted, as it cannot be located without individual "
        "contract data). GPM percentile bands on a trailing 252d window (.shift(1) -> past only): "
        "ENTER long crush <20th pctile, short crush >80th pctile; EXIT at trailing median OR a "
        "pre-registered 60-day max hold (Mitchell-2010: short-horizon deviations don't always revert "
        "-> horizon FROZEN). Vol gate (frozen): no new entry when 20d realised spread-vol is "
        "top-decile. FIXED 1:1:1 board ratio (gross-1 coefficient/notional weights), held -> turnover "
        "only on entry/exit. 1-day execution lag (W.shift(1)). Costs cost_bps/leg turnover.\n"
        "SCOPE=BROAD: the mechanism (processing-margin convergence anchored by a physical production "
        "function) is universal, so STAGE-2 (frozen signal, default params, holdout only): cattle "
        "feedlot crush (LE/GF/ZC), petroleum 3-2-1 crack (CL/RB/HO), hog feeding margin (HE/ZC) — all "
        "DISJOINT from the soy search universe; >=60% must be OOS-positive or this is a soy artifact.\n"
        "LEDGER: one trade per held leg-run; sectors are distinct commodity products; legs held "
        "together every position-day so single-name share ~1/N (<40%). EXPECTATIONS: 60d hold cap "
        "respected; vol gate keeps entries out of the top realised-vol decile."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default":      {},
        "band_15_85":   {"entry_lo": 0.15, "entry_hi": 0.85},
        "band_25_75":   {"entry_lo": 0.25, "entry_hi": 0.75},
        "hold_40":      {"max_hold": 40},
        "hold_90":      {"max_hold": 90},
        "lookback_189": {"lookback": 189},
    },
    scope="broad",
    generalization_universes=["cattle_crush", "crack_321", "hog_feed"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=3,
    expectations=[
        {"name": "max_hold_60",
         "claim": "no realised trade run exceeds the pre-registered 60-day max hold (<=63 w/ lag slack)",
         "check": _check_max_hold},
        {"name": "vol_gate_active",
         "claim": "fewer than the 10% unconditional rate of entries land in the top realised-vol decile",
         "check": _check_vol_gate},
    ],
)