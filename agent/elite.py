"""Elite pool — the forge's EVOLUTIONARY memory, as a MAP-ELITES GRID (quality-diversity).

Cell = family × universe-bucket × turnover-band; each cell keeps only its BEST occupant (fitness=DSR).
Diversity is STRUCTURAL: a 10th Amihud variant can only displace the Amihud cell's occupant — it can
never crowd out the seasonal/carry/quality cells. This replaces the old top-K pool + family-downweighted
sampling (two reactive patches from the value×mom-hammering incident); the queue-level theme cap stays
(different layer). Port of QuantEvolve/MAP-Elites, adapted 2026-06-12 (tasks/SYNTHESIS_PLAN.md Stage 1a).

Legacy items (no 'cell' key) are re-binned on first write/read; collisions keep the higher DSR.
"""
import json

from crucible_paths import ELITE as POOL, WIKI

CLOSED_FAMILIES = WIKI / "decisions" / "closed-families.txt"
MAX_CELLS = 24  # global safety cap (grid is the real bound; in practice far fewer cells are occupied)
MIN_FIT = 0.5   # only genuinely promising runs enter (DSR > 0.5)


# --------------------------------------------------------------------------- cell geometry
def _family(item: dict) -> str:
    from agent.families import family_bucket
    return family_bucket((item.get("title") or "") or (item.get("proposal") or {}).get("premium", ""))


def _universe_bucket(market: str) -> str:
    """Coarse market/universe axis from the proposal's free-text 'market' field."""
    t = (market or "").lower()
    if "crypto" in t or "btc" in t or "perp" in t:
        return "crypto"
    if "futures" in t or "commodit" in t:
        return "futures"
    if "etf" in t and "small" not in t:
        return "etf"
    if "small" in t or "micro" in t:
        return "us_small"
    if "large" in t or "s&p" in t or "sp500" in t:
        return "us_large"
    if "equit" in t or "stock" in t:
        return "us_equity"
    return "other"


def _turnover_band(n_trades) -> str:
    """Crude turnover axis from verdict n_trades (total over the backtest). Bands chosen from the
    observed run_log distribution; 'unknown' for legacy items lacking a verdict summary."""
    if n_trades is None:
        return "unknown"
    n = int(n_trades)
    return "low" if n < 300 else ("med" if n < 1500 else "high")


def cell_of(item: dict) -> str:
    """Stable cell key. Stored items carry it; legacy items are derived (migration path)."""
    if item.get("cell"):
        return item["cell"]
    s = item.get("summary") or {}
    return "|".join((_family(item),
                     _universe_bucket((item.get("proposal") or {}).get("market", "")),
                     _turnover_band(s.get("n_trades"))))


# --------------------------------------------------------------------------- fitness (unchanged)
def _closed_families() -> set:
    """Family buckets CLOSED by decision (cf. decisions/CLOSED.md). Elites in these families are
    never sampled and never (re-)recorded — the exploit loop must not keep evolving a falsified
    premium (the value×mom-hammering failure mode, closed 2026-06-10)."""
    if not CLOSED_FAMILIES.exists():
        return set()
    return {l.strip() for l in CLOSED_FAMILIES.read_text().splitlines()
            if l.strip() and not l.startswith("#")}


def _fitness(v: dict) -> float:
    if not v:
        return 0.0
    if v.get("beta_confound"):
        return 0.0   # long-only-beta confound -> never seed the evolutionary exploit loop with it
    if v.get("dsr") is not None:
        return float(v["dsr"])                 # the deflated, multiple-testing-aware Sharpe = the natural fitness
    s, h = v.get("search_sharpe"), v.get("holdout_sharpe")
    return float(min(s, h)) if (s and h and s > 0 and h > 0) else 0.0  # fallback: search/holdout consistency


def _load() -> list:
    return [json.loads(l) for l in POOL.read_text().splitlines() if l.strip()] if POOL.exists() else []


def _grid(items: list, closed: set | None = None) -> dict:
    """cell -> best occupant. Re-bins legacy items; collisions keep higher fitness (the migration)."""
    closed = closed if closed is not None else _closed_families()
    g: dict[str, dict] = {}
    for it in items:
        if _family(it) in closed:
            continue
        c = cell_of(it)
        if c not in g or it["fitness"] > g[c]["fitness"]:
            g[c] = {**it, "cell": c}
    return g


# --------------------------------------------------------------------------- record / sample
def record(outcome: dict) -> None:
    v = outcome.get("verdict") or {}
    fit = _fitness(v)
    if fit <= MIN_FIT:
        return
    if _family(outcome) in _closed_families():
        return  # falsified family — do not seed the evolutionary loop with it
    from sdk.locks import FileLock
    with FileLock("elite-pool", ttl=60):
        _record_locked(outcome, fit, v)


def _record_locked(outcome: dict, fit: float, v: dict) -> None:
    item = {"id": outcome.get("id"), "fitness": round(fit, 4), "title": outcome.get("title"),
            "proposal": outcome.get("proposal"), "ts": outcome.get("ts"),
            # small verdict summary: feeds cell binning + the refine/orthogonal/crossover prompts
            "summary": {k: v.get(k) for k in
                        ("dsr", "holdout_sharpe", "search_sharpe", "n_trades", "scope", "tier")}}
    item["cell"] = cell_of(item)
    g = _grid(_load())
    if item["cell"] in g and g[item["cell"]]["fitness"] >= item["fitness"]:
        return  # cell already holds an equal-or-better elite
    g[item["cell"]] = item
    items = sorted(g.values(), key=lambda x: x["fitness"], reverse=True)[:MAX_CELLS]
    POOL.parent.mkdir(parents=True, exist_ok=True)
    POOL.write_text("".join(json.dumps(i) + "\n" for i in items))


def sample(rng) -> dict | None:
    """UNIFORM over occupied cells, then that cell's occupant. No fitness weighting needed —
    the grid already guarantees each occupant is its niche's best; uniform-over-cells is the
    diversity property (this is what the old family-downweighting approximated)."""
    g = _grid(_load())
    if not g:
        return None
    return g[rng.choice(sorted(g.keys()))]


def sample_pair(rng) -> tuple[dict, dict] | None:
    """Two elites from DIFFERENT families (crossover parents). The different-family rule is
    enforced HERE, structurally — not by prompt instruction."""
    g = _grid(_load())
    by_fam: dict[str, list] = {}
    for it in g.values():
        by_fam.setdefault(_family(it), []).append(it)
    fams = sorted(by_fam.keys())
    if len(fams) < 2:
        return None
    f1, f2 = rng.sample(fams, 2)
    return rng.choice(by_fam[f1]), rng.choice(by_fam[f2])


def top(k: int = MAX_CELLS) -> list:
    return sorted(_grid(_load()).values(), key=lambda x: x["fitness"], reverse=True)[:k]
