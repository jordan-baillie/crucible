"""Elite pool — the forge's EVOLUTIONARY memory. Keeps the top-K strategies by fitness (DSR); the director
MUTATES these (exploit) alongside fresh proposals (explore), so good constructions compound instead of being
re-discovered each night. Succinct by design: one jsonl, fitness=DSR, fitness-weighted sampling."""
import json
from pathlib import Path

POOL = Path("/root/research-wiki/.elite/pool.jsonl")
K = 12          # pool size
MIN_FIT = 0.5   # only genuinely promising runs enter (DSR > 0.5)


def _fitness(v: dict) -> float:
    if not v:
        return 0.0
    if v.get("dsr") is not None:
        return float(v["dsr"])                 # the deflated, multiple-testing-aware Sharpe = the natural fitness
    s, h = v.get("search_sharpe"), v.get("holdout_sharpe")
    return float(min(s, h)) if (s and h and s > 0 and h > 0) else 0.0  # fallback: search/holdout consistency


def _load() -> list:
    return [json.loads(l) for l in POOL.read_text().splitlines() if l.strip()] if POOL.exists() else []


def record(outcome: dict) -> None:
    fit = _fitness(outcome.get("verdict"))
    if fit <= MIN_FIT:
        return
    items = _load()
    items.append({"id": outcome.get("id"), "fitness": round(fit, 4), "title": outcome.get("title"),
                  "proposal": outcome.get("proposal"), "ts": outcome.get("ts")})
    items.sort(key=lambda x: x["fitness"], reverse=True)
    POOL.parent.mkdir(parents=True, exist_ok=True)
    POOL.write_text("".join(json.dumps(i) + "\n" for i in items[:K]))


def sample(rng) -> dict | None:
    """Fitness-weighted pick of an elite to evolve."""
    items = _load()
    if not items:
        return None
    w = [max(i["fitness"], 0.01) for i in items]
    r = rng.random() * sum(w)
    c = 0.0
    for it, wi in zip(items, w):
        c += wi
        if r <= c:
            return it
    return items[0]


def top(k: int = K) -> list:
    return _load()[:k]
