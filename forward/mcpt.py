"""Monte Carlo Permutation Test (MCPT) — stage-2 fluke-confirmation (the statistical sibling of generalization).

Answers: "could this strategy have looked this good on data with NO real structure to exploit?" It permutes
the input panel (shuffles each asset's daily returns -> destroys serial + cross-sectional predictability while
preserving the marginal distribution), re-runs the SAME signal, and computes p = P(permuted Sharpe >= real).
A real edge -> LOW p (real result sits in the tail of the no-edge distribution); a curve-fit -> high p (the
real result is buried inside what structureless data produces).

Run on CANDIDATES only (it re-runs the signal N times). Best for price-panel strategies (sep_panel/yf_panel).
    python3 forward/mcpt.py strategies.<module> [N]
"""
import importlib
import sys

sys.path.insert(0, "/root/hephaestus")
sys.path.insert(0, "/root/hephaestus/forward")
import numpy as np
import pandas as pd


def _sharpe(r, ann=252):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * np.sqrt(ann)) if len(r) > 20 and r.std() > 0 else 0.0


def permute_panel(panel, rng):
    """Shuffle each asset's daily returns independently (destroy structure, keep marginal vol), then
    reconstruct a price panel for the signal to consume."""
    rets = panel.pct_change()
    out = {}
    for c in rets.columns:
        s = rets[c]
        mask = s.notna()
        perm = s.copy()
        perm[mask] = rng.permutation(s[mask].values)
        out[c] = perm
    sh = pd.DataFrame(out)
    px = (1 + sh.fillna(0)).cumprod()
    return px.div(px.bfill().iloc[0]).mul(panel.bfill().iloc[0])  # rescale to original start levels


def mcpt(module_name, n=50, seed=0):
    m = importlib.import_module(module_name)
    panel = m.load_data()
    real = _sharpe(m.signal(panel, **m.SPEC.default_params)[0])
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n):
        try:
            perm.append(_sharpe(m.signal(permute_panel(panel, rng), **m.SPEC.default_params)[0]))
        except Exception:
            continue
    perm = np.array(perm) if perm else np.array([0.0])
    pval = (np.sum(perm >= real) + 1) / (len(perm) + 1)
    verdict = "PASS (edge distinguishable from no-structure)" if pval <= 0.05 else \
              "FAIL (looks like luck on structureless data)"
    print(f"MCPT {module_name}: real Sharpe {real:.2f} | {len(perm)} perms | "
          f"perm mean {perm.mean():.2f} (95th {np.percentile(perm, 95):.2f}) | p-value {pval:.3f} -> {verdict}")
    return real, float(pval)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mcpt(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    else:
        print("usage: python3 forward/mcpt.py <strategy.module> [N]")
