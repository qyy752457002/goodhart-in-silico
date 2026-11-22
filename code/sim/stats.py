"""Statistical reporting helpers (methodology section 7): bootstrap confidence intervals, paired
Wilcoxon tests with Holm correction, rank agreement, and exact binomial intervals for coverage."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator = None) -> tuple[float, float]:
    """Percentile bootstrap interval for `stat` of i.i.d. values."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    if x.size == 1 or np.ptp(x) == 0:
        return (float(x[0]), float(x[0]))
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = stat(x[idx], axis=1) if stat in (np.mean, np.median) else np.array([stat(x[i]) for i in idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mean_ci(values, n_boot: int = 2000, alpha: float = 0.05, rng: np.random.Generator = None) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    lo, hi = bootstrap_ci(x, np.mean, n_boot, alpha, rng)
    return {"mean": float(x.mean()) if x.size else np.nan, "ci_lo": lo, "ci_hi": hi, "n": int(x.size)}


def cluster_bootstrap_ci(statistic, clusters, n_boot: int = 1000, alpha: float = 0.05,
                         rng: np.random.Generator = None) -> tuple[float, float]:
    """Percentile bootstrap that resamples whole clusters (organisations), for statistics of cells
    nested in organisations. `statistic(row_index_array) -> float`."""
    clusters = np.asarray(clusters)
    rng = rng if rng is not None else np.random.default_rng(0)
    ids, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == j) for j in range(len(ids))]
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(ids), size=len(ids))
        boot[b] = statistic(np.concatenate([members[j] for j in pick]))
    lo, hi = np.nanquantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_wilcoxon(x, y) -> dict:
    """Two-sided Wilcoxon signed-rank test of paired samples (x - y)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[~np.isnan(d)]
    out = {"n": int(d.size), "n_nonzero": int((d != 0).sum()),
           "mean_diff": float(d.mean()) if d.size else np.nan,
           "median_diff": float(np.median(d)) if d.size else np.nan}
    if out["n_nonzero"] == 0:
        return {**out, "statistic": 0.0, "p": 1.0}
    res = sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {**out, "statistic": float(res.statistic), "p": float(res.pvalue)}


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def spearman(x, y) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"rho": np.nan, "p": np.nan, "n": int(x.size)}
    res = sps.spearmanr(x, y)
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": int(x.size)}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial interval for a proportion (e.g. empirical coverage)."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(sps.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar p-value from the discordant counts (b: only A correct, c: only B correct)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * sps.binom.cdf(min(b, c), n, 0.5)))
