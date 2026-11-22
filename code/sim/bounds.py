"""Finite-population confidence bounds for a simple random sample drawn without replacement.

Setting: a fixed population of N values in [0, 1] with mean mu; a sample x_1, ..., x_n is drawn
without replacement (the pilot organisations of one stakeholder group). Every function returns an
upper confidence bound U with P(mu <= U) >= 1 - delta.

Methodology map (03_methodology.md): section 4.2 step 5.
- `betting_ucb_wor`        main choice: betting confidence sequence for sampling without replacement
                           (Waudby-Smith & Ramdas, JRSS-B 2024, section 5);
- `hoeffding_serfling_ucb` closed-form fallback (Serfling, 1974), used in the Day-1 precheck;
- `exceedance_count_ucb`   distribution-free bound on a *count*, for the GUARD-count ablation.

All mean bounds share the signature (x, N, delta) -> float so they are interchangeable.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def hoeffding_serfling_ucb(x: np.ndarray, N: int, delta: float) -> float:
    """UCB on the population mean of [0,1] values from a without-replacement sample (Serfling 1974)."""
    n = len(x)
    if n == 0:
        return 1.0
    f = (n - 1) / N
    return min(1.0, float(np.mean(x)) + np.sqrt((1 - f) * np.log(1 / delta) / (2 * n)))


def plugin_mean(x: np.ndarray, N: int, delta: float) -> float:
    """The sample mean itself: no confidence margin (used by the GUARD-est ablation)."""
    return float(np.mean(x)) if len(x) else 0.0


def _predictable_bets(x: np.ndarray, delta: float) -> np.ndarray:
    """Predictable plug-in empirical-Bernstein bets for a fixed sample size n:
    ell_t = sqrt(2 log(1/delta) / (n * var_{t-1})), with the regularised running variance of
    Waudby-Smith & Ramdas (2024, eq. 26). ell_t depends only on x_1..x_{t-1}."""
    n = x.size
    t = np.arange(1, n + 1)
    mu_hat = (0.5 + np.cumsum(x)) / (t + 1)
    var_hat = (0.25 + np.cumsum((x - mu_hat) ** 2)) / (t + 1)
    var_prev = np.concatenate(([0.25], var_hat[:-1]))
    return np.sqrt(2.0 * np.log(1.0 / delta) / (n * var_prev))


def betting_ucb_wor(x: np.ndarray, N: int, delta: float, *, c: float = 0.75, strategy: str = "prpl",
                    running: bool = True, tol: float = 1e-6) -> float:
    """Upper confidence bound from a betting confidence sequence for sampling without replacement.

    `x` must be in *sampling order*: a simple random sample listed in a uniformly random
    (exchangeable) order. The bound is not permutation-invariant, and an order that merely "does
    not look at the values" is not enough -- e.g. a stratified sample listed stratum by stratum
    breaks the martingale property and with it the guarantee.

    For a candidate mean m, the conditional mean of draw t given the past is
        mu_t(m) = (N m - sum_{i<t} x_i) / (N - t + 1),
    and the capital process
        K_t(m) = prod_{i<=t} (1 + lam_i(m) * (mu_i(m) - x_i)),   0 <= lam_i(m) <= c / (1 - mu_i(m)),
    is a non-negative martingale at the true mean, so P(exists t: K_t(mu) >= 1/delta) <= delta
    (Ville). K_t(m) is non-decreasing in m for the bets used here, so the rejected set is an
    up-set and the bound is the smallest rejected m.

    strategy: 'prpl'   predictable plug-in empirical-Bernstein bets, truncated at c / (1 - mu_t);
              'dkelly' equal-weight mixture of constant-fraction bets (diversified Kelly).
    running:  use max_t K_t (time-uniform, never worse) rather than K_n only.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return 1.0
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta}")
    if N < n:
        raise ValueError(f"population size N={N} is smaller than the sample size n={n}")
    if x.min() < -1e-12 or x.max() > 1 + 1e-12:
        raise ValueError("values must lie in [0, 1]")
    x = np.clip(x, 0.0, 1.0)
    total = float(x.sum())
    m_lo = total / N                       # the unseen values cannot be below 0 ...
    m_hi = min(1.0, (total + (N - n)) / N)  # ... nor above 1
    if n == N or m_hi - m_lo <= tol:
        return m_hi
    seen_before = np.concatenate(([0.0], np.cumsum(x)[:-1]))  # sum of the first t-1 draws
    remaining = (N - np.arange(n)).astype(float)               # population size before draw t
    log_threshold = np.log(1.0 / delta)

    if strategy == "prpl":
        bets = _predictable_bets(x, delta)
        fractions = None
    elif strategy == "dkelly":
        bets = None
        fractions = c * np.array([0.25, 0.5, 0.75, 1.0])
    else:
        raise ValueError(f"unknown betting strategy {strategy!r} (use 'prpl' or 'dkelly')")

    def rejected(m: np.ndarray) -> np.ndarray:
        mu = (N * m[:, None] - seen_before[None, :]) / remaining[None, :]
        mu = np.clip(mu, 0.0, 1.0)
        room = np.maximum(1.0 - mu, 1e-12)
        if fractions is None:
            lam = np.minimum(bets[None, :], c / room)
            log_k = np.cumsum(np.log1p(lam * (mu - x[None, :])), axis=1)
        else:
            # factor = 1 - theta + theta * (1 - x) / (1 - mu) for each constant fraction theta
            ratio = ((1.0 - x)[None, :] / room)[None, :, :]
            log_each = np.cumsum(np.log(1.0 - fractions[:, None, None] + fractions[:, None, None] * ratio),
                                 axis=2)
            peak = log_each.max(axis=0)
            log_k = peak + np.log(np.exp(log_each - peak[None, :, :]).mean(axis=0))
        stat = log_k.max(axis=1) if running else log_k[:, -1]
        return stat >= log_threshold

    if not rejected(np.array([m_hi]))[0]:
        return m_hi
    if rejected(np.array([m_lo]))[0]:
        return m_lo
    lo, hi = m_lo, m_hi  # invariant: lo not rejected, hi rejected
    while hi - lo > tol:
        grid = np.linspace(lo, hi, 34)[1:-1]
        rej = rejected(grid)
        if rej.any():
            first = int(np.argmax(rej))
            hi = grid[first]
            lo = grid[first - 1] if first > 0 else lo
        else:
            lo = grid[-1]
    return float(hi)


def t_ucb_wor(x: np.ndarray, N: int, delta: float) -> float:
    """Student-t upper bound with the finite-population correction. Asymptotic ("semi-safe" in the
    sense of Thomas et al., 2015): not a finite-sample guarantee. Values need not lie in [0, 1]."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.inf
    if n == 1:
        return np.inf
    fpc = max(0.0, 1.0 - n / N)
    se = float(np.std(x, ddof=1)) / np.sqrt(n) * np.sqrt(fpc)
    return float(np.mean(x) + sps.t.ppf(1.0 - delta, n - 1) * se)


def hoeffding_serfling_halfwidth(n: int, N: int, delta: float, value_range: float = 1.0) -> float:
    """One-sided Hoeffding-Serfling margin for values in an interval of length `value_range`."""
    if n == 0:
        return np.inf
    f = (n - 1) / N
    return float(value_range * np.sqrt((1 - f) * np.log(1 / delta) / (2 * n)))


UCB_METHODS = {
    "betting": betting_ucb_wor,
    "hoeffding_serfling": hoeffding_serfling_ucb,
    "plugin": plugin_mean,
}


def get_ucb(name: str, **kwargs):
    """Look up a mean bound by name; keyword arguments are bound to it (e.g. the betting strategy)."""
    if name not in UCB_METHODS:
        raise ValueError(f"unknown confidence bound {name!r}; choose from {sorted(UCB_METHODS)}")
    fn = UCB_METHODS[name]
    if not kwargs:
        return fn
    return lambda x, N, delta: fn(x, N, delta, **kwargs)


# ----------------------------------------------------------------------------------------------
# Counting bound (GUARD-count)
# ----------------------------------------------------------------------------------------------

def exceedance_tail(n_pop: int, n: int, rank: int, k: int) -> float:
    """P(at least k unsampled units exceed the `rank`-th smallest sampled value).

    Population of `n_pop` exchangeable units, `n` of them sampled without replacement. Order the
    population from largest to smallest: at least k unsampled units beat the rank-th smallest
    sampled value iff the top (k + n - rank) units contain at most (n - rank) sampled units, and
    the number of sampled units among the top j is Hypergeometric(n_pop, n, j). The probability is
    exact for distinct values and conservative with ties (strict exceedance).
    """
    if not 1 <= rank <= n <= n_pop:
        raise ValueError(f"need 1 <= rank <= n <= n_pop, got rank={rank}, n={n}, n_pop={n_pop}")
    if k <= 0:
        return 1.0
    if k > n_pop - n:
        return 0.0
    top = k + n - rank
    return float(sps.hypergeom.cdf(n - rank, n_pop, n, top))


def exceedance_count_ucb(n_pop: int, n: int, rank: int, delta: float) -> int:
    """Smallest K such that P(more than K unsampled units exceed the rank-th smallest sampled
    value) <= delta. With probability >= 1 - delta at most K unsampled units exceed it."""
    if not 1 <= rank <= n <= n_pop:
        raise ValueError(f"need 1 <= rank <= n <= n_pop, got rank={rank}, n={n}, n_pop={n_pop}")
    m = n_pop - n
    if m == 0:
        return 0
    k = np.arange(1, m + 1)                                         # events "at least k exceed"
    tails = sps.hypergeom.cdf(n - rank, n_pop, n, k + n - rank)     # same formula as exceedance_tail
    small = np.flatnonzero(tails <= delta)
    return int(small[0]) if small.size else m
