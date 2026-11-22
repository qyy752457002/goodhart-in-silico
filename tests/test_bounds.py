"""Finite-population confidence bounds (methodology 4.2 step 5)."""
import itertools
import zlib

import numpy as np
import pytest

from sim import bounds

POPULATIONS = {
    "low": lambda rng, N: rng.beta(1.0, 8.0, N),
    "bimodal": lambda rng, N: np.where(rng.random(N) < 0.25, rng.uniform(0.8, 1.0, N), rng.uniform(0.0, 0.1, N)),
    "uniform": lambda rng, N: rng.random(N),
    "high": lambda rng, N: rng.beta(8.0, 1.5, N),
}


def test_hoeffding_serfling_formula():
    x = np.array([0.1, 0.2, 0.3])
    expected = 0.2 + np.sqrt((1 - 2 / 10) * np.log(1 / 0.05) / 6)
    assert bounds.hoeffding_serfling_ucb(x, 10, 0.05) == pytest.approx(expected)
    assert bounds.hoeffding_serfling_ucb(np.array([]), 10, 0.05) == 1.0
    assert bounds.hoeffding_serfling_ucb(np.full(3, 0.99), 10, 0.05) == 1.0       # capped at 1
    assert bounds.plugin_mean(x, 10, 0.05) == pytest.approx(0.2)
    assert bounds.plugin_mean(np.array([]), 10, 0.05) == 0.0


def test_betting_edge_cases():
    x = np.array([0.2, 0.4, 0.1])
    assert bounds.betting_ucb_wor(x, 3, 0.1) == pytest.approx(x.mean())         # census: no uncertainty
    assert bounds.betting_ucb_wor(np.array([]), 5, 0.1) == 1.0
    u = bounds.betting_ucb_wor(x, 20, 0.1)
    assert x.sum() / 20 <= u <= (x.sum() + 17) / 20                              # logical limits
    assert bounds.betting_ucb_wor(x, 20, 0.01) >= bounds.betting_ucb_wor(x, 20, 0.2)
    assert bounds.betting_ucb_wor(x, 4, 0.1) <= (x.sum() + 1) / 4 + 1e-12       # one unseen unit left
    with pytest.raises(ValueError):
        bounds.betting_ucb_wor(x, 2, 0.1)
    with pytest.raises(ValueError):
        bounds.betting_ucb_wor(np.array([0.5, 1.5]), 10, 0.1)
    with pytest.raises(ValueError):
        bounds.betting_ucb_wor(x, 10, 1.5)
    with pytest.raises(ValueError):
        bounds.betting_ucb_wor(x, 10, 0.1, strategy="martingale")


def reference_betting_ucb(x, N, delta, c=0.75, grid=4001):
    """Direct transcription of the definition: scan candidate means, multiply the capital."""
    n = len(x)
    t = np.arange(1, n + 1)
    mu_hat = (0.5 + np.cumsum(x)) / (t + 1)
    var = (0.25 + np.cumsum((x - mu_hat) ** 2)) / (t + 1)
    var_prev = np.r_[0.25, var[:-1]]
    bets = np.sqrt(2 * np.log(1 / delta) / (n * var_prev))
    m_hi = min(1.0, (x.sum() + N - n) / N)
    for m in np.linspace(x.sum() / N, m_hi, grid):
        capital, peak, seen = 1.0, 1.0, 0.0
        for i in range(n):
            mu = (N * m - seen) / (N - i)
            lam = min(bets[i], c / max(1 - mu, 1e-12))
            capital *= 1 + lam * (mu - x[i])
            peak = max(peak, capital)
            seen += x[i]
        if peak >= 1 / delta:
            return m
    return m_hi


@pytest.mark.parametrize("seed", range(5))
def test_betting_matches_reference_implementation(seed):
    rng = np.random.default_rng(seed)
    x = rng.beta(1.5, 5.0, size=12)
    got = bounds.betting_ucb_wor(x, 40, 0.05)
    assert got == pytest.approx(reference_betting_ucb(x, 40, 0.05), abs=5e-4)


@pytest.mark.parametrize("strategy", ["prpl", "dkelly"])
@pytest.mark.parametrize("shape", sorted(POPULATIONS))
@pytest.mark.parametrize("N,n", [(30, 10), (15, 4)])
def test_betting_coverage(strategy, shape, N, n):
    """P(mu <= UCB) >= 1 - delta over repeated sampling without replacement."""
    rng = np.random.default_rng(zlib.crc32(f"{strategy}-{shape}-{N}".encode()))
    delta, reps = 0.1, 1500
    pop = np.clip(POPULATIONS[shape](rng, N), 0, 1)
    mu = pop.mean()
    misses = sum(bounds.betting_ucb_wor(rng.permutation(pop)[:n], N, delta, strategy=strategy) < mu - 1e-9
                 for _ in range(reps))
    assert misses / reps <= delta + 3 * np.sqrt(delta * (1 - delta) / reps)


def test_betting_is_tighter_than_hoeffding_on_concentrated_data():
    rng = np.random.default_rng(3)
    x = np.clip(rng.normal(0.1, 0.03, 25), 0, 1)
    assert bounds.betting_ucb_wor(x, 100, 0.05) < bounds.hoeffding_serfling_ucb(x, 100, 0.05)


def test_t_bound_and_registry():
    x = np.array([0.1, 0.2, 0.3, 0.2])
    assert bounds.t_ucb_wor(x, 4, 0.05) == pytest.approx(x.mean())              # census
    assert bounds.t_ucb_wor(x, 100, 0.05) > x.mean()
    assert np.isinf(bounds.t_ucb_wor(x[:1], 100, 0.05))
    assert bounds.get_ucb("hoeffding_serfling") is bounds.hoeffding_serfling_ucb
    fn = bounds.get_ucb("betting", strategy="dkelly")
    assert fn(x, 20, 0.1) == pytest.approx(bounds.betting_ucb_wor(x, 20, 0.1, strategy="dkelly"))
    with pytest.raises(ValueError):
        bounds.get_ucb("bootstrap")


def test_exceedance_tail_matches_enumeration():
    n_pop, n = 7, 3
    values = np.arange(n_pop)
    subsets = list(itertools.combinations(range(n_pop), n))
    for rank in (1, 2, 3):
        for k in range(0, n_pop - n + 2):
            hits = 0
            for s in subsets:
                threshold = np.sort(values[list(s)])[rank - 1]
                unsampled = np.setdiff1d(values, values[list(s)])
                hits += int((unsampled > threshold).sum() >= k)
            assert bounds.exceedance_tail(n_pop, n, rank, k) == pytest.approx(hits / len(subsets))


def test_exceedance_count_ucb():
    # with n = 5 of 15 sampled, at most K unsampled units exceed the sample maximum w.p. >= 1 - delta
    k = bounds.exceedance_count_ucb(15, 5, 5, 0.025)
    assert bounds.exceedance_tail(15, 5, 5, k + 1) <= 0.025 < bounds.exceedance_tail(15, 5, 5, k)
    assert bounds.exceedance_count_ucb(15, 5, 5, 0.5) <= k                       # less confidence -> fewer
    assert bounds.exceedance_count_ucb(15, 5, 1, 0.025) >= k                     # lower threshold -> more
    assert bounds.exceedance_count_ucb(6, 6, 6, 0.1) == 0                        # nothing unsampled
