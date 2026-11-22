"""Welfare algebra (methodology 1.1, 1.4) and package search (1.2, Proposition 3)."""
import numpy as np
import pytest

from sim import optimize, welfare
from sim.optimize import PackageSpace
from sim.welfare import Affine


def brute_force_utility(prob, i, a):
    """u_i(a) straight from the definition: mean over the org's cells of phi(s, a_q)."""
    cells = np.flatnonzero(prob.cell_org == i)
    vals = [prob.sigma[k] if a[prob.cell_dec[k]] == 1 else 1.0 - prob.sigma[k] for k in cells]
    return float(np.mean(vals))


def test_org_utility_matches_definition(small_problem, rng):
    prob = small_problem
    org = welfare.org_affine_from_sigma(prob, prob.sigma)
    for _ in range(20):
        a = rng.integers(0, 2, prob.n_dec)
        expected = [brute_force_utility(prob, i, a) for i in range(prob.n_org)]
        np.testing.assert_allclose(org.at(a), expected, atol=1e-12)


def test_group_welfare_is_mean_of_org_utilities(small_problem, rng):
    prob = small_problem
    org = welfare.org_affine_from_sigma(prob, prob.sigma)
    grp = welfare.group_mean(org, prob.g_index, prob.n_groups)
    a = rng.integers(0, 2, prob.n_dec)
    u = org.at(a)
    for g in range(prob.n_groups):
        assert grp.at(a)[g] == pytest.approx(u[prob.g_index == g].mean())
    # utilities stay in [0, 1]
    table = org.table()
    assert table.min() >= -1e-12 and table.max() <= 1 + 1e-12


def test_enumerate_scores_equals_matrix_product(rng):
    c, D = rng.normal(size=4), rng.normal(size=(4, 7))
    expected = welfare.score(welfare.all_packages(7), c, D)
    np.testing.assert_allclose(welfare.enumerate_scores(c, D), expected, atol=1e-12)
    aff = Affine(c, D)
    np.testing.assert_allclose(aff.table().min(0), aff.minimum_over_packages(), atol=1e-12)
    np.testing.assert_allclose(aff.table().max(0), aff.maximum_over_packages(), atol=1e-12)
    np.testing.assert_allclose(aff.at(welfare.all_packages(7)), expected, atol=1e-12)


def test_power_mean_objectives():
    W = np.array([[0.2, 0.8], [0.5, 0.5]])
    np.testing.assert_allclose(welfare.power_mean(W, 1), [0.5, 0.5])
    np.testing.assert_allclose(welfare.power_mean(W, 0), [0.4, 0.5])
    np.testing.assert_allclose(welfare.power_mean(W, -np.inf), [0.2, 0.5])
    # weights: equal weights reproduce the unweighted paths, unequal weights tilt the mean
    for p in (1, 0, -np.inf, -1.0, 2.0):
        np.testing.assert_allclose(welfare.power_mean(W, p, np.array([1.0, 1.0])),
                                   welfare.power_mean(W, p), atol=1e-12)
    np.testing.assert_allclose(welfare.power_mean(W, 1, np.array([3.0, 1.0])), [0.35, 0.5])
    # power means are ordered in p
    vals = [welfare.power_mean(W, p)[0] for p in (-np.inf, -1.0, 0, 1, 2.0, np.inf)]
    assert all(x <= y + 1e-12 for x, y in zip(vals, vals[1:]))
    with pytest.raises(ValueError):
        welfare.objective_p("median")


def test_package_space_indexing():
    space = PackageSpace(5)
    assert space.size == 32 and space.ed_index == 31 and space.ifrs_index == 0
    np.testing.assert_array_equal(space.package(space.ed_index), np.ones(5))
    np.testing.assert_array_equal(space.popcount, space.packages.sum(1))
    for k in (0, 7, 19, 31):
        assert space.index(space.package(k)) == k
    np.testing.assert_array_equal(space.hamming(space.ed_index), 5 - space.popcount)
    assert space.hamming_between(0b10100, 0b00110) == 2
    with pytest.raises(ValueError):
        space.index([0, 1, 2, 0, 0])


def test_argmax_breaks_ties_towards_the_draft():
    space = PackageSpace(3)
    values = np.zeros(8)
    assert space.argmax(values) == space.ed_index          # all tied -> keep every ED proposal
    values[[0b001, 0b110]] = 1.0
    assert space.argmax(values) == 0b110                    # tie between 1 and 2 retained proposals
    values[0b010] = 2.0
    assert space.argmax(values) == 0b010                    # a strict maximum always wins


@pytest.mark.parametrize("seed", range(6))
def test_exact_solvers_agree_with_enumeration(seed):
    rng = np.random.default_rng(seed)
    G, Q = 4, 8
    group = Affine(rng.uniform(0.2, 0.6, G), rng.normal(0, 0.1, (G, Q)))
    space = PackageSpace(Q)
    table = group.table()
    # utilitarian: per-proposal weighted majority
    a_cf = optimize.solve(group, 1, method="closed_form")
    assert group.at(a_cf).mean() == pytest.approx(table.mean(1).max())
    w = np.array([0.4, 0.3, 0.2, 0.1])
    a_w = optimize.solve_utilitarian(group, w)
    assert group.at(a_w) @ w == pytest.approx((table @ w).max())
    # max-min: MILP
    a_milp, val = optimize.solve_maxmin_milp(group)
    assert val == pytest.approx(table.min(1).max(), abs=1e-7)
    assert group.at(a_milp).min() == pytest.approx(val)
    # enumeration through the common entry point
    for p in (1, 0, -np.inf):
        a = optimize.solve(group, p, space=space, method="enumerate")
        assert welfare.power_mean(group.at(a), p) == pytest.approx(welfare.power_mean(table, p).max())


def test_best_of_n_law():
    M = 64
    for N in (1, 2, 8, 63, 64):
        pmf = optimize.best_of_n_pmf(M, N)
        assert len(pmf) == M - N + 1
        assert pmf.sum() == pytest.approx(1.0) and (pmf >= 0).all()
    np.testing.assert_allclose(optimize.best_of_n_pmf(M, 1), np.full(M, 1 / M))
    np.testing.assert_allclose(optimize.best_of_n_pmf(M, M), [1.0])
    # against literally drawing N packages without replacement
    rng = np.random.default_rng(0)
    N, reps = 8, 40000
    best = np.array([rng.choice(M, N, replace=False).min() for _ in range(reps)])
    empirical = np.bincount(best, minlength=M - N + 1) / reps
    assert np.abs(empirical - optimize.best_of_n_pmf(M, N)).max() < 0.01
    with pytest.raises(ValueError):
        optimize.best_of_n_pmf(M, M + 1)


def test_best_of_n_is_stable_at_full_scale():
    pmf = optimize.best_of_n_pmf(2 ** 18, 2 ** 10)
    assert pmf.sum() == pytest.approx(1.0)
    assert np.isfinite(pmf).all() and pmf[0] == pytest.approx(2 ** 10 / 2 ** 18, rel=1e-6)
    assert optimize.kl_best_of_n(1) == pytest.approx(0.0)


def test_hamming_ball_argmax():
    rng = np.random.default_rng(1)
    space = PackageSpace(6)
    values = rng.normal(size=space.size)
    idx = optimize.hamming_ball_argmax(space, values)
    assert idx[0] == space.ed_index and idx[-1] == int(np.argmax(values))
    best = values[idx]
    assert (np.diff(best) >= -1e-12).all()                  # a larger radius can only help
    for r, k in enumerate(idx):
        assert space.hamming_between(k, space.ed_index) <= r
