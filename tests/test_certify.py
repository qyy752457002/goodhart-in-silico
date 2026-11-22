"""Base sets, mis-set mass, Lemma 1 and the uniform bounds of Theorem 1 (methodology 4.2-4.3)."""
import numpy as np
import pytest

from sim import bounds, certify, welfare
from sim.data import STANCES
from sim.engine import DesignEngine, EngineSettings
from sim.guard import certificate_bounds
from sim.pilot import draw_pilot
from sim.simulators.base import SimulatorOutput


def random_sim(prob, seed):
    rng = np.random.default_rng(seed)
    return SimulatorOutput("random", rng.dirichlet(np.full(len(STANCES), 0.5), size=prob.n_cells))


def test_base_set_rules():
    sigma_vec = np.array([1.0, 0.75, 0.5, 0.25, 0.25, 0.0])
    probs = np.array([[0.5, 0.3, 0.1, 0.05, 0.05, 0.0],
                      [0.0, 0.0, 0.1, 0.2, 0.3, 0.4]])
    top1 = certify.base_sets(probs, sigma_vec, "top1")
    np.testing.assert_array_equal(top1.member.sum(1), [1, 1])
    np.testing.assert_allclose(top1.sigma_lo, [1.0, 0.0])
    np.testing.assert_allclose(top1.sigma_hi, [1.0, 0.0])
    top2 = certify.base_sets(probs, sigma_vec, "topk", k=2)
    np.testing.assert_allclose(top2.sigma_lo, [0.75, 0.0])
    np.testing.assert_allclose(top2.sigma_hi, [1.0, 0.25])
    mass = certify.base_sets(probs, sigma_vec, "mass", mass=0.85)
    np.testing.assert_array_equal(mass.member.sum(1), [3, 3])                    # 0.5+0.3+0.1, 0.4+0.3+0.2
    np.testing.assert_allclose(mass.sigma_lo, [0.5, 0.0])
    full = certify.base_sets(probs, sigma_vec, "mass", mass=1.0)
    assert full.member[1].sum() == 4 or full.member[1].sum() == 6                # zero-mass classes optional
    np.testing.assert_allclose(top1.beta_bar, [1.0, 1.0])
    np.testing.assert_allclose(top2.beta_bar, [0.75, 0.75])
    with pytest.raises(ValueError):
        certify.base_sets(probs, sigma_vec, "conformal")


@pytest.mark.parametrize("rule,kw", [("top1", {}), ("topk", {"k": 2}), ("mass", {"mass": 0.8})])
@pytest.mark.parametrize("miss_rule", ["sigma_hull", "label"])
def test_lemma_1_sandwich(small_problem, rule, kw, miss_rule):
    """lambda_i(a) - S_i^- <= u_i(a) <= upsilon_i(a) + S_i^+ for every organisation and package."""
    prob = small_problem
    sim = random_sim(prob, 5)
    bs = certify.base_sets(sim.probs, prob.sigma_vec, rule, **kw)
    lam, ups = certify.org_bound_affines(prob.cell_org, prob.cell_dec, prob.n_org, prob.n_dec, bs)
    miss = certify.cell_miss(bs, prob.sigma, prob.stance_idx, miss_rule)
    s_minus, s_plus, s_max = certify.org_misset_mass(prob.cell_org, prob.n_org, bs, miss)
    truth = welfare.org_affine_from_sigma(prob, prob.sigma).table()              # (M, n_org)
    assert (lam.table() - s_minus[None, :] <= truth + 1e-12).all()
    assert (truth <= ups.table() + s_plus[None, :] + 1e-12).all()
    np.testing.assert_allclose(s_minus, s_plus)                                  # symmetric cell utility
    assert (s_minus <= s_max + 1e-12).all()
    # the hull rule never counts more misses than the literal rule
    hull = certify.cell_miss(bs, prob.sigma, prob.stance_idx, "sigma_hull")
    label = certify.cell_miss(bs, prob.sigma, prob.stance_idx, "label")
    assert not (hull & ~label).any()


def _engine(prob, sim, **kw):
    return DesignEngine(prob, sim, EngineSettings(**kw))


def test_mean_bounds_reproduce_the_precheck_interface(small_problem, noisy_sim):
    """`mean_bounds` with Hoeffding-Serfling equals the Day-1 `guard_mean_bound`, except that tau is
    capped at its largest possible value instead of the number of targets."""
    prob = small_problem
    e = _engine(prob, noisy_sim)
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.5, np.random.default_rng(0))
    delta_g = 0.1 / prob.n_groups
    Cg, Dg, tau_old = certify.guard_mean_bound(pilot.known, e.gi, e.G, e.true_org.c, e.true_org.D,
                                               e.lam_org.c, e.lam_org.D, e.s_minus, delta_g)
    new = certify.mean_bounds(e.gi, e.G, pilot.known, pilot.calib_order, e.true_org, e.lam_org, e.ups_org,
                              e.s_minus, e.s_plus, e.s_max, delta_g, bounds.hoeffding_serfling_ucb)
    cap = np.array([e.s_max[(e.gi == g) & ~pilot.known].sum() for g in range(e.G)])
    np.testing.assert_allclose(new.tau_minus, np.minimum(tau_old, cap))
    np.testing.assert_allclose(new.lower.D, Dg)
    np.testing.assert_allclose(new.lower.c, Cg + (tau_old - new.tau_minus) / e.sizes)


def test_lower_bound_is_valid_whenever_tau_covers_the_true_mass(small_problem, noisy_sim):
    """The deterministic half of Theorem 1: tau_g >= sum of the targets' true mis-set mass implies
    L_g(a) <= W_g(a) <= U_g(a) for every package."""
    prob = small_problem
    e = _engine(prob, noisy_sim)
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.3, np.random.default_rng(1))
    true_tau = np.array([e.s_minus[(e.gi == g) & ~pilot.known].sum() for g in range(e.G)])
    exact = certify._assemble(e.gi, e.G, pilot.known, ~pilot.known, e.true_org, e.lam_org, e.ups_org,
                              true_tau, true_tau, "oracle", True)
    check = certify.uniform_validity(exact, e.true_group)
    assert check["lower_valid"] and check["upper_valid"]
    # brute force over all packages agrees with the closed-form check
    assert (exact.lower.table() <= e.W_true + 1e-9).all() and (e.W_true <= exact.upper.table() + 1e-9).all()
    # shrinking tau below the truth must eventually break validity for an imperfect simulator
    too_small = certify._assemble(e.gi, e.G, pilot.known, ~pilot.known, e.true_org, e.lam_org, e.ups_org,
                                  np.zeros(e.G), np.zeros(e.G), "zero", False)
    assert not certify.uniform_validity(too_small, e.true_group)["lower_valid"]


@pytest.mark.parametrize("kind", ["betting", "hoeffding_serfling", "count"])
def test_certified_mass_covers_the_truth(small_problem, noisy_sim, kind):
    """P(tau_g >= true target mass for all g) >= 1 - delta over random stratified pilots."""
    prob = small_problem
    e = _engine(prob, noisy_sim, delta=0.2)
    rng = np.random.default_rng(42)
    reps, fails = 400, 0
    for _ in range(reps):
        pilot = draw_pilot(prob.g_index, prob.n_groups, 0.4, rng)
        b = certificate_bounds(e, pilot, kind)
        true_tau = np.array([e.s_minus[(e.gi == g) & ~pilot.known].sum() for g in range(e.G)])
        ok = (b.tau_minus >= true_tau - 1e-9).all()
        assert certify.uniform_validity(b, e.true_group)["lower_valid"] or not ok
        fails += int(not ok)
    assert fails / reps <= 0.2 + 3 * np.sqrt(0.2 * 0.8 / reps)


def test_two_sided_split_is_more_conservative(small_problem, noisy_sim):
    prob = small_problem
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.4, np.random.default_rng(2))
    loose = certificate_bounds(_engine(prob, noisy_sim, split_two_sided=True), pilot, "betting")
    tight = certificate_bounds(_engine(prob, noisy_sim), pilot, "betting")
    assert (loose.tau_minus >= tight.tau_minus - 1e-12).all()
    np.testing.assert_allclose(tight.tau_minus, tight.tau_plus)


def test_pooled_bound_shares_one_total(small_problem, noisy_sim):
    prob = small_problem
    e = _engine(prob, noisy_sim)
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.4, np.random.default_rng(3))
    b = certificate_bounds(e, pilot, "pooled")
    assert not b.guaranteed and b.kind == "pooled"
    per_target = b.tau_minus / b.n_target
    np.testing.assert_allclose(per_target, per_target[0])                        # same mass per target


def test_trivial_clipping_and_certificate(small_problem, noisy_sim):
    prob = small_problem
    e = _engine(prob, noisy_sim)
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.2, np.random.default_rng(4))
    b = certificate_bounds(e, pilot, "hoeffding_serfling")
    a = np.ones(prob.n_dec)
    assert (b.lower_at(a) >= b.known_part.at(a) - 1e-12).all()
    assert (b.lower_at(a) >= b.lower_at(a, clip_trivial=False) - 1e-12).all()
    assert (b.upper_at(a) <= b.upper_at(a, clip_trivial=False) + 1e-12).all()
    assert (b.lower_at(a) >= 0).all() and (b.upper_at(a) <= 1 + 1e-12).all()
    for p in (1, 0, -np.inf):
        assert b.certificate(a, p) == pytest.approx(float(welfare.power_mean(b.lower_at(a), p)))
    # a census pilot pins the welfare exactly
    full = draw_pilot(prob.g_index, prob.n_groups, 1.0, np.random.default_rng(5))
    exact = certificate_bounds(e, full, "betting")
    np.testing.assert_allclose(exact.lower_at(a), e.true_group.at(a))
    np.testing.assert_allclose(exact.upper_at(a), e.true_group.at(a))
