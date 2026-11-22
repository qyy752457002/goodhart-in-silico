"""GUARD (Algorithm 1), its variants, and the paper's toy example (02 section 3)."""
import itertools

import numpy as np
import pytest

from sim import guard, welfare
from sim.engine import DesignEngine, EngineSettings
from sim.pilot import Pilot, draw_pilot, pilot_from_mask
from sim.simulators import synthetic as synth


def all_pilots(prob, sizes):
    """Every stratified pilot with the given group sizes."""
    per_group = [list(itertools.combinations(np.flatnonzero(prob.g_index == g), int(n)))
                 for g, n in enumerate(sizes)]
    for combo in itertools.product(*per_group):
        known = np.zeros(prob.n_org, dtype=bool)
        for idx in combo:
            known[list(idx)] = True
        yield Pilot(known=known, fit=np.zeros(prob.n_org, dtype=bool),
                    calib_order=tuple(np.array(idx) for idx in combo))


def test_rectified_estimate_is_unbiased(tiny_problem):
    """Averaged over every possible stratified pilot, the rectified welfare equals the truth
    exactly, for every package (it is affine, so it suffices to compare coefficients)."""
    prob = tiny_problem
    sim = synth.structured_misreading(prob, group=0, e_s=0.9, rng=np.random.default_rng(1), k=2)
    e = DesignEngine(prob, sim)
    c_sum, D_sum, n = np.zeros(e.G), np.zeros((e.G, e.Q)), 0
    for pilot in all_pilots(prob, (3, 2, 2)):
        rect = guard.rectified_system(e, pilot)
        c_sum += rect.c
        D_sum += rect.D
        n += 1
    assert n == 20 * 10 * 6
    np.testing.assert_allclose(c_sum / n, e.true_group.c, atol=1e-12)
    np.testing.assert_allclose(D_sum / n, e.true_group.D, atol=1e-12)
    # ... whereas the uncorrected simulator is biased on this problem
    assert np.abs(e.sim_group.D - e.true_group.D).max() > 0.01


def test_rectified_matches_the_precheck_formula(small_problem, noisy_sim):
    prob = small_problem
    e = DesignEngine(prob, noisy_sim)
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.3, np.random.default_rng(0))
    rect = guard.rectified_system(e, pilot)
    c_t, D_t, c_h, D_h = e.true_org.c, e.true_org.D, e.sim_org.c, e.sim_org.D
    for g in range(e.G):
        m = e.gi == g
        P, T = m & pilot.known, m & ~pilot.known
        dc = (c_h[P] - c_t[P]).mean()
        dD = (D_h[P] - D_t[P]).mean(0)
        assert rect.c[g] == pytest.approx((c_t[P].sum() + c_h[T].sum() - T.sum() * dc) / m.sum())
        np.testing.assert_allclose(rect.D[g], (D_t[P].sum(0) + D_h[T].sum(0) - T.sum() * dD) / m.sum())


def test_perfect_simulator_and_census_pilot(small_problem):
    prob = small_problem
    e = DesignEngine(prob, synth.oracle(prob))
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.3, np.random.default_rng(0))
    for obj in welfare.OBJECTIVES:
        res = guard.run_guard(e, pilot, obj)
        assert res.index == e.oracle_index(obj)
        assert res.estimate == pytest.approx(e.true_values(obj)[res.index])
        assert res.certificate <= e.true_values(obj)[res.index] + 1e-9
    assert e.s_minus.max() == 0.0                                                # nothing is mis-set
    census = draw_pilot(prob.g_index, prob.n_groups, 1.0, np.random.default_rng(0))
    noisy = DesignEngine(prob, synth.iid_noise(prob, 0.5, np.random.default_rng(3)))
    res = guard.run_guard(noisy, census, "maxmin")
    assert res.index == noisy.oracle_index("maxmin")
    assert res.certificate == pytest.approx(noisy.true_values("maxmin")[res.index])


def test_toy_example_reproduces_the_paper_table(toy):
    prob, sim = toy
    e = DesignEngine(prob, sim)
    space = e.space
    idx = {a: space.index(a) for a in [(1, 1), (1, 0), (0, 1), (0, 0)]}
    # true and simulated group welfare (W_A, W_B)
    np.testing.assert_allclose(e.W_true[idx[(1, 1)]], [0.50, 0.75])
    np.testing.assert_allclose(e.W_true[idx[(1, 0)]], [1.00, 0.25])
    np.testing.assert_allclose(e.W_true[idx[(0, 1)]], [0.00, 0.75])
    np.testing.assert_allclose(e.W_true[idx[(0, 0)]], [0.50, 0.25])
    np.testing.assert_allclose(e.W_sim[idx[(1, 1)]], [0.50, 0.45])
    np.testing.assert_allclose(e.W_sim[idx[(1, 0)]], [1.00, 0.55])
    np.testing.assert_allclose(e.W_sim[idx[(0, 0)]], [0.50, 0.55])
    # max-min plug-in: looks balanced (0.55), the misread group B is the real loser (0.25)
    k = e.plugin_index("maxmin")
    assert k == idx[(1, 0)] and e.oracle_index("maxmin") == idx[(1, 1)]
    m = e.evaluate(k, "maxmin", e.W_sim)
    assert m["est_welfare"] == pytest.approx(0.55) and m["true_welfare"] == pytest.approx(0.25)
    assert m["regret"] == pytest.approx(0.25) and m["overestimate"] == pytest.approx(0.30)
    assert m["delta[B]"] == pytest.approx(0.30) and m["delta[A]"] == pytest.approx(0.0)
    assert m["illusory_fairness"] == pytest.approx(0.30)
    assert m["true_worst_group"] == "B" and m["most_overestimated_group"] == "B"
    assert m["worst_is_most_overestimated"]
    # utilitarian plug-in: same package, zero regret (Proposition A)
    ku = e.plugin_index("utilitarian")
    assert ku == idx[(1, 0)]
    assert e.evaluate(ku, "utilitarian", e.W_sim)["regret"] == pytest.approx(0.0)
    # a small pilot reveals B's true position on q2: GUARD switches to (1, 1)
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.5, np.random.default_rng(0))
    res = guard.run_guard(e, pilot, "maxmin")
    assert res.index == idx[(1, 1)]
    assert res.estimate == pytest.approx(0.5) and res.revised(prob.decision_labels()) == []
    assert res.certificate <= 0.5 + 1e-12
    # Proposition C: lowering B's welfare by a package-independent constant cannot change the choice
    for kappa in (0.0, 0.05, 0.1, 0.2):
        shifted = e.W_sim - np.array([0.0, kappa])
        assert space.argmax(shifted.min(1)) != idx[(1, 1)]
        assert space.argmax(shifted.mean(1)) == ku


@pytest.mark.parametrize("ucb", ["betting", "hoeffding_serfling"])
def test_certificate_covers_the_selected_package(small_problem, structured_sim, ucb):
    """Empirical P(W_p(a-hat) >= c) >= 1 - delta although a-hat is chosen with the same pilot."""
    prob = small_problem
    e = DesignEngine(prob, structured_sim, EngineSettings(delta=0.2, ucb=ucb))
    rng = np.random.default_rng(9)
    reps = 150
    for obj in welfare.OBJECTIVES:
        fails = 0
        for _ in range(reps):
            pilot = draw_pilot(prob.g_index, prob.n_groups, 0.4, rng)
            res = guard.run_guard(e, pilot, obj)
            fails += int(e.true_values(obj)[res.index] < res.certificate - 1e-9)
            assert (res.lower <= res.upper + 1e-9).all()
        assert fails / reps <= 0.2 + 3 * np.sqrt(0.2 * 0.8 / reps)


def test_all_variants_run_and_share_one_certificate(small_problem, noisy_sim):
    prob = small_problem
    e = DesignEngine(prob, noisy_sim)
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.4, np.random.default_rng(5))
    results = {v: guard.run_guard(e, pilot, "maxmin", v) for v in
               ("guard", "guard_pess", "guard_est", "guard_cons")}
    # GUARD-pess maximises the certificate, so no other rule can have a larger one
    best = results["guard_pess"].certificate
    assert all(r.certificate <= best + 1e-9 for r in results.values())
    np.testing.assert_allclose(results["guard"].tau, results["guard_pess"].tau)
    with pytest.raises(ValueError):
        guard.run_guard(e, pilot, "maxmin", "guard_plus")
    for kind in ("count", "pooled", "hoeffding_serfling"):
        assert np.isfinite(guard.run_guard(e, pilot, "nash", certificate_kind=kind).certificate)


# ------------------------------------------------------------------ GUARD-cons

def true_improvement(e_vec, d, a, p, w):
    loss = d @ (1.0 - a)
    if p == 0:
        return float(np.log(np.clip(e_vec - loss, 1e-300, None)) @ w - np.log(e_vec) @ w)
    return float(welfare.power_mean(e_vec - loss, p, w) - welfare.power_mean(e_vec, p, w))


@pytest.mark.parametrize("objective", ["utilitarian", "maxmin", "nash"])
def test_improvement_bound_is_the_exact_inner_minimum(small_problem, noisy_sim, objective):
    prob = small_problem
    e = DesignEngine(prob, noisy_sim, EngineSettings(cons_bound="hoeffding_serfling"))
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.5, np.random.default_rng(6))
    region = guard.conservative_region(e, pilot)
    gain = guard.improvement_lower_bound(e, region, objective)
    p = welfare.objective_p(objective)
    assert gain[e.space.ed_index] == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(0)
    packages = [e.space.package(k).astype(float) for k in rng.choice(e.space.size, 12, replace=False)]
    for a in packages:
        k = e.space.index(a.astype(int))
        if not np.isfinite(gain[k]):
            continue
        # (i) never above the improvement at any point of the region
        for _ in range(60):
            e_vec = rng.uniform(region["e_lo"], region["e_hi"])
            d = rng.uniform(region["d_lo"], region["d_hi"])
            assert true_improvement(e_vec, d, a, p, e.weights) >= gain[k] - 1e-9
        # (ii) attained at a corner of the region
        corners = [np.where(np.arange(e.G) == g, region["e_lo"], region["e_hi"]) for g in range(e.G)]
        loss = region["d_hi"] @ (1.0 - a)
        corners.append(np.where(loss > 0, region["e_lo"], region["e_hi"]))
        attained = min(true_improvement(c, region["d_hi"], a, p, e.weights) for c in corners)
        assert attained == pytest.approx(gain[k], abs=1e-9)


def test_conservative_region_contains_the_truth(small_problem, noisy_sim):
    prob = small_problem
    e = DesignEngine(prob, noisy_sim, EngineSettings(cons_bound="hoeffding_serfling"))
    true_ed = e.true_group.c + e.true_group.D.sum(1)
    rng = np.random.default_rng(7)
    for _ in range(20):
        region = guard.conservative_region(e, draw_pilot(prob.g_index, prob.n_groups, 0.4, rng))
        assert (region["e_lo"] <= true_ed + 1e-9).all() and (true_ed <= region["e_hi"] + 1e-9).all()
        assert (region["d_lo"] <= e.true_group.D + 1e-9).all() and (e.true_group.D <= region["d_hi"] + 1e-9).all()
        assert all(np.isfinite(v).all() for v in region.values())


def test_conservative_rule_keeps_the_draft_without_evidence(small_problem, noisy_sim):
    prob = small_problem
    for bound in ("t", "hoeffding_serfling"):
        e = DesignEngine(prob, noisy_sim, EngineSettings(cons_bound=bound))
        one_each = np.zeros(prob.n_org, dtype=bool)
        for g in range(prob.n_groups):
            one_each[np.flatnonzero(prob.g_index == g)[0]] = True
        pilot = pilot_from_mask(prob.g_index, prob.n_groups, one_each)
        for obj in welfare.OBJECTIVES:
            assert guard.conservative_select(e, pilot, obj) == e.space.ed_index


def test_conservative_rule_with_an_exact_simulator(small_problem):
    """An exact simulator has zero calibration residuals. That alone must not be taken as proof:
    the default (moderated) interval stays open, so the rule may keep the draft but never does
    worse than it; only a census, or switching the prior off, makes it follow the simulator."""
    prob = small_problem
    exact = synth.oracle(prob)
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.5, np.random.default_rng(8))
    census = draw_pilot(prob.g_index, prob.n_groups, 1.0, np.random.default_rng(8))
    default = DesignEngine(prob, exact, EngineSettings(cons_bound="t"))
    trusting = DesignEngine(prob, exact, EngineSettings(cons_bound="t", cons_prior_obs=0.0))
    for obj in welfare.OBJECTIVES:
        truth = default.true_values(obj)
        best, draft = truth[default.oracle_index(obj)], truth[default.space.ed_index]
        assert truth[guard.conservative_select(default, pilot, obj)] >= draft - 1e-12
        assert truth[guard.conservative_select(default, census, obj)] == pytest.approx(best)
        assert truth[guard.conservative_select(trusting, pilot, obj)] == pytest.approx(best)
