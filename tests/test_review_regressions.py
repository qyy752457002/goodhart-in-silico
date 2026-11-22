"""Regression tests for defects found by an independent review of the statistical core.

Each test reproduces the scenario in which the original code went wrong.
"""
import numpy as np
import pytest

from sim import bounds, certify, goodhart, guard, methods, optimize, welfare
from sim.data import STANCE_INDEX, STANCE_SIGMA, Problem
from sim.engine import DesignEngine, EngineSettings
from sim.pilot import draw_pilot
from sim.simulators import synthetic as synth
from sim.simulators.base import SimulatorOutput
from sim.welfare import Affine

# ------------------------------------------------------------------ pooled calibration order

def _two_group_masses(seed=0, n=40):
    """Group 0 is simulated perfectly (mass 0), group 1 badly (mass in [0.3, 0.9])."""
    rng = np.random.default_rng(seed)
    gi = np.repeat([0, 1], n)
    s = np.r_[np.zeros(n), rng.uniform(0.3, 0.9, n)]
    zero = Affine(np.zeros(2 * n), np.zeros((2 * n, 1)))
    return gi, s, zero


def test_group_ordered_sample_breaks_the_betting_bound():
    """Why the order matters: a stratified sample listed group by group is not exchangeable."""
    gi, s, _ = _two_group_masses()
    rng = np.random.default_rng(1)
    reps, short_ordered, short_shuffled = 300, 0, 0
    for _ in range(reps):
        pilot = draw_pilot(gi, 2, 0.3, rng)
        idx = np.concatenate(pilot.calib_order)                      # group 0 first, then group 1
        n_pop, truth = len(gi), s.mean()
        short_ordered += bounds.betting_ucb_wor(s[idx], n_pop, 0.1) < truth
        short_shuffled += bounds.betting_ucb_wor(s[rng.permutation(idx)], n_pop, 0.1) < truth
    assert short_ordered / reps > 0.9                                # the bound is almost always too small
    assert short_shuffled / reps <= 0.1 + 3 * np.sqrt(0.09 / reps)


def test_pooled_bound_controls_the_total_whichever_group_is_misread():
    gi, s, zero = _two_group_masses()
    delta, reps = 0.1, 300
    for flip in (False, True):                                       # misread group last / first
        s_use = s[::-1].copy() if flip else s
        rng = np.random.default_rng(2)
        short = {"random order": 0, "no rng": 0}
        for _ in range(reps):
            pilot = draw_pilot(gi, 2, 0.3, rng)
            true_total = s_use[~pilot.known].sum()
            args = (gi, 2, pilot.known, pilot.calib_order, zero, zero, zero, s_use, np.ones_like(s_use), delta)
            with_rng = certify.pooled_bounds(*args, bounds.betting_ucb_wor, rng)
            without = certify.pooled_bounds(*args, bounds.betting_ucb_wor)       # falls back to Hoeffding-Serfling
            short["random order"] += with_rng.tau_minus.sum() < true_total - 1e-9
            short["no rng"] += without.tau_minus.sum() < true_total - 1e-9
            assert not with_rng.guaranteed and with_rng.kind == "pooled"
        for name, count in short.items():
            assert count / reps <= delta + 3 * np.sqrt(delta * (1 - delta) / reps), (flip, name)


def test_pooled_bound_without_rng_does_not_depend_on_group_order():
    gi, s, zero = _two_group_masses()
    pilot = draw_pilot(gi, 2, 0.3, np.random.default_rng(3))
    args = (zero, zero, zero, s, np.ones_like(s), 0.1)
    a = certify.pooled_bounds(gi, 2, pilot.known, pilot.calib_order, *args)
    b = certify.pooled_bounds(gi, 2, pilot.known, pilot.calib_order[::-1], *args)
    assert a.tau_minus.sum() == pytest.approx(b.tau_minus.sum())


def test_pooled_certificate_in_the_pipeline_uses_a_random_order(small_problem):
    prob = small_problem
    last = prob.n_groups - 1
    sim = synth.structured_misreading(prob, last, 1.0, np.random.default_rng(4), k=4, e_bg=0.0)
    e = DesignEngine(prob, sim, EngineSettings(delta=0.1))
    rng = np.random.default_rng(5)
    reps, short = 200, 0
    for _ in range(reps):
        pilot = draw_pilot(prob.g_index, prob.n_groups, 0.4, rng)
        ctx = methods.RepContext(e, pilot, rng)
        short += ctx.bounds("pooled").tau_minus.sum() < e.s_minus[pilot.target].sum() - 1e-9
    assert short / reps <= 0.1 + 3 * np.sqrt(0.09 / reps)


# ------------------------------------------------------------------ GUARD-cons intervals

def _mostly_right_scenario(n_per_group=20, n_agree=11, n_bad=2):
    """Two groups, two proposals, everyone answers. q1: all agree. q2: 11 agree, 9 disagree, so
    retaining q2 is better for both groups. The simulator is exact except that `n_bad` agreeing
    organisations per group are read as 'disagree' on q2, which makes revising q2 look better."""
    orgs, groups, cell_org, cell_dec, cell_q, stances, sim_label = [], [], [], [], [], [], []
    for g in ("A", "B"):
        for j in range(n_per_group):
            orgs.append(f"{g}{j:02d}")
            groups.append(g)
            second = "agree" if j < n_agree else "disagree"
            for d, (q, s) in enumerate(((1, "agree"), (2, second))):
                cell_org.append(len(orgs) - 1)
                cell_dec.append(d)
                cell_q.append(q)
                stances.append(s)
                sim_label.append("disagree" if (q == 2 and j < n_bad) else s)
    stances = np.array(stances, dtype=object)
    prob = Problem(orgs=orgs, groups=groups, group_names=["A", "B"], decisions=[1, 2],
                   cell_org=np.array(cell_org), cell_dec=np.array(cell_dec), cell_q=np.array(cell_q),
                   cell_stance=stances, sigma=np.array([STANCE_SIGMA[s] for s in stances]), name="scenario")
    sim = SimulatorOutput.from_labels("mostly_right", np.array([STANCE_INDEX[s] for s in sim_label]))
    return prob, sim


@pytest.mark.parametrize("n_bad,rho", [(2, 0.2), (3, 0.3)])
def test_conservative_rule_does_not_trust_a_sample_without_spread(n_bad, rho):
    """When the calibration sample misses every misread organisation its residuals are all zero.
    The interval must not collapse then: GUARD-cons has to keep the draft."""
    prob, sim = _mostly_right_scenario(n_bad=n_bad)
    e = DesignEngine(prob, sim, EngineSettings(cons_delta=0.1))
    assert e.true_group.D[0, 1] > 0 > e.sim_group.D[0, 1]            # the simulator points the wrong way
    true_ed = e.true_group.c + e.true_group.D.sum(1)
    rng = np.random.default_rng(0)
    reps, contains, worse, zero_width = 400, 0, 0, 0
    for _ in range(reps):
        pilot = draw_pilot(prob.g_index, prob.n_groups, rho, rng)
        region = guard.conservative_region(e, pilot)
        contains += ((region["e_lo"] <= true_ed + 1e-9).all() and (true_ed <= region["e_hi"] + 1e-9).all()
                     and (region["d_lo"] <= e.true_group.D + 1e-9).all()
                     and (e.true_group.D <= region["d_hi"] + 1e-9).all())
        zero_width += (region["d_hi"][:, 1] - region["d_lo"][:, 1] < 1e-12).any()
        for obj in welfare.OBJECTIVES:
            k = guard.conservative_select(e, pilot, obj, region)
            worse += e.true_values(obj)[k] < e.true_values(obj)[e.space.ed_index] - 1e-12
    assert zero_width == 0
    assert contains / reps >= 0.9
    assert worse / (3 * reps) <= 0.1
    # the unmoderated interval (k0 = 0) is the behaviour this guards against
    naive = DesignEngine(prob, sim, EngineSettings(cons_delta=0.1, cons_prior_obs=0.0))
    rng = np.random.default_rng(0)
    collapsed = 0
    for _ in range(100):
        region = guard.conservative_region(naive, draw_pilot(prob.g_index, 2, rho, rng))
        collapsed += (region["d_hi"][:, 1] - region["d_lo"][:, 1] < 1e-12).any()
    assert collapsed > 30


def test_conservative_region_is_exact_only_where_nothing_can_differ(small_problem):
    """A proposal that no unobserved organisation of a group answered leaves nothing to infer:
    zero width is right there and only there (even with an exact simulator, whose calibration
    residuals are all zero, the prior keeps every other coordinate open)."""
    prob = small_problem
    e = DesignEngine(prob, synth.oracle(prob))
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.4, np.random.default_rng(1))
    region = guard.conservative_region(e, pilot)
    width = region["d_hi"] - region["d_lo"]
    # a coordinate is open iff some *target* organisation of the group answered that proposal:
    # the pilot organisations' own contribution is known exactly
    by_target = np.zeros((prob.n_groups, prob.n_dec), dtype=bool)
    hidden = pilot.target[prob.cell_org]
    by_target[prob.g_index[prob.cell_org[hidden]], prob.cell_dec[hidden]] = True
    assert (width[by_target] > 0).all() and (width[~by_target] == 0).all()
    assert all(np.isfinite(v).all() for v in region.values())
    assert (region["d_lo"] <= e.true_group.D + 1e-9).all() and (e.true_group.D <= region["d_hi"] + 1e-9).all()


# ------------------------------------------------------------------ labels, streams, KL axis

def test_plugin_bounds_are_not_labelled_as_guaranteed(small_problem, noisy_sim):
    e = DesignEngine(small_problem, noisy_sim)
    pilot = draw_pilot(small_problem.g_index, small_problem.n_groups, 0.4, np.random.default_rng(0))
    labels = {k: (guard.certificate_bounds(e, pilot, k).kind, guard.certificate_bounds(e, pilot, k).guaranteed)
              for k in ("betting", "hoeffding_serfling", "count", "pooled", "est")}
    assert labels == {"betting": ("mean", True), "hoeffding_serfling": ("mean", True), "count": ("count", True),
                      "pooled": ("pooled", False), "est": ("est", False)}


def test_each_randomised_rule_has_its_own_stream(small_problem, structured_sim):
    """What HedgeTune draws must not depend on which other rules or certificates were evaluated."""
    prob = small_problem
    e = DesignEngine(prob, structured_sim)
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.4, np.random.default_rng(0))
    alone, _ = methods.run_replication(e, pilot, np.random.default_rng(9), ["hedgetune"], ["maxmin"],
                                       certificate_kinds=("betting",))
    everything, _ = methods.run_replication(e, pilot, np.random.default_rng(9), methods.available_methods(e),
                                            ["utilitarian", "nash", "maxmin"])
    together = next(r for r in everything if r["method"] == "hedgetune" and r["objective"] == "maxmin")
    assert alone[0]["package"] == together["package"]
    ctx = methods.RepContext(e, pilot, np.random.default_rng(9))
    assert ctx.stream("pooled") is ctx.stream("pooled")
    assert ctx.stream("pooled").random() != ctx.stream("hedgetune/maxmin").random()


def test_exact_selection_pressure():
    M = 64
    assert optimize.kl_selection(optimize.best_of_n_pmf(M, 1), M) == pytest.approx(0.0, abs=1e-12)
    assert optimize.kl_selection(optimize.best_of_n_pmf(M, M), M) == pytest.approx(np.log(M))
    assert optimize.kl_selection(optimize.best_of_n_pmf(4, 2), 4) == pytest.approx(0.375, abs=5e-4)
    previous = -1.0
    for n in (1, 2, 4, 8, 16, 32, 64):
        exact = optimize.kl_selection(optimize.best_of_n_pmf(M, n), M)
        assert exact >= float(optimize.kl_best_of_n(n)) - 1e-12      # the nominal formula understates it
        assert exact > previous                                      # more draws, more pressure
        previous = exact
    # with many packages and few draws the two agree
    big = optimize.kl_selection(optimize.best_of_n_pmf(2 ** 18, 2 ** 9), 2 ** 18)
    assert big == pytest.approx(float(optimize.kl_best_of_n(2 ** 9)), abs=2e-3)


def test_pressure_curve_reports_both_axes(small_problem, noisy_sim):
    e = DesignEngine(small_problem, noisy_sim)
    curve = goodhart.pressure_curve(e, "maxmin", reps=0)
    assert {"kl_bon", "d", "kl_gao", "d_gao"} <= set(curve.columns)
    np.testing.assert_allclose(curve["d"], np.sqrt(curve["kl_bon"]))
    assert (curve["kl_bon"] >= curve["kl_gao"] - 1e-12).all()
    assert curve["kl_bon"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert curve["kl_bon"].iloc[-1] == pytest.approx(np.log(e.space.size))


# ------------------------------------------------------------------ betting bound, all variants

def _capital_peak(x, N, m, delta, strategy, running, c=0.75):
    """Independent implementation of the capital process at candidate mean m."""
    n = len(x)
    t = np.arange(1, n + 1)
    mu_hat = (0.5 + np.cumsum(x)) / (t + 1)
    var = (0.25 + np.cumsum((x - mu_hat) ** 2)) / (t + 1)
    bets = np.sqrt(2 * np.log(1 / delta) / (n * np.r_[0.25, var[:-1]]))
    thetas = c * np.array([0.25, 0.5, 0.75, 1.0])
    capital, mix, seen, path = 1.0, np.ones(len(thetas)), 0.0, []
    for i in range(n):
        mu = min(1.0, max(0.0, (N * m - seen) / (N - i)))
        room = max(1 - mu, 1e-12)
        if strategy == "prpl":
            capital *= 1 + min(bets[i], c / room) * (mu - x[i])
            path.append(capital)
        else:
            mix = mix * (1 - thetas + thetas * (1 - x[i]) / room)
            path.append(mix.mean())
        seen += x[i]
    return max(path) if running else path[-1]


@pytest.mark.parametrize("strategy", ["prpl", "dkelly"])
@pytest.mark.parametrize("running", [True, False])
def test_betting_bound_is_the_smallest_rejected_mean(strategy, running):
    rng = np.random.default_rng(7)
    for trial in range(12):
        n = int(rng.integers(1, 12))
        N = n + int(rng.integers(1, 30))
        x = [rng.beta(0.6, 3.0, n), (rng.random(n) < 0.2).astype(float), np.round(rng.random(n), 1)][trial % 3]
        delta = float(rng.choice([0.2, 0.1, 0.025]))
        m_lo, m_hi = x.sum() / N, min(1.0, (x.sum() + N - n) / N)
        grid = np.linspace(m_lo, m_hi, 1501)
        stat = np.array([_capital_peak(x, N, m, delta, strategy, running) for m in grid])
        assert (np.diff(stat) >= -1e-9 * np.abs(stat[:-1])).all()    # non-decreasing in the candidate mean
        rejected = np.flatnonzero(stat >= 1 / delta)
        expected = grid[rejected[0]] if rejected.size else m_hi
        got = bounds.betting_ucb_wor(x, N, delta, strategy=strategy, running=running)
        assert got == pytest.approx(expected, abs=2 * (m_hi - m_lo) / 1500 + 2e-6)
