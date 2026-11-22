"""Decision-rule registry, baselines, pilot designs and the information discipline."""
from dataclasses import replace

import numpy as np
import pytest

from sim import baselines, methods, welfare
from sim.data import STANCES
from sim.engine import DesignEngine, EngineSettings
from sim.pilot import draw_pilot, pilot_sizes, tilt_weights

OBJECTIVES = list(welfare.OBJECTIVES)


def test_pilot_sizes_and_structure(small_problem):
    np.testing.assert_array_equal(pilot_sizes([38, 27, 20, 15], 0.3), [11, 8, 6, 4])
    np.testing.assert_array_equal(pilot_sizes([38, 27, 20, 15], 0.5), [19, 14, 10, 8])
    np.testing.assert_array_equal(pilot_sizes([3, 2], 0.01), [1, 1])            # at least one per group
    np.testing.assert_array_equal(pilot_sizes([3, 2], 1.0), [3, 2])
    with pytest.raises(ValueError):
        pilot_sizes([3, 2], 0.0)
    prob = small_problem
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.5, np.random.default_rng(0), fit_fraction=0.5)
    counts = pilot.group_counts(prob.g_index, prob.n_groups)
    np.testing.assert_array_equal(counts["n_known"], pilot_sizes(prob.group_sizes, 0.5))
    np.testing.assert_array_equal(counts["n_fit"] + counts["n_calib"], counts["n_known"])
    assert (counts["n_calib"] >= 1).all()
    assert not (pilot.fit & ~pilot.known).any() and not (pilot.calib & pilot.fit).any()
    np.testing.assert_array_equal(pilot.calib | pilot.fit, pilot.known)
    for g, idx in enumerate(pilot.calib_order):
        assert (prob.g_index[idx] == g).all()


def test_simple_random_sampling_is_uniform(small_problem):
    prob = small_problem
    rng = np.random.default_rng(1)
    hits = np.zeros(prob.n_org)
    reps = 4000
    for _ in range(reps):
        hits += draw_pilot(prob.g_index, prob.n_groups, 0.4, rng).known
    expected = (pilot_sizes(prob.group_sizes, 0.4) / prob.group_sizes)[prob.g_index]
    assert np.abs(hits / reps - expected).max() < 0.04


def test_tilted_pilot_over_samples_high_scores(small_problem):
    prob = small_problem
    score = np.arange(prob.n_org, dtype=float)
    w = tilt_weights(score, prob.g_index, prob.n_groups, gamma=2.0)
    np.testing.assert_allclose(tilt_weights(score, prob.g_index, prob.n_groups, 0.0), 1.0)
    rng = np.random.default_rng(2)
    mean_srs = np.mean([score[draw_pilot(prob.g_index, prob.n_groups, 0.3, rng).known].mean() for _ in range(300)])
    mean_tilt = np.mean([score[draw_pilot(prob.g_index, prob.n_groups, 0.3, rng, weights=w).known].mean()
                         for _ in range(300)])
    assert mean_tilt > mean_srs + 1.0


def test_run_replication_covers_every_rule(small_problem, structured_sim):
    prob = small_problem
    board = np.ones(prob.n_dec, dtype=np.uint8)
    board[:2] = 0
    e = DesignEngine(prob, structured_sim, board=board)
    names = methods.available_methods(e)
    assert set(names) == set(methods.METHODS)
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.4, np.random.default_rng(3))
    rows, record = methods.run_replication(e, pilot, np.random.default_rng(4), names, OBJECTIVES)
    assert len(rows) == len(names) * len(OBJECTIVES)
    by = {(r["objective"], r["method"]): r for r in rows}
    for obj in OBJECTIVES:
        assert by[(obj, "oracle")]["regret"] == pytest.approx(0.0)
        assert by[(obj, "ed")]["n_revised"] == 0 and by[(obj, "ifrs")]["n_revised"] == prob.n_dec
        assert by[(obj, "board")]["n_revised"] == 2 and by[(obj, "board")]["hamming_to_board"] == 0
        assert by[(obj, "sim_plugin")]["package"] == e.plugin_index(obj)
        assert 0 <= by[(obj, "hedgetune")]["hedgetune_log2n"] <= prob.n_dec
        for m in names:
            r = by[(obj, m)]
            assert r["regret"] >= -1e-12 and 0.0 <= r["true_welfare"] <= 1.0
            for kind in ("betting", "hoeffding_serfling", "count", "pooled", "ens"):
                assert f"cert_{kind}" in r and isinstance(r[f"covered_{kind}"], bool)
            # the guaranteed certificates never exceed the largest value any package can reach
            assert r["cert_betting"] <= e.true_values(obj).max() + 0.5
        # rules that optimise an estimate report how much they overestimate
        assert by[(obj, "guard")]["overestimate"] == pytest.approx(
            by[(obj, "guard")]["est_welfare"] - by[(obj, "guard")]["true_welfare"])
        assert "overestimate" not in by[(obj, "ed")]
    assert all(f"tau_betting[{g}]" in record for g in prob.group_names)
    assert all(f"width_at_oracle[{o}]" in record for o in OBJECTIVES)
    assert isinstance(record["uniform_valid_betting"], bool)
    # Theorem 2: when the bounds hold, GUARD-pess regret is at most the width at the oracle (p = 1, -inf)
    if record["uniform_valid_betting"]:
        for obj in ("utilitarian", "maxmin"):
            assert by[(obj, "guard_pess")]["regret"] <= record[f"width_at_oracle[{obj}]"] + 1e-9


def test_rules_without_prerequisites_are_dropped(small_problem, noisy_sim):
    e = DesignEngine(small_problem, noisy_sim)                                   # no members, no Board
    names = methods.available_methods(e)
    assert not set(names) & (methods.NEEDS_MEMBERS | methods.NEEDS_BOARD)
    assert methods.available_methods(e, ["guard", "ens_lcb", "board"]) == ["guard"]
    with pytest.raises(ValueError):
        methods.available_methods(e, ["guard", "magic"])
    pilot = draw_pilot(small_problem.g_index, small_problem.n_groups, 0.4, np.random.default_rng(0))
    rows, _ = methods.run_replication(e, pilot, np.random.default_rng(0), names, ["maxmin"],
                                      certificate_kinds=("betting",))
    assert "cert_ens" not in rows[0] and "cert_count" not in rows[0] and "cert_betting" in rows[0]


def test_ensemble_tables(small_problem, structured_sim):
    e = DesignEngine(small_problem, structured_sim, EngineSettings(ens_kappa=2.0))
    t = baselines.ensemble_tables(e)
    stack = np.stack([welfare.group_mean(m, e.gi, e.G).table() for m in e.member_org])
    np.testing.assert_allclose(t["mean"], stack.mean(0), atol=1e-12)
    np.testing.assert_allclose(t["wco"], stack.min(0), atol=1e-12)
    np.testing.assert_allclose(t["lcb"], stack.mean(0) - 2.0 * stack.std(0, ddof=1), atol=1e-9)
    assert (t["wco"] <= t["mean"] + 1e-12).all()
    pilot = draw_pilot(small_problem.g_index, small_problem.n_groups, 0.4, np.random.default_rng(0))
    mixed = baselines.pilot_ensemble_lcb_table(e, pilot)
    assert mixed.shape == e.W_true.shape
    census = draw_pilot(small_problem.g_index, small_problem.n_groups, 1.0, np.random.default_rng(0))
    np.testing.assert_allclose(baselines.pilot_ensemble_lcb_table(e, census), e.W_true, atol=1e-12)


def test_hedgetune_prefers_less_pressure_when_the_simulator_misleads(small_problem):
    prob = small_problem
    rng = np.random.default_rng(0)
    from sim.simulators import synthetic as synth
    e = DesignEngine(prob, synth.oracle(prob))
    # with a perfect simulator and the truth as the estimate, full pressure is optimal
    k, log2n = baselines.hedgetune_select(e, "maxmin", e.true_values("maxmin"), rng)
    assert log2n == prob.n_dec and k == e.oracle_index("maxmin")
    # if the estimate is the *negated* simulated welfare, any pressure hurts: choose N = 1
    _, log2n = baselines.hedgetune_select(e, "maxmin", -e.sim_values("maxmin"), rng)
    assert log2n == 0
    assert baselines.hedgetune_grid(e)[-1] == prob.n_dec


def scramble_targets(prob, pilot, seed):
    """A different world: organisations outside the pilot hold other stances."""
    rng = np.random.default_rng(seed)
    stance = prob.cell_stance.copy()
    hidden = ~pilot.known[prob.cell_org]
    stance[hidden] = rng.choice(STANCES, size=int(hidden.sum()))
    sigma = np.array([prob.sigma_map[s] for s in stance])
    return replace(prob, cell_stance=stance, sigma=sigma)


@pytest.mark.parametrize("fit_fraction", [0.0, 0.5])
def test_decisions_and_certificates_ignore_unobserved_stances(small_problem, structured_sim, fit_fraction):
    """No rule and no certificate may depend on the true stances of organisations outside the pilot."""
    prob = small_problem
    pilot = draw_pilot(prob.g_index, prob.n_groups, 0.4, np.random.default_rng(11), fit_fraction=fit_fraction)
    other = scramble_targets(prob, pilot, 99)
    assert (other.sigma != prob.sigma).any()
    e1, e2 = DesignEngine(prob, structured_sim), DesignEngine(other, structured_sim)
    blind = [m for m in methods.available_methods(e1) if m != "oracle"]
    c1, c2 = (methods.RepContext(e, pilot, np.random.default_rng(5)) for e in (e1, e2))
    for obj in OBJECTIVES:
        for m in blind:
            assert methods.METHODS[m](c1, obj).index == methods.METHODS[m](c2, obj).index, (m, obj)
    for kind in ("betting", "hoeffding_serfling", "count", "pooled", "est"):
        b1, b2 = c1.bounds(kind), c2.bounds(kind)
        np.testing.assert_allclose(b1.lower.c, b2.lower.c)
        np.testing.assert_allclose(b1.lower.D, b2.lower.D)
        np.testing.assert_allclose(b1.upper.c, b2.upper.c)
    for name in ("rectified", "pilot_sim", "pilot_only"):
        np.testing.assert_allclose(c1.system(name).c, c2.system(name).c)
        np.testing.assert_allclose(c1.system(name).D, c2.system(name).D)
    r1, r2 = c1.cons_region(), c2.cons_region()
    for key in r1:
        np.testing.assert_allclose(r1[key], r2[key])
