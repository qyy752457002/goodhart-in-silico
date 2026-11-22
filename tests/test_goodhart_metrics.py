"""Goodhart protocol (RQ2), validity metrics (RQ1) and statistical helpers."""
import numpy as np
import pandas as pd
import pytest

from sim import goodhart, metrics, stats
from sim.data import STANCES
from sim.engine import DesignEngine
from sim.simulators import synthetic as synth
from sim.simulators.base import SimulatorOutput

# ------------------------------------------------------------------ RQ2

def test_pressure_curve_end_points(small_problem, structured_sim):
    e = DesignEngine(small_problem, structured_sim)
    for obj in ("utilitarian", "maxmin"):
        curve = goodhart.pressure_curve(e, obj, log2_n=[0, 2, 4], reps=500, rng=np.random.default_rng(0))
        assert list(curve["log2_n"]) == [0, 2, 4, e.Q]                           # exact optimum always added
        first, last = curve.iloc[0], curve.iloc[-1]
        # N = 1 is a uniformly random package
        assert first["true_exact"] == pytest.approx(e.true_values(obj).mean())
        assert first["sim_exact"] == pytest.approx(e.sim_values(obj).mean())
        # N = 2^Q is the plug-in design
        k = e.plugin_index(obj)
        assert last["sim_exact"] == pytest.approx(e.sim_values(obj)[k])
        assert last["true_exact"] == pytest.approx(e.true_values(obj)[k])
        assert last["overestimate_exact"] == pytest.approx(e.evaluate(k, obj, e.W_sim)["overestimate"])
        assert last["true_lo"] == pytest.approx(last["true_hi"])                 # degenerate at full pressure
        # simulated welfare of the winner can only grow with pressure
        assert (np.diff(curve["sim_exact"]) >= -1e-12).all()
        np.testing.assert_allclose(curve["overestimate_exact"], curve["sim_exact"] - curve["true_exact"],
                                   atol=1e-12)
        # the group shares of "true worst group" form a distribution
        shares = curve[[f"p_true_worst[{g}]_exact" for g in small_problem.group_names]].sum(1)
        np.testing.assert_allclose(shares, 1.0)
        # Monte Carlo agrees with the exact expectation
        se = (curve["true_hi"] - curve["true_lo"]) / (2 * 1.96)
        assert (np.abs(curve["true_mean"] - curve["true_exact"]) <= 4 * se + 1e-9).all()


def test_pressure_curve_matches_literal_sampling(small_problem, noisy_sim):
    """The rank law reproduces 'draw N packages, keep the best simulated one'."""
    e = DesignEngine(small_problem, noisy_sim)
    rng = np.random.default_rng(1)
    sim_val, true_val = e.sim_values("maxmin"), e.true_values("maxmin")
    n, reps = 4, 20000
    picks = np.array([rng.choice(e.space.size, n, replace=False) for _ in range(reps)])
    winners = picks[np.arange(reps), sim_val[picks].argmax(1)]
    curve = goodhart.pressure_curve(e, "maxmin", log2_n=[2], reps=0)
    row = curve[curve["log2_n"] == 2].iloc[0]
    assert row["sim_exact"] == pytest.approx(sim_val[winners].mean(), abs=0.004)
    assert row["true_exact"] == pytest.approx(true_val[winners].mean(), abs=0.004)
    assert "true_mean" not in curve.columns


def test_hamming_curve_and_plugin_summary(small_problem, structured_sim):
    e = DesignEngine(small_problem, structured_sim)
    curve = goodhart.hamming_curve(e, "maxmin")
    assert len(curve) == e.Q + 1 and curve.iloc[0]["n_revised"] == 0
    assert curve.iloc[-1]["package"] == e.plugin_index("maxmin")
    assert (curve["n_revised"] <= curve["radius"]).all()
    assert (np.diff(curve["est_welfare"]) >= -1e-12).all()
    # Proposition A
    util = goodhart.plugin_summary(e, "utilitarian")
    assert util["overestimate"] == pytest.approx(util["weighted_delta"])
    mm = goodhart.plugin_summary(e, "maxmin")
    assert mm["overestimate"] <= mm["max_delta"] + 1e-12
    assert mm["regret"] == pytest.approx(mm["oracle_welfare"] - mm["true_welfare"])


def test_ndh_and_functional_form():
    assert goodhart.ndh([0.2, 0.5, 0.4], 1.0) == pytest.approx(0.1)             # rises, then falls
    assert goodhart.ndh([0.2, 0.3, 0.5], 1.0) == 0.0                            # monotone: no Goodharting
    assert goodhart.ndh([0.2, 0.5, 0.4], 0.5) == pytest.approx(0.2)
    d = np.linspace(0, 3, 12)
    fit = goodhart.fit_bon(d, d * (0.8 - 0.2 * d))
    assert fit["alpha"] == pytest.approx(0.8) and fit["beta"] == pytest.approx(0.2)
    assert fit["r2"] == pytest.approx(1.0) and fit["d_peak"] == pytest.approx(2.0)


def test_curve_summary(small_problem, structured_sim):
    e = DesignEngine(small_problem, structured_sim)
    curve = goodhart.pressure_curve(e, "maxmin", reps=0)
    s = goodhart.curve_summary(e, curve)
    assert 0.0 <= s["ndh"] <= 1.0 and s["objective"] == "maxmin"
    assert s["overestimate_at_full_pressure"] == pytest.approx(curve["overestimate_exact"].iloc[-1])


def test_iid_noise_shrinks_and_structure_inflates(small_problem):
    """The Day-1 finding in miniature: structured misreading of the binding group produces a
    positive illusory-fairness index at the plug-in, a perfect simulator produces none."""
    prob = small_problem
    exact = DesignEngine(prob, synth.oracle(prob))
    assert goodhart.plugin_summary(exact, "maxmin")["overestimate"] == pytest.approx(0.0)
    worst = int(np.argmin(exact.W_true[exact.oracle_index("maxmin")]))
    sim = synth.structured_misreading(prob, worst, 1.0, np.random.default_rng(0), k=3, e_bg=0.0)
    e = DesignEngine(prob, sim)
    s = goodhart.plugin_summary(e, "maxmin")
    assert s[f"delta[{prob.group_names[worst]}]"] > 0 and s["illusory_fairness"] >= 0


# ------------------------------------------------------------------ RQ1

def test_cell_metrics_by_hand():
    sigma_vec = np.array([1.0, 0.75, 0.5, 0.25, 0.25, 0.0])
    true_idx = np.array([0, 0, 5, 1])
    probs = np.array([[0.7, 0.3, 0, 0, 0, 0],      # correct, confident
                      [0.2, 0.8, 0, 0, 0, 0],      # wrong class, same 3-way side
                      [0.0, 0.0, 0, 0, 0, 1.0],    # correct
                      [0.5, 0.5, 0, 0, 0, 0]])     # tie between agree and qualified_agree
    m = metrics.cell_metrics(true_idx, probs, sigma_vec)
    assert m["accuracy"] == pytest.approx((1 + 0 + 1 + 0.5) / 4)
    assert m["accuracy3"] == pytest.approx(1.0)
    brier = [(0.3 ** 2) * 2, 0.8 ** 2 * 2, 0.0, 0.5 ** 2 * 2]
    assert m["brier"] == pytest.approx(np.mean(brier))
    sigma_hat = probs @ sigma_vec
    assert m["sigma_bias"] == pytest.approx((sigma_hat - sigma_vec[true_idx]).mean())
    # macro-F1 over classes that occur (argmax predictions: 0, 1, 5, 0)
    f1_agree = 2 * 1 / (2 * 1 + 1 + 1)       # tp=1 (row 0), fp=1 (row 3), fn=1 (row 1)
    f1_qagree = 0.0                           # predicted once (row 1) wrongly, missed once (row 3)
    f1_disagree = 1.0
    assert m["macro_f1"] == pytest.approx(np.mean([f1_agree, f1_qagree, f1_disagree]))
    # ECE: bins [0.5,0.6): conf .5 acc .5 | [0.7,0.8): conf .7 acc 1 | [0.8,0.9): conf .8 acc 0 | [1]: conf 1 acc 1
    assert m["ece"] == pytest.approx(0.25 * 0.0 + 0.25 * 0.3 + 0.25 * 0.8 + 0.25 * 0.0)
    assert metrics.cell_metrics(true_idx[:0], probs[:0], sigma_vec) == {"n_cells": 0}


def test_macro_f1_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 6, 300), rng.integers(0, 5, 300)
    assert metrics.macro_f1(y, p, 6) == pytest.approx(sklearn.f1_score(y, p, average="macro"))


def test_reference_simulators_and_perfect_scores(small_problem):
    prob = small_problem
    refs = metrics.reference_simulators(prob)
    uni = metrics.overall_metrics(prob, refs["uniform"])
    assert uni["accuracy"] == pytest.approx(1 / len(STANCES))
    assert uni["brier"] == pytest.approx(1 - 1 / len(STANCES))
    marg = metrics.overall_metrics(prob, refs["marginal"])
    assert marg["sigma_bias"] == pytest.approx(0.0, abs=1e-12)                   # calibrated in the mean
    assert marg["brier"] <= uni["brier"]
    grp = metrics.overall_metrics(prob, refs["group_marginal"])
    assert grp["brier"] <= marg["brier"] + 1e-12
    perfect = metrics.overall_metrics(prob, synth.oracle(prob), n_boot=50, rng=np.random.default_rng(0))
    assert perfect["accuracy"] == 1.0 and perfect["brier"] == 0.0 and perfect["ece"] == 0.0
    assert perfect["accuracy_lo"] == perfect["accuracy_hi"] == 1.0


def test_group_metrics_and_bias_matrix(small_problem):
    prob = small_problem
    sim = synth.structured_misreading(prob, 2, 1.0, np.random.default_rng(1), k=2, e_bg=0.0)
    e = DesignEngine(prob, sim)
    gm = metrics.group_metrics(prob, sim).set_index("group")
    name = prob.group_names[2]
    assert gm.loc[name, "sigma_bias"] > 0 and gm.loc[name, "welfare_bias_ed"] > 0
    for other in prob.group_names:
        if other != name:
            assert gm.loc[other, "sigma_bias"] == pytest.approx(0.0) and gm.loc[other, "accuracy"] == 1.0
    # welfare_bias_ed is exactly the simulator's overestimate of group welfare at the ED package
    ed = e.space.ed_index
    np.testing.assert_allclose(gm.loc[prob.group_names, "welfare_bias_ed"], e.W_sim[ed] - e.W_true[ed])
    bm = metrics.bias_matrix(prob, sim)
    hot = bm[bm["bias"].abs() > 1e-12]
    assert set(hot["group"]) == {name} and set(hot["decision"]) <= set(sim.meta["decisions"])
    assert bm["n_cells"].sum() == prob.n_cells
    ce = metrics.coefficient_error(e)
    assert ce.shape == (prob.n_groups, prob.n_dec)
    assert (ce.loc[[g for g in prob.group_names if g != name]].abs().to_numpy() < 1e-12).all()


def test_decision_metrics_and_contrast(small_problem, noisy_sim):
    prob = small_problem
    e = DesignEngine(prob, noisy_sim)
    dm = metrics.decision_metrics(e, ["utilitarian", "maxmin"])
    assert list(dm["objective"]) == ["utilitarian", "maxmin"]
    np.testing.assert_allclose(dm["selected_error"], dm["overestimate"].abs())
    c = metrics.paired_accuracy_contrast(prob, synth.oracle(prob), noisy_sim, n_boot=100,
                                         rng=np.random.default_rng(0))
    assert c["accuracy_a"] == 1.0 and c["diff"] > 0 and c["diff_lo"] > 0
    assert c["only_b_correct"] == 0 and c["mcnemar_p"] < 0.05
    same = metrics.paired_accuracy_contrast(prob, noisy_sim, noisy_sim, n_boot=50)
    assert same["diff"] == 0 and same["mcnemar_p"] == 1.0


def test_three_class_merge_matrix():
    np.testing.assert_allclose(metrics.MERGE3.sum(0), 1.0)                       # every class has one home
    assert metrics.MERGE3.shape == (3, len(STANCES))
    out = SimulatorOutput.uniform(4)
    np.testing.assert_allclose(out.probs @ metrics.MERGE3.T, [[2 / 6, 1 / 6, 3 / 6]] * 4)


# ------------------------------------------------------------------ statistics

def test_bootstrap_and_tests():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 400)
    lo, hi = stats.bootstrap_ci(x, rng=np.random.default_rng(1))
    assert lo < x.mean() < hi and hi - lo == pytest.approx(4 * 1.0 / np.sqrt(400), rel=0.25)
    assert stats.bootstrap_ci(np.full(5, 0.3)) == (0.3, 0.3)
    assert np.isnan(stats.bootstrap_ci([])[0])
    ci = stats.mean_ci(np.r_[x, np.nan])
    assert ci["n"] == 400 and ci["ci_lo"] < ci["mean"] < ci["ci_hi"]
    w = stats.paired_wilcoxon(x + 0.5, x + rng.normal(0, 0.1, 400))
    assert w["p"] < 1e-6 and w["mean_diff"] == pytest.approx(0.5, abs=0.05)
    assert stats.paired_wilcoxon(x, x)["p"] == 1.0
    np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(stats.holm([0.5, 0.9]), [1.0, 1.0])
    assert stats.spearman([1, 2, 3, 4], [1, 2, 3, 5])["rho"] == pytest.approx(1.0)
    assert np.isnan(stats.spearman([1, 2], [2, 1])["rho"])
    lo, hi = stats.clopper_pearson(95, 100)
    assert lo < 0.95 < hi and stats.clopper_pearson(0, 10)[0] == 0.0 and stats.clopper_pearson(10, 10)[1] == 1.0
    assert stats.mcnemar_exact(0, 0) == 1.0 and stats.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    cl = stats.cluster_bootstrap_ci(lambda idx: x[idx].mean(), np.repeat(np.arange(40), 10), n_boot=200)
    assert cl[0] < x.mean() < cl[1]
    assert isinstance(pd.DataFrame([ci]), pd.DataFrame)
