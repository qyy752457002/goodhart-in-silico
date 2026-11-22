"""Simulator outputs, synthetic error injection, fitted simulators (R0, R3) and the registry."""
from dataclasses import replace

import numpy as np
import pytest

from sim import metrics
from sim.data import STANCE_INDEX, STANCES
from sim.simulators import fitted, registry
from sim.simulators import synthetic as synth
from sim.simulators.base import SimulatorOutput, combine, from_frame, load_output, save_output, to_frame
from sim.testbeds.synthetic import make_synthetic_problem

K = len(STANCES)


# ------------------------------------------------------------------ container

def test_output_validation_and_views():
    probs = np.full((3, K), 1 / K)
    out = SimulatorOutput("u", probs)
    np.testing.assert_allclose(out.expected_sigma(np.arange(K)), np.arange(K).mean())
    assert out.n_members == 0 and out.top1().tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        out.member_sigma(np.arange(K))
    with pytest.raises(ValueError):
        SimulatorOutput("bad", np.full((3, K), 0.5))                 # rows do not sum to 1
    with pytest.raises(ValueError):
        SimulatorOutput("bad", np.full((3, K - 1), 1 / (K - 1)))     # wrong number of classes
    with pytest.raises(ValueError):
        SimulatorOutput("bad", probs, members=np.full((2, 4, K), 1 / K))
    bad = probs.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        SimulatorOutput("bad", bad)
    hot = SimulatorOutput.from_labels("hot", [0, 5, 2], member_idx=[[0, 5, 2], [1, 5, 2]])
    assert hot.top1().tolist() == [0, 5, 2] and hot.n_members == 2
    np.testing.assert_allclose(hot.member_sigma(np.array([1, .75, .5, .25, .25, 0])), [[1, 0, .5], [.75, 0, .5]])
    assert hot.subset(np.array([2, 0])).top1().tolist() == [2, 0]


def test_combine_is_the_mean_of_components():
    a = SimulatorOutput.from_labels("a", [0, 1])
    b = SimulatorOutput.from_labels("b", [0, 5], member_idx=[[0, 5], [1, 5]])
    ens = combine([a, b], "ens")
    np.testing.assert_allclose(ens.probs[0], np.eye(K)[0])
    np.testing.assert_allclose(ens.probs[1], (np.eye(K)[1] + np.eye(K)[5]) / 2)
    assert ens.n_members == 2 and ens.meta["components"] == ["a", "b"]
    assert combine([a, b], "ens", members="all").n_members == 4
    weighted = combine([a, b], "w", weights=[3, 1])
    assert weighted.probs[1, 1] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        combine([], "empty")


def test_save_load_round_trip(tmp_path, small_problem, structured_sim):
    prob = small_problem
    path = save_output(structured_sim, prob, str(tmp_path / "sims" / "s.parquet"))
    back = load_output(path, prob)
    np.testing.assert_allclose(back.probs, structured_sim.probs)
    np.testing.assert_allclose(back.members, structured_sim.members)
    assert back.meta["kind"] == "structured_misreading" and back.name == structured_sim.name
    # row order in the file does not matter: alignment is by (org, qid)
    frame = to_frame(structured_sim, prob).sample(frac=1.0, random_state=0)
    np.testing.assert_allclose(from_frame(frame, prob, "shuffled").probs, structured_sim.probs)
    # missing cells: explicit error by default, uniform fill on request
    partial = frame[frame["org"] != prob.orgs[0]]
    with pytest.raises(ValueError, match="no prediction"):
        from_frame(partial, prob, "partial")
    filled = from_frame(partial, prob, "partial", on_missing="uniform")
    n_missing = int((prob.cell_org == 0).sum())
    assert filled.meta["n_missing_cells"] == n_missing
    np.testing.assert_allclose(filled.probs[prob.cell_org == 0], 1 / K)
    with pytest.raises(FileNotFoundError):
        load_output(str(tmp_path / "nope.parquet"), prob)
    with pytest.raises(ValueError):
        to_frame(SimulatorOutput.uniform(3), prob)


# ------------------------------------------------------------------ synthetic injection

def test_iid_noise_rate(small_problem):
    prob = make_synthetic_problem(group_sizes=(60, 60), n_dec=10, response_rate=0.9, seed=1)
    for e in (0.0, 0.1, 0.4):
        sim = synth.iid_noise(prob, e, np.random.default_rng(0))
        wrong = (sim.top1() != prob.stance_idx).mean()
        assert wrong == pytest.approx(e, abs=0.04)
    assert metrics.overall_metrics(small_problem, synth.oracle(small_problem))["accuracy"] == 1.0


def test_corrupt_labels_never_returns_the_original():
    rng = np.random.default_rng(0)
    true_idx = rng.integers(0, K, 5000)
    out = synth.corrupt_labels(true_idx, 1.0, np.full(K, 1 / K), rng)
    assert (out != true_idx).all()
    # classes that never occur are never invented ...
    freq = np.array([0.5, 0.5, 0, 0, 0, 0])
    two = synth.corrupt_labels(rng.integers(0, 2, 2000), 1.0, freq, rng)
    assert set(np.unique(two)) <= {0, 1}
    # ... unless there is no alternative at all
    lone = synth.corrupt_labels(np.zeros(200, dtype=int), 1.0, np.array([1.0, 0, 0, 0, 0, 0]), rng)
    assert (lone != 0).all()


def test_structured_misreading_hits_only_the_target_cells():
    prob = make_synthetic_problem(group_sizes=(80, 80, 80), n_dec=8, response_rate=0.9, seed=2)
    g = 1
    decs = synth.most_opposed_decisions(prob, g, 3)
    assert len(decs) == 3
    # they really are the least supported decisions of that group
    cell_g = prob.g_index[prob.cell_org] == g
    dec_ids = np.asarray(prob.decisions)[prob.cell_dec]
    means = {d: prob.sigma[cell_g & (dec_ids == d)].mean() for d in prob.decisions}
    assert max(means[d] for d in decs) <= min(means[d] for d in prob.decisions if d not in decs) + 1e-12
    sim = synth.structured_misreading(prob, g, 0.8, np.random.default_rng(3), k=3, e_bg=0.0)
    target = cell_g & np.isin(dec_ids, decs)
    changed = sim.top1() != prob.stance_idx
    assert not changed[~target].any()
    assert (sim.top1()[target & changed] == STANCE_INDEX["agree"]).all()
    not_agree = target & (prob.stance_idx != STANCE_INDEX["agree"])
    assert changed[not_agree].mean() == pytest.approx(0.8, abs=0.08)
    # ensemble members share the misreading: their mean still over-reads the group
    ens = synth.structured_misreading(prob, g, 0.9, np.random.default_rng(4), k=3, e_bg=0.1, n_members=5)
    assert ens.n_members == 5
    bias = (ens.expected_sigma(prob.sigma_vec) - prob.sigma)[target].mean()
    assert bias > 0.2
    opt = synth.optimistic_bias(prob, g, 0.5, np.random.default_rng(5))
    err = opt.expected_sigma(prob.sigma_vec) - prob.sigma
    assert err[cell_g].mean() > 0.05 and (err[cell_g] >= -1e-12).all()


# ------------------------------------------------------------------ fitted simulators

def _held_out(prob, frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(prob.n_org) < frac


def test_type_prior_backs_off_gracefully():
    prob = make_synthetic_problem(group_sizes=(40, 40), n_dec=6, response_rate=0.9, disagreement=2.0, seed=4)
    train = _held_out(prob)
    out = fitted.type_prior(prob, train, alpha=1.0)
    test_cells = ~train[prob.cell_org]
    uni = SimulatorOutput.uniform(prob.n_cells)
    held = lambda sim: metrics.cell_metrics(prob.stance_idx[test_cells], sim.probs[test_cells], prob.sigma_vec)
    assert held(out)["brier"] < held(uni)["brier"]
    # no training data at all -> uniform
    empty = fitted.type_prior(prob, np.zeros(prob.n_org, dtype=bool))
    np.testing.assert_allclose(empty.probs, 1 / K)
    # one group without training organisations -> falls back to the pooled distribution
    only_g0 = (prob.g_index == 0) & train
    out0 = fitted.type_prior(prob, only_g0, alpha=1.0)
    g1_cells = prob.g_index[prob.cell_org] == 1
    assert np.ptp(out0.probs[g1_cells], axis=0).max() < 1e-12
    assert np.isfinite(out0.probs).all()


@pytest.mark.parametrize("model", ["mf", "gbdt"])
def test_relational_models_learn_group_structure(model):
    if model == "gbdt":
        pytest.importorskip("sklearn")
    prob = make_synthetic_problem(group_sizes=(60, 60, 60), n_dec=8, response_rate=0.85, disagreement=2.0,
                                  org_sd=0.3, seed=5)
    train = _held_out(prob, 0.6)
    out = fitted.relational(prob, train, fitted.RelationalSpec(model=model))
    assert out.probs.shape == (prob.n_cells, K) and np.isfinite(out.probs).all()
    test_cells = ~train[prob.cell_org]
    m = metrics.cell_metrics(prob.stance_idx[test_cells], out.probs[test_cells], prob.sigma_vec)
    base = metrics.cell_metrics(prob.stance_idx[test_cells], np.full((int(test_cells.sum()), K), 1 / K),
                                prob.sigma_vec)
    assert m["brier"] < base["brier"] and m["sigma_mae"] < base["sigma_mae"]


def test_relational_uses_context_answers():
    """With an informative context matrix (S2) held-out organisations are predicted better."""
    prob = make_synthetic_problem(group_sizes=(70, 70), n_dec=8, response_rate=0.9, disagreement=0.2,
                                  org_sd=1.5, seed=6)
    # context: the organisation's mean true support plus noise, on 6 context questions
    rng = np.random.default_rng(0)
    disposition = np.bincount(prob.cell_org, weights=prob.sigma, minlength=prob.n_org) / prob.cells_per_org
    context = np.clip(disposition[:, None] + rng.normal(0, 0.1, (prob.n_org, 6)), 0, 1)
    context[rng.random(context.shape) < 0.2] = np.nan
    train = _held_out(prob, 0.5, seed=1)
    test_cells = ~train[prob.cell_org]
    spec = fitted.RelationalSpec(model="mf", rank=2)
    mae = lambda out: np.abs(out.expected_sigma(prob.sigma_vec) - prob.sigma)[test_cells].mean()
    assert mae(fitted.relational(prob, train, spec, context)) < mae(fitted.relational(prob, train, spec, None))
    with pytest.raises(ValueError):
        fitted.relational(prob, train, spec, context[:5])
    with pytest.raises(ValueError):
        fitted.relational(prob, train, fitted.RelationalSpec(model="svm"))


def test_gbdt_accepts_a_context_question_no_training_organisation_answered():
    """With a small pilot, a context question can be unanswered by every training organisation. That
    all-missing training column crashed scikit-learn's binning (1.9: 'window shape cannot be larger
    than input array shape') in the first S2 run of RQ3. It carries no information, so it is dropped."""
    pytest.importorskip("sklearn")
    prob = make_synthetic_problem(group_sizes=(30, 30), n_dec=6, response_rate=0.9, seed=8)
    context = np.random.default_rng(1).random((prob.n_org, 4))
    train = _held_out(prob, 0.3, seed=2)
    spec = fitted.RelationalSpec(model="gbdt", gbdt_max_iter=20)
    # (a) answered by held-out organisations only: this is the case that crashed
    held_out_only = context.copy()
    held_out_only[train, 2] = np.nan
    out = fitted.relational(prob, train, spec, held_out_only)
    assert np.isfinite(out.probs).all()
    np.testing.assert_allclose(out.probs.sum(1), 1.0)
    # (b) answered by nobody: exactly the model fitted without that question
    nobody = context.copy()
    nobody[:, 2] = np.nan
    np.testing.assert_allclose(fitted.relational(prob, train, spec, nobody).probs,
                               fitted.relational(prob, train, spec, np.delete(context, 2, axis=1)).probs)
    # (c) no context answer at all among the training organisations
    none_seen = context.copy()
    none_seen[train] = np.nan
    assert np.isfinite(fitted.relational(prob, train, spec, none_seen).probs).all()


@pytest.mark.parametrize("spec", [{"kind": "type_prior"}, {"kind": "relational", "model": "mf"}])
def test_cross_fit_never_uses_an_organisations_own_stances(spec):
    prob = make_synthetic_problem(group_sizes=(12, 10), n_dec=5, response_rate=0.9, seed=7)
    base = fitted.cross_fit(spec, prob, n_folds=4, rng=np.random.default_rng(0))
    assert base.meta["cross_fitted"] and np.isfinite(base.probs).all()
    # flip every stance of organisation 3: its own predictions must not move
    flipped = prob.cell_stance.copy()
    own = prob.cell_org == 3
    flipped[own] = np.where(prob.cell_stance[own] == "agree", "disagree", "agree")
    other = replace(prob, cell_stance=flipped, sigma=np.array([prob.sigma_map[s] for s in flipped]))
    again = fitted.cross_fit(spec, other, n_folds=4, rng=np.random.default_rng(0))
    np.testing.assert_allclose(again.probs[own], base.probs[own])
    assert not np.allclose(again.probs[~own], base.probs[~own])                  # ... but others' do
    folds = fitted.stratified_folds(prob.g_index, 4, np.random.default_rng(0))
    for g in range(prob.n_groups):
        sizes = np.bincount(folds[prob.g_index == g], minlength=4)
        assert sizes.max() - sizes.min() <= 1
    with pytest.raises(ValueError):
        fitted.fit_predict({"kind": "persona"}, prob, np.ones(prob.n_org, dtype=bool))


# ------------------------------------------------------------------ registry

def test_registry_resolves_every_kind(tmp_path, small_problem):
    prob = small_problem
    cfg = {
        "uni": {"kind": "uniform"},
        "truth": {"kind": "oracle"},
        "noise": {"kind": "synthetic", "mode": "iid", "e": 0.2, "seed": 1},
        "misread": {"kind": "synthetic", "mode": "structured", "group": prob.group_names[1], "e_s": 0.9,
                    "k": 2, "n_members": 3},
        "rosy": {"kind": "synthetic", "mode": "optimistic", "group": 0, "e": 0.4},
        "llm": {"kind": "persona", "variant": "named", "model": "m"},
        "ens": {"kind": "ensemble", "members": ["noise", "misread"]},
        "loop": {"kind": "ensemble", "members": ["loop"]},
        "r0": {"kind": "type_prior"},
        "r3": {"kind": "relational", "model": "mf", "rank": 2},
        "weird": {"kind": "telepathy"},
    }
    args = (cfg, prob, str(tmp_path), "synthetic")
    assert registry.load_static("uni", *args).probs[0, 0] == pytest.approx(1 / K)
    assert (registry.load_static("truth", *args).top1() == prob.stance_idx).all()
    noise = registry.load_static("noise", *args)
    assert noise.name == "noise" and noise.meta["id"] == "noise"
    np.testing.assert_allclose(noise.probs, registry.load_static("noise", *args).probs)   # seeded
    assert registry.load_static("misread", *args).n_members == 3
    assert registry.load_static("rosy", *args).meta["kind"] == "optimistic_bias"
    ens = registry.load_static("ens", *args)
    assert ens.n_members == 2 and ens.name == "ens"
    with pytest.raises(registry.MissingSimulatorOutput, match="python -m sim simulate --simulator llm"):
        registry.load_static("llm", *args)
    save_output(noise, prob, registry.simulated_path(str(tmp_path), "synthetic", "llm"))
    loaded = registry.load_static("llm", *args)
    assert loaded.name == "llm" and loaded.meta["model"] == "m"
    with pytest.raises(ValueError, match="circular"):
        registry.load_static("loop", *args)
    with pytest.raises(ValueError):
        registry.load_static("r0", *args)
    with pytest.raises(ValueError):
        registry.load_static("weird", *args)
    with pytest.raises(KeyError):
        registry.load_static("undefined", *args)
    assert registry.is_fitted(cfg["r0"]) and not registry.is_fitted(cfg["noise"])
    r3 = registry.resolve("r3", *args, cv_folds=3, seed=0)
    assert r3.meta["cross_fitted"] and r3.name == "r3"
    assert registry.resolve("uni", *args).name == "uni"
