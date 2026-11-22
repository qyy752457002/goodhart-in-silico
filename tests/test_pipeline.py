"""End-to-end checks of the experiment pipeline on a small synthetic consultation with synthetic
simulators and the mock LLM backend. Nothing here touches the AASB data or a real model."""
import json
import os
import zlib

import numpy as np
import pandas as pd
import pytest
import yaml

from sim import cli, config, utils
from sim.experiments import (
    common,
    describe,
    rq1_validity,
    rq2_goodhart,
    rq3_guard,
    sensitivity,
    simulate,
    synthetic_injection,
    toy_example,
)
from sim.reporting import figures, tables
from sim.simulators import registry

SMALL = {
    "paths": {"results_dir": "results", "simulated_dir": "simulated", "cache_dir": "cache"},
    "testbed": {"name": "synthetic",
                "synthetic": {"group_sizes": [10, 9, 8, 7], "n_dec": 6, "response_rate": 0.75, "seed": 3}},
    "simulators": {
        "mock_named": {"kind": "persona", "variant": "named", "model": "mock"},
        "mock_anon": {"kind": "persona", "variant": "anonymous", "model": "mock"},
        "mock_ctx": {"kind": "persona", "variant": "context", "identity": "anonymous", "model": "mock"},
        "noise": {"kind": "synthetic", "mode": "iid", "e": 0.2, "seed": 1, "n_members": 3},
        "misread": {"kind": "synthetic", "mode": "structured", "group": 1, "e_s": 0.9, "k": 2, "e_bg": 0.0,
                    "seed": 2, "n_members": 3},
        "ens_mock": {"kind": "ensemble", "members": ["mock_named", "mock_anon"]},
    },
    "llm": {"persona": {"n_samples": 2}},
    "rq1": {"simulators": ["mock_named", "mock_anon", "noise", "misread", "r0_prior", "r3_mf", "uniform"],
            "cv_folds": 3, "n_boot": 40,
            "contrasts": [["mock_named", "mock_anon", "named vs anonymous"], ["noise", "absent", "skipped"]]},
    "rq2": {"simulators": ["noise", "misread"], "reps": 200, "log2_n": [0, 2, 4, 6, 8, 10]},
    "rq3": {"simulators": ["misread", "noise", "r0_prior"], "rhos": [0.3, 0.5], "reps": 6, "n_boot": 50},
    "synthetic": {"n_sim_seeds": 2, "rhos": [0.5], "reps": 3, "n_members": 3,
                  "scenarios": [{"name": "iid", "mode": "iid", "e": [0.1, 0.3]},
                                {"name": "structured", "mode": "structured", "groups": [0, "G2"], "k": 2,
                                 "e_s": [0.0, 0.9], "e_bg": 0.1},
                                {"name": "rosy", "mode": "optimistic", "groups": [1], "e": 0.4}]},
    "sensitivity": {"simulators": ["misread"], "rhos": [0.5], "reps": 4,
                    "variants": [{"name": "baseline", "set": {}},
                                 {"name": "binary", "set": {"welfare.sigma_map": "binary"}},
                                 {"name": "by_size", "set": {"welfare.group_weights": "size"}},
                                 {"name": "tilted", "set": {"rq3.pilot.design": "tilted",
                                                            "rq3.pilot.tilt_by": "misset", "rq3.pilot.gamma": -1.0}}]},
    "figures": {"formats": ["png"], "dpi": 70},
}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, root_dir):
    """A scratch project root: the real default configuration plus a small override file."""
    root = tmp_path_factory.mktemp("project")
    os.makedirs(root / "configs")
    with open(os.path.join(root_dir, "configs", "default.yaml"), encoding="utf-8") as f:
        (root / "configs" / "default.yaml").write_text(f.read(), encoding="utf-8")
    (root / "configs" / "small.yaml").write_text(yaml.safe_dump(SMALL), encoding="utf-8")
    cfg = config.load_config(str(root / "configs" / "small.yaml"), root=str(root))
    return root, cfg


# ------------------------------------------------------------------ configuration

def test_default_configuration_matches_the_methodology(root_dir):
    cfg = config.load_config(root=root_dir)
    assert cfg.guard.delta == 0.1 and cfg.guard.base_set == "top1" and cfg.welfare.group_weights == "equal"
    assert cfg.rq3.rhos == [0.2, 0.3, 0.4, 0.5] and cfg.rq3.reps == 500
    assert cfg.rq2.reps == 2000 and cfg.rq2.log2_n == list(range(0, 19, 2))
    assert cfg.llm.persona.temperature == 0.0 and cfg.llm.persona.n_samples == 5
    assert cfg.llm.persona.sample_temperature == 0.7
    assert cfg.welfare.objectives == ["utilitarian", "nash", "maxmin"]
    assert cfg.get_path("testbed.aasb.decision_set") == "main" and cfg.get_path("no.such.key", 7) == 7
    settings = common.engine_settings(cfg)
    assert settings.delta_per_group(4).tolist() == [0.025] * 4 and settings.ucb == "betting"
    # every simulator referenced by an experiment is defined, and every persona has a model entry
    for section in ("rq1", "rq2", "rq3", "sensitivity"):
        for sim_id in cfg[section]["simulators"]:
            assert sim_id in cfg["simulators"], (section, sim_id)
    for spec in cfg["simulators"].values():
        if spec["kind"] == "persona":
            assert spec["model"] in cfg["llm"]["models"]
        if spec["kind"] == "ensemble":
            assert all(m in cfg["simulators"] for m in spec["members"])
    with pytest.raises(AttributeError):
        cfg.rq3.no_such_key


def test_configuration_layers_and_rejects_typos(tmp_path, root_dir):
    user = tmp_path / "mine.yaml"
    user.write_text("rq3: {reps: 7}\nsimulators: {mine: {kind: uniform}}\n", encoding="utf-8")
    cfg = config.load_config(str(user), ["guard.delta=0.05", "rq3.rhos=[0.1, 0.9]", "simulators.extra.kind=oracle"],
                             root=root_dir)
    assert cfg.rq3.reps == 7 and cfg.guard.delta == 0.05 and cfg.rq3.rhos == [0.1, 0.9]
    assert cfg.simulators.mine.kind == "uniform" and cfg.simulators.extra.kind == "oracle"
    assert "r1a_llama" in cfg.simulators                                   # defaults are kept
    assert cfg.rq3.fit_fraction == 0.5
    bad = tmp_path / "typo.yaml"
    bad.write_text("rq3: {repz: 7}\n", encoding="utf-8")
    with pytest.raises(KeyError, match="rq3.repz"):
        config.load_config(str(bad), root=root_dir)
    with pytest.raises(KeyError, match="guard.detla"):
        config.load_config(overrides=["guard.detla=0.2"], root=root_dir)
    with pytest.raises(ValueError):
        config.load_config(overrides=["no-equals-sign"], root=root_dir)
    with pytest.raises(FileNotFoundError):
        config.load_config(root=str(tmp_path))
    changed = config.with_overrides(cfg, {"welfare.sigma_map": "binary"})
    assert changed.welfare.sigma_map == "binary" and cfg.welfare.sigma_map == "graded"
    assert "_root" not in config.public_view(cfg) and config.resolve_path(cfg, "data").startswith(root_dir)


def test_replication_streams_are_independent_of_order():
    a = utils.rep_rng(1, 2, 3).random(4)
    np.testing.assert_allclose(a, utils.rep_rng(1, 2, 3).random(4))
    assert not np.allclose(a, utils.rep_rng(1, 3, 2).random(4))
    assert not np.allclose(a, utils.rep_rng(2, 2, 3).random(4))
    # a name is a key too, through a hash that is the same in every process (unlike Python's hash())
    by_name = utils.rep_rng(1, 3, "r0_prior").random(4)
    np.testing.assert_allclose(by_name, utils.rep_rng(1, 3, zlib.crc32(b"r0_prior")).random(4))
    assert not np.allclose(by_name, utils.rep_rng(1, 3, "r3_mf").random(4))


def test_dotenv_loader(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("# comment\nSIM_TEST_KEY = 'abc'\nSIM_KEEP=new\nnot a pair\n", encoding="utf-8")
    monkeypatch.delenv("SIM_TEST_KEY", raising=False)
    monkeypatch.setenv("SIM_KEEP", "old")
    assert utils.load_dotenv(str(tmp_path)) == ["SIM_TEST_KEY"]
    assert os.environ["SIM_TEST_KEY"] == "abc" and os.environ["SIM_KEEP"] == "old"
    monkeypatch.delenv("SIM_TEST_KEY")
    assert utils.load_dotenv(str(tmp_path / "missing")) == []


# ------------------------------------------------------------------ pipeline

def test_step0_describe(workspace):
    root, cfg = workspace
    out = describe.run(cfg)
    assert out["overview"]["n_orgs"].iloc[0] == 34 and out["overview"]["n_decisions"].iloc[0] == 6
    pk = out["packages"].set_index("package")
    assert pk.loc["ED (retain all)", "n_revised"] == 0 and pk.loc["IFRS baseline (revise all)", "n_revised"] == 6
    for obj in ("utilitarian", "nash", "maxmin"):
        assert pk.loc[f"optimum: {obj}", f"W_{obj}"] == pytest.approx(pk[f"W_{obj}"].max())
    assert len(out["optima_hamming"]) == 3
    sup = out["support_by_group"]
    assert ((sup["spread"] >= 0) & (sup["spread"] <= 1)).all()
    np.testing.assert_allclose(out["stance_distribution"].drop(columns=["group", "n_cells"]).sum(1), 1.0)
    folder = root / "results" / "synthetic" / "describe"
    assert (folder / "packages.csv").exists() and (folder / "manifest.json").exists()
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["experiment"] == "describe" and (folder / "config.yaml").exists()


def test_step1_simulate_with_the_mock_backend(workspace):
    root, cfg = workspace
    tb = common.load_testbed(cfg)
    dry = simulate.run(cfg, ["mock_named"], dry_run=True)
    assert dry["mock_named"]["n_calls"] == tb.prob.n_cells * 3
    assert not (root / "simulated").exists()                               # a dry run writes nothing
    part = simulate.run(cfg, ["mock_named"], limit=7)
    assert part["mock_named"]["saved"] is False and part["mock_named"]["calls"] == 21
    done = simulate.run(cfg, ["mock_named", "mock_anon", "mock_ctx"])
    assert done["mock_named"]["cached"] == 21                              # the limited run was reused
    assert done["mock_named"]["calls"] == tb.prob.n_cells * 3 - 21
    path = registry.simulated_path(str(root / "simulated"), "synthetic", "mock_named")
    assert os.path.exists(path) and os.path.exists(path + ".meta.json")
    meta = json.loads(open(path + ".meta.json", encoding="utf-8").read())
    assert meta["is_mock"] is True and meta["variant"] == "named" and meta["n_members"] == 2
    assert simulate.run(cfg, ["mock_named"])["mock_named"]["status"] == "exists"
    again = simulate.run(cfg, ["mock_named"], force=True)
    assert again["mock_named"]["calls"] == 0                               # recomputed purely from the cache
    with pytest.raises(ValueError, match="only persona simulators"):
        simulate.run(cfg, ["noise"])
    with pytest.raises(KeyError):
        simulate.run(cfg, ["mock_named"], model_override="no_such_model")
    # an anonymous prompt never names the organisation, with or without context
    runner, _, _ = simulate.build_runner(cfg, tb, "mock_ctx")
    for cell in range(0, tb.prob.n_cells, 9):
        messages, _ = runner.messages_for(cell, "greedy0")
        assert tb.prob.orgs[tb.prob.cell_org[cell]] not in messages[1]["content"]


def test_step2_rq1(workspace):
    root, cfg = workspace
    out = rq1_validity.run(cfg)
    tab1 = out["tab1_simulators"].set_index("simulator")
    assert {"mock_named", "noise", "misread", "r0_prior", "r3_mf", "uniform", "marginal", "group_marginal"} <= set(
        tab1.index)
    assert tab1.loc["uniform", "accuracy"] == pytest.approx(1 / 6)
    # majority vote of three members that are each wrong with probability 0.2
    assert tab1.loc["noise", "accuracy"] == pytest.approx(0.8 ** 3 + 3 * 0.8 ** 2 * 0.2, abs=0.08)
    assert bool(tab1.loc["marginal", "is_reference"]) and not bool(tab1.loc["noise", "is_reference"])
    assert tab1.loc["mock_named", "parse_failure_rate"] == 0.0
    assert all(c in tab1.columns for c in ("regret_utilitarian", "regret_nash", "regret_maxmin", "accuracy_lo"))
    assert len(out["contrasts"]) == 1 and out["contrasts"]["contrast"].iloc[0] == "named vs anonymous"
    bias = out["bias_matrix"]
    hot = bias[(bias["simulator"] == "misread") & (bias["bias"] > 0.2)]
    assert set(hot["group"]) == {"G2"}                                    # only the misread group stands out
    assert set(out["rank_agreement"]["metric"]) == {"accuracy", "brier", "sigma_mae"}
    assert len(out["decision_centric"]) == len(tab1) * 3


def test_step3_rq2(workspace):
    root, cfg = workspace
    out = rq2_goodhart.run(cfg)
    curve = out["pressure_curve"]
    assert sorted(curve["log2_n"].unique()) == [0, 2, 4, 6]                # values above Q = 6 are dropped
    assert len(curve) == 2 * 3 * 4
    full = curve[curve["log2_n"] == 6].set_index(["simulator", "objective"])
    plug = out["plugin_summary"].set_index(["simulator", "objective"])
    np.testing.assert_allclose(full["overestimate_exact"], plug.loc[full.index, "overestimate"], atol=1e-12)
    assert len(out["hamming_curve"]) == 2 * 3 * 7
    assert (out["curve_summary"]["ndh"] >= 0).all()
    assert (plug["regret"] >= -1e-12).all()


def test_a_simulator_gets_the_same_numbers_wherever_it_is_listed(workspace):
    """Cross-fitting folds, bootstrap intervals and random packages are keyed by the simulator's id.
    They used to be keyed by its position in the experiment's list, so two scenarios with different
    lists (S1, S2) reported two different accuracies for the same type prior."""
    root, cfg = workspace
    tb = common.load_testbed(cfg)
    first = common.resolve_simulators(cfg, tb, ["uniform", "r0_prior", "r3_mf"], 3)
    second = common.resolve_simulators(cfg, tb, ["r3_mf", "noise", "r0_prior"], 3)
    for sim_id in ("r0_prior", "r3_mf"):
        np.testing.assert_array_equal(first[sim_id].probs, second[sim_id].probs)
    other_seed = config.with_overrides(cfg, {"project.seed": 7})                 # the project seed still matters
    assert not np.allclose(first["r0_prior"].probs,
                           common.resolve_simulators(other_seed, tb, ["r0_prior"], 3)["r0_prior"].probs)

    reordered = config.with_overrides(cfg, {"rq1.simulators": ["r3_mf", "noise", "r0_prior"], "rq1.contrasts": [],
                                            "rq2.simulators": ["misread", "noise"]})
    a = rq1_validity.run(cfg, str(root / "order_a"))["tab1_simulators"].set_index("simulator")
    b = rq1_validity.run(reordered, str(root / "order_b"))["tab1_simulators"].set_index("simulator")
    shared = ["noise", "r0_prior", "r3_mf", "marginal", "group_marginal"]
    numeric = [c for c in a.columns if c in b.columns and a[c].dtype.kind == "f"]   # b has no persona columns
    assert {"accuracy", "accuracy_lo", "brier_hi", "regret_maxmin"} <= set(numeric)
    pd.testing.assert_frame_equal(a.loc[shared, numeric], b.loc[shared, numeric])

    keys = ["simulator", "objective", "log2_n"]
    c = rq2_goodhart.run(cfg, str(root / "order_c"))["pressure_curve"].sort_values(keys).reset_index(drop=True)
    d = rq2_goodhart.run(reordered, str(root / "order_d"))["pressure_curve"].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(c, d)


def test_step4_rq3(workspace):
    root, cfg = workspace
    out = rq3_guard.run(cfg)
    runs = out["runs"]
    assert set(runs["simulator"]) == {"misread", "noise", "r0_prior"}
    # the synthetic testbed has no Board package; r0 has no ensemble members
    assert "board" not in set(runs["method"])
    assert "ens_lcb" in set(runs.loc[runs["simulator"] == "misread", "method"])
    assert "ens_lcb" not in set(runs.loc[runs["simulator"] == "r0_prior", "method"])
    n_methods = runs[runs["simulator"] == "misread"]["method"].nunique()
    assert len(runs[runs["simulator"] == "misread"]) == 2 * 6 * 3 * n_methods
    assert (runs.loc[runs["method"] == "oracle", "regret"].abs() < 1e-12).all()
    # the same pilots are used for every pilot-independent simulator (paired design)
    p = out["pilots"]
    a = p[p["simulator"] == "misread"].filter(like="n_known").to_numpy()
    b = p[p["simulator"] == "noise"].filter(like="n_known").to_numpy()
    np.testing.assert_array_equal(a, b)
    pilot_only = runs[runs["method"] == "pilot_only"].set_index(["simulator", "rho", "rep", "objective"])["package"]
    np.testing.assert_array_equal(pilot_only.loc["misread"].sort_index().to_numpy(),
                                  pilot_only.loc["noise"].sort_index().to_numpy())
    # a fitted simulator hands part of each pilot to fitting
    fitted_rows = p[p["simulator"] == "r0_prior"]
    assert (fitted_rows.filter(like="n_calib").to_numpy() < fitted_rows.filter(like="n_known").to_numpy()).any()
    tab2 = out["tab2_methods"]
    assert {"true_welfare", "true_welfare_lo", "regret", "recovered_share", "coverage_betting", "cert_betting",
            "slack_betting", "coverage_pooled"} <= set(tab2.columns)
    oracle = tab2[tab2["method"] == "oracle"]
    np.testing.assert_allclose(oracle["recovered_share"].dropna(), 1.0)
    cov = out["coverage"]
    assert set(cov["certificate"]) >= {"betting", "hoeffding_serfling", "count", "pooled", "ens"}
    assert cov.loc[cov["certificate"] == "pooled", "guaranteed"].eq(False).all()
    assert ((cov["coverage"] >= 0) & (cov["coverage"] <= 1)).all() and (cov["nominal"] == 0.9).all()
    tests = out["pairwise_tests"]
    assert (tests["reference"] == "guard").all() and "guard" not in set(tests["method"])
    assert (tests["p_holm"] >= tests["p"] - 1e-12).all()
    assert {"uniform_valid_betting", "lower_valid_count", "n"} <= set(out["uniform_validity"].columns)
    folder = root / "results" / "synthetic" / "rq3"
    assert (folder / "runs.parquet").exists() and (folder / "tab2_methods.csv").exists()
    back = utils.read_table(str(folder / "runs.parquet"))
    assert len(back) == len(runs)


def test_parallel_replications_match_sequential_ones(workspace):
    """Worker processes reproduce the single-process output exactly."""
    root, cfg = workspace
    tb = common.load_testbed(cfg)
    kw = {"rhos": [0.3, 0.5], "reps": 4, "methods": ["ed", "oracle", "sim_plugin", "hedgetune", "guard"],
          "certificates": ["betting", "count"], "seed_keys": (5,)}
    for sim_id in ("misread", "r0_prior"):
        seq_rows, seq_pilots = rq3_guard.run_pilots(cfg, tb, sim_id, n_jobs=1, **kw)
        par_rows, par_pilots = rq3_guard.run_pilots(cfg, tb, sim_id, n_jobs=2, **kw)
        pd.testing.assert_frame_equal(pd.DataFrame(seq_rows), pd.DataFrame(par_rows))
        pd.testing.assert_frame_equal(pd.DataFrame(seq_pilots), pd.DataFrame(par_pilots))
    # replications are keyed by (seed, rho index, rep): the experiment's own runs used the same streams
    runs = utils.read_table(str(root / "results" / "synthetic" / "rq3" / "runs.parquet"))
    mine = pd.DataFrame(rq3_guard.run_pilots(cfg, tb, "misread", n_jobs=1, **kw)[0])
    ref = runs[(runs["simulator"] == "misread") & (runs["method"] == "guard") & (runs["rep"] < 4)]
    got = mine[mine["method"] == "guard"]
    key = ["rho", "rep", "objective"]
    np.testing.assert_array_equal(ref.sort_values(key)["package"].to_numpy(),
                                  got.sort_values(key)["package"].to_numpy())
    # whole experiments sharing one pool: synthetic simulators are shipped to the workers, and the
    # sensitivity variants change the testbed between calls
    one = config.with_overrides(cfg, {"project.n_jobs": 1, "synthetic.n_sim_seeds": 1})
    two = config.with_overrides(cfg, {"project.n_jobs": 2, "synthetic.n_sim_seeds": 1})
    pd.testing.assert_frame_equal(synthetic_injection.run(one, str(root / "par_syn_1"))["runs"],
                                  synthetic_injection.run(two, str(root / "par_syn_2"))["runs"])
    pd.testing.assert_frame_equal(sensitivity.run(one, str(root / "par_sens_1"))["summary"],
                                  sensitivity.run(two, str(root / "par_sens_2"))["summary"])
    with rq3_guard.pilot_pool(cfg, 1) as none:
        assert none is None


def test_broken_worker_pool_falls_back_to_this_process(workspace):
    """If the workers cannot run (e.g. out of memory), the replications run in-process instead."""
    from concurrent.futures.process import BrokenProcessPool
    root, cfg = workspace
    tb = common.load_testbed(cfg)

    class DeadPool:
        n_workers = 2

        def map(self, *args):
            raise BrokenProcessPool("a worker died")

    kw = {"rhos": [0.5], "reps": 3, "methods": ["ed", "guard"], "certificates": ["betting"], "seed_keys": (5,)}
    expected = rq3_guard.run_pilots(cfg, tb, "noise", n_jobs=1, **kw)
    got = rq3_guard.run_pilots(cfg, tb, "noise", pool=DeadPool(), **kw)
    pd.testing.assert_frame_equal(pd.DataFrame(expected[0]), pd.DataFrame(got[0]))


def test_step5_synthetic_and_sensitivity(workspace):
    root, cfg = workspace
    out = synthetic_injection.run(cfg)
    summary = out["summary"]
    assert set(summary["scenario"]) == {"iid", "structured", "rosy"}
    structured = summary[summary["scenario"] == "structured"]
    assert set(structured["group_name"]) == {"G1", "G2"} and set(structured["e_s"]) == {0.0, 0.9}
    assert set(summary.loc[summary["scenario"] == "iid", "e"]) == {0.1, 0.3}
    misread = out["misread_is_worst"]
    assert set(misread["scenario"]) == {"structured", "rosy"}              # undefined without a misread group
    assert ((misread["worst_is_misread"] >= 0) & (misread["worst_is_misread"] <= 1)).all()
    assert len(out["runs"]) > 0 and "sim_seed" in out["runs"].columns
    sens = sensitivity.run(cfg)
    assert set(sens["summary"]["variant"]) == {"baseline", "binary", "by_size", "tilted"}
    assert set(sens["packages"]["variant"]) == set(sens["summary"]["variant"])
    assert not [c for c in sens["summary"].columns if "[" in c]


def test_step6_report_builds_tables_and_figures(workspace):
    root, cfg = workspace
    written = tables.build_all(cfg)
    names = {os.path.basename(p) for p in written}
    assert {"tab1_simulators.csv", "tab1_simulators.tex", "fig2_goodhart_curves.png",
            "fig3_group_decomposition.png", "fig4_pilot_size_misread.png",
            "fig5a_structured_misreading_maxmin.png", "fig5b_certificate_coverage_maxmin.png",
            "bias_heatmap_misread.png"} <= names
    assert all(os.path.getsize(p) > 500 for p in written)
    paper = root / "results" / "synthetic" / "paper"
    # every figure has a table of the values it shows
    for stem in ("fig2_goodhart_curves", "fig3_group_decomposition", "fig4_pilot_size_misread"):
        assert len(pd.read_csv(paper / f"{stem}.csv")) > 0
    tex = (paper / "tab1_simulators.tex").read_text()
    assert r"\toprule" in tex and r"\textbf{" in tex and "mock\\_named" in tex
    tab2 = [n for n in names if n.startswith("tab2_misread_rho0.3_maxmin")]
    assert len(tab2) == 2


def test_report_without_results_is_harmless(tmp_path, root_dir):
    cfg = config.load_config(overrides=[f"paths.results_dir={tmp_path.as_posix()}"], root=root_dir)
    assert tables.build_all(cfg) == []


def test_missing_persona_output_is_reported_clearly(tmp_path, root_dir):
    overrides = ["testbed.name=synthetic", "testbed.synthetic.n_dec=4", "testbed.synthetic.group_sizes=[6, 6]",
                 f"paths.results_dir={tmp_path.as_posix()}", f"paths.simulated_dir={(tmp_path / 's').as_posix()}",
                 "rq2.simulators=[r1a_llama]", "rq3.simulators=[r1a_llama]", "rq3.reps=1"]
    cfg = config.load_config(overrides=overrides, root=root_dir)
    with pytest.raises(registry.MissingSimulatorOutput, match="python -m sim simulate --simulator r1a_llama"):
        rq2_goodhart.run(cfg)
    with pytest.raises(registry.MissingSimulatorOutput):
        rq3_guard.run(cfg)
    skipping = config.load_config(overrides=overrides + ["on_missing_simulator=skip"], root=root_dir)
    with pytest.raises(RuntimeError, match="no simulator"):
        rq2_goodhart.run(skipping)
    with pytest.raises(RuntimeError, match="no simulator"):
        rq3_guard.run(skipping)


# ------------------------------------------------------------------ toy example and CLI

def test_toy_example_tables(tmp_path):
    out = toy_example.run(out=str(tmp_path))
    pk = out["packages"].set_index("package")
    assert pk.loc["(1, 0)", "sim_min"] == pytest.approx(0.55) and pk.loc["(1, 0)", "true_min"] == pytest.approx(0.25)
    sel = out["selections"].set_index(["objective", "rule"])
    assert sel.loc[("maxmin", "simulator plug-in"), "package"] == "(1, 0)"
    assert sel.loc[("maxmin", "GUARD"), "package"] == "(1, 1)"
    assert sel.loc[("maxmin", "simulator plug-in"), "true_worst_group"] == "B"
    assert sel.loc[("utilitarian", "simulator plug-in"), "regret"] == pytest.approx(0.0)
    assert (tmp_path / "fig1_packages.csv").exists()


def test_cli_commands(workspace, capsys, monkeypatch):
    root, cfg = workspace
    monkeypatch.setenv("SIM_ROOT", str(root))
    small = str(root / "configs" / "small.yaml")
    assert cli.main(["show-config", "-c", small, "--set", "rq3.reps=3"]) == 0
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["rq3"]["reps"] == 3 and shown["testbed"]["name"] == "synthetic" and "_root" not in shown
    assert cli.main(["toy", "-c", small, "--out", str(root / "toy_out")]) == 0
    assert "(1, 0)" in capsys.readouterr().out
    assert cli.main(["describe", "-c", small, "--out", str(root / "cli_describe")]) == 0
    assert "ED (retain all)" in capsys.readouterr().out
    assert cli.main(["simulate", "-c", small, "-m", "mock_named", "--dry-run"]) == 0
    assert "example prompt" in capsys.readouterr().out
    assert cli.main(["rq2", "-c", small, "--set", "rq2.reps=20", "--out", str(root / "cli_rq2")]) == 0
    assert (root / "cli_rq2" / "pressure_curve.csv").exists()
    with pytest.raises(SystemExit):
        cli.main(["no-such-command"])
    with pytest.raises(SystemExit):
        cli.main(["--version"])


def test_figures_reject_missing_selections(workspace):
    root, cfg = workspace
    curves = pd.read_csv(root / "results" / "synthetic" / "rq2" / "pressure_curve.csv")
    with pytest.raises(ValueError, match="none of the requested simulators"):
        figures.fig2_goodhart_curves(curves, str(root / "figs"), simulators=["absent"])
    paths = figures.fig2_goodhart_curves(curves, str(root / "figs"), simulators=["misread"],
                                         objectives=["maxmin"], formats=("png",), dpi=60)
    assert len(paths) == 1 and os.path.exists(paths[0])
