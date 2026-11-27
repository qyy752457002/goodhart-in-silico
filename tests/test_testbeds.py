"""Testbed loaders: synthetic, survey (OpinionQA-shaped) and AASB ED SR1.

The AASB checks need Dataset A on disk and are skipped otherwise. They only load the data, build
prompts and recompute the Day-1 precondition numbers (methodology section 12, P1): no experiment
is run and no model is called.
"""
import os

import numpy as np
import pandas as pd
import pytest
from conftest import ORG_GROUPS, requires_aasb

from sim import config, data
from sim.engine import DesignEngine
from sim.experiments import common, describe
from sim.simulators import prompts
from sim.simulators import synthetic as synth
from sim.simulators.backends import MockBackend
from sim.simulators.persona import PersonaRunner, PersonaSpec
from sim.testbeds import opinionqa
from sim.testbeds.synthetic import latent_to_stance, make_synthetic_problem

# ------------------------------------------------------------------ data helpers

def test_sigma_maps():
    graded = data.make_sigma_map("graded", 0.75)
    assert graded == data.STANCE_SIGMA
    wide = data.make_sigma_map("graded", 0.9)
    assert wide["qualified_agree"] == 0.9 and wide["qualified_disagree"] == pytest.approx(0.1)
    assert wide["alternative_proposal"] == pytest.approx(0.1) and wide["mixed"] == 0.5
    assert data.make_sigma_map("binary") == data.STANCE_SIGMA_BINARY
    with pytest.raises(ValueError):
        data.make_sigma_map("graded", 0.3)
    with pytest.raises(ValueError):
        data.make_sigma_map("ternary")
    assert set(data.STANCE3) == set(data.STANCES) and set(data.STANCE3.values()) == set(data.STANCE3_CLASSES)


def test_grouping_schemes_cover_every_fine_type():
    for name, scheme in data.GROUP_SCHEMES.items():
        assert set(scheme) == set(data.TYPE_RULES), name
    assert len(set(data.GROUP_SCHEMES["coarse4"].values())) == 4
    assert len(set(data.GROUP_SCHEMES["coarse3"].values())) == 3
    assert len(set(data.GROUP_SCHEMES["coarse2"].values())) == 2
    assert data.DECISION_SETS["main"][14] == 13 and len(set(data.DECISION_SETS["main"].values())) == 18
    assert len(set(data.DECISION_SETS["main+nfp"].values())) == 22


def test_problem_views(small_problem):
    prob = small_problem
    prob.validate()
    assert prob.n_cells == len(prob.cell_keys) == len(set(prob.cell_keys))
    np.testing.assert_allclose(prob.sigma, prob.sigma_vec[prob.stance_idx])
    binary = prob.with_sigma_map(data.STANCE_SIGMA_BINARY)
    assert set(np.unique(binary.sigma)) <= {0.0, 0.5, 1.0} and binary.n_cells == prob.n_cells
    np.testing.assert_allclose(binary.sigma_vec, [1, 1, 0.5, 0, 0, 0])
    merged = prob.with_groups({o: ("X" if g in ("G1", "G2") else "Y") for o, g in zip(prob.orgs, prob.groups)})
    assert merged.group_names == ["X", "Y"] and merged.group_sizes.sum() == prob.n_org
    assert prob.n_groups == 4                                             # the original is untouched
    assert prob.decision_labels()[0] == "Q1"


def test_synthetic_problem_generator():
    prob = make_synthetic_problem(group_sizes=(30, 20), n_dec=7, response_rate=0.5, seed=5,
                                  group_names=["Users", "Preparers"])
    prob.validate()
    assert prob.n_org == 50 and prob.n_dec == 7 and prob.group_names == ["Preparers", "Users"]
    assert prob.cells_per_org.min() >= 1 and 0.35 < prob.n_cells / (50 * 7) < 0.65
    same = make_synthetic_problem(group_sizes=(30, 20), n_dec=7, response_rate=0.5, seed=5,
                                  group_names=["Users", "Preparers"])
    assert (same.cell_stance == prob.cell_stance).all()
    assert latent_to_stance(np.array([2.0, 0.5, 0.0, -0.3, -0.7, -3.0])).tolist() == data.STANCES
    # with no group effect the groups agree; with a large one they do not
    flat = make_synthetic_problem(group_sizes=(400, 400), n_dec=6, response_rate=1.0, disagreement=0.0, seed=1)
    split = make_synthetic_problem(group_sizes=(400, 400), n_dec=6, response_rate=1.0, disagreement=2.5, seed=1)
    spread = lambda p: describe.support_tables(p)[0]["spread"].mean()
    assert spread(flat) < 0.08 < spread(split)
    with pytest.raises(ValueError):
        make_synthetic_problem(group_sizes=(5, 5), group_names=["only-one"])


def test_board_decisions_file(root_dir):
    path = os.path.join(root_dir, "data", "board_decisions.csv")
    table = pd.read_csv(path)
    assert set(table["decision_qid"]) == set(data.DECISION_SETS["main+nfp"].values())
    assert set(table["a"]) == {0, 1} and (table["a"] == (table["board_action"] == "retain")).all()
    main = sorted(set(data.MAIN_DECISIONS.values()))
    board = data.load_board_decisions(path, main)
    # 15 of the 18 main decisions were reverted to the IFRS baseline
    assert board.shape == (18,) and int((board == 0).sum()) == 15
    assert [q for q, a in zip(main, board) if a == 1] == [4, 11, 15]
    # Rows that are not in Appendix A were first inferred as "retained". The three main decisions among them were
    # confirmed on 2026-11-23 against the Basis for Conclusions on AASB S2 (BC28, BC40, BC60 and BC64) and are no
    # longer flagged; the three rows of the NFP extension set are still inferred and stay flagged for review.
    inferred = table[table["listed_in_appendix_a"] == 0]
    assert sorted(inferred["decision_qid"]) == [4, 11, 15, 26, 27, 29]
    assert sorted(table.loc[table["needs_review"] == 1, "decision_qid"]) == [26, 27, 29]
    assert data.load_board_decisions(path, [2, 3, 99]) is None
    assert data.load_board_decisions(os.path.join(root_dir, "nope.csv"), main) is None


# ------------------------------------------------------------------ survey testbed

def _write_wave(folder, n=240, seed=0):
    rng = np.random.default_rng(seed)
    party = rng.choice(["Republican", "Democrat", "Independent", "Refused"], size=n, p=[0.35, 0.35, 0.25, 0.05])
    lean = {"Republican": 0.25, "Democrat": 0.8, "Independent": 0.5, "Refused": 0.5}
    favour = np.array([lean[p] for p in party])
    four = ["Strongly favor", "Somewhat favor", "Somewhat oppose", "Strongly oppose"]

    def likert(p):
        probs = np.column_stack([p * 0.6, p * 0.4, (1 - p) * 0.4, (1 - p) * 0.6])
        return [four[rng.choice(4, p=row)] if rng.random() > 0.1 else "Refused" for row in probs]

    frame = pd.DataFrame({
        "QKEY": [f"id{j}" for j in range(n)], "POLPARTY": party,
        "AGE": rng.choice(["18-29", "30-49", "50-64", "65+"], size=n), "F SEX": rng.choice(["Male", "Female"], size=n),
        "GUNS_W1": likert(1 - favour), "CLIMATE_W1": likert(favour),
        "WALL_W1": [("Yes" if rng.random() < 1 - f else "No") for f in favour],
        "OPEN_W1": ["free text"] * n,
    })
    os.makedirs(folder, exist_ok=True)
    frame.to_csv(os.path.join(folder, "responses.csv"), index=False)
    pd.DataFrame({"key": ["GUNS_W1", "CLIMATE_W1", "WALL_W1"],
                  "question": ["Do you favor or oppose stricter gun laws?", "Do you favor a carbon tax?",
                               "Should the wall be expanded?"],
                  "references": [str(four), str(four), "['Yes', 'No']"]}).to_csv(os.path.join(folder, "info.csv"),
                                                                                  index=False)
    return frame


SURVEY_SPEC = {
    "responses_csv": "wave/responses.csv", "info_csv": "wave/info.csv", "respondent_id": "QKEY",
    "group_column": "POLPARTY",
    "groups": {"Republican": "Republican", "Democrat": "Democrat", "Independent": "Independent"},
    "persona_columns": ["AGE", "F SEX", "POLPARTY", "NOT_A_COLUMN"], "max_per_group": 50, "seed": 1,
    "proposals": [
        {"key": "GUNS_W1", "support": {"Strongly favor": "agree", "Somewhat favor": "qualified_agree",
                                       "Somewhat oppose": "qualified_disagree", "Strongly oppose": "disagree"}},
        {"key": "CLIMATE_W1", "text": "A carbon tax?",
         "support": {"Strongly favor": "agree", "Somewhat favor": "qualified_agree",
                     "Somewhat oppose": "qualified_disagree", "Strongly oppose": "disagree"}},
        {"key": "WALL_W1", "support": {"Yes": "agree", "No": "disagree"}},
    ],
}


def test_survey_testbed(tmp_path):
    raw = _write_wave(str(tmp_path / "wave"))
    tb = opinionqa.load_opinionqa(SURVEY_SPEC, root=str(tmp_path))
    prob = tb.prob
    assert tb.name == "opinionqa" and prob.group_names == ["Democrat", "Independent", "Republican"]
    assert prob.n_dec == 3 and prob.group_sizes.max() <= 50 and prob.n_org == prob.group_sizes.sum()
    assert not set(prob.orgs) & set(raw.loc[raw["POLPARTY"] == "Refused", "QKEY"])
    # stances are exactly the mapped raw answers; refusals are missing cells
    lookup = raw.set_index("QKEY")
    keys = ["GUNS_W1", "CLIMATE_W1", "WALL_W1"]
    for (org, qid), stance in list(zip(prob.cell_keys, prob.cell_stance))[::17]:
        assert SURVEY_SPEC["proposals"][qid - 1]["support"][lookup.loc[org, keys[qid - 1]]] == stance
    n_refused = sum(int((lookup.loc[prob.orgs, k] == "Refused").sum()) for k in keys)
    assert prob.n_cells == 3 * prob.n_org - n_refused
    assert prob.questions[1]["question"] == "Do you favor or oppose stricter gun laws?"
    assert prob.questions[2]["question"] == "A carbon tax?"
    # groups disagree by construction, so the objectives pick different packages or at least differ in value
    engine = DesignEngine(prob, synth.oracle(prob))
    assert describe.support_tables(prob)[0]["spread"].max() > 0.3
    assert engine.true_values("maxmin").max() < engine.true_values("utilitarian").max()
    # prompts: demographic persona, the survey's own answer options, mapped back to stance classes
    src = tb.source
    assert "AGE:" in src.persona(0, "named") and "F SEX:" in src.persona(0, "named")
    assert "AGE:" not in src.persona(0, "anonymous") and "POLPARTY:" in src.persona(0, "anonymous")
    assert [o.key for o in src.options(3)] == ["yes", "no"]
    assert [o.stance for o in src.options(1)] == ["agree", "qualified_agree", "qualified_disagree", "disagree"]
    out = PersonaRunner(prob, src, MockBackend(), PersonaSpec(n_samples=0)).run("survey", cells=np.arange(20))
    assert out.probs[:, data.STANCE_INDEX["mixed"]].sum() == 0.0          # no option maps to 'mixed'
    np.testing.assert_allclose(out.probs.sum(1), 1.0)
    listing = opinionqa.inspect_wave(str(tmp_path / "wave"))
    assert {"GUNS_W1", "CLIMATE_W1", "WALL_W1", "POLPARTY"} <= set(listing["column"])
    assert "OPEN_W1" not in set(listing["column"]) and "QKEY" not in set(listing["column"])
    assert listing.set_index("column").loc["WALL_W1", "question"] == "Should the wall be expanded?"
    assert opinionqa.parse_list_cell("['Yes', 'No']") == ["Yes", "No"] and opinionqa.parse_list_cell("oops") == []


def test_survey_spec_errors(tmp_path):
    _write_wave(str(tmp_path / "wave"))
    bad_column = {**SURVEY_SPEC, "proposals": [{"key": "MISSING_W1", "support": {"Yes": "agree"}}]}
    with pytest.raises(ValueError, match="columns missing"):
        opinionqa.load_opinionqa(bad_column, root=str(tmp_path))
    bad_stance = {**SURVEY_SPEC, "proposals": [{"key": "WALL_W1", "support": {"Yes": "strongly_agree"}}]}
    with pytest.raises(ValueError, match="unknown stance labels"):
        opinionqa.load_opinionqa(bad_stance, root=str(tmp_path))


def test_example_spec_is_a_valid_template(root_dir):
    import yaml
    with open(os.path.join(root_dir, "configs", "opinionqa_example.yaml"), encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    assert {"responses_csv", "group_column", "groups", "proposals"} <= set(spec)
    for p in spec["proposals"]:
        assert set(p["support"].values()) <= set(data.STANCES)
    assert len(set(spec["groups"].values())) == 4


# ------------------------------------------------------------------ AASB ED SR1 (needs Dataset A)

@pytest.fixture(scope="module")
def aasb(root_dir):
    cfg = config.load_config(root=root_dir)
    return cfg, common.load_testbed(cfg)


@requires_aasb
def test_aasb_problem_matches_the_methodology(aasb):
    """Section 1.1: 100 organisations, 1,177 responded cells, 18 decisions, groups 38/27/20/15."""
    cfg, tb = aasb
    prob = tb.prob
    assert prob.n_org == 100 and prob.n_cells == 1177 and prob.n_dec == 18 and prob.n_groups == 4
    assert dict(zip(prob.group_names, prob.group_sizes.tolist())) == {
        "Assurance+Advisory": 20, "Investors": 15, "Preparers": 38, "Public+Academic+Civil": 27}
    assert prob.decisions == [2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 23]
    assert set(np.unique(prob.cell_q)) == set(data.MAIN_DECISIONS)         # includes Q14 -> decision 13
    assert tb.board is not None and int((tb.board == 0).sum()) == 15
    assert tb.meta["n_orgs_needing_review"] == 0
    assert tb.context.shape == (100, len(tb.context_qids)) and not set(tb.context_qids) & set(data.MAIN_DECISIONS)
    assert np.nanmin(tb.context) >= 0 and np.nanmax(tb.context) <= 1 and np.isnan(tb.context).any()


@requires_aasb
def test_aasb_reproduces_the_day1_precondition_table(aasb, root_dir):
    """Section 12, P1: the refactored welfare engine gives the numbers recorded on Day 1."""
    cfg, tb = aasb
    engine = common.make_engine(cfg, tb, synth.oracle(tb.prob))
    packages, hamming = describe.package_table(engine, ["utilitarian", "nash", "maxmin"])
    pk = packages.set_index("package")
    expected = {  # package: (revised, utilitarian, max-min, worst group)
        "ED (retain all)": ("-", 0.628, 0.549, "Investors"),
        "IFRS baseline (revise all)": (None, 0.372, 0.244, "Preparers"),
        "optimum: utilitarian": ("Q16,Q20,Q21", 0.667, 0.613, "Public+Academic+Civil"),
        "optimum: nash": ("Q16,Q20,Q21", 0.667, 0.613, "Public+Academic+Civil"),
        "optimum: maxmin": ("Q8,Q10,Q20,Q21", 0.657, 0.647, "Public+Academic+Civil"),
    }
    for name, (revised, util, mm, worst) in expected.items():
        row = pk.loc[name]
        if revised is not None:
            assert row["revised"] == revised
        assert row["W_utilitarian"] == pytest.approx(util, abs=6e-4)
        assert row["W_maxmin"] == pytest.approx(mm, abs=6e-4)
        assert row["worst_group"] == worst
    ham = hamming.set_index(["a", "b"])["hamming"]
    assert ham[("utilitarian", "maxmin")] == 3 and ham[("utilitarian", "nash")] == 0
    # ... and, where the Day-1 output files are still on disk, their numbers to three decimals
    old = os.path.join(root_dir, "results", "precheck", "p1_packages.csv")
    if os.path.exists(old):
        before = pd.read_csv(old).set_index("package")
        rename = {"ED (retain all)": "ED (retain all)", "IFRS (revise all)": "IFRS baseline (revise all)"}
        for label, row in before.iterrows():
            new = pk.loc[rename.get(label, label)]
            for g in tb.prob.group_names:
                assert new[g] == pytest.approx(row[g], abs=6e-4)
            for obj in ("utilitarian", "nash", "maxmin"):
                assert new[f"W_{obj}"] == pytest.approx(row[f"W_{obj}"], abs=6e-4)
    support, counts = describe.support_tables(tb.prob)
    # Q8 has the largest gap: preparers 0.85 vs investors 0.22. The Day-1 table reports 0.63, the
    # difference of the two *rounded* means; the unrounded gap is 0.623.
    assert support.loc[8, "Preparers"] == pytest.approx(0.85, abs=0.005)
    assert support.loc[8, "Investors"] == pytest.approx(0.22, abs=0.005)
    assert round(round(support.loc[8, "Preparers"], 2) - round(support.loc[8, "Investors"], 2), 2) == 0.63
    assert support.loc[8, "spread"] == pytest.approx(0.623, abs=0.001)
    assert support["spread"].idxmax() == 8
    assert int(counts.to_numpy().sum()) == 1177
    # the Board's package under the stance-welfare objectives (diagnostic only)
    assert "Board's decision" in pk.index and pk.loc["Board's decision", "n_revised"] == 15


@requires_aasb
def test_aasb_prompts_never_leak(aasb, root_dir):
    """Every prompt of every persona variant is built for the real data and checked.

    (a) Named and anonymous prompts depend on the organisation only through the "Respondent:" line,
        so nothing about its answers can enter them.
    (b) Anonymous prompts never contain the organisation's name or acronym, context included.
    (c) Context items come from permitted questions only: never the question being predicted, never
        the question sharing its decision (Q13/Q14), and in 'outside' mode never a decision question.
    """
    cfg, tb = aasb
    prob, src = tb.prob, tb.source
    df = data.load_dataset_a(config.resolve_path(cfg, cfg["testbed"]["aasb"]["parquet_dir"]))
    responded = df[(df["answer"] != "no_response") & ~df["org"].isin(data.EXCLUDE_ORGS)]
    answered = responded.groupby("org")["qid"].apply(lambda q: {int(x) for x in q}).to_dict()
    rationale = {(r.org, int(r.qid)): " ".join(str(r.key_rationale).split()) for r in responded.itertuples()}
    twins = {13: 14, 14: 13}
    opts = prompts.stance_options()

    def without_persona(text):
        return "\n".join(line for line in text.splitlines() if not line.startswith("Respondent:"))

    shared = {"named": {}, "anonymous": {}}
    n_items = 0
    for cell in range(prob.n_cells):
        i, q = int(prob.cell_org[cell]), int(prob.cell_q[cell])
        org = prob.orgs[i]
        named = prompts.build_messages(src, i, q, "named", opts)[1]["content"]
        anon = prompts.build_messages(src, i, q, "anonymous", opts)[1]["content"]
        ctx_anon = prompts.build_messages(src, i, q, "context", opts, identity="anonymous",
                                          max_context_items=40)[1]["content"]
        assert org in named and src.question(q) in named and src.background(q).strip() in named
        assert not prompts.contains_alias(anon, org), (org, q)
        assert not prompts.contains_alias(ctx_anon, org), (org, q)
        shared["named"].setdefault(q, set()).add(without_persona(named))
        shared["anonymous"].setdefault(q, set()).add(without_persona(anon))
        # context: exactly the organisation's answers to the questions outside the decision set
        allowed = sorted(qq for qq in answered[org] if qq not in data.MAIN_DECISIONS)
        items = src.context(i, q)
        assert [it.question for it in items] == [src.question(qq) for qq in allowed], (org, q)
        for it, qq in zip(items, allowed):
            expected = rationale[(org, qq)]
            assert it.rationale == ("" if expected.upper() in ("", "NONE", "NAN") else expected)
        n_items += len(items)
    assert n_items > 1000
    for variant, by_question in shared.items():
        assert all(len(texts) == 1 for texts in by_question.values()), variant
    # 'loo' mode: every other answered question, minus the twin
    from sim.testbeds.aasb import AasbPromptSource
    loo = AasbPromptSource(prob, responded, src.questions, data.MAIN_DECISIONS, context_mode="loo")
    explicit = AasbPromptSource(prob, responded, src.questions, data.MAIN_DECISIONS, context_mode="explicit",
                                context_questions=[33, 34, 13, 14])
    for cell in range(0, prob.n_cells, 7):
        i, q = int(prob.cell_org[cell]), int(prob.cell_q[cell])
        org = prob.orgs[i]
        allowed = sorted(answered[org] - {q, twins.get(q)})
        assert [it.question for it in loo.context(i, q)] == [src.question(qq) for qq in allowed]
        chosen = sorted((answered[org] & {33, 34, 13, 14}) - {q, twins.get(q)})
        assert [it.question for it in explicit.context(i, q)] == [src.question(qq) for qq in chosen]
    with pytest.raises(ValueError):
        AasbPromptSource(prob, responded, src.questions, data.MAIN_DECISIONS, context_mode="everything")
    with pytest.raises(ValueError, match="reveals its name"):
        AasbPromptSource(prob, responded, src.questions, data.MAIN_DECISIONS,
                         profiles={"BHP": "a miner; BHP in all but name"})


@requires_aasb
def test_aasb_variants_load(root_dir):
    for overrides, groups, decisions in (
            (["testbed.aasb.grouping=coarse2"], 2, 18), (["testbed.aasb.grouping=coarse3"], 3, 18),
            (["testbed.aasb.grouping=fine"], 9, 18), (["testbed.aasb.decision_set=main+nfp"], 4, 22),
            (["welfare.sigma_map=binary"], 4, 18)):
        tb = common.load_testbed(config.load_config(overrides=overrides, root=root_dir))
        assert tb.prob.n_groups == groups and tb.prob.n_dec == decisions
        tb.prob.validate()
    with pytest.raises(ValueError, match="unknown decision_set"):
        common.load_testbed(config.load_config(overrides=["testbed.aasb.decision_set=all"], root=root_dir))


@requires_aasb
def test_org_table_is_consistent(root_dir):
    table = data.read_org_table(ORG_GROUPS)
    # the nine borderline assignments (data.UNCERTAIN) were reviewed by hand: none was changed, none is flagged now
    assert len(table) == 107 and table["needs_review"].sum() == 0
    assert (table["group"] == table["fine_type"].map(data.COARSE)).all()
    assert (table["fine_type"] == table["org"].map(data.fine_type)).all()
    profiles = pd.read_csv(os.path.join(root_dir, "data", "org_profiles.csv")).fillna("")
    assert set(profiles["org"]) == set(table["org"])
    for org, text in zip(profiles["org"], profiles["anon_description"]):
        assert not prompts.contains_alias(text, org)
