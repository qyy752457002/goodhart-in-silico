"""The AASB ED SR1 testbed: staff-labelled stances of organisations on Australia's draft
climate-related disclosure standard.

Methodology map (03_methodology.md): section 6 (testbed), 1.3 (information scenarios S1 / S2) and
2.1 (what each simulator may see).

Scenario S2 context. `context_mode` decides which of an organisation's *other* answers a context
persona (R2) or a relational model (R3) may use:
  'outside'   answers to questions outside the decision set (general matters etc.): the decision
              set stays whole, so the same 18-decision problem is evaluated;
  'loo'       every other answer, i.e. leave-one-question-out retrodiction;
  'explicit'  the question ids listed in `context_questions`.
The target question, and any question sharing its decision variable (Q13/Q14), is always excluded.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .. import data
from ..simulators import prompts
from .common import ProblemPromptSource, Testbed, strip_question_number

CONSULTATION = (
    "Australian Accounting Standards Board (AASB) Exposure Draft ED SR1, \"Australian Sustainability "
    "Reporting Standards - Disclosure of Climate-related Financial Information\" (October 2023). The "
    "draft standards are based on IFRS S1 and IFRS S2, with proposed Australian-specific modifications; "
    "respondents were asked whether they agree with each proposed modification."
)


class AasbPromptSource(ProblemPromptSource):
    consultation = CONSULTATION

    def __init__(self, prob: data.Problem, cells: pd.DataFrame, questions: dict, decision_map: dict,
                 context_mode: str = "outside", context_questions: list = None, profiles: dict = None,
                 use_rationale: bool = True):
        super().__init__(prob)
        if context_mode not in ("outside", "loo", "explicit"):
            raise ValueError(f"unknown context_mode {context_mode!r}")
        self.questions = questions            # qid -> {"question", "background"} for all questions
        self.decision_map = dict(decision_map)
        self.context_mode = context_mode
        self.context_questions = set(int(q) for q in (context_questions or []))
        self.use_rationale = use_rationale
        self.profiles = dict(profiles or {})
        for org, text in self.profiles.items():
            if prompts.contains_alias(text, org):
                raise ValueError(f"anonymous description of {org!r} reveals its name: {text!r}")
        self._by_org = {org: grp.sort_values("qid") for org, grp in cells.groupby("org")}

    def persona(self, org_index: int, variant: str) -> str:
        org = self.prob.orgs[org_index]
        fine = self.prob.fine_types[org_index] if self.prob.fine_types else None
        described = prompts.FINE_TYPE_DESCRIPTIONS.get(fine, "a respondent to the consultation")
        if variant == "anonymous":
            extra = self.profiles.get(org, "").strip()
            return f"{described} (name withheld).{' ' + extra if extra else ''}"
        return f"{org} -- {described}."

    def question(self, qid: int) -> str:
        return strip_question_number(self.questions[int(qid)]["question"])

    def background(self, qid: int) -> str:
        return self.questions[int(qid)].get("background", "")

    def _allowed(self, qid: int, other: int) -> bool:
        if other == qid:
            return False
        target_dec = self.decision_map.get(qid)
        if target_dec is not None and self.decision_map.get(other) == target_dec:
            return False
        if self.context_mode == "outside":
            return other not in self.decision_map
        if self.context_mode == "explicit":
            return other in self.context_questions
        return True

    def context(self, org_index: int, qid: int, anonymous: bool = False) -> list:
        org = self.prob.orgs[org_index]
        rows = self._by_org.get(org)
        if rows is None:
            return []
        items = []
        for r in rows.itertuples():
            if not self._allowed(int(qid), int(r.qid)):
                continue
            rationale = ""
            if self.use_rationale:
                rationale = " ".join(str(r.key_rationale).split())
                rationale = "" if rationale.upper() in ("", "NONE", "NAN") else rationale
            question = self.question(int(r.qid))
            if anonymous:
                rationale = prompts.redact(rationale, org)
            items.append(prompts.ContextItem(question=question, stance_text=prompts.STANCE_WORDS[r.answer],
                                             rationale=rationale))
        return items


def read_profiles(path: str) -> dict:
    """Optional per-organisation one-sentence anonymous descriptions: CSV with columns
    `org, anon_description`. Missing file or empty cells fall back to the stakeholder-type text."""
    if not path or not os.path.exists(path):
        return {}
    table = pd.read_csv(path, encoding="utf-8").fillna("")
    return {o: str(t) for o, t in zip(table["org"], table["anon_description"]) if str(t).strip()}


def context_matrix(prob: data.Problem, cells: pd.DataFrame, decision_map: dict) -> tuple[np.ndarray, list]:
    """(n_org, n_context) support on the questions outside the decision set, NaN where unanswered."""
    outside = cells[~cells["qid"].isin(decision_map) & cells["org"].isin(prob.orgs)]
    qids = sorted(int(q) for q in outside["qid"].unique())
    mat = np.full((prob.n_org, len(qids)), np.nan)
    oi = {o: k for k, o in enumerate(prob.orgs)}
    qi = {q: k for k, q in enumerate(qids)}
    for r in outside.itertuples():
        mat[oi[r.org], qi[int(r.qid)]] = prob.sigma_map[r.answer]
    return mat, qids


def load_aasb(tb_cfg: dict, welfare_cfg: dict, root: str = ".") -> Testbed:
    """Assemble the AASB testbed from the configuration (`testbed.aasb` and `welfare`)."""
    def path(key):
        value = tb_cfg.get(key)
        return None if not value else (value if os.path.isabs(value) else os.path.join(root, value))

    df = data.load_dataset_a(path("parquet_dir"))
    table = data.read_org_table(path("org_groups_csv"))
    decision_set = tb_cfg.get("decision_set", "main")
    if decision_set not in data.DECISION_SETS:
        raise ValueError(f"unknown decision_set {decision_set!r}; choose from {sorted(data.DECISION_SETS)}")
    decision_map = data.DECISION_SETS[decision_set]
    sigma_map = data.make_sigma_map(welfare_cfg.get("sigma_map", "graded"),
                                    welfare_cfg.get("qualified_weight", 0.75))
    groups = data.groups_for_scheme(table, tb_cfg.get("grouping", "coarse4"))
    responded = df[(df["answer"] != "no_response") & df["answer"].isin(sigma_map)
                   & ~df["org"].isin(data.EXCLUDE_ORGS)]
    missing = sorted(set(responded.loc[responded["qid"].isin(decision_map), "org"]) - set(groups))
    if missing:
        raise ValueError(f"organisations missing from {tb_cfg.get('org_groups_csv')}: {missing}")
    prob = data.build_problem(df, groups, decision_map, sigma_map,
                              fine_types=dict(zip(table["org"], table["fine_type"])))
    prob.name = "aasb"
    prob.validate()
    questions = {int(r.qid): {"question": r.question, "background": r.background}
                 for r in df.drop_duplicates("qid").itertuples()}
    source = AasbPromptSource(prob, responded, questions, decision_map,
                              context_mode=tb_cfg.get("context_mode", "outside"),
                              context_questions=tb_cfg.get("context_questions"),
                              profiles=read_profiles(path("org_profiles_csv")),
                              use_rationale=tb_cfg.get("context_use_rationale", True))
    ctx, ctx_qids = context_matrix(prob, responded, decision_map)
    board = data.load_board_decisions(path("board_decisions_csv"), prob.decisions)
    needs_review = int(table["needs_review"].sum()) if "needs_review" in table.columns else 0
    return Testbed(name="aasb", prob=prob, source=source, context=ctx, context_qids=ctx_qids, board=board,
                   meta={"decision_set": decision_set, "grouping": tb_cfg.get("grouping", "coarse4"),
                         "n_orgs_needing_review": needs_review})
