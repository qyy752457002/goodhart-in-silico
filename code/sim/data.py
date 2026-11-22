"""Load AASB ED SR1 stance data (Dataset A) and build the decision problem.

Decisions follow AASB internal question numbering (see the dataset card).
Main decision set: 18 proposal questions; Q14 (usefulness of the Q13 requirement)
shares Q13's decision variable. NFP-specific proposals form an extension set.

Methodology map (03_methodology.md): section 1.1 (notation) and section 8 (sensitivity:
stance-to-support mappings and grouping schemes).
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np
import pandas as pd

# question id -> decision id (questions sharing a decision map to the same id)
MAIN_DECISIONS = {q: q for q in [2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 23]}
MAIN_DECISIONS[14] = 13
NFP_DECISIONS = {q: q for q in [24, 26, 27, 29]}
DECISION_SETS = {
    "main": MAIN_DECISIONS,
    "nfp": NFP_DECISIONS,
    "main+nfp": {**MAIN_DECISIONS, **NFP_DECISIONS},
}

STANCE_SIGMA = {  # support for retaining the ED proposal
    "agree": 1.0, "qualified_agree": 0.75, "mixed": 0.5,
    "qualified_disagree": 0.25, "alternative_proposal": 0.25, "disagree": 0.0,
}
STANCE_SIGMA_BINARY = {
    "agree": 1.0, "qualified_agree": 1.0, "mixed": 0.5,
    "qualified_disagree": 0.0, "alternative_proposal": 0.0, "disagree": 0.0,
}
STANCES = list(STANCE_SIGMA)  # the 6 response classes (no_response = missing)
STANCE_INDEX = {s: k for k, s in enumerate(STANCES)}

# Coarse 3-way merge of the stance classes (methodology section 2.3).
STANCE3 = {
    "agree": "support", "qualified_agree": "support", "mixed": "mixed",
    "qualified_disagree": "oppose", "alternative_proposal": "oppose", "disagree": "oppose",
}
STANCE3_CLASSES = ["support", "mixed", "oppose"]

EXCLUDE_ORGS = {"Climate Disclosures Consultation Update"}  # AASB's own document

# Rule-based stakeholder typing by organisation-name prefix (review data/org_groups.csv).
TYPE_RULES = {
    "CORP": ["ADCO", "AGL", "ATCO", "Alinta", "Ampol", "Aurizon", "BHP", "CBA", "Chubb", "Microsoft",
             "National Australia Bank", "Origin", "Panasonic", "QBE", "Telstra", "Westpac", "Whitehaven"],
    "ASSOC": ["Australian Banking", "Australian Finance Industry", "Australian Financial Markets",
              "Australian Forrest", "Australian Information Industry", "Australian Institute of Company",
              "Australian Lot Feeders", "Australian Pork", "Australian Retailers", "Australian Steel",
              "Business Council for Sustainable", "Business Council of Australia", "Clean Energy Investor",
              "Clubs Australia", "Customer Owned Banking", "Energy Efficiency", "Financial Services Council",
              "Insurance Council", "Minerals Council", "National Farmers", "Property Council", "Red Meat",
              "The Australian Industry Group", "Water Services"],
    "INV": ["ACSI", "Adamantem", "Australian Retirement Trust", "Australian Super", "Australian Sustainable Finance",
            "British Columbia", "Impax", "Norges", "Principles for Responsible", "Responsible Investment Association",
            "T. Rowe", "The Investor Group", "UniSuper", "University Pension", "Yarra", "Zenith"],
    "ASSUR": ["ACCA", "Accountants and Actuaries", "BDO", "Deloitte", "EY", "Governance Institute", "Grant Thornton",
              "Institute of Public Accountants", "KPMG", "PwC"],
    "SERV": ["Anthesis", "Basford", "Bloomberg", "GHG Offset", "Gen Sustainability", "Greenbase", "Herbert Smith",
             "MSCI", "Scyne", "Thinkstep"],
    "PUBLIC": ["ABS", "APRA", "ASIC", "Australasian Council of Auditors", "Central Coast", "City of Newcastle",
               "City of Ryde", "Council of Financial Regulators", "HoTARAC", "Local Government Association",
               "Local Government NSW", "NSW Treasury", "Northern Beaches", "Waverly", "Green Power",
               "Sustainability Compliance team"],
    "ACAD": ["Curtin", "Deakin", "Department of Accounting", "Monash", "Professor Christine",
             "Social and Environmental Sustainability", "UNSW", "Victoria University"],
    "CIVIL": ["Climateworks", "Environment Institute", "Environmental Defenders", "Maritime Union"],
    "INDIV": ["David Hardidge", "Kate McNicholl"],
}
COARSE = {"CORP": "Preparers", "ASSOC": "Preparers", "INV": "Investors",
          "ASSUR": "Assurance+Advisory", "SERV": "Assurance+Advisory",
          "PUBLIC": "Public+Academic+Civil", "ACAD": "Public+Academic+Civil",
          "CIVIL": "Public+Academic+Civil", "INDIV": "Public+Academic+Civil"}
UNCERTAIN = {"Green Power", "Sustainability Compliance team", "Australian Sustainable Finance", "Clean Energy Investor",
             "Bloomberg", "MSCI", "Australian Institute of Company", "Environment Institute", "Maritime Union"}

# Grouping schemes for the sensitivity analysis (methodology section 8): fine type -> group.
# "coarse4" is the main specification. "coarse2" is the fallback named in 02 section 4 (preparers
# vs everyone else). "coarse3" is one reasonable intermediate scheme -- edit it if you prefer another.
GROUP_SCHEMES = {
    "coarse4": COARSE,
    "coarse3": {"CORP": "Preparers", "ASSOC": "Preparers",
                "INV": "Investors+Assurance+Advisory", "ASSUR": "Investors+Assurance+Advisory",
                "SERV": "Investors+Assurance+Advisory",
                "PUBLIC": "Public+Academic+Civil", "ACAD": "Public+Academic+Civil",
                "CIVIL": "Public+Academic+Civil", "INDIV": "Public+Academic+Civil"},
    "coarse2": {t: ("Preparers" if t in ("CORP", "ASSOC") else "Non-preparers") for t in TYPE_RULES},
    "fine": {t: t for t in TYPE_RULES},
}


def make_sigma_map(kind: str = "graded", qualified_weight: float = 0.75) -> dict:
    """Stance -> support score sigma(s) in [0, 1].

    graded: agree 1, qualified_agree w, mixed 0.5, qualified_disagree / alternative_proposal 1 - w,
            disagree 0 (w = 0.75 is the main specification; 0.6 and 0.9 are the sensitivity values).
    binary: qualified stances are rounded to full support / full opposition.
    """
    if kind == "binary":
        return dict(STANCE_SIGMA_BINARY)
    if kind == "graded":
        w = float(qualified_weight)
        if not 0.5 <= w <= 1.0:
            raise ValueError(f"qualified_weight must lie in [0.5, 1], got {w}")
        return {"agree": 1.0, "qualified_agree": w, "mixed": 0.5,
                "qualified_disagree": 1.0 - w, "alternative_proposal": 1.0 - w, "disagree": 0.0}
    raise ValueError(f"unknown sigma map kind {kind!r} (use 'graded' or 'binary')")


def clean_org(name: str) -> str:
    return re.sub(r"_(edsr1|aasb)_.*$", "", name)


def fine_type(org: str) -> str:
    hits = [t for t, pats in TYPE_RULES.items() for p in pats if org.startswith(p)]
    if len(hits) != 1:
        raise ValueError(f"cannot type organisation {org!r}: {hits}")
    return hits[0]


def load_dataset_a(parquet_dir: str) -> pd.DataFrame:
    """One row per (organisation, question): stance label plus the question/background text.

    `conditions` and `key_rationale` are LLM-extracted summaries of the organisation's own answer.
    They reveal the stance, so they must never enter a prompt that predicts the same cell; they are
    only used as context for *other* questions (simulator R2, scenario S2).
    """
    if not os.path.isdir(parquet_dir):
        raise FileNotFoundError(
            f"Dataset A not found at {parquet_dir!r}. Download it first: `python -m sim fetch-data`.")
    files = [os.path.join(parquet_dir, f) for f in sorted(os.listdir(parquet_dir)) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"no .parquet files in {parquet_dir!r}")
    df = pd.concat([pd.read_parquet(f).assign(split=os.path.basename(f).split("-")[0]) for f in files],
                   ignore_index=True)
    prob = df["problem"].map(json.loads)
    df["org"] = prob.map(lambda d: clean_org(d["organization"]))
    df["qid"] = prob.map(lambda d: int(re.search(r"\[Question (\d+)\]", d["question"]).group(1)))
    df["question"] = prob.map(lambda d: d["question"])
    df["background"] = prob.map(lambda d: d.get("background", ""))
    df["conditions"] = prob.map(lambda d: d.get("conditions", ""))
    df["key_rationale"] = prob.map(lambda d: d.get("key_rationale", ""))
    return df.drop(columns=["problem"])


def write_org_groups(orgs, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["org", "fine_type", "group", "needs_review"])
        for o in sorted(orgs):
            t = fine_type(o)
            w.writerow([o, t, COARSE[t], int(any(o.startswith(u) for u in UNCERTAIN))])


def read_org_groups(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return {r["org"]: r["group"] for r in csv.DictReader(f)}


def read_org_table(path: str) -> pd.DataFrame:
    """The curated organisation table: org, fine_type, group, needs_review (+ optional columns)."""
    table = pd.read_csv(path, encoding="utf-8")
    missing = {"org", "fine_type", "group"} - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if table["org"].duplicated().any():
        raise ValueError(f"{path}: duplicated organisations")
    return table


def groups_for_scheme(table: pd.DataFrame, scheme: str = "coarse4") -> dict:
    """org -> group under a grouping scheme.

    "coarse4" returns the `group` column as curated in the CSV (so manual corrections made there are
    respected); the other schemes are derived from `fine_type`.
    """
    if scheme == "coarse4":
        return dict(zip(table["org"], table["group"]))
    if scheme not in GROUP_SCHEMES:
        raise ValueError(f"unknown grouping scheme {scheme!r}; choose from {sorted(GROUP_SCHEMES)}")
    mapping = GROUP_SCHEMES[scheme]
    return {o: mapping[t] for o, t in zip(table["org"], table["fine_type"])}


@dataclass
class Problem:
    orgs: list            # organisations with >= 1 response in the decision set
    groups: list          # group name per org
    group_names: list
    decisions: list       # decision ids (columns of a package)
    cell_org: np.ndarray  # per responded cell: org index
    cell_dec: np.ndarray  # per responded cell: decision index
    cell_q: np.ndarray    # per responded cell: question id
    cell_stance: np.ndarray  # per responded cell: stance label
    sigma: np.ndarray     # per responded cell: support score in [0, 1]
    # --- optional metadata (not needed by the welfare maths) ---
    sigma_map: dict = field(default_factory=lambda: dict(STANCE_SIGMA))  # stance -> support score
    fine_types: list | None = None   # fine stakeholder type per org (prompts, regrouping)
    questions: dict | None = None    # qid -> {"question": str, "background": str}
    name: str = "aasb"

    @cached_property
    def g_index(self) -> np.ndarray:
        return np.array([self.group_names.index(g) for g in self.groups])

    @property
    def n_org(self) -> int:
        return len(self.orgs)

    @property
    def n_dec(self) -> int:
        return len(self.decisions)

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    @property
    def n_cells(self) -> int:
        return len(self.sigma)

    @cached_property
    def group_sizes(self) -> np.ndarray:
        return np.bincount(self.g_index, minlength=self.n_groups)

    @cached_property
    def cells_per_org(self) -> np.ndarray:
        return np.bincount(self.cell_org, minlength=self.n_org)

    @cached_property
    def sigma_vec(self) -> np.ndarray:
        """Support score of each stance class, in `STANCES` order."""
        return np.array([self.sigma_map[s] for s in STANCES], dtype=float)

    @cached_property
    def stance_idx(self) -> np.ndarray:
        """Class index (position in `STANCES`) of each cell's true stance."""
        return np.array([STANCE_INDEX[s] for s in self.cell_stance], dtype=int)

    @cached_property
    def cell_keys(self) -> list:
        """(organisation, question id) per cell -- the join key for simulator outputs."""
        return [(self.orgs[i], int(q)) for i, q in zip(self.cell_org, self.cell_q)]

    def decision_labels(self) -> list:
        return [f"Q{q}" for q in self.decisions]

    def validate(self) -> None:
        n = self.n_cells
        for name in ("cell_org", "cell_dec", "cell_q", "cell_stance"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has length {len(getattr(self, name))}, expected {n}")
        if n == 0:
            raise ValueError("the problem has no responded cells")
        if self.cells_per_org.min() < 1:
            raise ValueError("every organisation needs at least one responded cell")
        if len(self.groups) != self.n_org:
            raise ValueError("groups must have one entry per organisation")
        if self.sigma.min() < 0 or self.sigma.max() > 1:
            raise ValueError("support scores must lie in [0, 1]")
        if len(set(self.cell_keys)) != n:
            raise ValueError("duplicated (organisation, question) cells")

    def with_sigma_map(self, sigma_map: dict) -> Problem:
        """Same cells, different stance-to-support mapping (sensitivity analysis)."""
        sigma = np.array([sigma_map[s] for s in self.cell_stance], dtype=float)
        return replace(self, sigma=sigma, sigma_map=dict(sigma_map))

    def with_groups(self, org_groups: dict) -> Problem:
        """Same cells, different stakeholder grouping (sensitivity analysis)."""
        groups = [org_groups[o] for o in self.orgs]
        return replace(self, groups=groups, group_names=sorted(set(groups)))


def build_problem(df: pd.DataFrame, org_groups: dict, decision_map: dict = None,
                  sigma_map: dict = None, fine_types: dict = None) -> Problem:
    decision_map = decision_map or MAIN_DECISIONS
    sigma_map = sigma_map or STANCE_SIGMA
    d = df[df["qid"].isin(decision_map) & ~df["org"].isin(EXCLUDE_ORGS) & (df["answer"] != "no_response")]
    d = d[d["answer"].isin(sigma_map)].copy()
    orgs = sorted(d["org"].unique())
    decisions = sorted(set(decision_map.values()))
    oi = {o: k for k, o in enumerate(orgs)}
    di = {q: k for k, q in enumerate(decisions)}
    groups = [org_groups[o] for o in orgs]
    group_names = sorted(set(groups))
    questions = None
    if {"question", "background"} <= set(df.columns):
        q_rows = df[df["qid"].isin(decision_map)].drop_duplicates("qid")
        questions = {int(r.qid): {"question": r.question, "background": r.background}
                     for r in q_rows.itertuples()}
    return Problem(
        orgs=orgs, groups=groups, group_names=group_names, decisions=decisions,
        cell_org=d["org"].map(oi).to_numpy(), cell_dec=d["qid"].map(lambda q: di[decision_map[q]]).to_numpy(),
        cell_q=d["qid"].to_numpy(), cell_stance=d["answer"].to_numpy(),
        sigma=d["answer"].map(sigma_map).to_numpy(dtype=float),
        sigma_map=dict(sigma_map),
        fine_types=None if fine_types is None else [fine_types[o] for o in orgs],
        questions=questions,
    )


def load_board_decisions(path: str, decisions: list) -> np.ndarray | None:
    """The Board's actual package (1 = retained the ED proposal, 0 = reverted), aligned with
    `decisions`. Diagnostic only -- never ground truth (methodology section 5, item 10).

    Returns None when the file is missing or does not cover every decision.
    """
    if not path or not os.path.exists(path):
        return None
    table = pd.read_csv(path, encoding="utf-8")
    lookup = dict(zip(table["decision_qid"].astype(int), table["a"].astype(int)))
    if any(q not in lookup for q in decisions):
        return None
    return np.array([lookup[q] for q in decisions], dtype=np.uint8)
