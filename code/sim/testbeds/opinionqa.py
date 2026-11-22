"""Second testbed: individual survey respondents (OpinionQA / Pew American Trends Panel).

Mapping (methodology section 6): "organisation" -> respondent, "group" -> a demographic group,
"proposal" -> a policy question of one survey wave that can be read as "adopt or not".

The adapter is driven entirely by a YAML specification (see configs/opinionqa_example.yaml)
because which questions count as proposals, and how answer options translate into support, are
research decisions. It works for any respondent-by-question CSV with categorical answers.

STATUS: written against the documented layout of the OpinionQA release (one folder per wave with
`responses.csv` and `info.csv`) and tested on synthetic files of that shape only. The Pew data need
a free registration and were not available when this module was written: run
`python -m sim opinionqa-inspect --dir <wave folder>` first and check column names and answer
strings against your specification.
"""
from __future__ import annotations

import ast
import os
import re

import numpy as np
import pandas as pd

from ..data import STANCE_SIGMA, STANCES, Problem
from ..simulators import prompts
from .common import ProblemPromptSource, Testbed

CONSULTATION = "A public opinion survey of adults in the United States (Pew Research Center, American Trends Panel)."


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


class SurveyPromptSource(ProblemPromptSource):
    """Personas are demographic profiles; answer options are the survey's own response options."""
    consultation = CONSULTATION

    def __init__(self, prob: Problem, profiles: list, minimal: list, option_table: dict):
        super().__init__(prob)
        self._profiles = profiles     # per respondent: full demographic profile text
        self._minimal = minimal       # per respondent: the grouping attribute only
        self._options = option_table  # qid -> [Option]

    def persona(self, org_index: int, variant: str) -> str:
        """'named' = full demographic profile; 'anonymous' = the grouping attribute only (the
        survey analogue of the rich-vs-minimal persona contrast)."""
        return self._minimal[org_index] if variant == "anonymous" else self._profiles[org_index]

    def options(self, qid: int) -> list:
        return self._options[int(qid)]


def load_opinionqa(spec: dict, sigma_map: dict = None, root: str = ".") -> Testbed:
    """Build the testbed from a specification dictionary.

    Required keys: responses_csv, group_column, groups (raw value -> group name), proposals
    (list of {key, support: {answer: stance}, text?}). Optional: info_csv, respondent_id,
    persona_columns, max_per_group, seed.
    """
    sigma_map = dict(sigma_map or STANCE_SIGMA)

    def path(value):
        return value if os.path.isabs(value) else os.path.join(root, value)

    responses = pd.read_csv(path(spec["responses_csv"]), dtype=str, keep_default_na=False)
    group_col = spec["group_column"]
    proposals = spec["proposals"]
    needed = [group_col] + [p["key"] for p in proposals]
    absent = [c for c in needed if c not in responses.columns]
    if absent:
        raise ValueError(f"columns missing from {spec['responses_csv']}: {absent}")
    for p in proposals:
        bad = sorted(set(p["support"].values()) - set(STANCES))
        if bad:
            raise ValueError(f"proposal {p['key']}: unknown stance labels {bad}; use {STANCES}")

    group_map = {str(k): str(v) for k, v in spec["groups"].items()}
    frame = responses[responses[group_col].isin(group_map)].copy()
    frame["_group"] = frame[group_col].map(group_map)
    id_col = spec.get("respondent_id")
    frame["_id"] = frame[id_col].astype(str) if id_col else [f"R{j:06d}" for j in range(len(frame))]
    answered = np.column_stack([frame[p["key"]].isin(p["support"]).to_numpy() for p in proposals])
    frame = frame[answered.any(1)]
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    max_per_group = spec.get("max_per_group")
    if max_per_group:
        keep = []
        for _, sub in frame.groupby("_group"):
            idx = sub.index.to_numpy()
            keep.extend(rng.permutation(idx)[: int(max_per_group)])
        frame = frame.loc[sorted(keep)]
    frame = frame.sort_values("_id").reset_index(drop=True)
    if frame["_id"].duplicated().any():
        raise ValueError("respondent ids are not unique")

    info = {}
    if spec.get("info_csv"):
        table = pd.read_csv(path(spec["info_csv"]), dtype=str, keep_default_na=False)
        if {"key", "question"} <= set(table.columns):
            info = dict(zip(table["key"], table["question"]))

    decisions = list(range(1, len(proposals) + 1))
    cell_org, cell_dec, stances = [], [], []
    questions, option_table = {}, {}
    for d, p in enumerate(proposals):
        qid = decisions[d]
        questions[qid] = {"question": p.get("text") or info.get(p["key"], p["key"]), "background": ""}
        option_table[qid] = [prompts.Option(key=_slug(answer), text=str(answer), stance=stance)
                             for answer, stance in p["support"].items()]
        hit = frame[p["key"]].isin(p["support"]).to_numpy()
        cell_org.extend(np.flatnonzero(hit))
        cell_dec.extend([d] * int(hit.sum()))
        stances.extend(frame.loc[hit, p["key"]].map(p["support"]))
    stances = np.array(stances, dtype=object)
    cell_dec = np.array(cell_dec, dtype=int)
    groups = frame["_group"].tolist()
    prob = Problem(
        orgs=frame["_id"].tolist(), groups=groups, group_names=sorted(set(groups)), decisions=decisions,
        cell_org=np.array(cell_org, dtype=int), cell_dec=cell_dec, cell_q=cell_dec + 1,
        cell_stance=stances, sigma=np.array([sigma_map[s] for s in stances], dtype=float),
        sigma_map=sigma_map, questions=questions, name="opinionqa",
    )
    prob.validate()

    persona_cols = [c for c in spec.get("persona_columns", []) if c in frame.columns]
    profiles, minimal = [], []
    for rec in frame.to_dict("records"):
        traits = "; ".join(f"{c}: {rec[c]}" for c in persona_cols if str(rec[c]).strip())
        profiles.append(f"a US adult with this profile -- {traits}." if traits else "a US adult.")
        minimal.append(f"a US adult -- {group_col}: {rec[group_col]}.")
    source = SurveyPromptSource(prob, profiles, minimal, option_table)
    return Testbed(name="opinionqa", prob=prob, source=source,
                   meta={"n_respondents": prob.n_org, "proposal_keys": [p["key"] for p in proposals],
                         "group_column": group_col})


def inspect_wave(wave_dir: str, max_options: int = 8) -> pd.DataFrame:
    """List a wave's columns and candidate questions with their answer options, to help write the
    specification. Reads `info.csv` when present, otherwise summarises `responses.csv` itself."""
    responses_path = os.path.join(wave_dir, "responses.csv")
    responses = pd.read_csv(responses_path, dtype=str, keep_default_na=False)
    info_path = os.path.join(wave_dir, "info.csv")
    texts = {}
    if os.path.exists(info_path):
        info = pd.read_csv(info_path, dtype=str, keep_default_na=False)
        if {"key", "question"} <= set(info.columns):
            texts = dict(zip(info["key"], info["question"]))
    rows = []
    for col in responses.columns:
        values = responses[col][responses[col] != ""]
        counts = values.value_counts()
        if 2 <= len(counts) <= max_options:
            rows.append({"column": col, "n_answers": int(len(values)), "n_options": int(len(counts)),
                         "options": " | ".join(f"{k} ({v})" for k, v in counts.items()),
                         "question": texts.get(col, "")})
    return pd.DataFrame(rows, columns=["column", "n_answers", "n_options", "options", "question"])


def parse_list_cell(text: str) -> list:
    """Parse a list stored as a string in `info.csv` (e.g. the `references` column)."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return list(value) if isinstance(value, (list, tuple)) else []
