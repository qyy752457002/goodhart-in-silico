"""Describe a testbed: data statistics (Table 0) and precondition P1 -- do stakeholder groups
disagree enough that the egalitarian and utilitarian optima differ?

Methodology map (03_methodology.md): section 12 (P1) and section 6.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from .. import utils
from ..data import STANCES
from ..simulators import synthetic
from . import common

log = logging.getLogger(__name__)


def support_tables(prob) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean true support and number of responders per (decision, group)."""
    cell = pd.DataFrame({"decision": np.asarray(prob.decisions)[prob.cell_dec],
                         "group": np.asarray(prob.group_names)[prob.g_index[prob.cell_org]],
                         "sigma": prob.sigma})
    support = cell.pivot_table(index="decision", columns="group", values="sigma", aggfunc="mean")
    support = support.reindex(columns=prob.group_names)
    counts = cell.pivot_table(index="decision", columns="group", values="sigma", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=prob.group_names, fill_value=0)
    support["all"] = cell.groupby("decision")["sigma"].mean()
    support["spread"] = support[prob.group_names].max(1) - support[prob.group_names].min(1)
    return support, counts


def stance_distribution(prob) -> pd.DataFrame:
    """Share of each stance class per group."""
    cell_group = prob.g_index[prob.cell_org]
    rows = []
    for g, name in enumerate(prob.group_names):
        freq = np.bincount(prob.stance_idx[cell_group == g], minlength=len(STANCES))
        rows.append({"group": name, "n_cells": int(freq.sum()),
                     **{s: freq[k] / max(1, freq.sum()) for k, s in enumerate(STANCES)}})
    return pd.DataFrame(rows)


def package_table(engine, objectives: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """True welfare of the reference packages and of each objective's optimum, plus the Hamming
    distances between the optima."""
    e, prob = engine, engine.prob
    named = {"ED (retain all)": e.space.ed_index, "IFRS baseline (revise all)": e.space.ifrs_index}
    if e.board_index is not None:
        named["Board's decision"] = e.board_index
    optima = {obj: e.oracle_index(obj) for obj in objectives}
    named.update({f"optimum: {obj}": k for obj, k in optima.items()})
    rows = []
    for label, k in named.items():
        a = e.space.package(k)
        row = {"package": label,
               "revised": ",".join(d for d, x in zip(prob.decision_labels(), a) if x == 0) or "-",
               "n_revised": int((a == 0).sum())}
        row.update({g: float(v) for g, v in zip(prob.group_names, e.W_true[k])})
        row.update({f"W_{obj}": float(e.true_values(obj)[k]) for obj in objectives})
        row["worst_group"] = prob.group_names[int(np.argmin(e.W_true[k]))]
        rows.append(row)
    ham = [{"a": a, "b": b, "hamming": e.space.hamming_between(optima[a], optima[b])}
           for i, a in enumerate(objectives) for b in objectives[i + 1:]]
    return pd.DataFrame(rows), pd.DataFrame(ham, columns=["a", "b", "hamming"])


def run(cfg, out: str = None) -> dict:
    tb = common.load_testbed(cfg)
    prob = tb.prob
    out_dir = utils.results_dir(cfg, "describe", out)
    engine = common.make_engine(cfg, tb, synthetic.oracle(prob))
    objs = common.objectives(cfg)
    overview = pd.DataFrame([{**common.describe_problem(prob), **tb.meta}])
    support, counts = support_tables(prob)
    packages, hamming = package_table(engine, objs)
    tables = {"overview": overview, "support_by_group": support.reset_index(),
              "responders_by_group": counts.reset_index(), "stance_distribution": stance_distribution(prob),
              "packages": packages, "optima_hamming": hamming}
    for name, table in tables.items():
        utils.save_table(table, os.path.join(out_dir, f"{name}.csv"))
    utils.write_manifest(out_dir, cfg, {"experiment": "describe"})
    log.info("describe: %d organisations, %d cells, %d decisions -> %s", prob.n_org, prob.n_cells, prob.n_dec,
             out_dir)
    return tables
