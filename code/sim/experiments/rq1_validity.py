"""RQ1 -- simulator validity: how accurately do simulated stances reproduce real ones, and for whom
are they wrong?

Outputs (Table 1 and its diagnostics):
  tab1_simulators.csv   per simulator: accuracy, macro-F1, Brier, ECE, bias (+ bootstrap CIs)
  by_group.csv          the same per stakeholder group, plus the signed bias b_g
  bias_matrix.csv       group x proposal signed support error (the key diagnostic)
  coef_error.csv        error of the simulated welfare coefficients d_gq
  decision_centric.csv  plug-in package per objective: selected-package error, regret
  contrasts.csv         leakage controls: named vs anonymous, early vs late cut-off
  rank_agreement.csv    does ranking simulators by accuracy agree with ranking them by regret?

Methodology map (03_methodology.md): section 2.4 and Table 1 of section 7.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from .. import metrics, stats, utils
from . import common

log = logging.getLogger(__name__)

META_KEYS = ("kind", "model", "variant", "cutoff_group", "parse_failure_rate", "fallback_rate", "cross_fitted")


def run(cfg, out: str = None) -> dict:
    tb = common.load_testbed(cfg)
    prob = tb.prob
    rq = cfg["rq1"]
    out_dir = utils.results_dir(cfg, "rq1", out)
    seed = int(cfg["project"]["seed"])
    objs = common.objectives(cfg)
    sims = common.resolve_simulators(cfg, tb, rq["simulators"], int(rq["cv_folds"]))
    references = {k: v for k, v in metrics.reference_simulators(prob).items() if k not in sims}
    everything = {**sims, **references}

    overall, by_group, bias, coef, decision = [], [], [], [], []
    for sim_id, sim in everything.items():
        row = {"simulator": sim_id, "is_reference": sim_id in references}
        row.update({k: sim.meta[k] for k in META_KEYS if k in sim.meta})
        row.update(metrics.overall_metrics(prob, sim, int(rq["n_boot"]), utils.rep_rng(seed, 1, sim_id)))
        overall.append(row)
        by_group.append(metrics.group_metrics(prob, sim).assign(simulator=sim_id))
        bias.append(metrics.bias_matrix(prob, sim).assign(simulator=sim_id))
        engine = common.make_engine(cfg, tb, sim)
        ce = metrics.coefficient_error(engine).rename_axis("group").reset_index()
        coef.append(ce.melt(id_vars="group", var_name="decision", value_name="coef_error").assign(simulator=sim_id))
        decision.append(metrics.decision_metrics(engine, objs).assign(simulator=sim_id))
        log.info("rq1: %s accuracy=%.3f brier=%.3f", sim_id, row["accuracy"], row["brier"])

    tab1 = pd.DataFrame(overall)
    decision_df = pd.concat(decision, ignore_index=True)
    regret = decision_df.pivot(index="simulator", columns="objective", values="regret")
    tab1 = tab1.merge(regret.add_prefix("regret_").reset_index(), on="simulator", how="left")

    contrasts = []
    for a, b, label in rq.get("contrasts") or []:
        if a in sims and b in sims:
            contrasts.append({"contrast": label, **metrics.paired_accuracy_contrast(
                prob, sims[a], sims[b], int(rq["n_boot"]), utils.rep_rng(seed, 2, a, b))})
        else:
            log.warning("rq1: contrast %r skipped (needs %s and %s)", label, a, b)

    # "Mean accuracy ranking" versus "decision regret ranking" across the non-reference simulators
    agreement = []
    real = tab1[~tab1["is_reference"]]
    for obj in objs:
        for metric in ("accuracy", "brier", "sigma_mae"):
            agreement.append({"objective": obj, "metric": metric,
                              **stats.spearman(real[metric], real[f"regret_{obj}"])})

    tables = {"tab1_simulators": tab1, "by_group": pd.concat(by_group, ignore_index=True),
              "bias_matrix": pd.concat(bias, ignore_index=True), "coef_error": pd.concat(coef, ignore_index=True),
              "decision_centric": decision_df,
              "contrasts": pd.DataFrame(contrasts), "rank_agreement": pd.DataFrame(agreement)}
    for name, table in tables.items():
        utils.save_table(table, os.path.join(out_dir, f"{name}.csv"))
    utils.write_manifest(out_dir, cfg, {"experiment": "rq1", "simulators": list(sims),
                                        "references": list(references)})
    return tables
