"""Sensitivity analysis (methodology section 8): rerun the core comparison under alternative
modelling choices and report how the conclusions move.

Each variant is a set of configuration overrides: stance-to-support mapping, grouping scheme, group
weights, base-set rule, non-random pilot designs, ... (see `sensitivity.variants`).

Outputs:
  packages.csv   per variant: the reference packages and each objective's optimum (does P1 hold?)
  summary.csv    per variant x simulator x rho x objective x method: welfare, regret, coverage
  coverage.csv   per variant and certificate: coverage, to quantify what a biased pilot costs
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from .. import config as config_mod
from .. import utils
from ..simulators import registry, synthetic
from . import common, describe, rq3_guard

log = logging.getLogger(__name__)


def run_variant(cfg, variant: dict, sens: dict, pool=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """One variant: (its package table, its raw runs or None if no simulator could be run)."""
    name = variant["name"]
    vcfg = config_mod.with_overrides(cfg, variant.get("set") or {})
    tb = common.load_testbed(vcfg)
    objs = common.objectives(vcfg)
    truth = common.make_engine(vcfg, tb, synthetic.oracle(tb.prob))
    pk, _ = describe.package_table(truth, objs)
    # group columns differ between grouping schemes: keep the comparable ones
    keep = ["package", "revised", "n_revised", "worst_group"] + [f"W_{o}" for o in objs]
    packages = pk[keep].assign(variant=name, n_groups=tb.prob.n_groups)
    rows = []
    for sim_id in sens["simulators"]:
        try:
            r, _ = rq3_guard.run_pilots(vcfg, tb, sim_id, rhos=sens["rhos"], reps=int(sens["reps"]),
                                        methods=sens["methods"], certificates=vcfg["rq3"]["certificates"],
                                        seed_keys=(17,), pilot_cfg=vcfg["rq3"]["pilot"],
                                        fit_fraction=float(vcfg["rq3"]["fit_fraction"]), pool=pool)
        except registry.MissingSimulatorOutput as err:
            if cfg.get("on_missing_simulator", "error") != "skip":
                raise
            log.warning("sensitivity: skipping %s in variant %s: %s", sim_id, name, err)
            continue
        rows.extend(r)
    if not rows:
        return packages, None
    runs = pd.DataFrame(rows)
    # per-group columns are variant-specific; drop them so that variants stack
    return packages, runs[[c for c in runs.columns if "[" not in c]]


def run(cfg, out: str = None) -> dict:
    sens = cfg["sensitivity"]
    out_dir = utils.results_dir(cfg, "sensitivity", out)
    seed = int(cfg["project"]["seed"])
    keys = ["simulator", "rho", "objective", "method"]
    packages, summaries, coverages = [], [], []
    with rq3_guard.pilot_pool(cfg) as pool:
        for variant in sens["variants"]:
            name = variant["name"]
            pk, runs = run_variant(cfg, variant, sens, pool)
            packages.append(pk)
            if runs is None:
                continue
            vcfg = config_mod.with_overrides(cfg, variant.get("set") or {})
            summaries.append(rq3_guard.summarise_methods(runs, keys, int(vcfg["rq3"]["n_boot"]), seed)
                             .assign(variant=name))
            coverages.append(rq3_guard.coverage_table(runs, keys, float(vcfg["guard"]["delta"]))
                             .assign(variant=name))
            log.info("sensitivity: variant %s done", name)
    tables = {"packages": pd.concat(packages, ignore_index=True)}
    if summaries:
        tables["summary"] = pd.concat(summaries, ignore_index=True)
        tables["coverage"] = pd.concat(coverages, ignore_index=True)
    for name, table in tables.items():
        utils.save_table(table, os.path.join(out_dir, f"{name}.csv"))
    utils.write_manifest(out_dir, cfg, {"experiment": "sensitivity",
                                        "variants": [v["name"] for v in sens["variants"]]})
    return tables
