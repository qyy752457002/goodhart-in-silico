"""RQ2 -- Goodhart: how much does optimising against a simulator overstate welfare, how does the
gap grow with optimisation pressure, and which group bears it?

Outputs:
  pressure_curve.csv   Fig. 2: simulated and true welfare vs log2 N, per simulator and objective
  hamming_curve.csv    auxiliary axis: exact optimum within Hamming radius r of the ED
  plugin_summary.csv   Fig. 3: the plug-in design at full pressure -- overestimate, group
                       decomposition delta_g, illusory fairness, true worst group
  curve_summary.csv    normalised drop height and the fitted d(alpha - beta d) form

Methodology map (03_methodology.md): section 3 and Figs. 2-3 of section 7.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from .. import goodhart, utils
from ..optimize import PackageSpace
from . import common

log = logging.getLogger(__name__)


def run(cfg, out: str = None) -> dict:
    tb = common.load_testbed(cfg)
    rq = cfg["rq2"]
    out_dir = utils.results_dir(cfg, "rq2", out)
    seed = int(cfg["project"]["seed"])
    objs = common.objectives(cfg)
    sims = common.resolve_simulators(cfg, tb, rq["simulators"], int(rq["cv_folds"]))
    space = PackageSpace(tb.prob.n_dec)
    log2_n = [int(j) for j in rq["log2_n"] if int(j) <= tb.prob.n_dec]

    curves, hamming, plugin, summary = [], [], [], []
    for sim_id, sim in sims.items():
        engine = common.make_engine(cfg, tb, sim, space)
        for obj in objs:
            curve = goodhart.pressure_curve(engine, obj, log2_n, int(rq["reps"]), utils.rep_rng(seed, 3, sim_id, obj))
            curves.append(curve.assign(simulator=sim_id))
            hamming.append(goodhart.hamming_curve(engine, obj).assign(simulator=sim_id))
            plugin.append({"simulator": sim_id, **goodhart.plugin_summary(engine, obj)})
            summary.append({"simulator": sim_id, **goodhart.curve_summary(engine, curve)})
            log.info("rq2: %s / %s overestimate at full pressure = %+.3f, regret = %.3f", sim_id, obj,
                     plugin[-1]["overestimate"], plugin[-1]["regret"])

    tables = {"pressure_curve": pd.concat(curves, ignore_index=True),
              "hamming_curve": pd.concat(hamming, ignore_index=True),
              "plugin_summary": pd.DataFrame(plugin), "curve_summary": pd.DataFrame(summary)}
    for name, table in tables.items():
        utils.save_table(table, os.path.join(out_dir, f"{name}.csv"))
    utils.write_manifest(out_dir, cfg, {"experiment": "rq2", "simulators": list(sims)})
    return tables
