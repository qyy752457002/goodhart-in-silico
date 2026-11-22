"""The two-group, two-proposal example behind Figure 1 (02_gaps_and_contributions.md section 3).

Writes the package table and what each rule selects, so the figure and the text can be regenerated
from code rather than typed by hand.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .. import guard, utils
from ..engine import DesignEngine, EngineSettings
from ..pilot import draw_pilot
from ..testbeds.toy import toy_problem, toy_simulator


def run(cfg=None, out: str = None, n_per_group: int = 4, rho: float = 0.5, seed: int = 0) -> dict:
    prob = toy_problem(n_per_group)
    sim = toy_simulator(prob)
    engine = DesignEngine(prob, sim, EngineSettings())
    space = engine.space
    rows = []
    for a in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        k = space.index(a)
        rows.append({"package": str(a), "true_W_A": engine.W_true[k, 0], "true_W_B": engine.W_true[k, 1],
                     "true_min": engine.W_true[k].min(), "true_mean": engine.W_true[k].mean(),
                     "sim_W_A": engine.W_sim[k, 0], "sim_W_B": engine.W_sim[k, 1],
                     "sim_min": engine.W_sim[k].min(), "sim_mean": engine.W_sim[k].mean()})
    table = pd.DataFrame(rows)

    pilot = draw_pilot(prob.g_index, prob.n_groups, rho, np.random.default_rng(seed))
    choices = []
    for objective in ("maxmin", "utilitarian"):
        plug = engine.evaluate(engine.plugin_index(objective), objective, engine.W_sim)
        res = guard.run_guard(engine, pilot, objective)
        fixed = engine.evaluate(res.index, objective)
        for rule, m, extra in (("simulator plug-in", plug, {}),
                               ("GUARD", fixed, {"certificate": res.certificate, "estimate": res.estimate})):
            choices.append({"objective": objective, "rule": rule,
                            "package": str(tuple(int(x) for x in space.package(m["package"]))),
                            "true_welfare": m["true_welfare"], "regret": m["regret"],
                            "simulated_welfare": m.get("est_welfare"), "overestimate": m.get("overestimate"),
                            "illusory_fairness": m.get("illusory_fairness"),
                            "true_worst_group": m["true_worst_group"], **extra})
    selections = pd.DataFrame(choices)
    if cfg is not None or out is not None:
        out_dir = out or os.path.join(cfg["_root"], cfg["paths"]["results_dir"], "toy")
        os.makedirs(out_dir, exist_ok=True)
        utils.save_table(table, os.path.join(out_dir, "fig1_packages.csv"))
        utils.save_table(selections, os.path.join(out_dir, "fig1_selections.csv"))
    return {"packages": table, "selections": selections}
