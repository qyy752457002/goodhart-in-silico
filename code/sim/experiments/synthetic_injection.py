"""Synthetic error injection: controlled evidence for Propositions A-C (Fig. 5).

Simulators are built from the true stances with a known error mechanism, then pushed through the
same pilot protocol as RQ3.
  iid          independent label noise at rate e -- expected to produce no Goodhart gap;
  structured   one group read as "agree" on the k proposals it opposes most, with strength e_s --
               expected to produce illusory fairness that pessimism alone does not repair
               (Proposition C), that ensembles sharing the bias do not reveal, and that pooled
               calibration fails to certify (Proposition B).

Outputs:
  runs.parquet, pilots.parquet   raw replications, tagged with the scenario parameters
  summary.csv                    per scenario x parameters x rho x objective x method
  coverage.csv                   per certificate (compare `pooled` with the group-conditional ones)
  misread_is_worst.csv           how often the true worst group is the misread group

Methodology map (03_methodology.md): sections 3 ("synthetic injection"), 12 and Fig. 5.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from .. import utils
from ..optimize import PackageSpace
from ..simulators import synthetic as synth
from . import common, rq3_guard

log = logging.getLogger(__name__)


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def expand_scenarios(scenarios: list, prob) -> list:
    """Cartesian product of each scenario's parameter lists. Returns dicts with the keys
    scenario, mode, e, e_s, e_bg, k, group (index or None), group_name."""
    out = []
    for sc in scenarios:
        mode = sc.get("mode", "iid")
        if mode == "iid":
            for e in _as_list(sc.get("e", 0.1)):
                out.append({"scenario": sc["name"], "mode": mode, "e": float(e), "e_s": np.nan, "e_bg": np.nan,
                            "k": np.nan, "group": None, "group_name": "-"})
        elif mode in ("structured", "optimistic"):
            groups = sc.get("groups", "all")
            idx = (range(prob.n_groups) if groups == "all" else
                   [prob.group_names.index(g) if isinstance(g, str) else int(g) for g in _as_list(groups)])
            for g in idx:
                if mode == "structured":
                    for e_s in _as_list(sc.get("e_s", 0.9)):
                        out.append({"scenario": sc["name"], "mode": mode, "e": np.nan, "e_s": float(e_s),
                                    "e_bg": float(sc.get("e_bg", 0.1)), "k": int(sc.get("k", 4)), "group": int(g),
                                    "group_name": prob.group_names[g]})
                else:
                    for e in _as_list(sc.get("e", 0.3)):
                        out.append({"scenario": sc["name"], "mode": mode, "e": float(e), "e_s": np.nan,
                                    "e_bg": np.nan, "k": np.nan, "group": int(g),
                                    "group_name": prob.group_names[g]})
        else:
            raise ValueError(f"scenario {sc.get('name')!r}: unknown mode {mode!r}")
    return out


def build_simulator(prob, params: dict, rng: np.random.Generator, n_members: int):
    if params["mode"] == "iid":
        return synth.iid_noise(prob, params["e"], rng, n_members)
    if params["mode"] == "structured":
        return synth.structured_misreading(prob, params["group"], params["e_s"], rng, k=int(params["k"]),
                                           e_bg=params["e_bg"], n_members=n_members)
    return synth.optimistic_bias(prob, params["group"], params["e"], rng, n_members)


def run(cfg, out: str = None) -> dict:
    tb = common.load_testbed(cfg)
    prob = tb.prob
    sc_cfg = cfg["synthetic"]
    out_dir = utils.results_dir(cfg, "synthetic", out)
    seed = int(cfg["project"]["seed"])
    space = PackageSpace(prob.n_dec)
    grid = expand_scenarios(sc_cfg["scenarios"], prob)
    tag_cols = ["scenario", "mode", "group_name", "e", "e_s", "e_bg", "k", "sim_seed"]
    rows, pilot_rows = [], []
    with rq3_guard.pilot_pool(cfg) as pool:
        for c, params in enumerate(grid):
            timer = utils.Timer()
            for s in range(int(sc_cfg["n_sim_seeds"])):
                sim = build_simulator(prob, params, utils.rep_rng(seed, 11, c, s), int(sc_cfg["n_members"]))
                tags = {k: params[k] for k in ("scenario", "mode", "group_name", "e", "e_s", "e_bg", "k")}
                tags["sim_seed"] = s
                r, p = rq3_guard.run_pilots(cfg, tb, "synthetic", rhos=sc_cfg["rhos"], reps=int(sc_cfg["reps"]),
                                            methods=sc_cfg["methods"], certificates=sc_cfg["certificates"],
                                            static_sim=sim, space=space, seed_keys=(13, s), tags=tags, pool=pool)
                for row in r:
                    # is the true worst group the one that was misread? (hypothesis H2)
                    row["worst_is_misread"] = (row["true_worst_group"] == params["group_name"]
                                               if params["group"] is not None else np.nan)
                rows.extend(r)
                pilot_rows.extend(p)
            log.info("synthetic: %s group=%s e=%s e_s=%s done (%.0fs)", params["scenario"], params["group_name"],
                     params["e"], params["e_s"], timer.seconds())
    runs, pilots = pd.DataFrame(rows), pd.DataFrame(pilot_rows)
    keys = ["scenario", "group_name", "e", "e_s", "rho", "objective", "method"]
    summary = rq3_guard.summarise_methods(runs, keys, n_boot=int(cfg["rq3"]["n_boot"]), seed=seed)
    misread = (runs.dropna(subset=["worst_is_misread"])
               .assign(worst_is_misread=lambda d: d["worst_is_misread"].astype(float))
               .groupby(keys, sort=False, dropna=False)["worst_is_misread"].mean().reset_index())
    tables = {
        "summary": summary,
        "coverage": rq3_guard.coverage_table(runs, keys, float(cfg["guard"]["delta"])),
        "misread_is_worst": misread,
        "uniform_validity": rq3_guard.uniform_validity_table(pilots, ["scenario", "group_name", "e", "e_s", "rho"]),
    }
    utils.save_table(runs[tag_cols + [c for c in runs.columns if c not in tag_cols]],
                     os.path.join(out_dir, "runs.parquet"))
    utils.save_table(pilots, os.path.join(out_dir, "pilots.parquet"))
    for name, table in tables.items():
        utils.save_table(table, os.path.join(out_dir, f"{name}.csv"))
    utils.write_manifest(out_dir, cfg, {"experiment": "synthetic", "n_cells_of_grid": len(grid)})
    return {"runs": runs, "pilots": pilots, **tables}
