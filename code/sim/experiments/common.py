"""Shared set-up for the experiments: testbed, engine settings and simulators from a configuration."""
from __future__ import annotations

import logging
from dataclasses import fields

import numpy as np
import yaml

from .. import config as config_mod
from ..data import make_sigma_map
from ..engine import DesignEngine, EngineSettings
from ..optimize import PackageSpace
from ..simulators import registry
from ..simulators.persona import PersonaSpec
from ..testbeds.common import ProblemPromptSource, Testbed

log = logging.getLogger(__name__)


def load_testbed(cfg) -> Testbed:
    """The testbed named in `testbed.name`."""
    name = cfg["testbed"]["name"]
    welfare_cfg = cfg["welfare"]
    sigma_map = make_sigma_map(welfare_cfg.get("sigma_map", "graded"), welfare_cfg.get("qualified_weight", 0.75))
    if name == "aasb":
        from ..testbeds.aasb import load_aasb
        return load_aasb(cfg["testbed"]["aasb"], welfare_cfg, cfg["_root"])
    if name == "opinionqa":
        from ..testbeds.opinionqa import load_opinionqa
        spec = dict(cfg["testbed"]["opinionqa"])
        if spec.get("spec_file"):
            with open(config_mod.resolve_path(cfg, spec["spec_file"]), encoding="utf-8") as f:
                spec = {**yaml.safe_load(f), **{k: v for k, v in spec.items() if k != "spec_file"}}
        return load_opinionqa(spec, sigma_map, cfg["_root"])
    if name == "synthetic":
        from ..testbeds.synthetic import make_synthetic_problem
        prob = make_synthetic_problem(sigma_map=sigma_map, **cfg["testbed"]["synthetic"])
        return Testbed(name="synthetic", prob=prob, source=ProblemPromptSource(prob))
    if name == "toy":
        from ..testbeds.toy import toy_problem
        prob = toy_problem(int(cfg["testbed"]["toy"].get("n_per_group", 4)))
        return Testbed(name="toy", prob=prob, source=ProblemPromptSource(prob))
    raise ValueError(f"unknown testbed {name!r} (use 'aasb', 'opinionqa', 'synthetic' or 'toy')")


def engine_settings(cfg, **overrides) -> EngineSettings:
    """`EngineSettings` from the `guard` and `welfare` sections."""
    known = {f.name for f in fields(EngineSettings)}
    values = {k: v for k, v in cfg["guard"].items() if k in known}
    unknown = set(cfg["guard"]) - known
    if unknown:
        raise KeyError(f"unknown keys in the `guard` section: {sorted(unknown)}")
    values["group_weights"] = cfg["welfare"].get("group_weights", "equal")
    values["hedgetune_log2n"] = tuple(values.get("hedgetune_log2n") or ())
    values.update(overrides)
    return EngineSettings(**values)


def make_engine(cfg, tb: Testbed, sim, space: PackageSpace = None, settings: EngineSettings = None) -> DesignEngine:
    return DesignEngine(tb.prob, sim, settings or engine_settings(cfg), space, tb.board)


def simulated_dir(cfg) -> str:
    return config_mod.resolve_path(cfg, cfg["paths"]["simulated_dir"])


def resolve_simulators(cfg, tb: Testbed, sim_ids: list, cv_folds: int = 5) -> dict:
    """{id: SimulatorOutput} for RQ1/RQ2. Fitted simulators are cross-fitted, on folds that depend
    on the project seed and the simulator's id only (not on `sim_ids`). A persona simulator that has
    not been precomputed raises, or is skipped when `on_missing_simulator: skip`."""
    out = {}
    for sim_id in sim_ids:
        try:
            out[sim_id] = registry.resolve(sim_id, cfg["simulators"], tb.prob, simulated_dir(cfg), tb.name,
                                           context=tb.context, cv_folds=cv_folds,
                                           seed=int(cfg["project"]["seed"]))
        except registry.MissingSimulatorOutput as err:
            if cfg.get("on_missing_simulator", "error") != "skip":
                raise
            log.warning("skipping simulator %s: %s", sim_id, err)
        except ValueError as err:
            # an ensemble whose members are missing surfaces as MissingSimulatorOutput above;
            # anything else is a configuration error and must not be hidden
            raise ValueError(f"simulator {sim_id!r}: {err}") from err
    if not out:
        raise RuntimeError("no simulator is available. Precompute the persona simulators with "
                           "`python -m sim simulate --simulator <id>` or choose others in the configuration.")
    return out


def persona_spec(cfg, sim_spec: dict, model_cfg: dict) -> PersonaSpec:
    """Persona settings: `llm.persona` defaults <- the simulator's own entries <- the model's
    `max_tokens` (reasoning models need more room)."""
    known = {f.name for f in fields(PersonaSpec)}
    values = {k: v for k, v in cfg["llm"]["persona"].items() if k in known}
    values.update({k: v for k, v in sim_spec.items() if k in known})
    if "max_tokens" in model_cfg:
        values["max_tokens"] = int(model_cfg["max_tokens"])
    return PersonaSpec(**values)


def objectives(cfg) -> list:
    return list(cfg["welfare"]["objectives"])


def describe_problem(prob) -> dict:
    cells = prob.cells_per_org
    return {"testbed": prob.name, "n_orgs": prob.n_org, "n_cells": prob.n_cells, "n_decisions": prob.n_dec,
            "n_groups": prob.n_groups, "cells_per_org_min": int(cells.min()),
            "cells_per_org_median": float(np.median(cells)), "cells_per_org_max": int(cells.max()),
            **{f"n[{g}]": int(n) for g, n in zip(prob.group_names, prob.group_sizes)}}
