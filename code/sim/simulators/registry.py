"""Resolve simulator ids from the configuration (`simulators.<id>`) into `SimulatorOutput`s.

Kinds:
  persona      LLM persona simulator; must be precomputed with `python -m sim simulate` and is then
               loaded from `<paths.simulated_dir>/<testbed>/<id>.parquet`.
  ensemble     mean of other static simulators (`members: [ids]`).
  uniform      uniform guessing.
  oracle       the true stances (reference point).
  synthetic    controlled error injection (`mode: iid | structured | optimistic`, plus parameters).
  type_prior, relational
               fitted on observed stances (see `fitted.py`): cross-fitted for RQ1/RQ2, refitted on
               each pilot for RQ3.
"""
from __future__ import annotations

import os

import numpy as np

from .. import utils
from . import fitted, synthetic
from .base import SimulatorOutput, combine, load_output

STATIC_KINDS = ("persona", "ensemble", "uniform", "oracle", "synthetic")


class MissingSimulatorOutput(FileNotFoundError):
    """A persona simulator has not been precomputed yet."""


def simulated_path(simulated_dir: str, testbed: str, sim_id: str) -> str:
    return os.path.join(simulated_dir, testbed, f"{sim_id}.parquet")


def is_fitted(spec: dict) -> bool:
    return spec.get("kind") in fitted.FITTED_KINDS


def get_spec(simulators_cfg: dict, sim_id: str) -> dict:
    if sim_id not in simulators_cfg:
        raise KeyError(f"simulator {sim_id!r} is not defined under `simulators` in the configuration; "
                       f"defined: {sorted(simulators_cfg)}")
    return dict(simulators_cfg[sim_id])


def synthetic_from_spec(spec: dict, prob, name: str) -> SimulatorOutput:
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    mode = spec.get("mode", "iid")
    n_members = int(spec.get("n_members", 0))
    if mode == "iid":
        out = synthetic.iid_noise(prob, float(spec.get("e", 0.1)), rng, n_members)
    elif mode in ("structured", "optimistic"):
        group = spec.get("group", 0)
        g = prob.group_names.index(group) if isinstance(group, str) else int(group)
        if mode == "structured":
            out = synthetic.structured_misreading(prob, g, float(spec.get("e_s", 0.9)), rng,
                                                  k=int(spec.get("k", 4)), decisions=spec.get("decisions"),
                                                  e_bg=float(spec.get("e_bg", 0.1)), n_members=n_members)
        else:
            out = synthetic.optimistic_bias(prob, g, float(spec.get("e", 0.3)), rng, n_members)
    else:
        raise ValueError(f"unknown synthetic mode {mode!r}")
    out.name = name
    return out


def load_static(sim_id: str, simulators_cfg: dict, prob, simulated_dir: str, testbed: str,
                _stack: tuple = ()) -> SimulatorOutput:
    """A simulator that does not depend on the pilot."""
    if sim_id in _stack:
        raise ValueError(f"circular ensemble definition: {' -> '.join(_stack + (sim_id,))}")
    spec = get_spec(simulators_cfg, sim_id)
    kind = spec.get("kind")
    if kind == "persona":
        path = simulated_path(simulated_dir, testbed, sim_id)
        if not os.path.exists(path):
            raise MissingSimulatorOutput(
                f"no precomputed output for simulator {sim_id!r} at {path}. "
                f"Run: python -m sim simulate --simulator {sim_id}")
        out = load_output(path, prob, name=sim_id)
    elif kind == "ensemble":
        parts = [load_static(m, simulators_cfg, prob, simulated_dir, testbed, _stack + (sim_id,))
                 for m in spec["members"]]
        out = combine(parts, name=sim_id, weights=spec.get("weights"), members=spec.get("pool", "components"))
    elif kind == "uniform":
        out = SimulatorOutput.uniform(prob.n_cells, sim_id)
    elif kind == "oracle":
        out = synthetic.oracle(prob)
        out.name = sim_id
    elif kind == "synthetic":
        out = synthetic_from_spec(spec, prob, sim_id)
    elif kind in fitted.FITTED_KINDS:
        raise ValueError(f"simulator {sim_id!r} is fitted on stances; use cross_fit or a pilot")
    else:
        raise ValueError(f"simulator {sim_id!r}: unknown kind {kind!r}")
    out.meta.setdefault("id", sim_id)
    for key in ("model", "variant", "cutoff_group"):
        if key in spec:
            out.meta.setdefault(key, spec[key])
    return out


def resolve(sim_id: str, simulators_cfg: dict, prob, simulated_dir: str, testbed: str, *,
            context: np.ndarray = None, cv_folds: int = 5, seed: int = 0) -> SimulatorOutput:
    """A full-problem simulator output for RQ1/RQ2: static simulators are loaded or built, fitted
    ones are cross-fitted so that no organisation is predicted from its own stances.

    The folds depend only on (`seed`, `sim_id`): a simulator has the same cross-fitted output whichever
    experiment asks for it and whatever else is in that experiment's simulator list."""
    spec = get_spec(simulators_cfg, sim_id)
    if is_fitted(spec):
        out = fitted.cross_fit(spec, prob, cv_folds, utils.rep_rng(seed, 23, sim_id), context, name=sim_id)
        out.meta.setdefault("id", sim_id)
        return out
    return load_static(sim_id, simulators_cfg, prob, simulated_dir, testbed)
