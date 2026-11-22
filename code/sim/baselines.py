"""Baseline decision rules (methodology section 5).

  ED / IFRS / oracle / Board      fixed packages (handled in `methods.py`)
  simulator-only plug-in          `DesignEngine.plugin_index`
  best-of-N                       `best_of_n_select`
  pilot-only                      `pilot_only_system`
  pilot + simulator plug-in       `pilot_sim_system`
  Ensemble-LCB / worst case       `ensemble_tables`, `pilot_ensemble_lcb_table`   (Coste et al., 2024)
  HedgeTune                       `hedgetune_select`                              (Khalaf et al., 2025)
"""
from __future__ import annotations

import numpy as np

from . import welfare
from .engine import DesignEngine
from .pilot import Pilot
from .welfare import Affine


def pilot_sim_system(engine: DesignEngine, pilot: Pilot) -> Affine:
    """Pilot + simulator plug-in: true utility for known organisations, simulated utility for the
    rest, no rectification (same information as GUARD)."""
    e = engine
    total = e.group_sum(pilot.known, e.true_org) + e.group_sum(pilot.target, e.sim_org)
    return total.scaled(1.0 / e.sizes)


def pilot_only_system(engine: DesignEngine, pilot: Pilot) -> Affine:
    """Pilot only: extrapolate each group's welfare from the mean utility of its pilot organisations."""
    e = engine
    n_known = e.group_count(pilot.known)
    if (n_known == 0).any():
        raise ValueError("pilot-only needs at least one pilot organisation in every group")
    return e.group_sum(pilot.known, e.true_org).scaled(1.0 / n_known)


# ----------------------------------------------------------------------------------------------
# Ensemble pessimism (no validity guarantee)
# ----------------------------------------------------------------------------------------------

def _mean_sd_min(tables) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, sample standard deviation and minimum over a sequence of equally shaped arrays."""
    s1 = s2 = low = None
    k = 0
    for t in tables:
        if s1 is None:
            s1, s2, low = t.copy(), t * t, t.copy()
        else:
            s1 += t
            s2 += t * t
            np.minimum(low, t, out=low)
        k += 1
    if k == 0:
        raise ValueError("the simulator has no ensemble members")
    mean = s1 / k
    var = np.clip((s2 - k * mean * mean) / (k - 1), 0.0, None) if k > 1 else np.zeros_like(mean)
    return mean, np.sqrt(var), low


def ensemble_tables(engine: DesignEngine) -> dict:
    """Simulator-only ensemble tables, (M, G) each:
    'lcb'  mean_k W^k_g(a) - kappa * sd_k W^k_g(a)   (uncertainty-weighted optimisation),
    'wco'  min_k W^k_g(a)                            (worst-case optimisation),
    'mean' mean_k W^k_g(a).
    Cached on the engine: they do not depend on the pilot."""
    e = engine
    if "ensemble_tables" not in e._cache:
        mean, sd, low = _mean_sd_min(welfare.group_mean(m, e.gi, e.G).table() for m in e.member_org)
        e._cache["ensemble_tables"] = {"lcb": mean - e.settings.ens_kappa * sd, "wco": low, "mean": mean}
    return e._cache["ensemble_tables"]


def pilot_ensemble_lcb_table(engine: DesignEngine, pilot: Pilot) -> np.ndarray:
    """Ensemble-LCB given the same information as GUARD: known organisations enter with their true
    utility, and the ensemble mean minus kappa standard deviations is used for the targets only."""
    e = engine
    known = e.group_sum(pilot.known, e.true_org).scaled(1.0 / e.sizes).table()
    mean, sd, _ = _mean_sd_min(e.group_sum(pilot.target, m).scaled(1.0 / e.sizes).table()
                               for m in e.member_org)
    return known + mean - e.settings.ens_kappa * sd


# ----------------------------------------------------------------------------------------------
# Optimisation pressure as a tuning knob
# ----------------------------------------------------------------------------------------------

def best_of_n_select(engine: DesignEngine, objective: str, log2_n: int, rng: np.random.Generator) -> int:
    """One best-of-N draw: sample N = 2^log2_n packages uniformly, keep the best simulated one."""
    pmf = engine.best_of_n_pmf(int(log2_n))
    rank = int(rng.choice(len(pmf), p=pmf))
    return int(engine.sim_order(objective)[rank])


def hedgetune_grid(engine: DesignEngine) -> list:
    grid = list(engine.settings.hedgetune_log2n) or list(range(0, engine.Q + 1, 2))
    if engine.Q not in grid:
        grid.append(engine.Q)  # always allow the exact optimum
    return sorted(set(int(j) for j in grid if 0 <= j <= engine.Q))


def hedgetune_select(engine: DesignEngine, objective: str, est_values: np.ndarray,
                     rng: np.random.Generator) -> tuple[int, int]:
    """HedgeTune adapted to package design: choose the optimisation pressure N on the pilot.

    For each candidate N the expected *estimated* true welfare of best-of-N is computed exactly,
    E_N = sum_r P_N(rank = r) * est_values[package of simulated rank r], where `est_values` is a
    pilot-based welfare estimate of every package. The N with the largest E_N is used (ties: the
    smallest N, i.e. the least pressure) and one best-of-N draw is returned.

    Returns (package index, chosen log2 N).
    """
    order = engine.sim_order(objective)
    ranked = est_values[order]
    best_j, best_val = None, -np.inf
    for j in hedgetune_grid(engine):
        pmf = engine.best_of_n_pmf(j)
        val = float(pmf @ ranked[: len(pmf)])
        if val > best_val + 1e-12:
            best_j, best_val = j, val
    return best_of_n_select(engine, objective, best_j, rng), int(best_j)
