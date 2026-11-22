"""Pilot designs: which organisations have their true stances observed.

Methodology map (03_methodology.md): section 4.2 step 1 (simple random sampling without
replacement within each group) and section 8 (non-random pilots as a threat to validity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Pilot:
    """A pilot sample.

    known        (n_org,) bool: true stances observed.
    fit          (n_org,) bool, subset of `known`: used to fit a pilot-dependent simulator
                 (type prior R0, relational model R3). All False for pilot-independent simulators.
    calib_order  per group, the remaining known organisations (known & ~fit) in sampling order.
                 They calibrate the rectification and the certificate.
    """
    known: np.ndarray
    fit: np.ndarray
    calib_order: tuple

    @property
    def calib(self) -> np.ndarray:
        mask = np.zeros_like(self.known)
        for idx in self.calib_order:
            mask[idx] = True
        return mask

    @property
    def target(self) -> np.ndarray:
        return ~self.known

    def group_counts(self, g_index: np.ndarray, n_groups: int) -> dict:
        return {"n_known": np.bincount(g_index[self.known], minlength=n_groups),
                "n_fit": np.bincount(g_index[self.fit], minlength=n_groups),
                "n_calib": np.array([len(c) for c in self.calib_order]),
                "n_target": np.bincount(g_index[~self.known], minlength=n_groups)}


def pilot_sizes(group_sizes, rho: float) -> np.ndarray:
    """n_g = round(rho * N_g), at least 1 and at most N_g (Python's round, as in the precheck)."""
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"pilot fraction rho must lie in (0, 1], got {rho}")
    return np.array([min(int(n), max(1, int(round(rho * int(n))))) for n in group_sizes])


def draw_pilot(g_index: np.ndarray, n_groups: int, rho: float, rng: np.random.Generator, *,
               fit_fraction: float = 0.0, weights: np.ndarray = None) -> Pilot:
    """Stratified pilot: within each group draw n_g organisations without replacement.

    weights=None gives simple random sampling (the design assumed by Theorem 1). Otherwise
    organisations are drawn successively with probability proportional to `weights` -- a biased
    design for the sensitivity analysis.
    fit_fraction: share of each group's pilot handed to a pilot-fitted simulator (rounded down, and
    leaving at least one calibration organisation whenever the group's pilot has two or more).
    """
    n_org = len(g_index)
    known = np.zeros(n_org, dtype=bool)
    fit = np.zeros(n_org, dtype=bool)
    calib_order = []
    sizes = pilot_sizes(np.bincount(g_index, minlength=n_groups), rho)
    for g in range(n_groups):
        idx = np.flatnonzero(g_index == g)
        k = int(sizes[g])
        if weights is None:
            chosen = rng.permutation(idx)[:k]
        else:
            w = np.asarray(weights, dtype=float)[idx]
            chosen = rng.choice(idx, size=k, replace=False, p=w / w.sum())
        known[chosen] = True
        n_fit = int(np.floor(fit_fraction * k)) if fit_fraction > 0 else 0
        n_fit = min(n_fit, max(0, k - 1))
        fit[chosen[:n_fit]] = True
        calib_order.append(np.asarray(chosen[n_fit:], dtype=int))
    return Pilot(known=known, fit=fit, calib_order=tuple(calib_order))


def pilot_from_mask(g_index: np.ndarray, n_groups: int, known: np.ndarray,
                    rng: np.random.Generator = None) -> Pilot:
    """Wrap a given set of known organisations (e.g. a real pilot). The within-group order is
    shuffled with `rng` when given; pass rng=None only if index order is unrelated to the data."""
    known = np.asarray(known, dtype=bool)
    order = []
    for g in range(n_groups):
        idx = np.flatnonzero((g_index == g) & known)
        order.append(rng.permutation(idx) if rng is not None else idx)
    return Pilot(known=known, fit=np.zeros_like(known), calib_order=tuple(order))


def tilt_weights(score: np.ndarray, g_index: np.ndarray, n_groups: int, gamma: float) -> np.ndarray:
    """Sampling weights exp(gamma * z_i), with z_i the score standardised within the group.

    gamma = 0 recovers simple random sampling. With score = mis-set mass, gamma < 0 makes the pilot
    under-represent the organisations the simulator gets wrong (optimistic calibration).
    """
    z = np.zeros(len(score), dtype=float)
    for g in range(n_groups):
        m = g_index == g
        sd = score[m].std()
        z[m] = (score[m] - score[m].mean()) / sd if sd > 0 else 0.0
    return np.exp(gamma * z)
