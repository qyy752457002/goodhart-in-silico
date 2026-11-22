"""Utilities, group welfare and power means; exact optimisation by enumeration.

Every group welfare is affine in the package a: W_g(a) = c_g + sum_q d_gq a_q,
so all 2^Q packages are scored with one matrix product.

Methodology map (03_methodology.md): sections 1.1 and 1.4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def all_packages(n_dec: int) -> np.ndarray:
    """(2^Q, Q) uint8 matrix of all packages; row k encodes k in binary."""
    k = np.arange(2 ** n_dec, dtype=np.int64)[:, None]
    return ((k >> np.arange(n_dec)) & 1).astype(np.uint8)


def org_affine(cell_org, cell_dec, value1, value0, n_org, n_dec):
    """Per-org affine utility u_i(a) = c_i + sum_q D_iq a_q from per-cell values.

    value1 / value0: per-cell utility if the decision is retain (1) / revise (0).
    Each org's cells are averaged (equal weight per responded cell).
    """
    counts = np.bincount(cell_org, minlength=n_org).astype(float)
    w = 1.0 / counts[cell_org]
    c = np.bincount(cell_org, weights=w * value0, minlength=n_org)
    D = np.zeros((n_org, n_dec))
    np.add.at(D, (cell_org, cell_dec), w * (value1 - value0))
    return c, D


def group_affine(c_org, D_org, g_index, n_groups, org_weights=None):
    """Group welfare W_g(a) = mean over the group's orgs of u_i(a) (optionally reweighted)."""
    wts = np.ones(len(c_org)) if org_weights is None else org_weights
    Cg = np.zeros(n_groups)
    Dg = np.zeros((n_groups, D_org.shape[1]))
    for g in range(n_groups):
        m = g_index == g
        Cg[g] = (wts[m] * c_org[m]).sum() / m.sum()
        Dg[g] = (wts[m][:, None] * D_org[m]).sum(0) / m.sum()
    return Cg, Dg


def score(packages, Cg, Dg):
    """(n_packages, G) group welfare for every package (pass a float64 matrix to avoid re-casting)."""
    P = packages if packages.dtype == np.float64 else packages.astype(np.float64)
    return Cg[None, :] + P @ Dg.T


def power_mean(Wg: np.ndarray, p: float, weights: np.ndarray = None) -> np.ndarray:
    """Power mean M_p across groups (last axis).

    p = 1 utilitarian, p = 0 Nash (geometric mean), p = -inf max-min; other real p are supported.
    `weights` are group weights (normalised internally); None means equal weights.
    Non-positive welfare is clipped at 1e-12 for p <= 0 (lower bounds can be negative).
    """
    if weights is None:
        if p == 1:
            return Wg.mean(-1)
        if p == 0:
            return np.exp(np.log(np.clip(Wg, 1e-12, None)).mean(-1))
        if p == -np.inf:
            return Wg.min(-1)
        weights = np.full(Wg.shape[-1], 1.0 / Wg.shape[-1])
    w = np.asarray(weights, dtype=float)
    if w.shape != (Wg.shape[-1],) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative, one per group, and not all zero")
    w = w / w.sum()
    active = w > 0
    if p == 1:
        return Wg @ w
    if p == 0:
        return np.exp(np.log(np.clip(Wg, 1e-12, None)) @ w)
    if p == -np.inf:
        return Wg[..., active].min(-1)
    if p == np.inf:
        return Wg[..., active].max(-1)
    if not np.isfinite(p):
        raise ValueError(p)
    base = np.clip(Wg, 1e-12, None) if p < 0 else np.clip(Wg, 0.0, None)
    return ((base ** p) @ w) ** (1.0 / p)


OBJECTIVES = {"utilitarian": 1, "nash": 0, "maxmin": -np.inf}


def objective_p(objective) -> float:
    """Resolve an objective name ('utilitarian' | 'nash' | 'maxmin') or a number to the exponent p."""
    if isinstance(objective, str):
        if objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}")
        return OBJECTIVES[objective]
    return float(objective)


# ----------------------------------------------------------------------------------------------
# Affine systems
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Affine:
    """A stack of functions that are affine in the package: f_r(a) = c_r + sum_q D_rq a_q.

    Rows are organisations (utilities) or groups (welfare / bounds on welfare).
    """
    c: np.ndarray  # (R,)
    D: np.ndarray  # (R, Q)

    def __post_init__(self):
        c, D = np.asarray(self.c, dtype=float), np.asarray(self.D, dtype=float)
        if c.ndim != 1 or D.ndim != 2 or D.shape[0] != c.shape[0]:
            raise ValueError(f"inconsistent affine system: c {c.shape}, D {D.shape}")
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "D", D)

    @property
    def n_rows(self) -> int:
        return self.c.shape[0]

    @property
    def n_dec(self) -> int:
        return self.D.shape[1]

    def at(self, a) -> np.ndarray:
        """Values at one package (Q,) -> (R,), or at several packages (n, Q) -> (n, R)."""
        a = np.asarray(a, dtype=float)
        return self.c + a @ self.D.T

    def table(self) -> np.ndarray:
        """(2^Q, R) values at every package; row k is the package whose bits encode k."""
        return enumerate_scores(self.c, self.D)

    def minimum_over_packages(self) -> np.ndarray:
        """(R,) exact minimum of each row over the hypercube {0,1}^Q (closed form)."""
        return self.c + np.minimum(self.D, 0.0).sum(1)

    def maximum_over_packages(self) -> np.ndarray:
        return self.c + np.maximum(self.D, 0.0).sum(1)

    def __add__(self, other: Affine) -> Affine:
        return Affine(self.c + other.c, self.D + other.D)

    def __sub__(self, other: Affine) -> Affine:
        return Affine(self.c - other.c, self.D - other.D)

    def scaled(self, k) -> Affine:
        k = np.asarray(k, dtype=float)
        return Affine(self.c * k, self.D * (k[:, None] if k.ndim == 1 else k))

    def shifted(self, delta) -> Affine:
        """Add a package-independent constant per row."""
        return Affine(self.c + np.asarray(delta, dtype=float), self.D)


def enumerate_scores(c: np.ndarray, D: np.ndarray) -> np.ndarray:
    """(2^Q, R) values of the affine system at every package, in `all_packages` row order.

    Built by doubling (bit q of the row index switches decision q on), so no (2^Q, Q) matrix
    is needed; equivalent to `score(all_packages(Q), c, D)` up to floating-point rounding.
    """
    c = np.asarray(c, dtype=float)
    D = np.asarray(D, dtype=float)
    out = np.empty((2 ** D.shape[1], c.shape[0]))
    out[0] = c
    for q in range(D.shape[1]):
        n = 1 << q  # rows n..2n-1 are rows 0..n-1 with decision q switched on
        np.add(out[:n], D[:, q], out=out[n:2 * n])
    return out


def org_affine_from_sigma(prob, sigma: np.ndarray) -> Affine:
    """Organisation utilities u_i(a) from per-cell support scores (true or simulated).

    The cell utility is phi(s, 1) = sigma and phi(s, 0) = 1 - sigma (methodology section 1.1).
    """
    c, D = org_affine(prob.cell_org, prob.cell_dec, sigma, 1.0 - sigma, prob.n_org, prob.n_dec)
    return Affine(c, D)


def group_mean(org: Affine, g_index: np.ndarray, n_groups: int) -> Affine:
    """Group welfare W_g(a): the mean utility of the group's organisations."""
    c, D = group_affine(org.c, org.D, g_index, n_groups)
    return Affine(c, D)


def group_weights(group_sizes, mode: str = "equal") -> np.ndarray:
    """Group weights for M_p: 'equal' (main specification) or 'size' (proportional to N_g)."""
    sizes = np.asarray(group_sizes, dtype=float)
    if mode == "equal":
        return np.full(len(sizes), 1.0 / len(sizes))
    if mode == "size":
        return sizes / sizes.sum()
    raise ValueError(f"unknown group weight mode {mode!r} (use 'equal' or 'size')")
