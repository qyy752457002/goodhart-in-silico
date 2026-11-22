"""Search over policy packages: exhaustive enumeration, closed forms, a max-min MILP and the
best-of-N selection law.

Methodology map (03_methodology.md): section 1.2 (plug-in design, optimisation pressure N),
section 1.4 (utilitarian objective decomposes by proposal), Proposition 3 (computation) and the
Hamming-radius axis of section 3.

Package indexing convention (shared with `welfare.all_packages`): package k has a_q = bit q of k,
so the all-revise (IFRS baseline) package is index 0 and the ED package (retain all) is 2^Q - 1.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
from scipy.special import gammaln

from . import welfare
from .welfare import Affine

MAX_ENUM_DECISIONS = 24  # 2^24 packages x G floats is the practical ceiling for enumeration


class PackageSpace:
    """All 2^Q packages of a Q-decision problem."""

    def __init__(self, n_dec: int):
        if not 1 <= n_dec <= MAX_ENUM_DECISIONS:
            raise ValueError(f"enumeration supports 1..{MAX_ENUM_DECISIONS} decisions, got {n_dec}; "
                             "use the closed form (utilitarian) or the MILP (max-min) instead")
        self.n_dec = int(n_dec)
        self.size = 2 ** self.n_dec

    @property
    def ed_index(self) -> int:
        """Retain every ED proposal."""
        return self.size - 1

    @property
    def ifrs_index(self) -> int:
        """Revise every proposal (revert to the IFRS baseline)."""
        return 0

    @cached_property
    def packages(self) -> np.ndarray:
        """(2^Q, Q) uint8."""
        return welfare.all_packages(self.n_dec)

    @cached_property
    def popcount(self) -> np.ndarray:
        """(2^Q,) number of retained proposals per package."""
        pc = np.zeros(1, dtype=np.int16)
        for _ in range(self.n_dec):
            pc = np.concatenate([pc, pc + 1])
        return pc

    def index(self, a) -> int:
        a = np.asarray(a).astype(np.int64).ravel()
        if a.shape != (self.n_dec,) or not np.isin(a, (0, 1)).all():
            raise ValueError(f"a package must be a 0/1 vector of length {self.n_dec}")
        return int((a << np.arange(self.n_dec)).sum())

    def package(self, k: int) -> np.ndarray:
        return ((int(k) >> np.arange(self.n_dec)) & 1).astype(np.uint8)

    def hamming(self, ref: int) -> np.ndarray:
        """(2^Q,) Hamming distance of every package from package `ref`."""
        return self.popcount[np.arange(self.size) ^ int(ref)]

    def hamming_between(self, k1: int, k2: int) -> int:
        return int(self.popcount[int(k1) ^ int(k2)])

    def argmax(self, values: np.ndarray, tol: float = 1e-12) -> int:
        """Index of the best package. Ties (within `tol`) go to the package that keeps the most ED
        proposals -- when indifferent, do not change the draft -- and then to the lowest index."""
        values = np.asarray(values)
        if values.shape != (self.size,):
            raise ValueError(f"expected {self.size} values, got shape {values.shape}")
        best = np.nanmax(values)
        cand = np.flatnonzero(values >= best - tol)
        if cand.size == 1:
            return int(cand[0])
        pc = self.popcount[cand]
        return int(cand[pc == pc.max()][0])

    def argmax_within(self, values: np.ndarray, mask: np.ndarray, tol: float = 1e-12) -> int:
        """`argmax` restricted to the packages where `mask` is True."""
        masked = np.where(mask, values, -np.inf)
        if not np.isfinite(masked).any():
            raise ValueError("empty feasible set")
        return self.argmax(masked, tol)


# ----------------------------------------------------------------------------------------------
# Exact solvers that avoid enumeration (Proposition 3)
# ----------------------------------------------------------------------------------------------

def solve_utilitarian(group: Affine, weights: np.ndarray = None) -> np.ndarray:
    """Utilitarian optimum in closed form: retain proposal q iff sum_g w_g d_gq >= 0
    (a weighted majority per proposal; ties keep the ED proposal)."""
    w = _weights(weights, group.n_rows)
    return (w @ group.D >= 0).astype(np.uint8)


def solve_maxmin_milp(group: Affine, time_limit: float = 60.0) -> tuple[np.ndarray, float]:
    """max_a min_g (c_g + d_g . a) as a mixed-integer linear programme.

    maximise t  s.t.  t - d_g . a <= c_g for every group g,  a in {0,1}^Q.
    Returns (package, optimal value).
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    G, Q = group.D.shape
    cost = np.zeros(Q + 1)
    cost[-1] = -1.0  # milp minimises
    A = np.hstack([-group.D, np.ones((G, 1))])
    span = np.abs(group.D).sum(1).max() + np.abs(group.c).max() + 1.0
    res = milp(
        c=cost,
        constraints=LinearConstraint(A, -np.inf, group.c),
        integrality=np.r_[np.ones(Q), 0.0],
        bounds=Bounds(np.r_[np.zeros(Q), -span], np.r_[np.ones(Q), span]),
        options={"time_limit": time_limit},
    )
    if res.x is None:
        raise RuntimeError(f"max-min MILP failed: {res.message}")
    a = np.rint(res.x[:Q]).astype(np.uint8)
    return a, float(group.at(a).min())


def solve(group: Affine, p: float, weights: np.ndarray = None, space: PackageSpace = None,
          method: str = "auto") -> np.ndarray:
    """Best package for objective M_p of an affine group system.

    method: 'enumerate' (any p), 'closed_form' (p = 1), 'milp' (p = -inf, equal-weight support),
    or 'auto' (enumerate when Q is small enough, otherwise the exact method for p in {1, -inf}).
    """
    Q = group.n_dec
    if method == "auto":
        method = "enumerate" if Q <= MAX_ENUM_DECISIONS else {1: "closed_form", -np.inf: "milp"}.get(p)
        if method is None:
            raise NotImplementedError(f"no exact solver for p={p} with Q={Q} > {MAX_ENUM_DECISIONS}")
    if method == "enumerate":
        space = space or PackageSpace(Q)
        k = space.argmax(welfare.power_mean(group.table(), p, weights))
        return space.package(k)
    if method == "closed_form":
        if p != 1:
            raise ValueError("the closed form exists only for the utilitarian objective (p = 1)")
        return solve_utilitarian(group, weights)
    if method == "milp":
        if p != -np.inf:
            raise ValueError("the MILP is formulated for the max-min objective (p = -inf)")
        if weights is not None and (np.asarray(weights) <= 0).any():
            keep = np.asarray(weights) > 0
            group = Affine(group.c[keep], group.D[keep])
        return solve_maxmin_milp(group)[0]
    raise ValueError(f"unknown method {method!r}")


def _weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


# ----------------------------------------------------------------------------------------------
# Optimisation pressure: best-of-N and the Hamming ball around the ED
# ----------------------------------------------------------------------------------------------

def best_of_n_pmf(n_packages: int, n_draws: int) -> np.ndarray:
    """Law of the *rank* of the selected package under best-of-N.

    N packages are drawn uniformly without replacement from the M packages and the one with the
    highest simulated welfare is selected. Entry r (0-based; 0 = the simulator's optimum) is

        P(rank = r) = C(M - 1 - r, N - 1) / C(M, N),      r = 0, ..., M - N,

    i.e. rank r is drawn and the other N - 1 draws are all worse. N = M gives the exact optimum.
    Sampling a rank from this law is equivalent to literally drawing N packages.
    """
    M, N = int(n_packages), int(n_draws)
    if not 1 <= N <= M:
        raise ValueError(f"need 1 <= N <= M, got N={N}, M={M}")
    r = np.arange(M - N + 1, dtype=float)
    log_pmf = (gammaln(M - r) - gammaln(N) - gammaln(M - r - N + 1)
               - (gammaln(M + 1) - gammaln(N + 1) - gammaln(M - N + 1)))
    pmf = np.exp(log_pmf)
    return pmf / pmf.sum()


def kl_best_of_n(n_draws) -> np.ndarray:
    """The nominal best-of-N pressure of Gao et al. (2023): log N - (N - 1)/N, the KL divergence
    from the sampling distribution when N candidates are drawn *with* replacement from a continuous
    score distribution. It is not the KL of the selection law used here (N distinct packages out
    of M): that one is larger, by up to one nat at N = M, where selection is deterministic and the
    exact value is log M. Use `kl_selection` for the exact value; this formula is kept as the
    conventional axis for comparison with the literature."""
    n = np.asarray(n_draws, dtype=float)
    return np.log(n) - (n - 1.0) / n


def kl_selection(pmf: np.ndarray, n_packages: int) -> float:
    """Exact KL(selection law || uniform over the M packages) for a rank law from `best_of_n_pmf`:
    sum_r p_r log(M p_r). 0 for N = 1 (uniform), log M for N = M (the optimum with certainty)."""
    p = np.asarray(pmf, dtype=float)
    p = p[p > 0]
    return float(np.sum(p * (np.log(p) + np.log(n_packages))))


def hamming_ball_argmax(space: PackageSpace, values: np.ndarray, ref: int = None) -> np.ndarray:
    """(Q + 1,) index of the best package within Hamming radius r = 0..Q of `ref` (default: ED)."""
    ref = space.ed_index if ref is None else int(ref)
    dist = space.hamming(ref)
    return np.array([space.argmax_within(values, dist <= r) for r in range(space.n_dec + 1)])
