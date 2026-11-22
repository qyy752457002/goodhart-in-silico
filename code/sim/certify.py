"""Finite-population upper confidence bounds and the GUARD-mean group welfare bound.

Pilot = simple random sample without replacement within each group.

Methodology map (03_methodology.md): section 4.2 steps 4-7, Lemma 1 and Theorem 1.

Three roles of organisations inside a group (see `sim.pilot.Pilot`):
  known   true stances observed (the pilot). They enter every bound with their true utility.
  calib   the part of the pilot used to *measure* simulator error. For a simulator that does not
          depend on the pilot (S1: persona simulators) this is the whole pilot. For a simulator
          fitted on pilot data, the fitting part is excluded so that calib is still a simple random
          sample of {calib + target} given the fitted simulator.
  target  everything else: true stances unknown, bounded through the simulator's base sets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bounds
from .bounds import hoeffding_serfling_ucb  # noqa: F401  (re-exported: used by the Day-1 precheck)
from .welfare import Affine, org_affine, power_mean


def guard_mean_bound(pilot_mask, g_index, n_groups, c_true, D_true, c_lam, D_lam, s_minus, delta_g, ucb=None):
    """Affine lower bounds L_g(a) = Cg + Dg a for each group.

    pilot orgs contribute their true utility; target orgs contribute the base-set lower utility
    lambda_i(a); the certified target mis-set mass tau_g is subtracted.

    (Day-1 interface, kept for `precheck_p1_p3.py`. New code uses `mean_bounds`.)
    """
    ucb = ucb or hoeffding_serfling_ucb
    Cg = np.zeros(n_groups)
    Dg = np.zeros((n_groups, D_true.shape[1]))
    tau = np.zeros(n_groups)
    for g in range(n_groups):
        m = g_index == g
        P = m & pilot_mask
        T = m & ~pilot_mask
        Ng = m.sum()
        mu_ucb = ucb(s_minus[P], Ng, delta_g)
        tau[g] = max(0.0, Ng * mu_ucb - s_minus[P].sum())
        tau[g] = min(tau[g], s_minus[T].size)  # cannot exceed the number of target orgs
        Cg[g] = (c_true[P].sum() + c_lam[T].sum() - tau[g]) / Ng
        Dg[g] = (D_true[P].sum(0) + D_lam[T].sum(0)) / Ng
    return Cg, Dg, tau


# ----------------------------------------------------------------------------------------------
# Base sets, cell-level bounds and mis-set mass (section 4.2 step 4)
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class BaseSets:
    """Per-cell base set B_iq of stance classes and the support interval it spans."""
    member: np.ndarray    # (n_cells, K) bool: class k is in the base set
    sigma_lo: np.ndarray  # (n_cells,) min support over the base set
    sigma_hi: np.ndarray  # (n_cells,) max support over the base set

    @property
    def beta_bar(self) -> np.ndarray:
        """max_x beta(x): the most utility a covered cell is guaranteed, hence the most that a
        *missed* cell can cost the lower bound. beta(1) = sigma_lo, beta(0) = 1 - sigma_hi."""
        return np.maximum(self.sigma_lo, 1.0 - self.sigma_hi)

    @property
    def one_minus_gamma_low(self) -> np.ndarray:
        """1 - min_x gamma(x), with gamma(1) = sigma_hi and gamma(0) = 1 - sigma_lo.
        Identically equal to `beta_bar` because phi(s, 0) = 1 - phi(s, 1)."""
        return 1.0 - np.minimum(self.sigma_hi, 1.0 - self.sigma_lo)


def base_sets(probs: np.ndarray, sigma_vec: np.ndarray, rule: str = "top1", k: int = 1,
              mass: float = 0.9) -> BaseSets:
    """Build base sets from simulator distributions.

    rule: 'top1' (main specification) the most probable class;
          'topk' the k most probable classes;
          'mass' the smallest set of most probable classes whose probability reaches `mass`.
    Ties are broken by class order, so the result is deterministic.
    """
    probs = np.asarray(probs, dtype=float)
    n, K = probs.shape
    order = np.argsort(-probs, axis=1, kind="stable")
    if rule == "top1":
        size = np.ones(n, dtype=int)
    elif rule == "topk":
        if not 1 <= k <= K:
            raise ValueError(f"k must lie in 1..{K}")
        size = np.full(n, int(k))
    elif rule == "mass":
        if not 0.0 < mass <= 1.0:
            raise ValueError("mass must lie in (0, 1]")
        cum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
        size = np.minimum((cum < mass - 1e-12).sum(1) + 1, K)
    else:
        raise ValueError(f"unknown base-set rule {rule!r} (use 'top1', 'topk' or 'mass')")
    in_set_sorted = np.arange(K)[None, :] < size[:, None]
    member = np.zeros((n, K), dtype=bool)
    np.put_along_axis(member, order, in_set_sorted, axis=1)
    sig = np.broadcast_to(np.asarray(sigma_vec, dtype=float)[None, :], (n, K))
    sigma_lo = np.where(member, sig, np.inf).min(1)
    sigma_hi = np.where(member, sig, -np.inf).max(1)
    return BaseSets(member=member, sigma_lo=sigma_lo, sigma_hi=sigma_hi)


def cell_miss(bs: BaseSets, true_sigma: np.ndarray, true_class: np.ndarray,
              rule: str = "sigma_hull") -> np.ndarray:
    """Which cells the base set fails to cover.

    'label'       the literal definition: the true stance class is not in B_iq.
    'sigma_hull'  the true support lies outside [sigma_lo, sigma_hi]. Lemma 1 only needs
                  phi(s, x) in [beta(x), gamma(x)], which holds whenever sigma(s) is inside the
                  hull, so this rule is valid and never counts more misses than 'label'. With
                  top-1 sets it differs only for classes that share a support score.
    """
    if rule == "label":
        return ~bs.member[np.arange(len(true_class)), true_class]
    if rule == "sigma_hull":
        return (true_sigma < bs.sigma_lo - 1e-12) | (true_sigma > bs.sigma_hi + 1e-12)
    raise ValueError(f"unknown miss rule {rule!r} (use 'sigma_hull' or 'label')")


def org_bound_affines(cell_org, cell_dec, n_org, n_dec, bs: BaseSets) -> tuple[Affine, Affine]:
    """Base-set lower and upper utilities lambda_i(a) and upsilon_i(a), affine in the package."""
    lam = org_affine(cell_org, cell_dec, bs.sigma_lo, 1.0 - bs.sigma_hi, n_org, n_dec)
    ups = org_affine(cell_org, cell_dec, bs.sigma_hi, 1.0 - bs.sigma_lo, n_org, n_dec)
    return Affine(*lam), Affine(*ups)


def org_misset_mass(cell_org, n_org, bs: BaseSets, miss: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mis-set mass per organisation (section 4.2 step 4).

    S_minus_i = (1/|Q_i|) sum_q beta_bar_iq 1[miss_iq]          (enters the lower bound)
    S_plus_i  = (1/|Q_i|) sum_q (1 - gamma_low_iq) 1[miss_iq]   (enters the upper bound)
    S_max_i   = (1/|Q_i|) sum_q beta_bar_iq                     (value if every cell were missed;
                                                                 known without any true stance)
    S_minus and S_plus need the true stances, so only pilot entries may be used by a method.
    """
    counts = np.bincount(cell_org, minlength=n_org).astype(float)
    miss = miss.astype(float)
    s_minus = np.bincount(cell_org, weights=bs.beta_bar * miss, minlength=n_org) / counts
    s_plus = np.bincount(cell_org, weights=bs.one_minus_gamma_low * miss, minlength=n_org) / counts
    s_max = np.bincount(cell_org, weights=bs.beta_bar, minlength=n_org) / counts
    return s_minus, s_plus, s_max


# ----------------------------------------------------------------------------------------------
# Uniform group bounds (section 4.2 steps 5-6, Theorem 1)
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class GroupBounds:
    """Bounds L_g(a) <= W_g(a) <= U_g(a) meant to hold for every package simultaneously."""
    lower: Affine            # (G rows)
    upper: Affine
    tau_minus: np.ndarray    # (G,) certified total mis-set mass of the target organisations
    tau_plus: np.ndarray
    known_part: Affine       # (1/N_g) sum_{known} u_i(a): trivial lower bound (targets >= 0)
    n_target: np.ndarray     # (G,) number of target organisations
    group_sizes: np.ndarray  # (G,)
    kind: str = "mean"
    guaranteed: bool = True

    def lower_at(self, a, clip_trivial: bool = True) -> np.ndarray:
        """(G,) lower bounds at package a. A target organisation's utility is at least 0, so the
        known part alone is always a valid bound; clipping never invalidates the certificate."""
        low = self.lower.at(a)
        return np.maximum(low, self.known_part.at(a)) if clip_trivial else low

    def upper_at(self, a, clip_trivial: bool = True) -> np.ndarray:
        up = self.upper.at(a)
        if not clip_trivial:
            return up
        return np.minimum(up, self.known_part.at(a) + self.n_target / self.group_sizes)

    def certificate(self, a, p: float, weights: np.ndarray = None, clip_trivial: bool = True) -> float:
        """c = M_p(L(a)): a lower bound on the true welfare of package a (section 4.2 step 7)."""
        return float(power_mean(self.lower_at(a, clip_trivial), p, weights))


def _group_sums(gi, n_groups, mask, org: Affine) -> Affine:
    """Per-group sums of an organisation-level affine system over the organisations in `mask`."""
    Z = (gi[None, :] == np.arange(n_groups)[:, None]) & mask[None, :]
    Zf = Z.astype(float)
    return Affine(Zf @ org.c, Zf @ org.D)


def _assemble(gi, n_groups, known, target, true_org, lam, ups, tau_minus, tau_plus, kind, guaranteed):
    sizes = np.bincount(gi, minlength=n_groups).astype(float)
    known_sum = _group_sums(gi, n_groups, known, true_org)
    lam_sum = _group_sums(gi, n_groups, target, lam)
    ups_sum = _group_sums(gi, n_groups, target, ups)
    inv = 1.0 / sizes
    lower = (known_sum + lam_sum).shifted(-tau_minus).scaled(inv)
    upper = (known_sum + ups_sum).shifted(tau_plus).scaled(inv)
    n_target = np.bincount(gi[target], minlength=n_groups).astype(float)
    return GroupBounds(lower=lower, upper=upper, tau_minus=tau_minus, tau_plus=tau_plus,
                       known_part=known_sum.scaled(inv), n_target=n_target, group_sizes=sizes,
                       kind=kind, guaranteed=guaranteed)


def _tau_from_ucb(values_calib, n_pop, delta, ucb, cap):
    """tau = N_pop * UCB(mean) - sum(calibration values), clipped to [0, cap]."""
    if n_pop == len(values_calib):  # no targets left
        return 0.0
    mu = ucb(values_calib, n_pop, delta)
    return float(min(max(0.0, n_pop * mu - values_calib.sum()), cap))


def mean_bounds(gi, n_groups, known, calib_order, true_org: Affine, lam: Affine, ups: Affine,
                s_minus, s_plus, s_max, delta_g, ucb=None, split_two_sided: bool = False,
                kind: str = "mean", guaranteed: bool = True) -> GroupBounds:
    """The main GUARD certificate: group-conditional bounds on mean mis-set mass.

    For each group g, with calibration organisations C_g (in sampling order), targets T_g and
    N_pop = |C_g| + |T_g|:

        tau_g^- = N_pop * UCB(S^- on C_g; N_pop, delta) - sum_{C_g} S_i^-      (>= sum_{T_g} S_i^-)
        L_g(a)  = (1/N_g) [ sum_{known} u_i(a) + sum_{T_g} lambda_i(a) - tau_g^- ]
        U_g(a)  = (1/N_g) [ sum_{known} u_i(a) + sum_{T_g} upsilon_i(a) + tau_g^+ ]

    By Lemma 1 the event {tau_g^- >= sum_{T_g} S_i^- for all g} implies L_g(a) <= W_g(a) for every
    package at once, so the bound survives any data-dependent choice of the package (Theorem 1).

    delta_g: (G,) error budget per group. S^+ equals S^- under the symmetric cell utility, so one
    confidence event covers both sides; set split_two_sided=True to spend delta_g / 2 per side as
    written in the methodology (more conservative, never needed for validity).
    tau is also capped at sum_{T_g} S_max_i, its largest possible value.

    `kind` / `guaranteed` label the result; pass guaranteed=False when `ucb` is not a confidence
    bound (the plug-in mean of GUARD-est).
    """
    ucb = ucb or bounds.betting_ucb_wor
    delta_g = np.broadcast_to(np.asarray(delta_g, dtype=float), (n_groups,))
    target = ~known
    tau_minus, tau_plus = np.zeros(n_groups), np.zeros(n_groups)
    for g in range(n_groups):
        calib = np.asarray(calib_order[g], dtype=int)
        tgt = np.flatnonzero((gi == g) & target)
        n_pop = len(calib) + len(tgt)
        cap = float(s_max[tgt].sum())
        same = np.array_equal(s_minus[calib], s_plus[calib])
        d = delta_g[g] if (same and not split_two_sided) else delta_g[g] / 2.0
        tau_minus[g] = _tau_from_ucb(s_minus[calib], n_pop, d, ucb, cap)
        tau_plus[g] = tau_minus[g] if same else _tau_from_ucb(s_plus[calib], n_pop, d, ucb, cap)
    return _assemble(gi, n_groups, known, target, true_org, lam, ups, tau_minus, tau_plus, kind, guaranteed)


def count_bounds(gi, n_groups, known, calib_order, true_org: Affine, lam: Affine, ups: Affine,
                 s_minus, s_max, delta_g, rank="best") -> GroupBounds:
    """GUARD-count: a transductive, distribution-free certificate that counts organisations.

    Let Gamma be the r-th smallest calibration value of S^-. With probability >= 1 - delta at most
    K target organisations have S_i > Gamma (`bounds.exceedance_count_ucb`); every other target has
    S_i <= min(Gamma, S_max_i), and the K exceptions have S_i <= S_max_i. Hence

        sum_{T_g} S_i  <=  sum_{T_g} min(Gamma, S_max_i) + (sum of the K largest S_max_i - min(...)).

    rank: 'max' (r = n), a quantile in (0, 1] (r = ceil(q n)), or 'best' -- try every rank at level
    delta / n (Bonferroni) and keep the smallest tau.
    """
    delta_g = np.broadcast_to(np.asarray(delta_g, dtype=float), (n_groups,))
    target = ~known
    tau = np.zeros(n_groups)
    for g in range(n_groups):
        calib = np.asarray(calib_order[g], dtype=int)
        tgt = np.flatnonzero((gi == g) & target)
        n, m = len(calib), len(tgt)
        cap_each = s_max[tgt]
        if m == 0:
            continue
        if n == 0:
            tau[g] = cap_each.sum()
            continue
        sorted_calib = np.sort(s_minus[calib])
        if rank == "best":
            ranks, level = range(1, n + 1), delta_g[g] / n
        elif rank == "max":
            ranks, level = [n], delta_g[g]
        else:
            ranks, level = [int(min(n, max(1, np.ceil(float(rank) * n))))], delta_g[g]
        best = cap_each.sum()
        for r in ranks:
            gamma = sorted_calib[r - 1]
            k_max = bounds.exceedance_count_ucb(n + m, n, r, level)
            within = np.minimum(gamma, cap_each)
            excess = np.sort(cap_each - within)[::-1][:k_max]
            best = min(best, float(within.sum() + excess.sum()))
        tau[g] = best
    return _assemble(gi, n_groups, known, target, true_org, lam, ups, tau, tau.copy(), "count", True)


def pooled_bounds(gi, n_groups, known, calib_order, true_org: Affine, lam: Affine, ups: Affine,
                  s_minus, s_max, delta, ucb=None, rng: np.random.Generator = None) -> GroupBounds:
    """GUARD-pooled (ablation for Proposition B): calibrate mis-set mass *without* groups.

    One bound on the total target mis-set mass is computed from the pooled calibration sample, as
    if it were a simple random sample of all non-fit organisations, and then shared out in
    proportion to the number of targets per group. It aims at the average over organisations
    only: nothing holds group by group, so the certificate is not guaranteed for objectives that
    depend on individual groups (max-min, Nash).

    Order matters: the betting bound needs an exchangeable order, and the per-group samples
    concatenated group by group are not one (if later groups are simulated worse, the bound is far
    too small; if earlier ones are, far too large). With `rng` the pooled sample is put in a
    uniformly random order; without it the order-free Hoeffding-Serfling bound is used whatever
    `ucb` says.
    """
    target = ~known
    calib_all = np.concatenate([np.asarray(c, dtype=int) for c in calib_order]) if calib_order else np.array([], int)
    if rng is not None:
        ucb = ucb or bounds.betting_ucb_wor
        calib_all = rng.permutation(calib_all)
    else:
        ucb = bounds.hoeffding_serfling_ucb
    n_target = np.bincount(gi[target], minlength=n_groups).astype(float)
    m_total = n_target.sum()
    tau_total = _tau_from_ucb(s_minus[calib_all], len(calib_all) + int(m_total), delta, ucb,
                              float(s_max[target].sum()))
    tau = tau_total * n_target / m_total if m_total > 0 else np.zeros(n_groups)
    return _assemble(gi, n_groups, known, target, true_org, lam, ups, tau, tau.copy(), "pooled", False)


def uniform_validity(b: GroupBounds, true_group: Affine, tol: float = 1e-9) -> dict:
    """Does L_g(a) <= W_g(a) <= U_g(a) hold for *every* package and group (the event of Theorem 1)?

    W - L and U - W are affine in a, so their minima over {0,1}^Q are available in closed form.
    Evaluation only: this uses the true welfare of all organisations.
    """
    slack_low = (true_group - b.lower).minimum_over_packages()
    slack_up = (b.upper - true_group).minimum_over_packages()
    return {"lower_valid": bool((slack_low >= -tol).all()), "upper_valid": bool((slack_up >= -tol).all()),
            "min_lower_slack": float(slack_low.min()), "min_upper_slack": float(slack_up.min())}
