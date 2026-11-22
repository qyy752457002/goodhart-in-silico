"""GUARD: rectify-then-certify (Algorithm 1).

A small stratified pilot of real stances is used twice:

  rectify   estimate the simulator's residual per group and per proposal and correct the simulated
            welfare before optimising (fixes directional misreadings);
  certify   bound the true welfare of *every* package simultaneously, so the bound holds for the
            selected package no matter how it was chosen (Theorem 1).

Methodology map (03_methodology.md): section 4.2 (construction), 4.4 (variants), 4.5 (Algorithm 1).
Variants implemented here:
  GUARD        decide with the rectified estimate W~           `rectified_system`
  GUARD-pess   decide with the lower bound L                   `certificate_bounds(kind=main)`
  GUARD-est    decide with a plug-in (no-margin) lower bound   `certificate_bounds(kind='est')`
  GUARD-cons   only revise proposals with a certified gain     `conservative_select`
  GUARD-count / GUARD-pooled are alternative *certificates*    `certificate_bounds(kind=...)`
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from . import bounds, certify, welfare
from .engine import DesignEngine
from .pilot import Pilot
from .welfare import Affine

CERTIFICATE_KINDS = ("betting", "hoeffding_serfling", "count", "pooled")


# ----------------------------------------------------------------------------------------------
# Rectify (section 4.2 step 2)
# ----------------------------------------------------------------------------------------------

def rectified_system(engine: DesignEngine, pilot: Pilot) -> Affine:
    """Rectified group welfare, affine in the package:

        W~_g(a) = (1/N_g) [ sum_{known} u_i(a) + sum_{T_g} u-hat_i(a) - m_g (dc_g + dD_g . a) ],

    where (dc_g, dD_g) is the mean residual (simulated minus true affine utility) over the
    calibration organisations of group g. For each fixed package this is a prediction-powered
    estimate: unbiased under simple random sampling of the calibration set.
    """
    e = engine
    known_sum = e.group_sum(pilot.known, e.true_org)
    target_sim = e.group_sum(pilot.target, e.sim_org)
    m = e.group_count(pilot.target)
    dc = np.zeros(e.G)
    dD = np.zeros((e.G, e.Q))
    for g, calib in enumerate(pilot.calib_order):
        if len(calib):
            dc[g] = (e.sim_org.c[calib] - e.true_org.c[calib]).mean()
            dD[g] = (e.sim_org.D[calib] - e.true_org.D[calib]).mean(0)
    c = (known_sum.c + target_sim.c - m * dc) / e.sizes
    D = (known_sum.D + target_sim.D - m[:, None] * dD) / e.sizes[:, None]
    return Affine(c, D)


# ----------------------------------------------------------------------------------------------
# Certify (section 4.2 steps 5-7)
# ----------------------------------------------------------------------------------------------

def certificate_bounds(engine: DesignEngine, pilot: Pilot, kind: str = None,
                       rng: np.random.Generator = None) -> certify.GroupBounds:
    """Uniform bounds on group welfare.

    kind: 'betting' | 'hoeffding_serfling'  group-conditional mean bounds (guaranteed);
          'count'                           GUARD-count (guaranteed);
          'pooled'                          GUARD-pooled (not guaranteed per group);
          'est'                             plug-in mis-set mass without a confidence margin
                                            (a decision rule for GUARD-est, not a certificate).
    None means the main certificate chosen in the engine settings.
    rng: used only by 'pooled', to put the pooled calibration sample in a random order; without it
    the pooled bound falls back to Hoeffding-Serfling (see `certify.pooled_bounds`).
    """
    e, s = engine, engine.settings
    kind = kind or s.ucb
    args = (e.gi, e.G, pilot.known, pilot.calib_order, e.true_org, e.lam_org, e.ups_org)
    delta_g = s.delta_per_group(e.G)
    if kind == "betting":
        ucb = bounds.get_ucb("betting", c=s.betting_c, strategy=s.betting_strategy)
        return certify.mean_bounds(*args, e.s_minus, e.s_plus, e.s_max, delta_g, ucb, s.split_two_sided)
    if kind == "hoeffding_serfling":
        return certify.mean_bounds(*args, e.s_minus, e.s_plus, e.s_max, delta_g,
                                   bounds.hoeffding_serfling_ucb, s.split_two_sided)
    if kind == "est":
        return certify.mean_bounds(*args, e.s_minus, e.s_plus, e.s_max, delta_g, bounds.plugin_mean,
                                   kind="est", guaranteed=False)
    if kind == "count":
        return certify.count_bounds(*args, e.s_minus, e.s_max, delta_g, s.count_rank)
    if kind == "pooled":
        ucb = (bounds.get_ucb("betting", c=s.betting_c, strategy=s.betting_strategy)
               if s.ucb == "betting" else bounds.hoeffding_serfling_ucb)
        return certify.pooled_bounds(*args, e.s_minus, e.s_max, s.delta, ucb, rng)
    raise ValueError(f"unknown certificate kind {kind!r}")


def lower_table(engine: DesignEngine, b: certify.GroupBounds) -> np.ndarray:
    """(M, G) lower bound on group welfare at every package, exactly as the certificate reports it.

    With `clip_trivial` the bound is max(L_g(a), known part): still valid for all packages at once,
    no longer affine. Enumeration handles that; the MILP of Proposition 3 uses the affine L itself.
    """
    table = b.lower.table()
    return np.maximum(table, b.known_part.table()) if engine.settings.clip_trivial else table


# ----------------------------------------------------------------------------------------------
# GUARD-cons (section 4.2 step 8)
# ----------------------------------------------------------------------------------------------

def conservative_region(engine: DesignEngine, pilot: Pilot) -> dict:
    """Simultaneous confidence region for each group's ED welfare e_g = W_g(ED) and proposal
    coefficients d_gq (so that W_g(a) = e_g - sum_{q revised} d_gq).

    Built from the calibration residuals of the simulator, one interval per coordinate with a
    Bonferroni split of `cons_delta` over the G (Q + 1) two-sided intervals. Every interval is
    clipped to its a-priori range, so it is always finite.

    cons_bound:
      't'  a *moderated* Student-t interval with finite-population correction. The sample variance
           s^2 of the n calibration residuals is combined with a prior variance v0 = the mean
           squared a-priori deviation of an organisation (the largest variance the residual could
           have), carrying k0 = `cons_prior_obs` degrees of freedom:
               s~^2 = ((n - 1) s^2 + k0 v0) / (n - 1 + k0),    df = n - 1 + k0.
           Without the prior (k0 = 0) the interval has zero width whenever the calibration
           residuals happen to be identical -- typically all zero, because the sampled
           organisations were read correctly -- and the rule then trusts the simulator completely.
           The interval is asymptotic, not a finite-sample guarantee.
      'hoeffding_serfling'  finite-sample valid, but far wider with a handful of organisations.
    """
    e, s = engine, engine.settings
    level = s.cons_delta / (2.0 * e.G * (e.Q + 1))
    true_ed = e.true_org.c + e.true_org.D.sum(1)
    sim_ed = e.sim_org.c + e.sim_org.D.sum(1)
    known_sum = e.group_sum(pilot.known, e.true_org)
    e_lo, e_hi = np.zeros(e.G), np.zeros(e.G)
    d_lo, d_hi = np.zeros((e.G, e.Q)), np.zeros((e.G, e.Q))
    for g, calib in enumerate(pilot.calib_order):
        tgt = np.flatnonzero((e.gi == g) & pilot.target)
        n, m = len(calib), len(tgt)
        n_pop = n + m
        # residual r_i = simulated - true, for (u_i(ED), D_i1, ..., D_iQ)
        resid = np.column_stack([sim_ed[calib] - true_ed[calib], e.sim_org.D[calib] - e.true_org.D[calib]])
        sim_tgt = np.r_[sim_ed[tgt].sum(), e.sim_org.D[tgt].sum(0)]
        # a-priori range of the target residual sums: u_i(ED) in [0, 1], |D_iq| <= coef_range_iq
        width = np.r_[float(m), e.coef_range[tgt].sum(0)]
        prior_lo = np.r_[sim_tgt[0] - m, sim_tgt[1:] - width[1:]]
        prior_hi = np.r_[sim_tgt[0], sim_tgt[1:] + width[1:]]
        if m == 0:
            sum_lo = sum_hi = np.zeros(e.Q + 1)
        elif n == 0:
            sum_lo, sum_hi = prior_lo, prior_hi
        else:
            mean = resid.mean(0)
            pop = np.r_[calib, tgt]
            if s.cons_bound == "t":
                df = (n - 1) + float(s.cons_prior_obs)
                if df > 0:
                    var = resid.var(0, ddof=1) if n >= 2 else np.zeros(e.Q + 1)
                    # largest possible residual of an organisation: 1 for u_i(ED), 2 * coef_range for D_iq
                    prior_var = np.r_[1.0, ((2.0 * e.coef_range[pop]) ** 2).mean(0)]
                    moderated = ((n - 1) * var + float(s.cons_prior_obs) * prior_var) / df
                    se = np.sqrt(moderated / n) * np.sqrt(max(0.0, 1.0 - n / n_pop))
                    half = sps.t.ppf(1.0 - level, df) * se
                else:
                    half = np.full(e.Q + 1, np.inf)
            elif s.cons_bound == "hoeffding_serfling":
                value_range = np.r_[2.0, 4.0 * e.coef_range[pop].max(0)]
                half = value_range * np.sqrt((1.0 - (n - 1) / n_pop) * np.log(1.0 / level) / (2.0 * n))
            else:
                raise ValueError(f"unknown cons_bound {s.cons_bound!r}")
            # clip the interval into the a-priori range (if it lies wholly outside, it collapses
            # onto the nearest end of that range rather than onto a point beyond it)
            sum_lo = np.clip(n_pop * (mean - half) - resid.sum(0), prior_lo, prior_hi)
            sum_hi = np.clip(n_pop * (mean + half) - resid.sum(0), prior_lo, prior_hi)
        base = np.r_[known_sum.c[g] + known_sum.D[g].sum(), known_sum.D[g]] + sim_tgt
        hi = (base - sum_lo) / e.sizes[g]
        lo = (base - sum_hi) / e.sizes[g]
        e_lo[g], e_hi[g] = lo[0], hi[0]
        d_lo[g], d_hi[g] = lo[1:], hi[1:]
    return {"e_lo": e_lo, "e_hi": e_hi, "d_lo": d_lo, "d_hi": d_hi}


def improvement_lower_bound(engine: DesignEngine, region: dict, objective) -> np.ndarray:
    """(M,) smallest welfare gain of each package over the ED that is compatible with the region:
    I(a) = min_{region} [ M_p(W(a)) - M_p(W(ED)) ]. I(ED) = 0. It is a lower confidence bound on
    the true gain exactly to the extent that the region covers the truth (see
    `conservative_region`: finite-sample with Hoeffding-Serfling, asymptotic with the t-interval).

    Group g loses at most s_g(a) = sum_{q revised} d_hi[g, q] when a is adopted. Then
      p = 1     I(a) = - sum_g w_g s_g(a)                          (decomposes by proposal);
      p = -inf  I(a) = min_g [ kappa_g - s_g(a) ],  kappa_g = max(0, e_lo[g] - min_{h != g} e_hi[h]),
                i.e. a group that is certainly not the worst may give up its margin kappa_g;
      p = 0     the gain is measured on the log scale: sum_g w_g log(1 - s_g(a) / e*_g), with
                e*_g = e_lo[g] if the group loses and e_hi[g] if it gains (exact inner minimum);
      other p   the generic bound M_p(e_lo - s(a)) - M_p(e_hi).
    """
    e = engine
    p = welfare.objective_p(objective)
    w = e.weights
    d_hi = region["d_hi"]
    loss = welfare.enumerate_scores(d_hi.sum(1), -d_hi)  # (M, G): s_g(a)
    if p == 1:
        return -(loss @ w)
    active = w > 0
    if p == -np.inf:
        kappa = np.zeros(e.G)
        for g in range(e.G):
            others = region["e_hi"][np.arange(e.G) != g]
            kappa[g] = max(0.0, region["e_lo"][g] - others.min()) if others.size else 0.0
        return (kappa[None, :] - loss)[:, active].min(1)
    if p == 0:
        e_star = np.where(loss > 0, region["e_lo"][None, :], region["e_hi"][None, :])
        ratio = 1.0 - loss / np.clip(e_star, 1e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.where(ratio > 0, np.log(np.where(ratio > 0, ratio, 1.0)), -np.inf)
        return logs[:, active] @ w[active]
    low = welfare.power_mean(region["e_lo"][None, :] - loss, p, w)
    return low - float(welfare.power_mean(region["e_hi"], p, w))


def conservative_select(engine: DesignEngine, pilot: Pilot, objective, region: dict = None) -> int:
    """GUARD-cons: the package with the largest lower confidence bound on its gain over the ED; the
    ED itself unless some revision has a strictly positive bound ("when in doubt, keep the draft").
    The package it returns still gets the ordinary certificate, which does not depend on this rule."""
    region = region or conservative_region(engine, pilot)
    gain = improvement_lower_bound(engine, region, objective)
    k = engine.space.argmax(gain)
    return k if gain[k] > 1e-12 else engine.space.ed_index


# ----------------------------------------------------------------------------------------------
# Algorithm 1, single shot
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class GuardResult:
    variant: str
    objective: str
    index: int               # package index in the PackageSpace
    package: np.ndarray      # (Q,) 1 = retain the ED proposal, 0 = revise
    certificate: float       # M_p(L(a-hat)): lower bound on true welfare, valid w.p. >= 1 - delta
    lower: np.ndarray        # (G,) L_g(a-hat)
    upper: np.ndarray        # (G,) U_g(a-hat)
    estimate: float          # M_p(W~(a-hat)): rectified point estimate
    estimate_groups: np.ndarray
    tau: np.ndarray          # (G,) certified total mis-set mass of the targets
    delta: float
    certificate_kind: str

    def revised(self, decision_labels: list) -> list:
        return [d for d, a in zip(decision_labels, self.package) if a == 0]


def run_guard(engine: DesignEngine, pilot: Pilot, objective: str = "maxmin", variant: str = "guard",
              certificate_kind: str = None, rng: np.random.Generator = None) -> GuardResult:
    """Algorithm 1 (GUARD, rectify-then-certify) and its decision variants.

    variant: 'guard' (main) | 'guard_pess' | 'guard_est' | 'guard_cons'.
    The certificate is computed the same way for every variant: Theorem 1 makes it valid for any
    rule that picks the package.
    """
    e = engine
    rect = rectified_system(e, pilot)                                   # lines 1-3: rectify
    cert = certificate_bounds(e, pilot, certificate_kind, rng)          # lines 5-7: certify
    if variant == "guard":
        k = e.space.argmax(e.values(rect.table(), objective))           # line 4: decide
    elif variant == "guard_pess":
        k = e.space.argmax(e.values(lower_table(e, cert), objective))
    elif variant == "guard_est":
        k = e.space.argmax(e.values(lower_table(e, certificate_bounds(e, pilot, "est")), objective))
    elif variant == "guard_cons":
        k = conservative_select(e, pilot, objective)
    else:
        raise ValueError(f"unknown GUARD variant {variant!r}")
    a = e.space.package(k).astype(float)
    p = welfare.objective_p(objective)
    clip = e.settings.clip_trivial
    return GuardResult(
        variant=variant, objective=objective, index=int(k), package=e.space.package(k),
        certificate=cert.certificate(a, p, e._w(), clip),
        lower=cert.lower_at(a, clip), upper=cert.upper_at(a, clip),
        estimate=e.value_at(rect.at(a), objective), estimate_groups=rect.at(a),
        tau=cert.tau_minus, delta=e.settings.delta, certificate_kind=certificate_kind or e.settings.ucb,
    )
