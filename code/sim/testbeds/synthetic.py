"""Fully synthetic consultations, for controlled checks that need more (or differently shaped) data
than the AASB testbed: coverage, pilot-size curves at larger N, Propositions A-C.

(The synthetic *simulators* -- label noise and structured misreading applied to a real or synthetic
problem -- live in `sim.simulators.synthetic`.)

Methodology map (03_methodology.md): section 6, "synthetic data".
"""
from __future__ import annotations

import numpy as np

from ..data import STANCE_SIGMA, Problem

# latent attitude -> stance (cut points on the latent scale, from most to least supportive)
_CUTS = [(1.0, "agree"), (0.3, "qualified_agree"), (-0.1, "mixed"),
         (-0.6, "qualified_disagree"), (-0.9, "alternative_proposal"), (-np.inf, "disagree")]


def latent_to_stance(z: np.ndarray) -> np.ndarray:
    out = np.empty(z.shape, dtype=object)
    assigned = np.zeros(z.shape, dtype=bool)
    for cut, label in _CUTS:
        hit = (~assigned) & (z > cut)
        out[hit] = label
        assigned |= hit
    return out


def make_synthetic_problem(group_sizes=(38, 27, 20, 15), n_dec: int = 18, response_rate: float = 0.56,
                           disagreement: float = 1.0, org_sd: float = 0.6, mean_support: float = 0.6,
                           group_names: list = None, sigma_map: dict = None, seed: int = 0) -> Problem:
    """Draw a consultation from a latent-attitude model.

    z_iq = mu_q + disagreement * eta_gq + org_i + noise_iq, then cut into the six stance classes.
      mu_q   ~ N(mean_support, 0.8^2)  how popular proposal q is overall
      eta_gq ~ N(0, 1)                 group-by-proposal disagreement (0 = groups agree)
      org_i  ~ N(0, org_sd^2)          an organisation's general disposition
    Each (organisation, proposal) cell is answered with probability `response_rate`; every
    organisation answers at least one proposal. Defaults mimic the AASB testbed's shape.
    """
    rng = np.random.default_rng(seed)
    sigma_map = dict(sigma_map or STANCE_SIGMA)
    G = len(group_sizes)
    group_names = list(group_names) if group_names else [f"G{g + 1}" for g in range(G)]
    if len(group_names) != G:
        raise ValueError("group_names must match group_sizes")
    mu = rng.normal(mean_support, 0.8, size=n_dec)
    eta = rng.normal(0.0, 1.0, size=(G, n_dec))
    orgs, groups = [], []
    cell_org, cell_dec, z_all = [], [], []
    for g, n in enumerate(group_sizes):
        for j in range(int(n)):
            i = len(orgs)
            orgs.append(f"{group_names[g]}_{j + 1:03d}")
            groups.append(group_names[g])
            answered = rng.random(n_dec) < response_rate
            if not answered.any():
                answered[rng.integers(n_dec)] = True
            disposition = rng.normal(0.0, org_sd)
            for q in np.flatnonzero(answered):
                cell_org.append(i)
                cell_dec.append(int(q))
                z_all.append(mu[q] + disagreement * eta[g, q] + disposition + rng.normal())
    stances = latent_to_stance(np.array(z_all))
    decisions = list(range(1, n_dec + 1))
    cell_dec = np.array(cell_dec)
    return Problem(
        orgs=orgs, groups=groups, group_names=sorted(set(groups)), decisions=decisions,
        cell_org=np.array(cell_org), cell_dec=cell_dec, cell_q=cell_dec + 1,
        cell_stance=stances, sigma=np.array([sigma_map[s] for s in stances], dtype=float),
        sigma_map=sigma_map,
        questions={q: {"question": f"Do you agree with proposal {q}?", "background": ""} for q in decisions},
        name="synthetic",
    )
