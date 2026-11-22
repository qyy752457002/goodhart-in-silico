"""Synthetic simulators built from the true stances, for controlled error injection.

  iid_noise              independent label noise at rate e (no structure across groups/proposals);
  structured_misreading  one group is read as "agree" on the proposals it opposes most, with
                         strength e_s, on top of background label noise e_bg;
  optimistic_bias        the Day-1 "biased" variant: one group's opposition is read as agreement.

Methodology map (03_methodology.md): section 3 ("synthetic injection") and section 12. The precheck
found that independent noise does not produce a Goodhart gap while structured misreading produces
illusory fairness; these generators reproduce both regimes (with their own random streams).
"""
from __future__ import annotations

import numpy as np

from ..data import STANCE_INDEX, STANCES
from .base import SimulatorOutput


def marginal_frequencies(prob) -> np.ndarray:
    """(K,) share of each stance class among the true stances."""
    return np.bincount(prob.stance_idx, minlength=len(STANCES)) / prob.n_cells


def corrupt_labels(true_idx: np.ndarray, rate, freq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace each label, with probability `rate` (scalar or per cell), by a *different* label
    drawn in proportion to the marginal frequencies of the other classes."""
    K = len(freq)
    trans = np.tile(np.asarray(freq, dtype=float), (K, 1))
    np.fill_diagonal(trans, 0.0)
    trans[trans.sum(1) == 0] = 1.0  # a class whose alternatives never occur: uniform over the others
    np.fill_diagonal(trans, 0.0)
    cdf = np.cumsum(trans / trans.sum(1, keepdims=True), axis=1)
    out = true_idx.copy()
    hit = rng.random(len(true_idx)) < rate
    u = rng.random(int(hit.sum()))
    new = (u[:, None] > cdf[true_idx[hit]]).sum(1)
    out[hit] = np.minimum(new, K - 1)
    return out


def most_opposed_decisions(prob, group: int, k: int) -> list:
    """Ids of the k decisions with the lowest mean true support in group `group`."""
    in_group = prob.g_index[prob.cell_org] == group
    sums = np.bincount(prob.cell_dec[in_group], weights=prob.sigma[in_group], minlength=prob.n_dec)
    counts = np.bincount(prob.cell_dec[in_group], minlength=prob.n_dec)
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)
    order = np.argsort(mean, kind="stable")[:k]
    return [prob.decisions[d] for d in order]


def _wrap(name: str, draws: list, meta: dict) -> SimulatorOutput:
    """One draw -> point predictions; several draws -> ensemble members with their mean as p-hat."""
    if len(draws) == 1:
        return SimulatorOutput.from_labels(name, draws[0], meta=meta)
    members = np.eye(len(STANCES))[np.stack(draws)]
    return SimulatorOutput(name, members.mean(0), members, meta)


def iid_noise(prob, e: float, rng: np.random.Generator, n_members: int = 0) -> SimulatorOutput:
    """Independent label noise: each cell is wrong with probability e."""
    freq = marginal_frequencies(prob)
    draws = [corrupt_labels(prob.stance_idx, e, freq, rng) for _ in range(max(1, n_members))]
    return _wrap(f"iid_noise(e={e:g})", draws, {"kind": "iid_noise", "e": e})


def structured_misreading(prob, group: int, e_s: float, rng: np.random.Generator, *, k: int = 4,
                          decisions: list = None, e_bg: float = 0.1, read_as: str = "agree",
                          n_members: int = 0) -> SimulatorOutput:
    """A systematic misreading of one group's position.

    On the target cells (group `group`, decisions `decisions` -- by default the k it opposes most)
    the simulator says `read_as` with probability e_s and is otherwise correct; every other cell
    gets independent label noise at rate e_bg. Ensemble members are independent draws of the same
    mechanism, so they *share* the misreading (their disagreement does not reveal it).
    """
    decisions = list(decisions) if decisions is not None else most_opposed_decisions(prob, group, k)
    dec_ids = np.asarray(prob.decisions)[prob.cell_dec]
    target = (prob.g_index[prob.cell_org] == group) & np.isin(dec_ids, decisions)
    freq = marginal_frequencies(prob)
    draws = []
    for _ in range(max(1, n_members)):
        labels = corrupt_labels(prob.stance_idx, np.where(target, 0.0, e_bg), freq, rng)
        flip = target & (rng.random(prob.n_cells) < e_s)
        labels[flip] = STANCE_INDEX[read_as]
        draws.append(labels)
    meta = {"kind": "structured_misreading", "group": prob.group_names[group], "decisions": decisions,
            "e_s": e_s, "e_bg": e_bg, "read_as": read_as}
    return _wrap(f"structured(g={prob.group_names[group]},e_s={e_s:g})", draws, meta)


def optimistic_bias(prob, group: int, e: float, rng: np.random.Generator, n_members: int = 0) -> SimulatorOutput:
    """In `group`, an opposing stance (support < 0.5) is read as 'agree' with probability e; the
    other groups get independent label noise at rate e / 2."""
    in_group = prob.g_index[prob.cell_org] == group
    freq = marginal_frequencies(prob)
    draws = []
    for _ in range(max(1, n_members)):
        labels = corrupt_labels(prob.stance_idx, np.where(in_group, 0.0, e / 2.0), freq, rng)
        flip = in_group & (prob.sigma < 0.5) & (rng.random(prob.n_cells) < e)
        labels[flip] = STANCE_INDEX["agree"]
        draws.append(labels)
    meta = {"kind": "optimistic_bias", "group": prob.group_names[group], "e": e}
    return _wrap(f"optimistic(g={prob.group_names[group]},e={e:g})", draws, meta)


def oracle(prob) -> SimulatorOutput:
    """A perfect simulator (the true stances): a reference point for tests and plots."""
    return SimulatorOutput.from_labels("oracle_simulator", prob.stance_idx, meta={"kind": "oracle"})
