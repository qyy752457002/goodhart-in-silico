"""The two-group, two-proposal example of 02_gaps_and_contributions.md section 3 (paper Figure 1).

Groups A and B have equal size. Truth: A wants to retain q1 and revise q2; B is indifferent on q1
and strongly wants to retain q2. The simulator reads A correctly but believes B leans towards
revising q2 (expected support 0.4 instead of 1).

  package (a1, a2)   true (W_A, W_B)   true min   simulated (W_A, W_B)   simulated min
  (1, 1)             (0.50, 0.75)      0.50       (0.50, 0.45)           0.45
  (1, 0)             (1.00, 0.25)      0.25       (1.00, 0.55)           0.55
  (0, 1)             (0.00, 0.75)      0.00       (0.00, 0.45)           0.00
  (0, 0)             (0.50, 0.25)      0.25       (0.50, 0.55)           0.50

The max-min plug-in picks (1, 0): it looks balanced in simulation (0.55) while the misread group B
is the real loser (0.25). The utilitarian plug-in picks the same package at zero regret.
"""
from __future__ import annotations

import numpy as np

from ..data import STANCE_INDEX, STANCE_SIGMA, STANCES, Problem
from ..simulators.base import SimulatorOutput

TRUE_STANCE = {("A", 1): "agree", ("A", 2): "disagree", ("B", 1): "mixed", ("B", 2): "agree"}
B_Q2_SIMULATED_SUPPORT = 0.4


def toy_problem(n_per_group: int = 4) -> Problem:
    orgs, groups, cell_org, cell_dec, cell_q, stances = [], [], [], [], [], []
    for g in ("A", "B"):
        for j in range(n_per_group):
            orgs.append(f"{g}{j + 1}")
            groups.append(g)
            for d, q in enumerate((1, 2)):
                cell_org.append(len(orgs) - 1)
                cell_dec.append(d)
                cell_q.append(q)
                stances.append(TRUE_STANCE[(g, q)])
    stances = np.array(stances, dtype=object)
    return Problem(
        orgs=orgs, groups=groups, group_names=["A", "B"], decisions=[1, 2],
        cell_org=np.array(cell_org), cell_dec=np.array(cell_dec), cell_q=np.array(cell_q),
        cell_stance=stances, sigma=np.array([STANCE_SIGMA[s] for s in stances]),
        questions={1: {"question": "Retain proposal q1?", "background": ""},
                   2: {"question": "Retain proposal q2?", "background": ""}},
        name="toy",
    )


def toy_simulator(prob: Problem) -> SimulatorOutput:
    """Correct everywhere except group B on q2, where support is read as 0.4 (0.4 agree, 0.6 disagree)."""
    probs = np.zeros((prob.n_cells, len(STANCES)))
    for k, ((org, q), s) in enumerate(zip(prob.cell_keys, prob.cell_stance)):
        if org.startswith("B") and q == 2:
            probs[k, STANCE_INDEX["agree"]] = B_Q2_SIMULATED_SUPPORT
            probs[k, STANCE_INDEX["disagree"]] = 1.0 - B_Q2_SIMULATED_SUPPORT
        else:
            probs[k, STANCE_INDEX[s]] = 1.0
    return SimulatorOutput("toy_simulator", probs, None, {"kind": "toy"})
