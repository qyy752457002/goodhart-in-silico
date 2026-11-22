"""Simulator validity metrics (RQ1).

Methodology map (03_methodology.md): section 2.4.
  per cell          accuracy, macro-F1, Brier score, ECE                       `cell_metrics`
  per group         signed bias b_g = mean(sigma-hat - sigma), accuracy, ECE   `group_metrics`
  group x proposal  bias matrix -- only structured (group x proposal) bias
                    produces a Goodhart gap; i.i.d. noise does not             `bias_matrix`
  decision-centric  error at the selected package, regret                      `decision_metrics`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats
from .data import STANCE3, STANCE3_CLASSES, STANCES
from .engine import DesignEngine
from .simulators.base import SimulatorOutput

# (3, 6) matrix that merges the six stance classes into support / mixed / oppose
MERGE3 = np.array([[1.0 if STANCE3[s] == c else 0.0 for s in STANCES] for c in STANCE3_CLASSES])


def _tie_aware_hit(true_idx: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Credit 1/|argmax set| when the true class is among the most probable classes, so that a
    uniform distribution scores 1/K instead of whatever class happens to be listed first."""
    top = probs.max(1, keepdims=True)
    is_top = probs >= top - 1e-12
    return is_top[np.arange(len(true_idx)), true_idx] / is_top.sum(1)


def macro_f1(true_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int) -> float:
    """Unweighted mean F1 over the classes that occur in the truth or the predictions."""
    scores = []
    for k in range(n_classes):
        tp = int(((pred_idx == k) & (true_idx == k)).sum())
        fp = int(((pred_idx == k) & (true_idx != k)).sum())
        fn = int(((pred_idx != k) & (true_idx == k)).sum())
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else np.nan


def expected_calibration_error(confidence: np.ndarray, hit: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE with equal-width bins: sum_b (n_b / n) |accuracy_b - confidence_b|."""
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - confidence[m].mean())
    return float(ece)


def cell_metrics(true_idx: np.ndarray, probs: np.ndarray, sigma_vec: np.ndarray, n_bins: int = 10) -> dict:
    """Per-cell fidelity of a set of predicted distributions (6 classes and the 3-class merge)."""
    n, K = probs.shape
    if n == 0:
        return {"n_cells": 0}
    onehot = np.eye(K)[true_idx]
    hit = _tie_aware_hit(true_idx, probs)
    sigma_true = sigma_vec[true_idx]
    sigma_hat = probs @ sigma_vec
    probs3 = probs @ MERGE3.T
    true3 = MERGE3[:, true_idx].argmax(0)
    return {
        "n_cells": int(n),
        "accuracy": float(hit.mean()),
        "macro_f1": macro_f1(true_idx, probs.argmax(1), K),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "ece": expected_calibration_error(probs.max(1), hit, n_bins),
        "nll": float(-np.log(np.clip(probs[np.arange(n), true_idx], 1e-6, None)).mean()),
        "sigma_bias": float((sigma_hat - sigma_true).mean()),
        "sigma_mae": float(np.abs(sigma_hat - sigma_true).mean()),
        "accuracy3": float(_tie_aware_hit(true3, probs3).mean()),
        "macro_f1_3": macro_f1(true3, probs3.argmax(1), 3),
    }


def overall_metrics(prob, sim: SimulatorOutput, n_boot: int = 0, rng: np.random.Generator = None) -> dict:
    """Cell metrics over the whole problem; with n_boot > 0, organisation-clustered bootstrap
    intervals for accuracy, Brier score and the mean support bias."""
    out = cell_metrics(prob.stance_idx, sim.probs, prob.sigma_vec)
    if n_boot > 0:
        for key in ("accuracy", "brier", "sigma_bias"):
            lo, hi = stats.cluster_bootstrap_ci(
                lambda idx, key=key: cell_metrics(prob.stance_idx[idx], sim.probs[idx], prob.sigma_vec)[key],
                prob.cell_org, n_boot=n_boot, rng=rng)
            out[f"{key}_lo"], out[f"{key}_hi"] = lo, hi
    return out


def group_metrics(prob, sim: SimulatorOutput) -> pd.DataFrame:
    """One row per stakeholder group.

    sigma_bias       b_g: mean over the group's cells of (simulated - true support);
    welfare_bias_ed  the same error as it enters welfare (each organisation weighted equally):
                     the simulator's overestimate of W_g at the ED package, delta_g(ED).
    """
    sigma_hat = sim.expected_sigma(prob.sigma_vec)
    err = sigma_hat - prob.sigma
    org_err = np.bincount(prob.cell_org, weights=err, minlength=prob.n_org) / prob.cells_per_org
    cell_group = prob.g_index[prob.cell_org]
    rows = []
    for g, name in enumerate(prob.group_names):
        m = cell_group == g
        row = {"group": name, "n_orgs": int(prob.group_sizes[g])}
        row.update(cell_metrics(prob.stance_idx[m], sim.probs[m], prob.sigma_vec))
        row["welfare_bias_ed"] = float(org_err[prob.g_index == g].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def bias_matrix(prob, sim: SimulatorOutput) -> pd.DataFrame:
    """Long table over (group, decision): mean signed support error and the number of cells."""
    err = sim.expected_sigma(prob.sigma_vec) - prob.sigma
    frame = pd.DataFrame({"group": np.asarray(prob.group_names)[prob.g_index[prob.cell_org]],
                          "decision": np.asarray(prob.decisions)[prob.cell_dec], "err": err,
                          "abs_err": np.abs(err)})
    out = frame.groupby(["group", "decision"]).agg(bias=("err", "mean"), mae=("abs_err", "mean"),
                                                   n_cells=("err", "size")).reset_index()
    return out


def coefficient_error(engine: DesignEngine) -> pd.DataFrame:
    """Error of the simulated welfare coefficients d-hat_gq - d_gq (what actually moves the
    decision): positive means the simulator overstates how much group g gains from retaining q."""
    diff = engine.sim_group.D - engine.true_group.D
    return pd.DataFrame(diff, index=engine.prob.group_names, columns=engine.prob.decision_labels())


def decision_metrics(engine: DesignEngine, objectives: list) -> pd.DataFrame:
    """Decision-centric evaluation: plug-in package per objective, its selected-package error
    |W-hat(a-hat) - W(a-hat)| and its regret."""
    rows = []
    for obj in objectives:
        k = engine.plugin_index(obj)
        m = engine.evaluate(k, obj, engine.W_sim)
        rows.append({"objective": obj, "selected_error": abs(m["overestimate"]), **m})
    return pd.DataFrame(rows)


def reference_simulators(prob) -> dict:
    """Label-free and oracle-marginal reference points for Table 1.

    uniform          uniform guessing over the six classes (Dominguez-Olmedo et al., 2024);
    marginal         the pooled stance frequencies (uses the truth: an upper bound on what a
                     persona-blind simulator can achieve);
    group_marginal   the stance frequencies of the organisation's own group (also uses the truth).
    """
    K = len(STANCES)
    marginal = np.bincount(prob.stance_idx, minlength=K) / prob.n_cells
    cell_group = prob.g_index[prob.cell_org]
    by_group = np.stack([np.bincount(prob.stance_idx[cell_group == g], minlength=K) / max(1, (cell_group == g).sum())
                         for g in range(prob.n_groups)])
    return {
        "uniform": SimulatorOutput.uniform(prob.n_cells),
        "marginal": SimulatorOutput("marginal", np.tile(marginal, (prob.n_cells, 1)), None, {"kind": "reference"}),
        "group_marginal": SimulatorOutput("group_marginal", by_group[cell_group], None, {"kind": "reference"}),
    }


def paired_accuracy_contrast(prob, sim_a: SimulatorOutput, sim_b: SimulatorOutput, n_boot: int = 1000,
                             rng: np.random.Generator = None) -> dict:
    """Is simulator A more accurate than B on the same cells? (named vs anonymous persona, early
    vs late training cut-off.) Reports the accuracy difference with an organisation-clustered
    bootstrap interval and an exact McNemar test on top-1 correctness."""
    hit_a = sim_a.top1() == prob.stance_idx
    hit_b = sim_b.top1() == prob.stance_idx
    diff = hit_a.astype(float) - hit_b.astype(float)
    lo, hi = stats.cluster_bootstrap_ci(lambda idx: diff[idx].mean(), prob.cell_org, n_boot=n_boot, rng=rng)
    return {"a": sim_a.name, "b": sim_b.name, "accuracy_a": float(hit_a.mean()), "accuracy_b": float(hit_b.mean()),
            "diff": float(diff.mean()), "diff_lo": lo, "diff_hi": hi,
            "only_a_correct": int((hit_a & ~hit_b).sum()), "only_b_correct": int((~hit_a & hit_b).sum()),
            "mcnemar_p": stats.mcnemar_exact(int((hit_a & ~hit_b).sum()), int((~hit_a & hit_b).sum()))}
