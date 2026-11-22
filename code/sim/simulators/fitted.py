"""Simulators that are *fitted on observed stances* rather than prompted: the type prior R0 and the
relational models R3 (low-rank matrix factorisation, gradient-boosted trees).

Methodology map (03_methodology.md): section 2.1.
  R0  type prior        mean stance of the same-group organisations whose stances are known
  R3  relational model  organisation x question stance matrix: low-rank MF / GBDT (scenario S2)

All of them share one interface, `fit_predict(prob, train_orgs, ...) -> SimulatorOutput`, where
`train_orgs` marks the organisations whose stances on the decision set may be used. Two protocols:

  * inside a pilot replication (RQ3): train_orgs is the *fit* part of the pilot, and the remaining
    pilot organisations calibrate the rectification and the certificate (see `sim.pilot.Pilot`);
  * for validity metrics and Goodhart curves (RQ1, RQ2): `cross_fit` predicts every organisation
    from a model that never saw its decision-set stances.

`context` is an optional (n_org, n_context_questions) matrix of support scores on questions
*outside* the decision set (NaN = no answer), known for every organisation in scenario S2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import STANCES
from .base import SimulatorOutput

K = len(STANCES)


# ----------------------------------------------------------------------------------------------
# R0: type prior
# ----------------------------------------------------------------------------------------------

def _shrunk_counts(counts: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    """Dirichlet posterior mean: (counts + alpha * prior) / (n + alpha), row-wise."""
    return (counts + alpha * prior) / (counts.sum(-1, keepdims=True) + alpha)


def type_prior(prob, train_orgs: np.ndarray, alpha: float = 1.0, name: str = "r0_type_prior") -> SimulatorOutput:
    """R0: for cell (i, q) predict the stance distribution of the *training* organisations of i's
    group on question q, shrunk towards the group's overall distribution, then the pooled one, then
    uniform (so groups or questions without training data still get a prediction)."""
    train_cell = np.asarray(train_orgs, dtype=bool)[prob.cell_org]
    g_cell = prob.g_index[prob.cell_org]
    qids, q_cell = np.unique(prob.cell_q, return_inverse=True)
    counts = np.zeros((prob.n_groups, len(qids), K))
    np.add.at(counts, (g_cell[train_cell], q_cell[train_cell], prob.stance_idx[train_cell]), 1.0)
    pooled = _shrunk_counts(counts.sum((0, 1)), np.full(K, 1.0 / K), alpha)
    by_group = _shrunk_counts(counts.sum(1), pooled[None, :], alpha)
    by_cell = _shrunk_counts(counts, by_group[:, None, :], alpha)
    probs = by_cell[g_cell, q_cell]
    return SimulatorOutput(name, probs, None, {"kind": "type_prior", "alpha": alpha,
                                               "n_train_orgs": int(np.sum(train_orgs))})


# ----------------------------------------------------------------------------------------------
# R3: relational models
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RelationalSpec:
    model: str = "mf"            # 'mf' | 'gbdt'
    rank: int = 3                # MF: number of latent factors
    ridge: float = 1.0           # MF: L2 penalty on the factors
    n_iter: int = 30             # MF: alternating least-squares sweeps
    shrink: float = 2.0          # pseudo-count shrinking group-by-question means to question means
    min_bandwidth: float = 0.1   # MF: floor on the predictive standard deviation
    gbdt_max_iter: int = 100
    gbdt_max_depth: int = 3
    gbdt_learning_rate: float = 0.1
    seed: int = 0


def _baseline_support(prob, train_cell: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shrunk mean support per (group, decision question) from the training cells, with back-off
    to the question mean and the global mean. Returns (per-cell baseline, per-question mean, table)."""
    g_cell = prob.g_index[prob.cell_org]
    qids, q_cell = np.unique(prob.cell_q, return_inverse=True)
    sig = prob.sigma
    grand = sig[train_cell].mean() if train_cell.any() else 0.5
    q_sum = np.bincount(q_cell[train_cell], weights=sig[train_cell], minlength=len(qids))
    q_n = np.bincount(q_cell[train_cell], minlength=len(qids))
    q_mean = (q_sum + shrink * grand) / (q_n + shrink)
    gq_sum = np.zeros((prob.n_groups, len(qids)))
    gq_n = np.zeros((prob.n_groups, len(qids)))
    np.add.at(gq_sum, (g_cell[train_cell], q_cell[train_cell]), sig[train_cell])
    np.add.at(gq_n, (g_cell[train_cell], q_cell[train_cell]), 1.0)
    table = (gq_sum + shrink * q_mean[None, :]) / (gq_n + shrink)
    return table[g_cell, q_cell], q_mean[q_cell], table


def _support_to_distribution(sigma_hat: np.ndarray, sigma_vec: np.ndarray, class_prior: np.ndarray,
                             bandwidth: float) -> np.ndarray:
    """Turn a predicted support score into a distribution over stance classes: a Gaussian kernel
    around the prediction, weighted by how common each class is."""
    z = (sigma_vec[None, :] - sigma_hat[:, None]) / bandwidth
    weights = class_prior[None, :] * np.exp(-0.5 * z * z)
    total = weights.sum(1, keepdims=True)
    return np.where(total > 0, weights / np.where(total > 0, total, 1.0), 1.0 / K)


def _als(R: np.ndarray, observed: np.ndarray, rank: int, ridge: float, n_iter: int,
         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Alternating least squares for R ~ U V^T on the observed entries."""
    n_rows, n_cols = R.shape
    U = rng.normal(0, 0.1, (n_rows, rank))
    V = rng.normal(0, 0.1, (n_cols, rank))
    eye = ridge * np.eye(rank)
    for _ in range(n_iter):
        for i in range(n_rows):
            m = observed[i]
            U[i] = np.linalg.solve(V[m].T @ V[m] + eye, V[m].T @ R[i, m]) if m.any() else 0.0
        for j in range(n_cols):
            m = observed[:, j]
            V[j] = np.linalg.solve(U[m].T @ U[m] + eye, U[m].T @ R[m, j]) if m.any() else 0.0
    return U, V


def _fit_mf(prob, train_cell, context, spec: RelationalSpec) -> np.ndarray:
    """Low-rank completion of the residual support matrix (organisations x questions).

    Residuals are taken from the shrunk group-by-question baseline. Decision-set columns are
    observed for training organisations only; context columns (if any) for everyone, which is what
    lets the model place a held-out organisation in the latent space.
    """
    rng = np.random.default_rng(spec.seed)
    base_cell, _, _ = _baseline_support(prob, train_cell, spec.shrink)
    qids, q_cell = np.unique(prob.cell_q, return_inverse=True)
    n_dec_q = len(qids)
    n_ctx = 0 if context is None else context.shape[1]
    R = np.zeros((prob.n_org, n_dec_q + n_ctx))
    observed = np.zeros_like(R, dtype=bool)
    R[prob.cell_org[train_cell], q_cell[train_cell]] = (prob.sigma - base_cell)[train_cell]
    observed[prob.cell_org[train_cell], q_cell[train_cell]] = True
    if n_ctx:
        seen = ~np.isnan(context)
        filled = np.where(seen, context, 0.0)
        col_mean = np.where(seen.any(0), filled.sum(0) / np.maximum(seen.sum(0), 1), 0.5)
        R[:, n_dec_q:] = np.where(seen, filled - col_mean[None, :], 0.0)
        observed[:, n_dec_q:] = seen
    U, V = _als(R, observed, spec.rank, spec.ridge, spec.n_iter, rng)
    fitted = (U @ V.T)[prob.cell_org, q_cell]
    sigma_hat = np.clip(base_cell + fitted, 0.0, 1.0)
    resid = (prob.sigma - sigma_hat)[train_cell]
    bandwidth = max(spec.min_bandwidth, float(np.sqrt(np.mean(resid ** 2))) if resid.size else 0.25)
    class_prior = (np.bincount(prob.stance_idx[train_cell], minlength=K) + 0.5) / (train_cell.sum() + 0.5 * K)
    return _support_to_distribution(sigma_hat, prob.sigma_vec, class_prior, bandwidth)


def _fit_gbdt(prob, train_cell, context, spec: RelationalSpec) -> np.ndarray:
    """Gradient-boosted classifier on (group, question, baselines, context answers)."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as err:  # pragma: no cover - exercised only without scikit-learn
        raise ImportError("the GBDT relational model needs scikit-learn (pip install -e .[ml])") from err
    base_cell, q_mean_cell, _ = _baseline_support(prob, train_cell, spec.shrink)
    g_onehot = np.eye(prob.n_groups)[prob.g_index[prob.cell_org]]
    _, q_cell = np.unique(prob.cell_q, return_inverse=True)
    feats = [g_onehot, q_cell[:, None].astype(float), base_cell[:, None], q_mean_cell[:, None]]
    if context is not None and context.shape[1]:
        seen = ~np.isnan(context)
        ctx_mean = np.where(seen.any(1), np.where(seen, context, 0.0).sum(1) / np.maximum(seen.sum(1), 1), np.nan)
        feats += [ctx_mean[prob.cell_org][:, None], seen.sum(1)[prob.cell_org][:, None].astype(float),
                  context[prob.cell_org]]
    X = np.column_stack(feats)
    # A context question that no training organisation answered gives an all-NaN training column:
    # it carries no information, and recent scikit-learn (seen on 1.9) crashes while binning it. Drop it.
    informative = ~np.isnan(X[train_cell]).all(0)
    X = X[:, informative]
    y = prob.stance_idx
    if train_cell.sum() < 2 or len(np.unique(y[train_cell])) < 2:
        return type_prior(prob, np.bincount(prob.cell_org[train_cell], minlength=prob.n_org) > 0).probs
    model = HistGradientBoostingClassifier(max_iter=spec.gbdt_max_iter, max_depth=spec.gbdt_max_depth,
                                           learning_rate=spec.gbdt_learning_rate, min_samples_leaf=5,
                                           random_state=spec.seed)
    model.fit(X[train_cell], y[train_cell])
    probs = np.zeros((prob.n_cells, K))
    probs[:, model.classes_] = model.predict_proba(X)
    probs = 0.98 * probs + 0.02 / K   # never rule a class out completely
    return probs / probs.sum(1, keepdims=True)


def relational(prob, train_orgs: np.ndarray, spec: RelationalSpec = None, context: np.ndarray = None,
               name: str = None) -> SimulatorOutput:
    """R3: predict every cell from the stances of the training organisations (and, in scenario S2,
    from every organisation's answers to the context questions)."""
    spec = spec or RelationalSpec()
    train_cell = np.asarray(train_orgs, dtype=bool)[prob.cell_org]
    if context is not None and context.shape[0] != prob.n_org:
        raise ValueError("context must have one row per organisation")
    if spec.model == "mf":
        probs = _fit_mf(prob, train_cell, context, spec)
    elif spec.model == "gbdt":
        probs = _fit_gbdt(prob, train_cell, context, spec)
    else:
        raise ValueError(f"unknown relational model {spec.model!r} (use 'mf' or 'gbdt')")
    meta = {"kind": "relational", "model": spec.model, "n_train_orgs": int(np.sum(train_orgs)),
            "uses_context": context is not None}
    return SimulatorOutput(name or f"r3_{spec.model}", probs, None, meta)


# ----------------------------------------------------------------------------------------------
# Common entry points
# ----------------------------------------------------------------------------------------------

FITTED_KINDS = ("type_prior", "relational")


def fit_predict(spec: dict, prob, train_orgs: np.ndarray, context: np.ndarray = None,
                name: str = None) -> SimulatorOutput:
    """Fit a pilot-dependent simulator described by a configuration entry (`simulators.<id>`)."""
    kind = spec.get("kind")
    if kind == "type_prior":
        return type_prior(prob, train_orgs, float(spec.get("alpha", 1.0)), name or "r0_type_prior")
    if kind == "relational":
        fields = {k: v for k, v in spec.items() if k in RelationalSpec.__dataclass_fields__}
        ctx = context if spec.get("use_context", True) else None
        return relational(prob, train_orgs, RelationalSpec(**fields), ctx, name)
    raise ValueError(f"simulator kind {kind!r} is not fitted on stances; expected one of {FITTED_KINDS}")


def stratified_folds(g_index: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Fold id per organisation, balanced within each group."""
    fold = np.zeros(len(g_index), dtype=int)
    for g in np.unique(g_index):
        idx = rng.permutation(np.flatnonzero(g_index == g))
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold


def cross_fit(spec: dict, prob, n_folds: int = 5, rng: np.random.Generator = None,
              context: np.ndarray = None, name: str = None) -> SimulatorOutput:
    """Out-of-fold predictions: each organisation is predicted by a model fitted on the other folds,
    so no prediction ever uses the organisation's own decision-set stances."""
    rng = rng if rng is not None else np.random.default_rng(0)
    fold = stratified_folds(prob.g_index, n_folds, rng)
    probs = np.zeros((prob.n_cells, K))
    cell_fold = fold[prob.cell_org]
    for f in range(n_folds):
        out = fit_predict(spec, prob, fold != f, context)
        held_out = cell_fold == f
        probs[held_out] = out.probs[held_out]
    meta = {"kind": spec.get("kind"), "cross_fitted": True, "n_folds": n_folds, **{k: v for k, v in spec.items()
                                                                                 if k != "kind"}}
    return SimulatorOutput(name or f"{spec.get('kind')}_cv", probs, None, meta)
