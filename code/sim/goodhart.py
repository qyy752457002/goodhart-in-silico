"""Goodhart measurement protocol (RQ2).

How far does the simulated welfare of the selected package exceed its true welfare, how does the
gap grow with optimisation pressure, and which group carries it?

Methodology map (03_methodology.md): section 1.2 (measures) and section 3 (protocol).
  optimisation pressure   `pressure_curve`   best-of-N for log2 N = 0, 2, ..., Q (N = 2^Q is exact)
  Hamming-radius axis     `hamming_curve`    exact optimum within radius r of the ED package
  group decomposition     `DesignEngine.evaluate` (delta_g, illusory fairness, worst group)
  curve summaries         `ndh` (normalised drop height) and `fit_bon` (the d(alpha - beta d) form)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimize
from .engine import DesignEngine


def per_package_quantities(engine: DesignEngine, objective: str) -> dict:
    """Arrays over all packages of everything the protocol tracks for the selected package."""
    e = engine
    sim_val, true_val = e.sim_values(objective), e.true_values(objective)
    delta = e.W_sim - e.W_true
    true_worst = e.W_true.argmin(1)
    out = {
        "sim": sim_val,
        "true": true_val,
        "overestimate": sim_val - true_val,                    # Delta(a)
        "regret": true_val.max() - true_val,                   # R(a)
        "illusory_fairness": e.W_sim.min(1) - e.W_true.min(1),  # IF(a)
        "true_min_group": e.W_true.min(1),
        "n_revised": (e.Q - e.space.popcount).astype(float),
        "worst_is_most_overestimated": (true_worst == delta.argmax(1)).astype(float),
    }
    for g, name in enumerate(e.prob.group_names):
        out[f"delta[{name}]"] = delta[:, g]
        out[f"p_true_worst[{name}]"] = (true_worst == g).astype(float)
    return out


def pressure_curve(engine: DesignEngine, objective: str, log2_n: list = None, reps: int = 2000,
                   rng: np.random.Generator = None) -> pd.DataFrame:
    """Goodhart curve: one row per optimisation pressure N = 2^j.

    For each N the law of the selected package under best-of-N is known exactly
    (`optimize.best_of_n_pmf`), so every quantity gets
      *_exact               its exact expectation, and
      *_mean, *_lo, *_hi    the Monte-Carlo mean over `reps` best-of-N draws with a 95% normal
                            interval (the protocol in the methodology; reps = 0 skips it).

    Pressure axes: `log2_n`; `d` = sqrt(`kl_bon`) with `kl_bon` the exact KL divergence of the
    selection law from uniform; `d_gao` = sqrt(`kl_gao`) with the nominal formula log N - (N-1)/N
    of Gao et al. (2023), which understates the pressure as N approaches the number of packages.
    """
    e = engine
    rng = rng if rng is not None else np.random.default_rng(0)
    grid = sorted(set(int(j) for j in (log2_n if log2_n is not None else range(0, e.Q + 1, 2)))
                  | {e.Q})
    order = e.sim_order(objective)
    quantities = per_package_quantities(e, objective)
    rows = []
    for j in grid:
        if not 0 <= j <= e.Q:
            raise ValueError(f"log2 N must lie in 0..{e.Q}, got {j}")
        pmf = e.best_of_n_pmf(j)
        ranked = order[: len(pmf)]
        kl = optimize.kl_selection(pmf, e.space.size)            # exact for this selection law
        kl_gao = float(optimize.kl_best_of_n(2 ** j))            # nominal formula of Gao et al.
        row = {"objective": objective, "log2_n": j, "n": 2 ** j, "kl_bon": kl, "d": float(np.sqrt(max(kl, 0.0))),
               "kl_gao": kl_gao, "d_gao": float(np.sqrt(kl_gao))}
        draws = rng.choice(len(pmf), size=reps, p=pmf) if reps > 0 else None
        for name, values in quantities.items():
            v = values[ranked]
            row[f"{name}_exact"] = float(pmf @ v)
            if draws is not None:
                s = v[draws]
                half = 1.96 * s.std(ddof=1) / np.sqrt(reps) if reps > 1 else 0.0
                row[f"{name}_mean"] = float(s.mean())
                row[f"{name}_lo"] = float(s.mean() - half)
                row[f"{name}_hi"] = float(s.mean() + half)
        rows.append(row)
    return pd.DataFrame(rows)


def hamming_curve(engine: DesignEngine, objective: str) -> pd.DataFrame:
    """Auxiliary pressure axis: the simulator's exact optimum within Hamming radius r of the ED."""
    e = engine
    idx = optimize.hamming_ball_argmax(e.space, e.sim_values(objective))
    rows = []
    for r, k in enumerate(idx):
        rows.append({"objective": objective, "radius": r, **e.evaluate(int(k), objective, e.W_sim)})
    return pd.DataFrame(rows)


def plugin_summary(engine: DesignEngine, objective: str) -> dict:
    """The plug-in design at full pressure, with the terms of Proposition A.

    Utilitarian: the overestimate equals sum_g w_g delta_g. Max-min: it is at most max_g delta_g.
    """
    e = engine
    k = e.plugin_index(objective)
    out = {"objective": objective, **e.evaluate(k, objective, e.W_sim)}
    delta = e.W_sim[k] - e.W_true[k]
    out["weighted_delta"] = float(delta @ e.weights)
    out["max_delta"] = float(delta.max())
    out["oracle_welfare"] = float(e.true_values(objective).max())
    out["ed_welfare"] = float(e.true_values(objective)[e.space.ed_index])
    out["revised"] = ",".join(d for d, a in zip(e.prob.decision_labels(), e.space.package(k)) if a == 0) or "-"
    return out


def ndh(true_curve, true_range: float) -> float:
    """Normalised drop height (after Karwowski et al., 2024): how much true welfare is lost by
    optimising past the best point of the curve, as a share of the attainable range.

    `true_curve` is the expected true welfare in order of increasing pressure; `true_range` is
    max_a W_p(a) - min_a W_p(a). NDH > 0 means the curve rises and then falls (Goodharting).
    """
    curve = np.asarray(true_curve, dtype=float)
    if curve.size == 0 or true_range <= 0:
        return 0.0
    return float((curve.max() - curve[-1]) / true_range)


def fit_bon(d, gain) -> dict:
    """Least-squares fit of the best-of-N functional form of Gao et al. (2023),
    gain(d) = d (alpha - beta d), with d = sqrt(KL_bon). alpha is the initial slope ("regressional"
    part), beta the curvature ("extremal" part); the fitted curve peaks at d = alpha / (2 beta)."""
    d, gain = np.asarray(d, dtype=float), np.asarray(gain, dtype=float)
    X = np.column_stack([d, -d * d])
    coef, *_ = np.linalg.lstsq(X, gain, rcond=None)
    pred = X @ coef
    ss_tot = float(((gain - gain.mean()) ** 2).sum())
    r2 = 1.0 - float(((gain - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    alpha, beta = float(coef[0]), float(coef[1])
    return {"alpha": alpha, "beta": beta, "r2": r2, "d_peak": alpha / (2 * beta) if beta > 0 else np.inf}


def curve_summary(engine: DesignEngine, curve: pd.DataFrame) -> dict:
    """NDH and the fitted functional forms for one objective's pressure curve."""
    objective = curve["objective"].iloc[0]
    true_vals = engine.true_values(objective)
    curve = curve.sort_values("log2_n")
    d = curve["d"].to_numpy()
    out = {"objective": objective, "ndh": ndh(curve["true_exact"], float(true_vals.max() - true_vals.min()))}
    for which in ("true", "sim"):
        y = curve[f"{which}_exact"].to_numpy()
        fit = fit_bon(d, y - y[0])
        out.update({f"{which}_{k}": v for k, v in fit.items()})
    peak = curve.loc[curve["true_exact"].idxmax()]
    out["best_log2_n"] = int(peak["log2_n"])
    out["overestimate_at_full_pressure"] = float(curve["overestimate_exact"].iloc[-1])
    return out
