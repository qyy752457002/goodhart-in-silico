"""Simulator outputs: one stance distribution per responded cell.

Every simulator -- LLM persona, relational model, type prior, synthetic -- produces a
`SimulatorOutput` whose rows line up with the cells of a `Problem`. The rest of the pipeline
(metrics, Goodhart protocol, GUARD) only ever sees this object, so simulators are interchangeable.

Methodology map (03_methodology.md): section 1.1 (p-hat_iq, conditional on the organisation having
responded) and section 2.1 (the simulator family; `combine` is the ensemble "Ens").
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import STANCES

PROB_COLUMNS = [f"p_{s}" for s in STANCES]
AGGREGATE = -1  # value of the `member` column for the aggregate distribution


@dataclass(eq=False)
class SimulatorOutput:
    """Stance distributions aligned with a problem's cells.

    probs    (n_cells, K) the simulator's distribution p-hat_iq over the K stance classes.
    members  optional (n_members, n_cells, K): individual ensemble members or temperature samples,
             used for dispersion-based baselines (Ensemble-LCB).
    """
    name: str
    probs: np.ndarray
    members: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.probs = _check_probs(np.asarray(self.probs, dtype=float), f"{self.name}.probs")
        if self.members is not None:
            members = np.asarray(self.members, dtype=float)
            if members.ndim != 3 or members.shape[1:] != self.probs.shape:
                raise ValueError(f"{self.name}: members must have shape (n_members, {self.probs.shape})")
            self.members = np.stack([_check_probs(m, f"{self.name}.members[{j}]") for j, m in enumerate(members)])

    @property
    def n_cells(self) -> int:
        return self.probs.shape[0]

    @property
    def n_members(self) -> int:
        return 0 if self.members is None else self.members.shape[0]

    def expected_sigma(self, sigma_vec: np.ndarray) -> np.ndarray:
        """(n_cells,) simulated support E_{p-hat}[sigma(s)]."""
        return self.probs @ np.asarray(sigma_vec, dtype=float)

    def member_sigma(self, sigma_vec: np.ndarray) -> np.ndarray:
        """(n_members, n_cells) simulated support of each member."""
        if self.members is None:
            raise ValueError(f"simulator {self.name!r} has no ensemble members")
        return self.members @ np.asarray(sigma_vec, dtype=float)

    def top1(self) -> np.ndarray:
        """(n_cells,) most probable class (ties -> first class in `STANCES` order)."""
        return self.probs.argmax(1)

    def subset(self, cell_idx: np.ndarray, name: str = None) -> SimulatorOutput:
        members = None if self.members is None else self.members[:, cell_idx]
        return SimulatorOutput(name or self.name, self.probs[cell_idx], members, dict(self.meta))

    @staticmethod
    def from_labels(name: str, class_idx: np.ndarray, member_idx: np.ndarray = None,
                    meta: dict = None) -> SimulatorOutput:
        """Point predictions as one-hot distributions (member_idx: (n_members, n_cells))."""
        eye = np.eye(len(STANCES))
        members = None if member_idx is None else eye[np.asarray(member_idx, dtype=int)]
        return SimulatorOutput(name, eye[np.asarray(class_idx, dtype=int)], members, meta or {})

    @staticmethod
    def uniform(n_cells: int, name: str = "uniform") -> SimulatorOutput:
        K = len(STANCES)
        return SimulatorOutput(name, np.full((n_cells, K), 1.0 / K), None, {"kind": "uniform"})


def _check_probs(p: np.ndarray, label: str) -> np.ndarray:
    if p.ndim != 2 or p.shape[1] != len(STANCES):
        raise ValueError(f"{label}: expected shape (n_cells, {len(STANCES)}), got {p.shape}")
    if not np.isfinite(p).all() or (p < -1e-9).any():
        raise ValueError(f"{label}: probabilities must be finite and non-negative")
    total = p.sum(1)
    if (np.abs(total - 1.0) > 1e-6).any():
        raise ValueError(f"{label}: every row must sum to 1 (max deviation {np.abs(total - 1).max():.2e})")
    p = np.clip(p, 0.0, None)
    return p / p.sum(1, keepdims=True)


def combine(outputs: list, name: str = "ensemble", weights: np.ndarray = None,
            members: str = "components") -> SimulatorOutput:
    """Ensemble of simulators ("Ens"): the (weighted) mean distribution.

    members='components'  each component's distribution is one ensemble member;
    members='all'         additionally pools the components' own members (temperature samples).
    """
    if not outputs:
        raise ValueError("cannot combine an empty list of simulators")
    stack = np.stack([o.probs for o in outputs])
    w = np.full(len(outputs), 1.0 / len(outputs)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    probs = np.tensordot(w, stack, axes=1)
    pool = [stack]
    if members == "all":
        pool += [o.members for o in outputs if o.members is not None]
    elif members != "components":
        raise ValueError(f"unknown members mode {members!r}")
    meta = {"kind": "ensemble", "components": [o.name for o in outputs], "weights": w.tolist()}
    return SimulatorOutput(name, probs, np.concatenate(pool), meta)


# ----------------------------------------------------------------------------------------------
# Persistence: a long table keyed by (org, qid), so files survive re-ordering of the cells
# ----------------------------------------------------------------------------------------------

def to_frame(out: SimulatorOutput, prob) -> pd.DataFrame:
    if out.n_cells != prob.n_cells:
        raise ValueError(f"{out.name}: {out.n_cells} rows but the problem has {prob.n_cells} cells")
    keys = pd.DataFrame(prob.cell_keys, columns=["org", "qid"])
    parts = [keys.assign(member=AGGREGATE, **dict(zip(PROB_COLUMNS, out.probs.T)))]
    for j in range(out.n_members):
        parts.append(keys.assign(member=j, **dict(zip(PROB_COLUMNS, out.members[j].T))))
    return pd.concat(parts, ignore_index=True)


def save_output(out: SimulatorOutput, prob, path: str) -> str:
    """Write `<path>` (parquet) and `<path>.meta.json`. Returns the parquet path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    to_frame(out, prob).to_parquet(path, index=False)
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"name": out.name, "n_cells": out.n_cells, "n_members": out.n_members,
                   "classes": STANCES, **out.meta}, f, indent=2, ensure_ascii=False, default=str)
    return path


def from_frame(frame: pd.DataFrame, prob, name: str, meta: dict = None,
               on_missing: str = "error") -> SimulatorOutput:
    """Align a long table with the problem's cells.

    on_missing: 'error' (default) or 'uniform' -- fill cells the simulator did not cover with the
    uniform distribution and record their number in meta['n_missing_cells'].
    """
    meta = dict(meta or {})
    keys = pd.MultiIndex.from_tuples(prob.cell_keys, names=["org", "qid"])
    K = len(STANCES)

    def align(sub: pd.DataFrame) -> tuple[np.ndarray, int]:
        sub = sub.set_index(["org", "qid"])
        if sub.index.duplicated().any():
            raise ValueError(f"{name}: duplicated (org, qid) rows")
        arr = sub.reindex(keys)[PROB_COLUMNS].to_numpy(dtype=float)
        missing = np.isnan(arr).any(1)
        if missing.any():
            if on_missing != "uniform":
                raise ValueError(f"{name}: {int(missing.sum())} of {len(keys)} cells have no prediction "
                                 f"(first missing: {keys[int(np.argmax(missing))]})")
            arr[missing] = 1.0 / K
        return arr, int(missing.sum())

    probs, n_missing = align(frame[frame["member"] == AGGREGATE])
    member_ids = sorted(m for m in frame["member"].unique() if m != AGGREGATE)
    members = np.stack([align(frame[frame["member"] == m])[0] for m in member_ids]) if member_ids else None
    if n_missing:
        meta["n_missing_cells"] = n_missing
    return SimulatorOutput(name, probs, members, meta)


def load_output(path: str, prob, name: str = None, on_missing: str = "error") -> SimulatorOutput:
    if not os.path.exists(path):
        raise FileNotFoundError(f"simulator output not found: {path}")
    meta = {}
    if os.path.exists(path + ".meta.json"):
        with open(path + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
    name = name or meta.get("name") or os.path.splitext(os.path.basename(path))[0]
    return from_frame(pd.read_parquet(path), prob, name, meta, on_missing)
