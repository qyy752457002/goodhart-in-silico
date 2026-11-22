"""Registry of decision rules and certificates, evaluated for one pilot draw.

`RepContext` holds one (engine, pilot) pair and builds each affine system / table at most once, so
that all rules and objectives of a replication share the work.

A rule maps (context, objective) to a `Selection`: the chosen package and, where the rule optimises
an estimate of welfare, the (M, G) table of that estimate (used to report the overestimate).

Methodology map (03_methodology.md): section 4.4 (GUARD variants) and section 5 (baselines).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from . import baselines, certify, guard, welfare
from .engine import DesignEngine
from .pilot import Pilot


@dataclass(frozen=True, eq=False)
class Selection:
    index: int
    est_table: np.ndarray | None = None   # (M, G) estimate optimised by the rule, if any
    extra: dict | None = None             # rule-specific diagnostics


class RepContext:
    """Lazily built systems, tables and bounds for one pilot draw."""

    def __init__(self, engine: DesignEngine, pilot: Pilot, rng: np.random.Generator = None):
        self.engine = engine
        self.pilot = pilot
        rng = rng if rng is not None else np.random.default_rng(0)
        self._seed = int(rng.integers(0, 2 ** 63 - 1))
        self._cache = {}

    def _get(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def stream(self, name: str) -> np.random.Generator:
        """A random stream of its own for each randomised component (pooled order, HedgeTune draw),
        so that what one rule draws never depends on which other rules are evaluated."""
        return self._get(("stream", name),
                         lambda: np.random.default_rng([self._seed, zlib.crc32(name.encode("utf-8"))]))

    # ------------------------------------------------------------------ building blocks
    def bounds(self, kind: str = None) -> certify.GroupBounds:
        kind = kind or self.engine.settings.ucb
        return self._get(("bounds", kind), lambda: guard.certificate_bounds(
            self.engine, self.pilot, kind, self.stream("pooled") if kind == "pooled" else None))

    def system(self, name: str) -> welfare.Affine:
        builders = {
            "pilot_sim": lambda: baselines.pilot_sim_system(self.engine, self.pilot),
            "pilot_only": lambda: baselines.pilot_only_system(self.engine, self.pilot),
            "rectified": lambda: guard.rectified_system(self.engine, self.pilot),
        }
        if name not in builders:
            raise ValueError(f"unknown system {name!r}")
        return self._get(("system", name), builders[name])

    def table(self, name: str) -> np.ndarray:
        """(M, G) table of group values at every package."""
        special = {
            "pilot_ens_lcb": lambda: baselines.pilot_ensemble_lcb_table(self.engine, self.pilot),
            "lower": lambda: guard.lower_table(self.engine, self.bounds()),
            "lower_est": lambda: guard.lower_table(self.engine, self.bounds("est")),
        }
        return self._get(("table", name), special.get(name, lambda: self.system(name).table()))

    def argmax(self, name: str, objective: str) -> int:
        return self._get(("argmax", name, objective),
                         lambda: self.engine.space.argmax(self.engine.values(self.table(name), objective)))

    def cons_region(self) -> dict:
        return self._get("cons_region", lambda: guard.conservative_region(self.engine, self.pilot))

    # ------------------------------------------------------------------ certificates
    def certificates(self, k: int, objective: str, kinds=guard.CERTIFICATE_KINDS) -> dict:
        """Every certificate evaluated at package k. By Theorem 1 the guaranteed ones are valid for
        any package, so they can be attached to any rule's selection."""
        e = self.engine
        a = e.space.package(k).astype(float)
        p = welfare.objective_p(objective)
        true_val = float(e.true_values(objective)[int(k)])
        out = {}
        for kind in kinds:
            c = self.bounds(kind).certificate(a, p, e._w(), e.settings.clip_trivial)
            out[f"cert_{kind}"] = c
            out[f"covered_{kind}"] = bool(true_val >= c - 1e-12)
        if e.member_org:
            c = e.value_at(baselines.ensemble_tables(e)["lcb"][int(k)], objective)
            out["cert_ens"] = c
            out["covered_ens"] = bool(true_val >= c - 1e-12)
        return out

    def pilot_record(self, kinds=guard.CERTIFICATE_KINDS) -> dict:
        """Pilot-level diagnostics: certified mis-set mass, uniform validity (the event of
        Theorem 1) and, per objective, the width that bounds GUARD-pess regret (Theorem 2)."""
        e = self.engine
        rec = {}
        counts = self.pilot.group_counts(e.gi, e.G)
        for g, name in enumerate(e.prob.group_names):
            rec[f"n_known[{name}]"] = int(counts["n_known"][g])
            rec[f"n_calib[{name}]"] = int(counts["n_calib"][g])
        true_tau = np.zeros(e.G)
        for g in range(e.G):
            true_tau[g] = e.s_minus[(e.gi == g) & self.pilot.target].sum()
        for kind in kinds:
            b = self.bounds(kind)
            valid = certify.uniform_validity(b, e.true_group)
            rec[f"uniform_valid_{kind}"] = valid["lower_valid"] and valid["upper_valid"]
            rec[f"lower_valid_{kind}"] = valid["lower_valid"]
            for g, name in enumerate(e.prob.group_names):
                rec[f"tau_{kind}[{name}]"] = float(b.tau_minus[g])
        for g, name in enumerate(e.prob.group_names):
            rec[f"tau_true[{name}]"] = float(true_tau[g])
        main = self.bounds()
        for objective in welfare.OBJECTIVES:
            a_star = e.space.package(e.oracle_index(objective)).astype(float)
            clip = e.settings.clip_trivial
            rec[f"width_at_oracle[{objective}]"] = float(
                (main.upper_at(a_star, clip) - main.lower_at(a_star, clip)).max())
        return rec


# ----------------------------------------------------------------------------------------------
# Decision rules
# ----------------------------------------------------------------------------------------------

def _fixed(index_of):
    return lambda ctx, obj: Selection(index_of(ctx.engine, obj))


def _from_system(name):
    return lambda ctx, obj: Selection(ctx.argmax(name, obj), ctx.table(name))


def _sim_plugin(ctx, obj):
    return Selection(ctx.engine.plugin_index(obj), ctx.engine.W_sim)


def _board(ctx, obj):
    if ctx.engine.board_index is None:
        raise ValueError("the Board's package is not available for this testbed")
    return Selection(ctx.engine.board_index)


def _ens(which):
    def rule(ctx, obj):
        table = baselines.ensemble_tables(ctx.engine)[which]
        key = ("ens_argmax", which, obj)
        if key not in ctx.engine._cache:
            ctx.engine._cache[key] = ctx.engine.space.argmax(ctx.engine.values(table, obj))
        return Selection(ctx.engine._cache[key], table)
    return rule


def _hedgetune(ctx, obj):
    e = ctx.engine
    est = "pilot_only" if e.settings.hedgetune_estimate == "pilot_only" else "rectified"
    k, log2n = baselines.hedgetune_select(e, obj, e.values(ctx.table(est), obj), ctx.stream(f"hedgetune/{obj}"))
    return Selection(k, e.W_sim, {"hedgetune_log2n": log2n})


def _guard_cons(ctx, obj):
    return Selection(guard.conservative_select(ctx.engine, ctx.pilot, obj, ctx.cons_region()),
                     ctx.table("rectified"))


METHODS = {
    # fixed packages
    "ed": _fixed(lambda e, obj: e.space.ed_index),
    "ifrs": _fixed(lambda e, obj: e.space.ifrs_index),
    "oracle": _fixed(lambda e, obj: e.oracle_index(obj)),
    "board": _board,
    # baselines
    "sim_plugin": _sim_plugin,
    "pilot_only": _from_system("pilot_only"),
    "pilot_sim_plugin": _from_system("pilot_sim"),
    "ens_lcb": _ens("lcb"),
    "ens_wco": _ens("wco"),
    "pilot_ens_lcb": lambda ctx, obj: Selection(ctx.argmax("pilot_ens_lcb", obj), ctx.table("pilot_ens_lcb")),
    "hedgetune": _hedgetune,
    # GUARD and its ablations
    "guard": _from_system("rectified"),
    "guard_pess": _from_system("lower"),
    "guard_est": _from_system("lower_est"),
    "guard_cons": _guard_cons,
}

NEEDS_MEMBERS = {"ens_lcb", "ens_wco", "pilot_ens_lcb"}
NEEDS_BOARD = {"board"}
PILOT_FREE = {"ed", "ifrs", "oracle", "board", "sim_plugin", "ens_lcb", "ens_wco"}

METHOD_LABELS = {
    "ed": "ED (retain all)", "ifrs": "IFRS baseline (revise all)", "oracle": "Oracle",
    "board": "Board's decision", "sim_plugin": "Simulator plug-in", "pilot_only": "Pilot only",
    "pilot_sim_plugin": "Pilot + simulator plug-in", "ens_lcb": "Ensemble-LCB",
    "ens_wco": "Ensemble worst case", "pilot_ens_lcb": "Pilot + Ensemble-LCB",
    "hedgetune": "HedgeTune", "guard": "GUARD (rectify)", "guard_pess": "GUARD-pess",
    "guard_est": "GUARD-est", "guard_cons": "GUARD-cons",
}


def available_methods(engine: DesignEngine, requested=None) -> list:
    """Requested rules that can run on this engine (drops ensemble rules without members and the
    Board diagnostic without a Board package). Unknown names raise."""
    requested = list(METHODS) if requested is None else list(requested)
    unknown = [m for m in requested if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {sorted(METHODS)}")
    out = []
    for m in requested:
        if m in NEEDS_MEMBERS and not engine.member_org:
            continue
        if m in NEEDS_BOARD and engine.board_index is None:
            continue
        out.append(m)
    return out


def run_replication(engine: DesignEngine, pilot: Pilot, rng: np.random.Generator, methods: list,
                    objectives: list, certificate_kinds=guard.CERTIFICATE_KINDS) -> tuple[list, dict]:
    """Evaluate every rule under every objective for one pilot.

    Returns (rows, pilot_record): one row per (objective, method) with the truth-based metrics of
    the selected package and every certificate evaluated at it.
    """
    ctx = RepContext(engine, pilot, rng)
    rows = []
    for objective in objectives:
        for method in methods:
            sel = METHODS[method](ctx, objective)
            row = {"objective": objective, "method": method}
            row.update(engine.evaluate(sel.index, objective, sel.est_table))
            row.update(ctx.certificates(sel.index, objective, certificate_kinds))
            if sel.extra:
                row.update(sel.extra)
            rows.append(row)
    return rows, ctx.pilot_record(certificate_kinds)
