"""RQ3 -- rectify-then-certify: can a small pilot repair the decision and certify it, and how large
must the pilot be?

Protocol (methodology section 7): for every pilot fraction rho, draw `reps` stratified random
pilots; for each pilot apply every decision rule under every objective and evaluate the selected
package on the held-out truth; attach every certificate to every selection (Theorem 1 makes the
guaranteed ones valid for any rule). The same pilots are used for all simulators and rules, so
comparisons are paired.

Outputs:
  runs.parquet          one row per (simulator, rho, rep, objective, method)
  pilots.parquet        one row per (simulator, rho, rep): certified mass, uniform validity
  tab2_methods.csv      Table 2: true welfare, regret, worst-group welfare, coverage, tightness
  coverage.csv          empirical P(W_p(a-hat) >= c) per certificate, with exact binomial intervals
  pairwise_tests.csv    paired Wilcoxon tests against the reference method, Holm-corrected
  pilot_size_curve.csv  Fig. 4: certificate and true welfare as functions of rho
  uniform_validity.csv  how often the bounds hold for all packages at once (the event of Theorem 1)
"""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

from .. import config as config_mod
from .. import methods as methods_mod
from .. import stats, utils
from ..optimize import PackageSpace
from ..pilot import draw_pilot, tilt_weights
from ..simulators import fitted, registry
from . import common

log = logging.getLogger(__name__)

GUARANTEED = {"betting": True, "hoeffding_serfling": True, "count": True, "pooled": False, "ens": False}
MEAN_COLUMNS = ("true_min_group", "n_revised", "overestimate", "illusory_fairness", "hamming_to_oracle",
                "hamming_to_board", "worst_is_most_overestimated", "hedgetune_log2n")


def pilot_weights(pilot_cfg: dict, engine, prob):
    """Sampling weights for a biased pilot design, or None for simple random sampling."""
    design = pilot_cfg.get("design", "srs")
    if design == "srs":
        return None
    if design != "tilted":
        raise ValueError(f"unknown pilot design {design!r} (use 'srs' or 'tilted')")
    by = pilot_cfg.get("tilt_by", "misset")
    if by == "misset":
        if engine is None:
            raise ValueError("a pilot tilted by mis-set mass needs a pilot-independent simulator")
        score = engine.s_minus
    elif by == "support":
        score = prob.sigma  # mean true support for the ED per organisation
        score = np.bincount(prob.cell_org, weights=score, minlength=prob.n_org) / prob.cells_per_org
    elif by == "n_responses":
        score = prob.cells_per_org.astype(float)
    else:
        raise ValueError(f"unknown tilt_by {by!r}")
    return tilt_weights(np.asarray(score, dtype=float), prob.g_index, prob.n_groups, float(pilot_cfg.get("gamma", 0)))


def _run_tasks(cfg, tb, sim_id: str, tasks: list, *, methods: list, certificates: list, static_sim, space,
               seed_keys: tuple, tags: dict, settings, pilot_cfg: dict, fit_fraction: float) -> tuple[list, list]:
    """The replications listed in `tasks` = [(rho index, rho, rep)], in order."""
    prob = tb.prob
    seed = int(cfg["project"]["seed"])
    objs = common.objectives(cfg)
    space = space or PackageSpace(prob.n_dec)
    settings = settings or common.engine_settings(cfg)
    spec = None if static_sim is not None else registry.get_spec(cfg["simulators"], sim_id)
    is_fitted = spec is not None and registry.is_fitted(spec)

    engine = None
    if not is_fitted:
        sim = static_sim if static_sim is not None else registry.load_static(
            sim_id, cfg["simulators"], prob, common.simulated_dir(cfg), tb.name)
        engine = common.make_engine(cfg, tb, sim, space, settings)
    weights = pilot_weights(pilot_cfg, engine, prob)
    kinds = tuple(certificates)
    rows, pilot_rows = [], []
    use = None
    for r_i, rho, rep in tasks:
        rng = utils.rep_rng(seed, *seed_keys, r_i, rep)
        pilot = draw_pilot(prob.g_index, prob.n_groups, float(rho), rng,
                           fit_fraction=fit_fraction if is_fitted else 0.0, weights=weights)
        if is_fitted:
            sim = fitted.fit_predict(spec, prob, pilot.fit, tb.context, name=sim_id)
            engine = common.make_engine(cfg, tb, sim, space, settings)
        if use is None:
            use = methods_mod.available_methods(engine, methods)
            dropped = [m for m in methods if m not in use]
            if dropped and (r_i, rep) == (0, 0):
                log.info("rq3: %s cannot run %s (no ensemble members / no Board package)", sim_id, dropped)
        rep_rows, record = methods_mod.run_replication(engine, pilot, rng, use, objs, kinds)
        for row in rep_rows:
            obj = row["objective"]
            row.update({"simulator": sim_id, "rho": float(rho), "rep": rep,
                        "oracle_welfare": float(engine.true_values(obj)[engine.oracle_index(obj)]),
                        "ed_welfare": float(engine.true_values(obj)[engine.space.ed_index]), **tags})
        rows.extend(rep_rows)
        pilot_rows.append({"simulator": sim_id, "rho": float(rho), "rep": rep, **tags, **record})
    return rows, pilot_rows


_WORKER_TESTBED = {}  # per worker process: {testbed key: Testbed}


def _worker(payload: dict, tasks: list) -> tuple[list, list]:
    """Process-pool entry point: rebuild the testbed from the configuration (once per process)."""
    cfg = config_mod.Config(payload["cfg"])
    key = json.dumps({k: payload["cfg"][k] for k in ("testbed", "welfare")}, sort_keys=True, default=str)
    if key not in _WORKER_TESTBED:
        _WORKER_TESTBED.clear()
        _WORKER_TESTBED[key] = common.load_testbed(cfg)
    return _run_tasks(cfg, _WORKER_TESTBED[key], payload["sim_id"], tasks, **payload["kwargs"])


@contextlib.contextmanager
def pilot_pool(cfg, n_jobs: int = None):
    """A process pool to share between several `run_pilots` calls; yields None when
    `project.n_jobs` (or `n_jobs`) is 1, in which case everything runs in this process."""
    n = int(n_jobs if n_jobs is not None else cfg["project"].get("n_jobs", 1))
    if n <= 1:
        yield None
        return
    with ProcessPoolExecutor(max_workers=n) as pool:
        pool.n_workers = n
        yield pool


def run_pilots(cfg, tb, sim_id: str, *, rhos: list, reps: int, methods: list, certificates: list,
               static_sim=None, space: PackageSpace = None, seed_keys: tuple = (), tags: dict = None,
               settings=None, pilot_cfg: dict = None, fit_fraction: float = 0.5,
               n_jobs: int = None, pool=None) -> tuple[list, list]:
    """Replicated pilots for one simulator. Returns (run rows, pilot rows).

    static_sim: a ready `SimulatorOutput` (synthetic experiments); otherwise `sim_id` is looked up
    in the configuration. Simulators fitted on stances are refitted on the fit part of each pilot.

    Parallelism: pass a pool from `pilot_pool(cfg)` (shared across calls), or set n_jobs (default
    `project.n_jobs`) to let this call start its own. Every replication draws from its own random
    stream keyed by (seed, seed_keys, rho index, rep), so the output is identical however the
    work is split. Workers rebuild the testbed from `cfg`, so `tb` must be what
    `common.load_testbed(cfg)` returns.
    """
    kwargs = {"methods": list(methods), "certificates": list(certificates), "static_sim": static_sim,
              "seed_keys": tuple(seed_keys), "tags": dict(tags or {}), "settings": settings,
              "pilot_cfg": dict(pilot_cfg if pilot_cfg is not None else cfg["rq3"]["pilot"]),
              "fit_fraction": float(fit_fraction)}
    tasks = [(r_i, float(rho), rep) for r_i, rho in enumerate(rhos) for rep in range(int(reps))]

    def in_workers(workers) -> tuple[list, list]:
        try:
            return _map_tasks(workers, cfg, sim_id, tasks, kwargs)
        except (BrokenProcessPool, MemoryError) as err:
            # e.g. a worker could not start, or ran out of memory: one process needs far less
            log.warning("worker processes failed (%s: %s); running %d replications in this process instead",
                        type(err).__name__, err, len(tasks))
            return _run_tasks(cfg, tb, sim_id, tasks, space=space, **kwargs)

    if pool is None:
        with pilot_pool(cfg, n_jobs) as own:
            if own is None or len(tasks) < 2 * own.n_workers:
                return _run_tasks(cfg, tb, sim_id, tasks, space=space, **kwargs)
            return in_workers(own)
    return in_workers(pool)


def _map_tasks(pool, cfg, sim_id: str, tasks: list, kwargs: dict) -> tuple[list, list]:
    """Split the replications into contiguous chunks and run them on the pool, keeping their order."""
    edges = np.linspace(0, len(tasks), min(len(tasks), 4 * pool.n_workers) + 1).astype(int)
    chunks = [tasks[a:b] for a, b in zip(edges[:-1], edges[1:]) if b > a]
    payload = {"cfg": copy.deepcopy(dict(cfg)), "sim_id": sim_id, "kwargs": {**kwargs, "space": None}}
    rows, pilot_rows = [], []
    for part_rows, part_pilots in pool.map(_worker, [payload] * len(chunks), chunks):
        rows.extend(part_rows)
        pilot_rows.extend(part_pilots)
    return rows, pilot_rows


# ----------------------------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------------------------

def summarise_methods(runs: pd.DataFrame, keys: list, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Table 2: one row per `keys` combination with means, bootstrap intervals and, per certificate,
    coverage, mean value and mean slack (true welfare minus certificate)."""
    kinds = [c[len("cert_"):] for c in runs.columns if c.startswith("cert_")]
    out = []
    for j, (key, grp) in enumerate(runs.groupby(keys, sort=False, dropna=False)):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row["n"] = len(grp)
        for col in ("true_welfare", "regret"):
            ci = stats.mean_ci(grp[col], n_boot, rng=utils.rep_rng(seed, 7, j, len(col)))
            row[col] = ci["mean"]
            row[f"{col}_lo"], row[f"{col}_hi"] = ci["ci_lo"], ci["ci_hi"]
        for col in MEAN_COLUMNS:
            if col in grp.columns and grp[col].notna().any():
                row[col] = float(grp[col].astype(float).mean())
        ed = float(grp["ed_welfare"].mean())
        gain = float(grp["oracle_welfare"].mean()) - ed
        # share of the oracle's gain over the ED that the rule recovers (undefined if the ED is optimal)
        row["recovered_share"] = (row["true_welfare"] - ed) / gain if gain > 1e-12 else np.nan
        for kind in kinds:
            cert = grp[f"cert_{kind}"].astype(float)
            if cert.notna().any():
                row[f"coverage_{kind}"] = float(grp[f"covered_{kind}"].astype(float).mean())
                row[f"cert_{kind}"] = float(cert.mean())
                row[f"slack_{kind}"] = float((grp["true_welfare"] - cert).mean())
        out.append(row)
    return pd.DataFrame(out)


def coverage_table(runs: pd.DataFrame, keys: list, delta: float) -> pd.DataFrame:
    """Long table: empirical coverage of every certificate with an exact (Clopper-Pearson) interval."""
    kinds = [c[len("cert_"):] for c in runs.columns if c.startswith("cert_")]
    out = []
    for key, grp in runs.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for kind in kinds:
            covered = grp[f"covered_{kind}"].dropna().astype(bool)
            if covered.empty:
                continue
            k, n = int(covered.sum()), int(len(covered))
            lo, hi = stats.clopper_pearson(k, n)
            out.append({**dict(zip(keys, key)), "certificate": kind, "guaranteed": GUARANTEED.get(kind, False),
                        "coverage": k / n, "coverage_lo": lo, "coverage_hi": hi, "n": n, "nominal": 1.0 - delta,
                        "below_nominal": hi < 1.0 - delta,
                        "mean_certificate": float(grp[f"cert_{kind}"].mean()),
                        "mean_slack": float((grp["true_welfare"] - grp[f"cert_{kind}"]).mean())})
    return pd.DataFrame(out)


def pairwise_tests(runs: pd.DataFrame, keys: list, reference: str, rep_cols: list,
                   metric: str = "true_welfare") -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests of `reference` against every other method on the same
    pilots, with Holm correction inside each `keys` family."""
    out = []
    for key, grp in runs.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        wide = grp.pivot_table(index=rep_cols, columns="method", values=metric, aggfunc="first")
        if reference not in wide.columns:
            continue
        family = []
        for method in wide.columns:
            if method == reference:
                continue
            pair = wide[[reference, method]].dropna()
            family.append({**dict(zip(keys, key)), "reference": reference, "method": method, "metric": metric,
                           **stats.paired_wilcoxon(pair[reference], pair[method])})
        if family:
            adjusted = stats.holm([f["p"] for f in family])
            for f, p_adj in zip(family, adjusted):
                f["p_holm"] = float(p_adj)
            out.extend(family)
    return pd.DataFrame(out)


def uniform_validity_table(pilots: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Share of pilots on which L_g(a) <= W_g(a) <= U_g(a) held for every package and group."""
    cols = [c for c in pilots.columns if c.startswith(("uniform_valid_", "lower_valid_"))]
    out = pilots.groupby(keys, sort=False, dropna=False)[cols].mean().reset_index()
    out["n"] = pilots.groupby(keys, sort=False, dropna=False).size().to_numpy()
    return out


def run(cfg, out: str = None) -> dict:
    tb = common.load_testbed(cfg)
    rq = cfg["rq3"]
    out_dir = utils.results_dir(cfg, "rq3", out)
    space = PackageSpace(tb.prob.n_dec)
    rows, pilot_rows, done = [], [], []
    with pilot_pool(cfg) as pool:
        for sim_id in rq["simulators"]:
            timer = utils.Timer()
            try:
                r, p = run_pilots(cfg, tb, sim_id, rhos=rq["rhos"], reps=int(rq["reps"]), methods=rq["methods"],
                                  certificates=rq["certificates"], space=space, seed_keys=(5,),
                                  fit_fraction=float(rq["fit_fraction"]), pool=pool)
            except registry.MissingSimulatorOutput as err:
                if cfg.get("on_missing_simulator", "error") != "skip":
                    raise
                log.warning("rq3: skipping %s: %s", sim_id, err)
                continue
            rows.extend(r)
            pilot_rows.extend(p)
            done.append(sim_id)
            log.info("rq3: %s done (%d rows, %.0fs)", sim_id, len(r), timer.seconds())
    if not rows:
        raise RuntimeError("rq3: no simulator could be run (precompute the persona simulators first)")
    runs, pilots = pd.DataFrame(rows), pd.DataFrame(pilot_rows)
    tables = summarise(runs, pilots, cfg, rq["reference_method"], int(rq["n_boot"]))
    utils.save_table(runs, os.path.join(out_dir, "runs.parquet"))
    utils.save_table(pilots, os.path.join(out_dir, "pilots.parquet"))
    for name, table in tables.items():
        utils.save_table(table, os.path.join(out_dir, f"{name}.csv"))
    utils.write_manifest(out_dir, cfg, {"experiment": "rq3", "simulators": done})
    return {"runs": runs, "pilots": pilots, **tables}


def summarise(runs: pd.DataFrame, pilots: pd.DataFrame, cfg, reference: str, n_boot: int,
              extra_keys: list = None, rep_cols: list = None) -> dict:
    """All RQ3 summary tables from the raw runs."""
    extra = list(extra_keys or [])
    keys = extra + ["simulator", "rho", "objective", "method"]
    seed = int(cfg["project"]["seed"])
    tab2 = summarise_methods(runs, keys, n_boot, seed)
    main = cfg["guard"]["ucb"]
    curve_cols = [c for c in keys + ["true_welfare", "true_welfare_lo", "true_welfare_hi", "regret",
                                     f"cert_{main}", f"coverage_{main}", f"slack_{main}", "recovered_share"]
                  if c in tab2.columns]
    return {
        "tab2_methods": tab2,
        "coverage": coverage_table(runs, keys, float(cfg["guard"]["delta"])),
        "pairwise_tests": pairwise_tests(runs, extra + ["simulator", "rho", "objective"], reference,
                                         rep_cols or ["rep"]),
        "pilot_size_curve": tab2[curve_cols],
        "uniform_validity": uniform_validity_table(pilots, extra + ["simulator", "rho"]),
    }
