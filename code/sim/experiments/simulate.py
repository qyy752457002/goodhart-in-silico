"""Precompute LLM persona simulators (step 1 of the pipeline) and run the optional recall probe.

`python -m sim simulate --simulator r1a_llama` asks the model for a stance distribution on every
responded cell and stores the result in `<paths.simulated_dir>/<testbed>/<id>.parquet`, which every
other experiment then reads. Replies are cached per model, so runs resume and repeat exactly.

  --dry-run   build every prompt, report the number of calls and prompt sizes, print two example
              prompts; nothing is sent to any model;
  --limit N   simulate only the first N cells and report parse statistics (nothing is saved).

Methodology map (03_methodology.md): sections 2.1-2.3 and 9 (call volume, local vs API).
"""
from __future__ import annotations

import logging
import os

import numpy as np

from .. import config as config_mod
from .. import metrics, utils
from ..simulators import leakage, registry
from ..simulators.backends import make_backend
from ..simulators.base import save_output
from ..simulators.cache import CallCache
from ..simulators.persona import PersonaRunner
from . import common

log = logging.getLogger(__name__)


def _model_cfg(cfg, model_id: str) -> dict:
    models = cfg["llm"]["models"]
    if model_id not in models:
        raise KeyError(f"model {model_id!r} is not defined under llm.models; defined: {sorted(models)}")
    return dict(models[model_id])


def build_runner(cfg, tb, sim_id: str, model_override: str = None) -> tuple[PersonaRunner, dict, str]:
    spec = registry.get_spec(cfg["simulators"], sim_id)
    if spec.get("kind") != "persona":
        raise ValueError(f"simulator {sim_id!r} has kind {spec.get('kind')!r}; only persona simulators call an LLM")
    if tb.source is None:
        raise ValueError(f"testbed {tb.name!r} has no prompt source")
    model_id = model_override or spec["model"]
    model_cfg = _model_cfg(cfg, model_id)
    pspec = common.persona_spec(cfg, spec, model_cfg)
    backend_cfg = {k: v for k, v in model_cfg.items() if k != "max_tokens"}
    backend = make_backend(backend_cfg)
    cache_path = os.path.join(config_mod.resolve_path(cfg, cfg["paths"]["cache_dir"]), tb.name, f"{model_id}.jsonl")
    runner = PersonaRunner(tb.prob, tb.source, backend, pspec, CallCache(cache_path))
    return runner, model_cfg, model_id


def run(cfg, sim_ids: list, dry_run: bool = False, limit: int = None, model_override: str = None,
        force: bool = False) -> dict:
    """Simulate the given persona simulators. Returns {id: summary dict}."""
    utils.load_dotenv(cfg["_root"])
    tb = common.load_testbed(cfg)
    summaries = {}
    for sim_id in sim_ids:
        runner, model_cfg, model_id = build_runner(cfg, tb, sim_id, model_override)
        path = registry.simulated_path(common.simulated_dir(cfg), tb.name, sim_id)
        if dry_run:
            info = runner.dry_run()
            example = info.pop("examples")
            log.info("dry run %s (model %s): %s", sim_id, model_id, info)
            for ex in example:
                print(f"\n----- example prompt: {ex['org']} / Q{ex['qid']} -----")
                for m in ex["messages"]:
                    print(f"[{m['role']}]\n{m['content']}\n")
            summaries[sim_id] = info
            continue
        if os.path.exists(path) and not force and limit is None:
            log.info("%s already simulated at %s (use --force to recompute from the cache)", sim_id, path)
            summaries[sim_id] = {"status": "exists", "path": path}
            continue
        if model_cfg.get("backend") == "mock":
            log.warning("%s uses the MOCK backend: the output is random and must not be reported", sim_id)
        cells = None if limit is None else np.arange(min(int(limit), tb.prob.n_cells))
        output = runner.run(sim_id, max_workers=int(model_cfg.get("max_workers", 1)), cells=cells)
        output.meta.update({"model_id": model_id, "testbed": tb.name, "cutoff": model_cfg.get("cutoff"),
                            "is_mock": model_cfg.get("backend") == "mock"})
        summary = {"model": model_id, **output.meta["counts"],
                   "parse_failure_rate": output.meta["parse_failure_rate"],
                   "fallback_rate": output.meta["fallback_rate"],
                   "temperature_honoured": output.meta["temperature_honoured"], "seconds": output.meta["seconds"]}
        if cells is not None and output.meta["seconds"] > 0 and output.meta["counts"]["calls"] > 0:
            # extrapolate from the calls actually made in this trial (cached ones cost nothing)
            per_call = output.meta["seconds"] / output.meta["counts"]["calls"]
            total_calls = tb.prob.n_cells * runner.spec.calls_per_cell()
            summary["estimated_minutes_for_full_run"] = round(per_call * total_calls / 60.0, 1)
        if cells is None:
            save_output(output, tb.prob, path)
            summary["path"] = path
            summary.update({k: v for k, v in metrics.overall_metrics(tb.prob, output).items()
                            if k in ("accuracy", "brier", "ece")})
        else:
            sub = metrics.cell_metrics(tb.prob.stance_idx[cells], output.probs, tb.prob.sigma_vec)
            summary.update({"n_cells": len(cells), "accuracy_on_subset": sub["accuracy"], "saved": False})
        log.info("simulated %s: %s", sim_id, summary)
        summaries[sim_id] = summary
    return summaries


def recall_probe(cfg, out: str = None) -> dict:
    """Verbatim-recall probe for each model listed in `recall_probe.models` (methodology 2.2)."""
    import pandas as pd

    utils.load_dotenv(cfg["_root"])
    rp = cfg["recall_probe"]
    out_dir = utils.results_dir(cfg, "recall_probe", out)
    responses_dir = config_mod.resolve_path(cfg, cfg["testbed"]["aasb"]["responses_dir"])
    if not os.path.isdir(responses_dir):
        raise FileNotFoundError(f"response texts not found at {responses_dir} (testbed.aasb.responses_dir)")
    passages = leakage.load_passages(responses_dir, int(rp["min_words"]))
    rows = []
    for model_id in rp["models"]:
        model_cfg = _model_cfg(cfg, model_id)
        backend = make_backend({k: v for k, v in model_cfg.items() if k != "max_tokens"})
        table = leakage.recall_probe(passages, backend, int(rp["n_passages"]),
                                     utils.rep_rng(int(cfg["project"]["seed"]), 19),  # same passages per model
                                     int(rp["prefix_words"]), int(rp["suffix_words"]))
        rows.append(table.assign(model=model_id, cutoff=model_cfg.get("cutoff")))
        log.info("recall probe %s: mean run gap %.2f, mean ratio gap %.3f", model_id, table["run_gap"].mean(),
                 table["ratio_gap"].mean())
    detail = pd.concat(rows, ignore_index=True)
    summary = detail.groupby("model")[["ratio_true", "ratio_control", "ratio_gap", "run_true", "run_control",
                                       "run_gap"]].mean().reset_index()
    utils.save_table(detail, os.path.join(out_dir, "recall_probe.csv"))
    utils.save_table(summary, os.path.join(out_dir, "recall_probe_summary.csv"))
    utils.write_manifest(out_dir, cfg, {"experiment": "recall_probe"})
    return {"detail": detail, "summary": summary}


def fetch_data(cfg) -> list:
    """Download Dataset A (two parquet files and the dataset card) from the Hugging Face hub.

    The dataset is private: the download needs a Hugging Face token with granted access, read from
    the environment variable `HF_TOKEN` (or from `.env`).
    """
    import urllib.error
    import urllib.request

    utils.load_dotenv(cfg["_root"])
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    tb = cfg["testbed"]["aasb"]
    target = config_mod.resolve_path(cfg, tb["parquet_dir"])
    os.makedirs(target, exist_ok=True)
    base = f"https://huggingface.co/datasets/{tb['hf_repo']}/resolve/main"
    files = [("data/train-00000-of-00001.parquet", os.path.join(target, "train-00000-of-00001.parquet")),
             ("data/test-00000-of-00001.parquet", os.path.join(target, "test-00000-of-00001.parquet")),
             ("README.md", os.path.join(os.path.dirname(target), "README.md"))]
    done = []
    for remote, local in files:
        if os.path.exists(local):
            log.info("already present: %s", local)
            continue
        log.info("downloading %s/%s", base, remote)
        request = urllib.request.Request(f"{base}/{remote}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as err:
            if err.code in (401, 403, 404):
                raise RuntimeError(
                    f"Cannot download {tb['hf_repo']} (HTTP {err.code}). The dataset is private: request "
                    "access from the authors, then set HF_TOKEN in the environment or in .env.") from err
            raise
        with open(local, "wb") as f:
            f.write(payload)
        done.append(local)
    return done
