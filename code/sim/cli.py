"""Command-line interface: `python -m sim <command>` (or `sim <command>` after `pip install -e .`).

Pipeline order:
  fetch-data -> simulate (the only step that calls a model) -> all
where `all` = describe -> rq1 -> rq2 -> rq3 -> synthetic -> sensitivity -> report, each of which can
also be run on its own. `status` shows what has been produced so far.

Every command takes `--config FILE` (layered on configs/default.yaml) and `--set key.sub=value`.
Commands that do work also write their log to `<results_dir>/<testbed>/logs/`, so a run can be
inspected afterwards without having kept the terminal output.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from . import __version__, config, utils

log = logging.getLogger("sim")

STEPS = ("describe", "rq1", "rq2", "rq3", "synthetic", "sensitivity", "report")
LOGGED = set(STEPS) | {"all", "simulate", "recall-probe"}
PREVIEW = {"describe": "packages", "rq1": "tab1_simulators", "rq2": "plugin_summary", "rq3": "coverage",
           "synthetic": "summary", "sensitivity": "packages"}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", default=None, help="YAML file layered on configs/default.yaml")
    parser.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                        help="override one configuration value, e.g. --set rq3.reps=50 (repeatable)")
    parser.add_argument("--out", "-o", default=None, help="output folder (default: results/<testbed>/<experiment>)")
    parser.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sim", description="Goodhart in Silico: rectify-then-certify policy "
                                                             "design with LLM-simulated stakeholders.")
    parser.add_argument("--version", action="version", version=f"sim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        _add_common(p)
        return p

    command("show-config", "print the resolved configuration")
    command("status", "show which simulators have been precomputed and which results exist")
    command("fetch-data", "download Dataset A (AASB ED SR1 stances) from the Hugging Face hub")
    p = command("simulate", "precompute LLM persona simulators (calls the configured model unless --dry-run)")
    p.add_argument("--simulator", "-m", action="append", required=True, dest="simulators",
                   help="simulator id from the configuration (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="build prompts and count calls; send nothing")
    p.add_argument("--limit", type=int, default=None, help="simulate only the first N cells; save nothing")
    p.add_argument("--model", default=None, help="use this model id instead of the simulator's own")
    p.add_argument("--force", action="store_true", help="recompute even if the output file exists")
    p = command("all", "run the CPU pipeline in order: " + ", ".join(STEPS) + " (no model is called)")
    p.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS), metavar="STEP",
                   help="run only these steps, in pipeline order (default: all of them)")
    command("describe", "testbed statistics and precondition P1 (do the groups disagree?)")
    command("rq1", "RQ1: simulator validity (Table 1)")
    command("rq2", "RQ2: Goodhart curves and group decomposition (Figs. 2-3)")
    command("rq3", "RQ3: rectify-then-certify against baselines (Table 2, Fig. 4)")
    command("synthetic", "synthetic error injection (Fig. 5, Propositions A-C)")
    command("sensitivity", "sensitivity analysis (section 8)")
    command("report", "build every table and figure for which results exist")
    command("toy", "the two-group example behind Figure 1")
    command("recall-probe", "verbatim-recall probe for data leakage (optional; calls the models)")
    p = command("opinionqa-inspect", "list the columns and answer options of an OpinionQA wave folder")
    p.add_argument("--dir", required=True, help="wave folder containing responses.csv (and info.csv)")
    return parser


def _tolerant_streams() -> None:
    """Never crash on a character the output encoding cannot represent (e.g. a non-ASCII project
    path printed while output is redirected on Windows): escape it instead."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


def _results_root(cfg) -> str:
    return os.path.join(config.resolve_path(cfg, cfg["paths"]["results_dir"]), cfg["testbed"]["name"])


def _run_step(step: str, cfg, out: str = None) -> None:
    """One CPU step of the pipeline, with a short preview of its main table."""
    if step == "report":
        from .reporting import tables
        written = tables.build_all(cfg, out)
        print(f"report: {len(written)} files written")
        return
    from .experiments import describe, rq1_validity, rq2_goodhart, rq3_guard, sensitivity, synthetic_injection
    runners = {"describe": describe.run, "rq1": rq1_validity.run, "rq2": rq2_goodhart.run, "rq3": rq3_guard.run,
               "synthetic": synthetic_injection.run, "sensitivity": sensitivity.run}
    tables_out = runners[step](cfg, out)
    table = tables_out.get(PREVIEW[step])
    if table is not None and len(table):
        print(table.head(40).round(3).to_string(index=False, max_colwidth=60))
    log.info("%s finished: outputs in %s", step, utils.results_dir(cfg, step, out))


def status(cfg) -> dict:
    """What exists on disk: precomputed simulators, cached replies, result folders. Reads no data."""
    from .simulators import registry

    testbed = cfg["testbed"]["name"]
    simulated = config.resolve_path(cfg, cfg["paths"]["simulated_dir"])
    cache_dir = os.path.join(config.resolve_path(cfg, cfg["paths"]["cache_dir"]), testbed)
    sims = []
    for sim_id, spec in cfg["simulators"].items():
        row = {"simulator": sim_id, "kind": spec.get("kind"), "model": spec.get("model", "")}
        if spec.get("kind") == "persona":
            path = registry.simulated_path(simulated, testbed, sim_id)
            row["precomputed"] = os.path.exists(path)
            if os.path.exists(path + ".meta.json"):
                with open(path + ".meta.json", encoding="utf-8") as f:
                    meta = json.load(f)
                row.update({k: meta.get(k) for k in ("n_members", "parse_failure_rate", "fallback_rate",
                                                     "temperature_honoured", "seconds", "is_mock")})
        elif spec.get("kind") == "ensemble":
            row["precomputed"] = "needs: " + ", ".join(spec.get("members", []))
        else:
            row["precomputed"] = "built on the fly"
        sims.append(row)
    caches = {}
    if os.path.isdir(cache_dir):
        for name in sorted(os.listdir(cache_dir)):
            if name.endswith(".jsonl"):
                with open(os.path.join(cache_dir, name), encoding="utf-8") as f:
                    caches[name[:-len(".jsonl")]] = sum(1 for line in f if line.strip())
    results = {}
    root = _results_root(cfg)
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            manifest = os.path.join(root, name, "manifest.json")
            if os.path.exists(manifest):
                with open(manifest, encoding="utf-8") as f:
                    results[name] = json.load(f).get("created")
            elif name == "paper" and os.path.isdir(os.path.join(root, name)):
                results[name] = f"{len(os.listdir(os.path.join(root, name)))} files"
    return {"testbed": testbed, "simulators": sims, "cached_replies": caches, "results": results,
            "results_dir": root}


def _print_status(info: dict) -> None:
    import pandas as pd

    print(f"testbed: {info['testbed']}\n")
    print(pd.DataFrame(info["simulators"]).fillna("").to_string(index=False))
    print("\ncached model replies:", info["cached_replies"] or "none")
    print(f"\nresults in {info['results_dir']}:")
    for name in STEPS[:-1] + ("recall_probe", "paper"):   # the report step writes into paper/
        print(f"  {name:12s} {info['results'].get(name, '-')}")


def _dispatch(args, cfg) -> int:
    if args.command == "show-config":
        import yaml
        yaml.safe_dump(config.public_view(cfg), sys.stdout, sort_keys=False, allow_unicode=True)
    elif args.command == "status":
        _print_status(status(cfg))
    elif args.command == "opinionqa-inspect":
        from .testbeds.opinionqa import inspect_wave
        print(inspect_wave(args.dir).to_string(index=False, max_colwidth=90))
    elif args.command == "toy":
        from .experiments import toy_example
        out = toy_example.run(cfg, args.out)
        print(out["packages"].round(3).to_string(index=False))
        print()
        print(out["selections"].round(3).to_string(index=False))
    elif args.command == "fetch-data":
        from .experiments import simulate
        print("\n".join(simulate.fetch_data(cfg)) or "nothing to download")
    elif args.command == "simulate":
        from .experiments import simulate
        summaries = simulate.run(cfg, args.simulators, dry_run=args.dry_run, limit=args.limit,
                                 model_override=args.model, force=args.force)
        print(json.dumps(summaries, indent=2, default=str))
    elif args.command == "recall-probe":
        from .experiments import simulate
        print(simulate.recall_probe(cfg, args.out)["summary"].round(3).to_string(index=False))
    elif args.command == "all":
        chosen = [s for s in STEPS if s in args.steps]
        for j, step in enumerate(chosen, 1):
            timer = utils.Timer()
            log.info("=== step %d/%d: %s ===", j, len(chosen), step)
            _run_step(step, cfg)
            log.info("=== %s done in %.0fs ===", step, timer.seconds())
        print(f"\nall done: {', '.join(chosen)} -> {_results_root(cfg)}")
    else:
        _run_step(args.command, cfg, args.out)
    return 0


def main(argv: list = None) -> int:
    _tolerant_streams()
    args = build_parser().parse_args(argv)
    cfg = config.load_config(args.config, args.overrides)
    utils.setup_logging(args.log_level or cfg["project"]["log_level"])
    handler = None
    if args.command in LOGGED and not getattr(args, "dry_run", False):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        handler = utils.add_log_file(os.path.join(_results_root(cfg), "logs", f"{args.command}-{stamp}.log"))
        log.info("sim %s", " ".join(argv if argv is not None else sys.argv[1:]))
    try:
        return _dispatch(args, cfg)
    except Exception:
        if handler is None:
            raise
        # the traceback goes to the terminal and to the log file; exit code 1 tells scripts it failed
        log.exception("%s failed", args.command)
        return 1
    finally:
        if handler is not None:
            log.info("log file: %s", handler.baseFilename)
            utils.remove_log_file(handler)


if __name__ == "__main__":
    sys.exit(main())
