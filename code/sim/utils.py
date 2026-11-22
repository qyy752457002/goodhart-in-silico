"""Run plumbing: logging, reproducible random streams, output folders and run manifests."""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
import zlib

import numpy as np
import pandas as pd
import yaml

from . import config as config_mod


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, str(level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S",
                        force=True)


def add_log_file(path: str) -> logging.Handler:
    """Also write the log to `path` (UTF-8). Returns the handler; pass it to `remove_log_file`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def remove_log_file(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def _stream_key(key) -> int:
    """An integer is used as it is; a name becomes the CRC-32 of its UTF-8 bytes (Python's own
    `hash` is salted per process, so it would not give the same stream twice)."""
    return zlib.crc32(key.encode("utf-8")) if isinstance(key, str) else int(key)


def rep_rng(seed: int, *keys) -> np.random.Generator:
    """An independent, reproducible stream for one replication.

    The stream depends only on (seed, keys) -- not on how many other replications ran before it --
    so results do not change when the grid is reordered, subsetted or parallelised.

    A key is an integer or a name. Anything that belongs to one simulator (its cross-fitting folds,
    its bootstrap intervals, its random packages) is keyed by the simulator's id, not by its position
    in a list: the same simulator then gets the same numbers in every experiment and configuration.
    """
    return np.random.default_rng(np.random.SeedSequence(entropy=int(seed),
                                                        spawn_key=tuple(_stream_key(k) for k in keys)))


def load_dotenv(root: str) -> list:
    """Load KEY=VALUE lines from `<root>/.env` into the environment (existing variables win).
    Returns the names that were set. The file is git-ignored; keys never go into results."""
    path = os.path.join(root, ".env")
    loaded = []
    if not os.path.exists(path):
        return loaded
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def git_revision(root: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def results_dir(cfg: dict, experiment: str, out: str = None) -> str:
    """`<results_dir>/<testbed>/<experiment>` (or the explicit `out` folder), created if needed."""
    path = out or os.path.join(config_mod.resolve_path(cfg, cfg["paths"]["results_dir"]),
                               cfg["testbed"]["name"], experiment)
    path = path if os.path.isabs(path) else os.path.join(cfg["_root"], path)
    os.makedirs(path, exist_ok=True)
    return path


def write_manifest(out_dir: str, cfg: dict, extra: dict = None) -> str:
    """Snapshot of what produced the files in `out_dir`: resolved configuration, code revision,
    library versions, command line and time."""
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config_mod.public_view(cfg), f, sort_keys=False, allow_unicode=True)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(sys.argv),
        "git_revision": git_revision(cfg.get("_root", ".")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config_sources": [os.path.basename(s) for s in cfg.get("_sources", [])],
        "config_overrides": cfg.get("_overrides", []),
        **(extra or {}),
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def save_table(df: pd.DataFrame, path: str, index: bool = False) -> str:
    """Write a table as .csv or .parquet (chosen by the extension)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index, encoding="utf-8")
    return path


def read_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, encoding="utf-8")


class Timer:
    def __init__(self):
        self.start = time.time()

    def seconds(self) -> float:
        return round(time.time() - self.start, 1)
