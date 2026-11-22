"""Configuration: `configs/default.yaml` holds every parameter of the methodology; a run may
layer a second YAML file and `--set key.sub=value` overrides on top of it.

Unknown keys are rejected (a typo should not silently fall back to a default), except inside the
free-form sections listed in `FREE_SECTIONS`, whose entries are user-defined.
"""
from __future__ import annotations

import copy
import os

import yaml

FREE_SECTIONS = ("simulators", "llm.models", "synthetic.scenarios", "sensitivity.variants",
                 "testbed.opinionqa", "rq1.contrasts")


class Config(dict):
    """A dict with attribute access (`cfg.rq3.reps`), nested dicts wrapped on the fly."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as err:
            raise AttributeError(f"no configuration key {name!r}") from err
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def get_path(self, dotted: str, default=None):
        node = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def project_root() -> str:
    """The repository root: $SIM_ROOT, else the directory above `code/` if it holds `configs/`,
    else the current working directory."""
    env = os.environ.get("SIM_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.dirname(os.path.dirname(here))
    if os.path.exists(os.path.join(candidate, "configs", "default.yaml")):
        return candidate
    return os.getcwd()


def _read_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level of a configuration file must be a mapping")
    return data


def _is_free(prefix: str) -> bool:
    return any(prefix == s or prefix.startswith(s + ".") for s in FREE_SECTIONS)


def deep_merge(base: dict, override: dict, prefix: str = "", strict: bool = True) -> dict:
    """Recursively overlay `override` on `base`. With strict=True, keys absent from `base` raise
    unless they sit inside a free-form section."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in out:
            if strict and not _is_free(prefix) and not _is_free(path):
                raise KeyError(f"unknown configuration key {path!r}")
            out[key] = copy.deepcopy(value)
        elif isinstance(out[key], dict) and isinstance(value, dict) and not _is_free(path):
            out[key] = deep_merge(out[key], value, path, strict)
        elif isinstance(out[key], dict) and isinstance(value, dict):
            merged = copy.deepcopy(out[key])
            for k, v in value.items():
                merged[k] = ({**merged[k], **v} if isinstance(merged.get(k), dict) and isinstance(v, dict)
                             else copy.deepcopy(v))
            out[key] = merged
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(cfg: dict, assignment: str, strict: bool = True) -> None:
    """Apply one `a.b.c=value` override in place; the value is parsed as YAML."""
    if "=" not in assignment:
        raise ValueError(f"override {assignment!r} must look like key.sub=value")
    dotted, raw = assignment.split("=", 1)
    parts = dotted.strip().split(".")
    node = cfg
    for depth, part in enumerate(parts[:-1]):
        if part not in node or not isinstance(node[part], dict):
            prefix = ".".join(parts[: depth + 1])
            if strict and not _is_free(prefix) and not _is_free(".".join(parts[:depth])):
                raise KeyError(f"unknown configuration key {prefix!r}")
            node[part] = {}
        node = node[part]
    leaf_prefix = ".".join(parts[:-1])
    if strict and parts[-1] not in node and not _is_free(leaf_prefix) and not _is_free(dotted.strip()):
        raise KeyError(f"unknown configuration key {dotted.strip()!r}")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str = None, overrides: list = None, root: str = None) -> Config:
    """Defaults <- optional file <- `--set` overrides. `cfg['_root']` is the project root against
    which relative paths are resolved; `cfg['_sources']` records where values came from."""
    root = os.path.abspath(root or project_root())
    default_path = os.path.join(root, "configs", "default.yaml")
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"default configuration not found: {default_path} "
                                "(run from the repository root or set SIM_ROOT)")
    cfg = _read_yaml(default_path)
    sources = [default_path]
    if path and os.path.abspath(path) != os.path.abspath(default_path):
        cfg = deep_merge(cfg, _read_yaml(path))
        sources.append(os.path.abspath(path))
    for item in overrides or []:
        apply_override(cfg, item)
    cfg["_root"] = root
    cfg["_sources"] = sources
    cfg["_overrides"] = list(overrides or [])
    return Config(cfg)


def with_overrides(cfg: dict, assignments: dict) -> Config:
    """A copy of `cfg` with `{dotted.key: value}` applied (used by the sensitivity analysis)."""
    out = copy.deepcopy(dict(cfg))
    for dotted, value in assignments.items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = copy.deepcopy(value)
    return Config(out)


def resolve_path(cfg: dict, value: str) -> str:
    """Absolute path of a (possibly relative) path from the configuration."""
    if value is None:
        return None
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(cfg["_root"], value))


def public_view(cfg: dict) -> dict:
    """The configuration without internal bookkeeping keys, as plain dicts (for snapshots)."""
    def plain(node):
        if isinstance(node, dict):
            return {k: plain(v) for k, v in node.items() if not str(k).startswith("_")}
        if isinstance(node, (list, tuple)):
            return [plain(v) for v in node]
        return node
    return plain(dict(cfg))
