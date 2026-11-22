"""Parse a verbalised probability distribution out of an LLM reply.

Models rarely return clean JSON: the object may be wrapped in a code fence, followed by an
explanation, use percentages, drop a key or answer with a bare label. `parse_distribution` accepts
all of these and reports how the answer had to be repaired, so that parse quality can be audited.
"""
from __future__ import annotations

import ast
import json
import re

import numpy as np

_WRAPPER_KEYS = ("probabilities", "distribution", "stance_probabilities", "probs", "answer", "result")


def _normalise_key(key) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def extract_json_object(text: str) -> dict | None:
    """The first balanced {...} block in `text` that parses as a JSON (or Python-literal) object."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, in_string, quote, escape = 0, False, "", False
        for pos in range(start, len(text)):
            ch = text[pos]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
            elif ch in "\"'":
                in_string, quote = True, ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    block = text[start: pos + 1]
                    for loader in (json.loads, ast.literal_eval):
                        try:
                            obj = loader(block)
                        except (ValueError, SyntaxError, MemoryError, RecursionError):
                            continue
                        if isinstance(obj, dict):
                            return obj
                    break
        start = text.find("{", start + 1)
    return None


def _to_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(%)?\s*", value)
        if m:
            return float(m.group(1)) / (100.0 if m.group(2) else 1.0)
    return None


def parse_distribution(text: str, keys: list) -> tuple[np.ndarray | None, str]:
    """Probabilities over `keys` (in that order) and a status string.

    status: 'ok'          a complete, properly normalised distribution;
            'normalised'  parsed, but it had to be rescaled, was missing keys or had extra ones;
            'label_only'  the reply was a single category rather than a distribution (one-hot);
            'fail:<why>'  nothing usable (probabilities are None).
    """
    norm = {_normalise_key(k): j for j, k in enumerate(keys)}
    obj = extract_json_object(text or "")
    if obj is not None:
        for wrapper in _WRAPPER_KEYS:  # {"probabilities": {...}}
            inner = next((v for k, v in obj.items() if _normalise_key(k) == wrapper), None)
            if isinstance(inner, dict):
                obj = inner
                break
        if len(obj) == 1 and isinstance(next(iter(obj.values())), str):
            # {"stance": "agree"}: a label wrapped in an object
            label = _normalise_key(next(iter(obj.values())))
            if label in norm:
                out = np.zeros(len(keys))
                out[norm[label]] = 1.0
                return out, "label_only"
        probs = np.zeros(len(keys))
        seen, extra = set(), 0
        for k, v in obj.items():
            j = norm.get(_normalise_key(k))
            if j is None:
                extra += 1
                continue
            number = _to_number(v)
            if number is None or not np.isfinite(number):
                return None, "fail:non_numeric"
            if number < 0:
                return None, "fail:negative"
            probs[j] += number
            seen.add(j)
        if not seen:
            return None, "fail:no_known_keys"
        total = probs.sum()
        if total <= 0:
            return None, "fail:zero_mass"
        clean = len(seen) == len(keys) and extra == 0 and abs(total - 1.0) <= 0.02
        return probs / total, ("ok" if clean else "normalised")
    label = _normalise_key(text or "")
    if label in norm:
        out = np.zeros(len(keys))
        out[norm[label]] = 1.0
        return out, "label_only"
    return None, "fail:no_json"
