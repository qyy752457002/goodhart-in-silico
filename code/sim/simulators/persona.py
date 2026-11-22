"""LLM persona simulators R1a (named), R1b (anonymous) and R2 (with context).

For every responded cell the model is asked for a distribution over the stance classes:
  * `n_shuffles` calls at `temperature` (0 by default), each with its own option order; their mean
    is the simulator's distribution p-hat;
  * `n_samples` further calls at `sample_temperature` (K = 5 at T = 0.7), kept as ensemble members
    for dispersion-based baselines.

Every call goes through the cache, so an interrupted run resumes where it stopped and a finished
run can be reproduced without calling the model again.

Methodology map (03_methodology.md): sections 2.1-2.3 and 9 (call volume).
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from ..data import STANCE_INDEX, STANCES
from . import prompts
from .backends import ChatBackend
from .base import SimulatorOutput
from .cache import CallCache, call_key
from .parsing import parse_distribution

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonaSpec:
    variant: str = "named"            # 'named' (R1a) | 'anonymous' (R1b) | 'context' (R2)
    identity: str = "named"           # context variant only: is the persona itself named?
    temperature: float = 0.0
    n_shuffles: int = 1               # calls per cell at `temperature`, each with a new option order
    n_samples: int = 5                # sampled calls per cell (ensemble members); 0 disables them
    sample_temperature: float = 0.7
    max_tokens: int = 200
    seed: int = 0
    max_parse_retries: int = 2        # extra attempts (with a new option order) after a bad reply
    json_mode: bool = True
    max_context_items: int = 12
    prompt_version: str = prompts.PROMPT_VERSION

    def calls_per_cell(self) -> int:
        return self.n_shuffles + self.n_samples


def _to_stances(option_probs: np.ndarray, options: list) -> np.ndarray:
    """Sum option probabilities into the canonical stance classes."""
    out = np.zeros(len(STANCES))
    for p, option in zip(option_probs, options):
        out[STANCE_INDEX[option.stance]] += p
    return out


class PersonaRunner:
    def __init__(self, prob, source: prompts.PromptSource, backend: ChatBackend, spec: PersonaSpec,
                 cache: CallCache = None):
        self.prob = prob
        self.source = source
        self.backend = backend
        self.spec = spec
        self.cache = cache if cache is not None else CallCache(None)
        self._lock = threading.Lock()
        self.counts = {"calls": 0, "cached": 0, "parse_ok": 0, "parse_repaired": 0, "parse_failed": 0,
                       "fallback_uniform": 0, "default_temperature": 0, "prompt_tokens": 0,
                       "completion_tokens": 0}

    # ------------------------------------------------------------------ one request
    def messages_for(self, cell: int, tag: str, attempt: int = 0) -> tuple[list, list]:
        """(messages, options in the order shown) for one call. `tag` identifies the call within
        the cell, e.g. 'greedy0' or 'sample3'."""
        org_index, qid = int(self.prob.cell_org[cell]), int(self.prob.cell_q[cell])
        options = prompts.shuffled(self.source.options(qid), self.spec.seed, self.prob.orgs[org_index], qid,
                                   tag, attempt)
        messages = prompts.build_messages(self.source, org_index, qid, self.spec.variant, options,
                                          self.spec.identity, self.spec.max_context_items)
        return messages, options

    def _ask(self, cell: int, tag: str, temperature: float, call_seed: int) -> tuple[np.ndarray, bool]:
        """One distribution over the stance classes, with parse retries and a uniform fallback, and
        whether the request carried the temperature that was asked for (False: the model only runs
        at its default temperature)."""
        spec = self.spec
        at_temperature = True
        for attempt in range(spec.max_parse_retries + 1):
            messages, options = self.messages_for(cell, tag, attempt)
            params = {"temperature": temperature, "max_tokens": spec.max_tokens, "seed": call_seed + attempt,
                      "json_mode": spec.json_mode, "prompt_version": spec.prompt_version}
            key = call_key(self.backend.fingerprint(), messages, **params)
            rec = self.cache.get(key)
            if rec is None:
                reply = self.backend.complete(messages, temperature=temperature, max_tokens=spec.max_tokens,
                                              seed=call_seed + attempt, json_mode=spec.json_mode)
                info = {} if reply.temperature_sent else {"temperature_sent": False}
                rec = self.cache.put(key, reply.text, reply.usage, **info)
                self._count("calls")
                self._count("prompt_tokens", rec["usage"].get("prompt_tokens") or 0)
                self._count("completion_tokens", rec["usage"].get("completion_tokens") or 0)
            else:
                self._count("cached")
            at_temperature = bool(rec.get("temperature_sent", True))
            if not at_temperature:
                self._count("default_temperature")
            probs, status = parse_distribution(rec["text"], [o.key for o in options])
            if probs is not None:
                self._count("parse_ok" if status == "ok" else "parse_repaired")
                return _to_stances(probs, options), at_temperature
            self._count("parse_failed")
            log.debug("unparseable reply (%s) for cell %d %s attempt %d: %r", status, cell, tag, attempt,
                      rec["text"][:200])
        self._count("fallback_uniform")
        return np.full(len(STANCES), 1.0 / len(STANCES)), at_temperature

    def _count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counts[name] += amount

    def simulate_cell(self, cell: int) -> tuple[np.ndarray, np.ndarray]:
        """(p-hat, members) for one cell.

        p-hat is the mean of the `n_shuffles` replies at `temperature` (0 by default). If the model
        does not accept a temperature, those replies are ordinary draws at its default temperature,
        exactly like the samples, so p-hat is then the mean of *all* replies of the cell: one draw
        would be a needlessly noisy estimate. The samples are returned as members either way.
        """
        spec = self.spec
        greedy = [self._ask(cell, f"greedy{j}", spec.temperature, spec.seed + j) for j in range(spec.n_shuffles)]
        samples = [self._ask(cell, f"sample{j}", spec.sample_temperature, spec.seed + 1000 + j)
                   for j in range(spec.n_samples)]
        sampled = [p for p, _ in samples]
        if greedy and all(ok for _, ok in greedy):
            probs = np.mean([p for p, _ in greedy], axis=0)
        else:
            probs = np.mean([p for p, _ in greedy] + sampled, axis=0)
        return probs, (np.stack(sampled) if sampled else np.zeros((0, len(STANCES))))

    # ------------------------------------------------------------------ whole problem
    def run(self, name: str, max_workers: int = 1, cells: np.ndarray = None,
            progress_every: int = 100) -> SimulatorOutput:
        """Simulate `cells` (default: every cell). Returns the output for exactly those cells."""
        spec = self.spec
        if spec.n_shuffles + spec.n_samples < 1:
            raise ValueError("a persona simulator needs at least one call per cell")
        cells = np.arange(self.prob.n_cells) if cells is None else np.asarray(cells, dtype=int)
        started = time.time()
        probs = np.zeros((len(cells), len(STANCES)))
        members = np.zeros((spec.n_samples, len(cells), len(STANCES)))
        # Set as soon as any cell fails. Workers check it before each cell, so after a failure only the
        # requests already in flight (at most max_workers - 1) are sent, however late the main thread
        # gets to cancel the queue.
        stop = threading.Event()

        def work(pos: int) -> int:
            if stop.is_set():
                return pos
            try:
                p, m = self.simulate_cell(int(cells[pos]))
            except BaseException:
                stop.set()
                raise
            probs[pos] = p
            if spec.n_samples:
                members[:, pos] = m
            return pos

        if max_workers <= 1:
            for pos in range(len(cells)):
                work(pos)
                if progress_every and (pos + 1) % progress_every == 0:
                    log.info("%s: %d/%d cells (%d new calls, %d cached)", name, pos + 1, len(cells),
                             self.counts["calls"], self.counts["cached"])
        else:
            pool = ThreadPoolExecutor(max_workers=max_workers)
            try:
                for done, _ in enumerate(pool.map(work, range(len(cells))), 1):
                    if progress_every and done % progress_every == 0:
                        log.info("%s: %d/%d cells (%d new calls, %d cached)", name, done, len(cells),
                                 self.counts["calls"], self.counts["cached"])
            except BaseException:
                # a failed request or Ctrl+C: drop the queued cells instead of sending them anyway
                # (finished replies are already in the cache, so the run can be resumed)
                stop.set()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            pool.shutdown()
        n_requests = max(1, self.counts["parse_ok"] + self.counts["parse_repaired"] + self.counts["parse_failed"])
        honoured = self.counts["default_temperature"] == 0
        if not honoured:
            log.warning("%s: the model ran at its default temperature (it does not accept one), so p-hat is the "
                        "mean of all %d replies per cell instead of the temperature-%g reply", name,
                        spec.calls_per_cell(), spec.temperature)
        meta = {
            "kind": "persona", "variant": spec.variant, "backend": self.backend.fingerprint(),
            "spec": asdict(spec), "counts": dict(self.counts),
            "parse_failure_rate": self.counts["parse_failed"] / n_requests,
            "fallback_rate": self.counts["fallback_uniform"] / max(1, len(cells) * spec.calls_per_cell()),
            "temperature_honoured": honoured,
            "p_hat": (f"mean of the {spec.n_shuffles} reply(ies) at temperature {spec.temperature:g}" if honoured
                      else "mean of all replies of the cell (model default temperature)"),
            "seconds": round(time.time() - started, 1),
        }
        return SimulatorOutput(name, probs, members if spec.n_samples else None, meta)

    def dry_run(self, n_examples: int = 2) -> dict:
        """Build every prompt without calling the model: call count, prompt sizes and examples."""
        spec = self.spec
        n_chars = []
        for cell in range(self.prob.n_cells):
            messages, _ = self.messages_for(cell, "greedy0")
            n_chars.append(sum(len(m["content"]) for m in messages))
        n_chars = np.array(n_chars)
        examples = []
        for cell in np.linspace(0, self.prob.n_cells - 1, min(n_examples, self.prob.n_cells)).astype(int):
            messages, _ = self.messages_for(int(cell), "greedy0")
            examples.append({"org": self.prob.orgs[int(self.prob.cell_org[cell])],
                             "qid": int(self.prob.cell_q[cell]), "messages": messages})
        return {"n_cells": int(self.prob.n_cells), "calls_per_cell": spec.calls_per_cell(),
                "n_calls": int(self.prob.n_cells * spec.calls_per_cell()),
                "prompt_chars_mean": float(n_chars.mean()), "prompt_chars_max": int(n_chars.max()),
                "approx_prompt_tokens_total": int(n_chars.sum() / 4 * spec.calls_per_cell()),
                "examples": examples}
