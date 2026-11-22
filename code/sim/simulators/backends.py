"""Chat-completion backends for the persona simulators.

Methodology map (03_methodology.md): section 2.2 (model choice: an early-cut-off local model and a
recent API model) and section 10 item 5 (backend still to be chosen -- so both are supported).

  OllamaBackend        local models through Ollama's native /api/chat. Preferred for local runs
                       because the context window (`num_ctx`) can be set: with the default window
                       long prompts are silently truncated from the front.
  OpenAICompatBackend  any server that speaks the OpenAI chat-completions protocol
                       (api.openai.com, llama.cpp server, vLLM, LM Studio, Ollama's /v1, ...).
  MockBackend          deterministic fake for tests and dry runs. Never use its output as a result.

Only the standard library is used for HTTP, so no SDK is required. API keys are read from the
environment variable named in the model configuration and are never written to disk.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from .prompts import keys_in_prompt

log = logging.getLogger(__name__)

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class BackendError(RuntimeError):
    """A request failed in a way that retrying will not fix (or retries were exhausted)."""


class BackendHTTPError(BackendError):
    """An HTTP error reply that is not retried; `status` and `body` keep the server's answer."""

    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} from {url}: {body[:500]}")
        self.status = status
        self.url = url
        self.body = body


@dataclass
class Completion:
    text: str
    usage: dict = field(default_factory=dict)
    temperature_sent: bool = True   # False if the request went out without a temperature (model default)


def unsupported_parameter(body: str) -> str | None:
    """The request field that an OpenAI-style error reply rejects as unsupported by the model
    (either the field itself or the value sent for it), or None if the reply says something else."""
    try:
        err = json.loads(body).get("error") or {}
    except (ValueError, AttributeError):
        err = {}
    if not isinstance(err, dict):
        err = {}
    message = str(err.get("message") or body)
    code = str(err.get("code") or "")
    if code not in ("unsupported_parameter", "unsupported_value") and not re.search(
            r"(?i)unsupported|not supported|does not support|unrecognized request argument|unknown parameter",
            message):
        return None
    param = err.get("param")
    if not param:
        found = re.search(r"'([A-Za-z_][A-Za-z_.]*)'", message)
        param = found.group(1) if found else None
    return str(param).split(".")[0] if param else None


def post_json(url: str, payload: dict, headers: dict = None, timeout: float = 120.0, max_retries: int = 6,
              backoff: float = 1.0, max_backoff: float = 60.0, sleep=time.sleep) -> dict:
    """POST JSON and return the decoded JSON reply, retrying transient failures with exponential
    backoff and jitter (and honouring Retry-After)."""
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    last_error = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")
            # an exhausted account is reported with status 429 but waiting does not help
            if err.code not in RETRY_STATUS or "insufficient_quota" in detail or "billing_hard_limit" in detail:
                raise BackendHTTPError(err.code, url, detail) from err
            last_error = f"HTTP {err.code}: {detail[:500]}"
            retry_after = err.headers.get("Retry-After") if err.headers else None
            wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as err:
            last_error, wait = repr(err), None
        if attempt == max_retries:
            break
        if wait is None:
            wait = min(max_backoff, backoff * 2 ** attempt) * (0.5 + random.random() / 2)
        log.warning("request to %s failed (%s); retry %d/%d in %.1fs", url, last_error, attempt + 1,
                    max_retries, wait)
        sleep(wait)
    raise BackendError(f"request to {url} failed after {max_retries + 1} attempts: {last_error}")


class ChatBackend:
    """Interface: `complete` returns the assistant's text for a list of chat messages."""
    kind = "abstract"

    def __init__(self, model: str):
        self.model = model

    def fingerprint(self) -> dict:
        """Everything about the backend that can change a reply (part of the cache key)."""
        return {"kind": self.kind, "model": self.model}

    def complete(self, messages: list, *, temperature: float = 0.0, max_tokens: int = 200,
                 seed: int = None, json_mode: bool = True) -> Completion:
        raise NotImplementedError


class OpenAICompatBackend(ChatBackend):
    """OpenAI-compatible /chat/completions.

    Model families differ in which request fields they accept, so these are configurable:
      send_temperature   False for models that only run at their default temperature;
      send_seed          False for servers that reject `seed`;
      max_tokens_field   'max_tokens' or 'max_completion_tokens';
      extra_body         merged into every request (e.g. {"reasoning_effort": "none"}).

    The backend also adapts on its own: when the server answers 400 because `temperature`, `seed` or
    `response_format` is not supported by the model, or because it wants the other token-limit
    field, the request is repeated without that field (or with the other field), a warning is
    logged once, and all later requests use the adapted form. A request that went out without a
    temperature is marked (`Completion.temperature_sent`), because its reply is a draw at the
    model's default temperature rather than at the temperature the caller asked for.
    Adaptation never changes `fingerprint()`, so cached replies stay valid across runs.
    """
    kind = "openai_compat"
    DROPPABLE = ("temperature", "seed", "response_format")
    TOKEN_FIELDS = ("max_tokens", "max_completion_tokens")

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key_env: str = "OPENAI_API_KEY",
                 send_temperature: bool = True, send_seed: bool = True, max_tokens_field: str = "max_tokens",
                 extra_body: dict = None, timeout: float = 120.0, max_retries: int = 6):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.send_temperature = send_temperature
        self.send_seed = send_seed
        self.max_tokens_field = max_tokens_field
        self.extra_body = dict(extra_body or {})
        self.timeout = timeout
        self.max_retries = max_retries
        self.dropped = set()                      # fields the server turned out not to accept
        self._tokens_field = max_tokens_field     # may be switched once by adaptation
        self._adapt_lock = threading.Lock()

    def fingerprint(self) -> dict:
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url,
                "send_temperature": self.send_temperature, "extra_body": self.extra_body}

    def _payload(self, messages, temperature, max_tokens, seed, json_mode) -> dict:
        payload = {"model": self.model, "messages": messages, self._tokens_field: int(max_tokens)}
        if self.send_temperature and "temperature" not in self.dropped:
            payload["temperature"] = float(temperature)
        if self.send_seed and seed is not None and "seed" not in self.dropped:
            payload["seed"] = int(seed)
        if json_mode and "response_format" not in self.dropped:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)
        return payload

    def _adapt(self, param: str | None, payload: dict) -> bool:
        """Change the request form after the server rejected `param`. True if something changed."""
        with self._adapt_lock:
            if param in self.DROPPABLE:
                if param in self.dropped:
                    return param in payload               # adapted by another thread meanwhile: retry
                if param not in payload:
                    return False                          # rejected although we did not send it
                self.dropped.add(param)
                note = ("replies are now draws at the model's default temperature"
                        if param == "temperature" else "continuing without it")
                log.warning("model %s does not accept `%s`: %s", self.model, param, note)
                return True
            if param in self.TOKEN_FIELDS:
                if param != self._tokens_field:
                    return param in payload               # switched by another thread meanwhile: retry
                self._tokens_field = self.TOKEN_FIELDS[1 - self.TOKEN_FIELDS.index(param)]
                log.warning("model %s does not accept `%s`: using `%s` instead", self.model, param,
                            self._tokens_field)
                return True
        return False

    def _headers(self) -> dict:
        if not self.api_key_env:
            return {}
        key = os.environ.get(self.api_key_env)
        if not key:
            raise BackendError(f"environment variable {self.api_key_env} is not set "
                               f"(needed for model {self.model!r}; see .env.example)")
        return {"Authorization": f"Bearer {key}"}

    def complete(self, messages, *, temperature=0.0, max_tokens=200, seed=None, json_mode=True) -> Completion:
        url = f"{self.base_url}/chat/completions"
        for _ in range(len(self.DROPPABLE) + 2):        # each field can be adapted at most once
            payload = self._payload(messages, temperature, max_tokens, seed, json_mode)
            try:
                reply = post_json(url, payload, self._headers(), self.timeout, self.max_retries)
                break
            except BackendHTTPError as err:
                if "insufficient_quota" in err.body or "billing_hard_limit" in err.body:
                    raise BackendError(f"the account behind {self.api_key_env} has no credit left (HTTP "
                                       f"{err.status}): add credit, then rerun -- finished calls are cached") from err
                if err.status != 400 or not self._adapt(unsupported_parameter(err.body), payload):
                    raise
        else:
            raise BackendError(f"could not find a request form that {self.model} accepts")
        try:
            text = reply["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as err:
            raise BackendError(f"unexpected reply from {self.base_url}: {str(reply)[:300]}") from err
        return Completion(text=text, usage=reply.get("usage") or {}, temperature_sent="temperature" in payload)


class OllamaBackend(ChatBackend):
    """Ollama's native chat endpoint (http://localhost:11434/api/chat by default)."""
    kind = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434", num_ctx: int = 8192,
                 keep_alive: str = "30m", options: dict = None, timeout: float = 600.0, max_retries: int = 3):
        super().__init__(model)
        self.host = host.rstrip("/")
        self.num_ctx = int(num_ctx)
        self.keep_alive = keep_alive
        self.options = dict(options or {})
        self.timeout = timeout
        self.max_retries = max_retries

    def fingerprint(self) -> dict:
        return {"kind": self.kind, "model": self.model, "num_ctx": self.num_ctx, "options": self.options}

    def complete(self, messages, *, temperature=0.0, max_tokens=200, seed=None, json_mode=True) -> Completion:
        options = {"temperature": float(temperature), "num_ctx": self.num_ctx, "num_predict": int(max_tokens),
                   **self.options}
        if seed is not None:
            options["seed"] = int(seed)
        payload = {"model": self.model, "messages": messages, "stream": False, "options": options,
                   "keep_alive": self.keep_alive}
        if json_mode:
            payload["format"] = "json"
        try:
            reply = post_json(f"{self.host}/api/chat", payload, None, self.timeout, self.max_retries)
        except BackendError as err:
            raise BackendError(f"{err} -- is the Ollama server running at {self.host}, and has the model been "
                               f"pulled (`ollama pull {self.model}`)?") from err
        try:
            text = reply["message"]["content"] or ""
        except (KeyError, TypeError) as err:
            raise BackendError(f"unexpected reply from Ollama: {str(reply)[:300]}") from err
        usage = {"prompt_tokens": reply.get("prompt_eval_count"), "completion_tokens": reply.get("eval_count")}
        if usage["prompt_tokens"] is not None and usage["prompt_tokens"] >= self.num_ctx:
            log.warning("prompt filled the whole context window (num_ctx=%d): it was probably truncated; "
                        "raise num_ctx for model %s", self.num_ctx, self.model)
        return Completion(text=text, usage=usage)


class MockBackend(ChatBackend):
    """Deterministic stand-in: returns a pseudo-random distribution over the keys listed in the
    prompt, as a function of the prompt and the seed. For tests and dry runs only.

    responder: optional callable (messages, temperature, seed) -> str that replaces the default.
    """
    kind = "mock"

    def __init__(self, model: str = "mock", responder=None, concentration: float = 0.6):
        super().__init__(model)
        self.responder = responder
        self.concentration = concentration
        self.calls = 0

    def complete(self, messages, *, temperature=0.0, max_tokens=200, seed=None, json_mode=True) -> Completion:
        self.calls += 1
        if self.responder is not None:
            return Completion(text=self.responder(messages, temperature, seed))
        keys = keys_in_prompt(messages)
        tag = json.dumps(messages, sort_keys=True) + (f"|{seed}" if temperature > 0 else "")
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:8], "big"))
        probs = rng.dirichlet(np.full(len(keys), self.concentration))
        probs = np.round(probs, 3)
        probs[int(np.argmax(probs))] += round(1.0 - probs.sum(), 3)
        return Completion(text=json.dumps({k: round(float(p), 3) for k, p in zip(keys, probs)}))


def make_backend(spec: dict) -> ChatBackend:
    """Build a backend from a model entry of the configuration (`llm.models.<id>`)."""
    spec = dict(spec)
    kind = spec.pop("backend", None)
    spec.pop("cutoff", None)       # documentation fields, not constructor arguments
    spec.pop("notes", None)
    spec.pop("max_workers", None)
    if kind == "ollama":
        return OllamaBackend(**spec)
    if kind == "openai_compat":
        return OpenAICompatBackend(**spec)
    if kind == "mock":
        return MockBackend(**{k: v for k, v in spec.items() if k in ("model", "concentration")})
    raise ValueError(f"unknown backend {kind!r} (use 'ollama', 'openai_compat' or 'mock')")
