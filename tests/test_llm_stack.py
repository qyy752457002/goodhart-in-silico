"""Prompts, reply parsing, the call cache, HTTP backends (against a local fake server) and the
persona runner. No real model is ever called."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from sim.data import STANCE_INDEX, STANCES
from sim.simulators import backends, leakage, prompts
from sim.simulators.cache import CallCache, call_key
from sim.simulators.parsing import extract_json_object, parse_distribution
from sim.simulators.persona import PersonaRunner, PersonaSpec
from sim.testbeds.common import ProblemPromptSource

K = len(STANCES)


# ------------------------------------------------------------------ prompts

def test_option_shuffle_is_deterministic_and_varies():
    opts = prompts.stance_options()
    a = [o.key for o in prompts.shuffled(opts, 0, "Org", 7, "greedy0", 0)]
    assert a == [o.key for o in prompts.shuffled(opts, 0, "Org", 7, "greedy0", 0)]
    assert sorted(a) == sorted(STANCES)
    orders = {tuple(o.key for o in prompts.shuffled(opts, 0, "Org", q, "greedy0", 0)) for q in range(40)}
    assert len(orders) > 20                                         # the order really changes per call
    # every class is listed first about equally often (no position is privileged)
    first = [prompts.shuffled(opts, 1, "Org", q, "t", 0)[0].key for q in range(1200)]
    counts = pd.Series(first).value_counts()
    assert counts.min() > 150 and counts.max() < 250


def test_aliases_and_redaction():
    org = "Minerals Council of Australia (MCA)"
    assert set(prompts.org_aliases(org)) == {org, "Minerals Council of Australia", "MCA"}
    text = "MCA supports this. The Minerals Council of Australia notes that McA-style acronyms differ."
    red = prompts.redact(text, org)
    assert "MCA" not in red and "Minerals Council" not in red and "McA-style" in red
    assert prompts.contains_alias("as noted by the minerals council of australia", org)
    assert not prompts.contains_alias("a mining industry association", org)
    # short all-caps names are matched case-sensitively and only as whole words
    assert prompts.redact("BHP agrees; bhp is not a word; ABHPC stays", "BHP") == \
        "the respondent agrees; bhp is not a word; ABHPC stays"


def test_messages_named_anonymous_and_context(small_problem):
    prob = small_problem
    source = ProblemPromptSource(prob)
    cell = 5
    i, q = int(prob.cell_org[cell]), int(prob.cell_q[cell])
    opts = prompts.shuffled(source.options(q), 0, prob.orgs[i], q, "greedy0", 0)
    named = prompts.build_messages(source, i, q, "named", opts)
    anon = prompts.build_messages(source, i, q, "anonymous", opts)
    assert [m["role"] for m in named] == ["system", "user"]
    assert prob.orgs[i] in named[1]["content"] and prob.orgs[i] not in anon[1]["content"]
    assert prob.groups[i] in anon[1]["content"]
    assert prompts.keys_in_prompt(named) == [o.key for o in opts]
    for o in opts:
        assert f'"{o.key}": {o.text}' in named[1]["content"]
    assert "did answer this question" in named[1]["content"]
    ctx = prompts.build_messages(source, i, q, "context", opts, identity="anonymous")
    assert "positions on other questions" in ctx[1]["content"] and prob.orgs[i] not in ctx[1]["content"]
    with pytest.raises(ValueError):
        prompts.build_messages(source, i, q, "oracle", opts)
    with pytest.raises(ValueError):
        prompts.build_messages(source, i, q, "context", opts, identity="pseudonymous")
    with pytest.raises(ValueError):
        prompts.keys_in_prompt([{"role": "user", "content": "no keys here"}])


def test_context_never_contains_the_target_or_its_twin(small_problem):
    """Leave-one-out context: the answer being predicted, and any question sharing its decision,
    must be absent."""
    from dataclasses import replace
    prob = small_problem
    # make questions 1 and 2 share decision 1 (like Q13/Q14 in the AASB data)
    twin_dec = np.where(prob.cell_dec == 1, 0, prob.cell_dec)
    twin = replace(prob, cell_dec=twin_dec)
    source = ProblemPromptSource(twin)
    for cell in range(twin.n_cells):
        i, q = int(twin.cell_org[cell]), int(twin.cell_q[cell])
        items = source.context(i, q)
        answered = {int(twin.cell_q[k]) for k in np.flatnonzero(twin.cell_org == i)}
        blocked = {q} | ({1, 2} if q in (1, 2) else set())
        assert len(items) == len(answered - blocked)
        assert all(source.question(b) not in [it.question for it in items] for b in blocked)
    assert prompts.context_block([]) == ""
    long_items = [prompts.ContextItem("Q" * 500, "agreed", "R" * 500)] * 20
    block = prompts.context_block(long_items, max_items=3, max_chars=50)
    assert block.count("\n") == 5 and "..." in block


# ------------------------------------------------------------------ parsing

KEYS = ["agree", "qualified_agree", "mixed", "qualified_disagree", "alternative_proposal", "disagree"]


@pytest.mark.parametrize("text,expected,status", [
    ('{"agree": 0.5, "qualified_agree": 0.2, "mixed": 0.1, "qualified_disagree": 0.1, '
     '"alternative_proposal": 0.05, "disagree": 0.05}', [.5, .2, .1, .1, .05, .05], "ok"),
    ('```json\n{"agree": 0.6, "disagree": 0.4}\n```\nBecause preparers ...', [.6, 0, 0, 0, 0, .4], "normalised"),
    ('{"Agree": 60, "Qualified Agree": 30, "Mixed": 10, "qualified-disagree": 0, '
     '"alternative proposal": 0, "DISAGREE": 0}', [.6, .3, .1, 0, 0, 0], "normalised"),
    ('{"agree": "70%", "disagree": "30%"}', [.7, 0, 0, 0, 0, .3], "normalised"),
    ("{'agree': 0.25, 'disagree': 0.75}", [.25, 0, 0, 0, 0, .75], "normalised"),
    ('{"probabilities": {"agree": 1.0}}', [1, 0, 0, 0, 0, 0], "normalised"),
    ('{"agree": 0.9, "disagree": 0.1, "abstain": 0.5}', [.9, 0, 0, 0, 0, .1], "normalised"),
    ('{"stance": "qualified agree"}', [0, 1, 0, 0, 0, 0], "label_only"),
    ("Disagree", [0, 0, 0, 0, 0, 1], "label_only"),
    ('Thinking {not json} ... {"agree": 0.3, "mixed": 0.7}', [.3, 0, .7, 0, 0, 0], "normalised"),
])
def test_parse_distribution_accepts_messy_replies(text, expected, status):
    probs, got = parse_distribution(text, KEYS)
    assert got == status
    np.testing.assert_allclose(probs, expected, atol=1e-9)
    assert probs.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("text,why", [
    ("", "no_json"), ("I cannot answer that.", "no_json"), ('{"agree": -0.2, "disagree": 1.2}', "negative"),
    ('{"agree": "likely"}', "non_numeric"), ('{"yes": 1.0}', "no_known_keys"),
    ('{"agree": 0, "disagree": 0}', "zero_mass"), ('{"agree": 0.5', "no_json"), ('{"agree": true}', "non_numeric"),
])
def test_parse_distribution_rejects_garbage(text, why):
    probs, status = parse_distribution(text, KEYS)
    assert probs is None and status == f"fail:{why}"


def test_extract_json_object_handles_braces_inside_strings():
    assert extract_json_object('x {"a": "}{", "b": 2} y') == {"a": "}{", "b": 2}
    assert extract_json_object('{"a": {"b": 1}}') == {"a": {"b": 1}}
    assert extract_json_object("[1, 2]") is None and extract_json_object(None) is None


# ------------------------------------------------------------------ cache

def test_cache_persists_and_survives_a_torn_write(tmp_path):
    path = str(tmp_path / "cache" / "calls.jsonl")
    cache = CallCache(path)
    fp = {"kind": "mock", "model": "m"}
    msgs = [{"role": "user", "content": "hello"}]
    k1 = call_key(fp, msgs, temperature=0.0, seed=1)
    assert k1 == call_key(fp, msgs, seed=1, temperature=0.0)                     # order-insensitive
    assert k1 != call_key(fp, msgs, temperature=0.7, seed=1)
    assert k1 != call_key({"kind": "mock", "model": "other"}, msgs, temperature=0.0, seed=1)
    assert cache.get(k1) is None
    cache.put(k1, "reply", {"prompt_tokens": 3})
    assert cache.get(k1)["text"] == "reply" and (cache.hits, cache.misses) == (1, 1)
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"key": "torn", "text": "unfin')                                # crash mid-write
    again = CallCache(path)
    assert len(again) == 1 and again.get(k1)["usage"] == {"prompt_tokens": 3}
    memory_only = CallCache(None)
    memory_only.put("k", "v")
    assert memory_only.get("k")["text"] == "v"


# ------------------------------------------------------------------ HTTP backends

class _FakeLLM(BaseHTTPRequestHandler):
    plan = []          # status codes to return before succeeding
    requests = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
        status = type(self).plan.pop(0) if type(self).plan else 200
        if status != 200:
            payload = {"error": {"message": "unsupported parameter: temperature"}}
        elif self.path.endswith("/api/chat"):
            payload = {"message": {"role": "assistant", "content": '{"agree": 1.0}'},
                       "prompt_eval_count": 11, "eval_count": 5, "done": True}
        else:
            payload = {"choices": [{"message": {"role": "assistant", "content": '{"agree": 1.0}'}}],
                       "usage": {"prompt_tokens": 11, "completion_tokens": 5}}
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_server():
    _FakeLLM.plan, _FakeLLM.requests = [], []
    server = HTTPServer(("127.0.0.1", 0), _FakeLLM)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", _FakeLLM
    server.shutdown()
    server.server_close()


MESSAGES = [{"role": "user", "content": "hi"}]


def test_openai_compatible_backend(fake_server, monkeypatch):
    url, fake = fake_server
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    backend = backends.OpenAICompatBackend("m1", base_url=url + "/v1", api_key_env="FAKE_KEY")
    reply = backend.complete(MESSAGES, temperature=0.0, max_tokens=50, seed=7)
    assert reply.text == '{"agree": 1.0}' and reply.usage["prompt_tokens"] == 11
    sent = fake.requests[-1]
    assert sent["path"] == "/v1/chat/completions" and sent["auth"] == "Bearer sk-test"
    assert sent["body"] == {"model": "m1", "messages": MESSAGES, "max_tokens": 50, "temperature": 0.0, "seed": 7,
                            "response_format": {"type": "json_object"}}
    # a model family with different request fields
    picky = backends.OpenAICompatBackend("m2", base_url=url + "/v1", api_key_env="FAKE_KEY",
                                         send_temperature=False, send_seed=False,
                                         max_tokens_field="max_completion_tokens",
                                         extra_body={"reasoning_effort": "low"})
    picky.complete(MESSAGES, temperature=0.7, max_tokens=64, seed=1, json_mode=False)
    assert fake.requests[-1]["body"] == {"model": "m2", "messages": MESSAGES, "max_completion_tokens": 64,
                                         "reasoning_effort": "low"}
    assert picky.fingerprint() != backend.fingerprint()
    monkeypatch.delenv("FAKE_KEY")
    with pytest.raises(backends.BackendError, match="FAKE_KEY is not set"):
        backend.complete(MESSAGES)
    keyless = backends.OpenAICompatBackend("m3", base_url=url + "/v1", api_key_env=None)
    keyless.complete(MESSAGES)
    assert fake.requests[-1]["auth"] is None


def test_ollama_backend(fake_server):
    url, fake = fake_server
    backend = backends.OllamaBackend("llama3.1:8b", host=url, num_ctx=4096)
    reply = backend.complete(MESSAGES, temperature=0.7, max_tokens=80, seed=3)
    assert reply.text == '{"agree": 1.0}' and reply.usage == {"prompt_tokens": 11, "completion_tokens": 5}
    body = fake.requests[-1]["body"]
    assert fake.requests[-1]["path"] == "/api/chat" and body["stream"] is False and body["format"] == "json"
    assert body["options"] == {"temperature": 0.7, "num_ctx": 4096, "num_predict": 80, "seed": 3}
    backend.complete(MESSAGES, json_mode=False)
    assert "format" not in fake.requests[-1]["body"]


def test_retries_then_success_and_hard_failure(fake_server):
    url, fake = fake_server
    waits = []
    fake.plan = [500, 429]
    out = backends.post_json(url + "/v1/chat/completions", {"x": 1}, max_retries=3, sleep=waits.append)
    assert out["choices"][0]["message"]["content"] == '{"agree": 1.0}'
    assert len(waits) == 2 and all(w > 0 for w in waits) and len(fake.requests) == 3
    fake.plan = [400]
    with pytest.raises(backends.BackendError, match="HTTP 400.*unsupported parameter"):
        backends.post_json(url + "/v1/chat/completions", {"x": 1}, sleep=waits.append)
    assert len(waits) == 2                                                       # a 400 is not retried
    fake.plan = [503, 503, 503]
    with pytest.raises(backends.BackendError, match="failed after 3 attempts"):
        backends.post_json(url + "/v1/chat/completions", {"x": 1}, max_retries=2, sleep=waits.append)
    with pytest.raises(backends.BackendError):
        backends.post_json("http://127.0.0.1:9/none", {"x": 1}, max_retries=1, timeout=0.5, sleep=waits.append)


def test_make_backend():
    assert isinstance(backends.make_backend({"backend": "mock"}), backends.MockBackend)
    ollama = backends.make_backend({"backend": "ollama", "model": "llama3.1:8b", "num_ctx": 2048,
                                    "cutoff": "2023-12", "notes": "local", "max_workers": 1})
    assert isinstance(ollama, backends.OllamaBackend) and ollama.num_ctx == 2048
    api = backends.make_backend({"backend": "openai_compat", "model": "x", "base_url": "http://h/v1/",
                                 "api_key_env": None})
    assert api.base_url == "http://h/v1"
    with pytest.raises(ValueError):
        backends.make_backend({"backend": "telepathy", "model": "x"})


# ------------------------------------------------------------------ persona runner

def test_persona_runner_with_mock_backend(tmp_path, small_problem):
    prob = small_problem
    source = ProblemPromptSource(prob)
    spec = PersonaSpec(variant="named", n_samples=3, seed=4)
    cache_path = str(tmp_path / "calls.jsonl")
    backend = backends.MockBackend()
    runner = PersonaRunner(prob, source, backend, spec, CallCache(cache_path))
    out = runner.run("r1a_mock")
    assert out.probs.shape == (prob.n_cells, K) and out.members.shape == (3, prob.n_cells, K)
    np.testing.assert_allclose(out.probs.sum(1), 1.0)
    assert backend.calls == prob.n_cells * 4 and runner.counts["calls"] == backend.calls
    assert out.meta["counts"]["parse_failed"] == 0 and out.meta["fallback_rate"] == 0.0
    assert out.meta["variant"] == "named" and out.meta["backend"]["kind"] == "mock"
    # sampled members differ from the greedy answer and from each other
    assert not np.allclose(out.members[0], out.members[1]) and not np.allclose(out.members[0], out.probs)
    # resume: a second runner on the same cache makes no new calls and reproduces the output
    backend2 = backends.MockBackend()
    again = PersonaRunner(prob, source, backend2, spec, CallCache(cache_path)).run("r1a_mock", max_workers=4)
    assert backend2.calls == 0
    np.testing.assert_allclose(again.probs, out.probs)
    np.testing.assert_allclose(again.members, out.members)
    # a different persona variant is a different set of prompts, hence new calls
    backend3 = backends.MockBackend()
    PersonaRunner(prob, source, backend3, PersonaSpec(variant="anonymous", n_samples=0, seed=4),
                  CallCache(cache_path)).run("r1b_mock", cells=np.arange(10))
    assert backend3.calls == 10


def test_persona_runner_recovers_from_bad_replies(small_problem):
    prob = small_problem
    source = ProblemPromptSource(prob)
    state = {"n": 0}

    def flaky(messages, temperature, seed):
        state["n"] += 1
        if state["n"] % 3 == 1:
            return "Sorry, I can't comply."                                      # first attempt of each cell
        return json.dumps({k: 1.0 if k == "disagree" else 0.0 for k in prompts.keys_in_prompt(messages)})

    runner = PersonaRunner(prob, source, backends.MockBackend(responder=flaky),
                           PersonaSpec(n_samples=0, max_parse_retries=1))
    out = runner.run("flaky", cells=np.arange(12))
    assert out.n_cells == 12 and out.members is None
    assert (out.top1() == STANCE_INDEX["disagree"]).mean() > 0.6
    assert runner.counts["parse_failed"] > 0 and out.meta["parse_failure_rate"] > 0
    # a model that never answers properly -> uniform fallback, clearly flagged
    never = PersonaRunner(prob, source, backends.MockBackend(responder=lambda *a: "no"),
                          PersonaSpec(n_samples=0, max_parse_retries=2))
    lost = never.run("never", cells=np.arange(5))
    np.testing.assert_allclose(lost.probs, 1 / K)
    assert lost.meta["fallback_rate"] == 1.0 and never.counts["calls"] == 15
    with pytest.raises(ValueError):
        PersonaRunner(prob, source, backends.MockBackend(), PersonaSpec(n_shuffles=0, n_samples=0)).run("x")


def test_options_are_mapped_to_stance_classes(small_problem):
    """Several answer options may map to one stance class (survey testbeds)."""
    prob = small_problem

    class TwoWay(ProblemPromptSource):
        def options(self, qid):
            return [prompts.Option("favor_strongly", "Strongly favor", "agree"),
                    prompts.Option("favor", "Favor", "agree"), prompts.Option("oppose", "Oppose", "disagree")]

    reply = lambda messages, t, s: '{"favor_strongly": 0.2, "favor": 0.3, "oppose": 0.5}'
    out = PersonaRunner(prob, TwoWay(prob), backends.MockBackend(responder=reply),
                        PersonaSpec(n_samples=0)).run("survey", cells=np.arange(3))
    np.testing.assert_allclose(out.probs[:, STANCE_INDEX["agree"]], 0.5)
    np.testing.assert_allclose(out.probs[:, STANCE_INDEX["disagree"]], 0.5)


def test_dry_run_counts_calls_without_calling(small_problem):
    prob = small_problem
    backend = backends.MockBackend()
    runner = PersonaRunner(prob, ProblemPromptSource(prob), backend, PersonaSpec(n_shuffles=2, n_samples=5))
    info = runner.dry_run(n_examples=2)
    assert backend.calls == 0 and info["n_calls"] == prob.n_cells * 7 and len(info["examples"]) == 2
    assert info["prompt_chars_max"] >= info["prompt_chars_mean"] > 200


# ------------------------------------------------------------------ recall probe

def test_recall_probe_separates_memorised_text(tmp_path):
    rng = np.random.default_rng(0)
    vocab = [f"w{j}" for j in range(400)]
    records = {org: [{"question_id": "Q7", "organization": f"{org}_edsr1_x",
                      "response_text": " ".join(rng.choice(vocab, 150))}] for org in ("Alpha", "Beta", "Gamma")}
    for org, recs in records.items():
        with open(tmp_path / f"{org}_edsr1_x.json", "w", encoding="utf-8") as f:
            json.dump(recs, f)
    passages = leakage.load_passages(str(tmp_path), min_words=80)
    assert sorted(passages["org"]) == ["Alpha", "Beta", "Gamma"]
    truth = {r["org"]: leakage.split_passage(r["text"])[1] for _, r in passages.iterrows()}

    def memoriser(messages, temperature, seed):
        org = next(o for o in truth if o in messages[1]["content"])
        return truth[org]

    hit = leakage.recall_probe(passages, backends.MockBackend(responder=memoriser), n=3)
    miss = leakage.recall_probe(passages, backends.MockBackend(responder=lambda *a: "something else entirely"),
                                n=3)
    assert (hit["ratio_true"] > 0.99).all() and (hit["run_gap"] > 20).all()
    assert (miss["ratio_true"] < 0.1).all() and miss["run_gap"].abs().max() <= 1
    assert leakage.overlap_scores("", "abc") == {"ratio": 0.0, "longest_run": 0}
    with pytest.raises(ValueError):
        leakage.recall_probe(passages.iloc[:0], backends.MockBackend())


def test_a_failing_backend_stops_the_run_and_keeps_finished_replies(tmp_path, small_problem):
    """A request error aborts the run without sending the queued cells; a rerun resumes from the cache."""
    prob = small_problem
    source = ProblemPromptSource(prob)

    def dying_backend(n_good=5):
        state = {"n": 0}                       # one counter per backend: requests still in flight when a
                                               # run aborts must not disturb the next phase of the test
        def responder(messages, temperature, seed):
            state["n"] += 1
            if state["n"] > n_good:
                raise backends.BackendError("service unavailable")
            return json.dumps({k: 1.0 if k == "agree" else 0.0 for k in prompts.keys_in_prompt(messages)})

        return backends.MockBackend(responder=responder), state

    spec = PersonaSpec(n_samples=0)
    for workers in (1, 3):
        backend, state = dying_backend()
        runner = PersonaRunner(prob, source, backend, spec, CallCache(str(tmp_path / f"calls{workers}.jsonl")))
        with pytest.raises(backends.BackendError):
            runner.run("dying", max_workers=workers, cells=np.arange(60))
        assert state["n"] < 30                                                 # the other cells were never sent
    # resume: replies obtained before the failure are reused
    cache_path = str(tmp_path / "calls.jsonl")
    backend, _ = dying_backend()
    first = PersonaRunner(prob, source, backend, spec, CallCache(cache_path))
    with pytest.raises(backends.BackendError):
        first.run("dying", cells=np.arange(60))
    healthy = backends.MockBackend()
    second = PersonaRunner(prob, source, healthy, spec, CallCache(cache_path))
    out = second.run("resumed", cells=np.arange(60))
    assert second.counts["cached"] == 5 and healthy.calls == 55 and out.n_cells == 60


def test_ollama_connection_failure_explains_itself():
    backend = backends.OllamaBackend("llama3.1:8b", host="http://127.0.0.1:9", max_retries=0, timeout=0.5)
    with pytest.raises(backends.BackendError, match="is the Ollama server running.*ollama pull llama3.1:8b"):
        backend.complete(MESSAGES)
