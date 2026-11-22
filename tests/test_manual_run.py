"""What matters when the pipeline is run by hand, unattended: the API backend adapts to what a
model accepts instead of stopping, failures say what to do and land in a log file, and one command
runs the whole CPU pipeline. No real model is called."""
import glob
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import yaml

from sim import cli, config
from sim.simulators import backends
from sim.simulators.cache import CallCache
from sim.simulators.persona import PersonaRunner, PersonaSpec
from sim.simulators.prompts import keys_in_prompt
from sim.testbeds.common import ProblemPromptSource

MESSAGES = [{"role": "user", "content": "hi"}]


# ------------------------------------------------------------------ a server that is picky like a real one

class _PickyAPI(BaseHTTPRequestHandler):
    reject = set()       # request fields this "model" does not support
    quota = False        # answer every request with "insufficient_quota"
    status = None        # answer every request with this status and a plain body
    requests = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        cls.requests.append(body)
        bad = next((f for f in body if f in cls.reject), None)
        if cls.status is not None:
            code, payload = cls.status, {"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}
        elif cls.quota:
            code, payload = 429, {"error": {"message": "You exceeded your current quota, please check your plan "
                                                       "and billing details.", "type": "insufficient_quota",
                                            "param": None, "code": "insufficient_quota"}}
        elif bad is not None:
            code, payload = 400, {"error": {"message": f"Unsupported parameter: '{bad}' is not supported with this "
                                                       "model.", "type": "invalid_request_error", "param": bad,
                                            "code": "unsupported_parameter"}}
        else:
            code, payload = 200, {"choices": [{"message": {"role": "assistant", "content": '{"agree": 1.0}'}}],
                                  "usage": {"prompt_tokens": 9, "completion_tokens": 4}}
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def picky():
    _PickyAPI.reject, _PickyAPI.quota, _PickyAPI.status, _PickyAPI.requests = set(), False, None, []
    server = HTTPServer(("127.0.0.1", 0), _PickyAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", _PickyAPI
    server.shutdown()
    server.server_close()


def test_backend_adapts_to_fields_the_model_rejects(picky, caplog):
    url, api = picky
    api.reject = {"temperature", "max_tokens", "response_format"}
    backend = backends.OpenAICompatBackend("picky", base_url=url, api_key_env=None, max_tokens_field="max_tokens",
                                           extra_body={"reasoning_effort": "none"})
    before = backend.fingerprint()
    with caplog.at_level(logging.WARNING, logger="sim.simulators.backends"):
        reply = backend.complete(MESSAGES, temperature=0.0, max_tokens=50, seed=3)
    assert reply.text == '{"agree": 1.0}' and reply.temperature_sent is False
    assert len(api.requests) == 4                                    # three rejections, then success
    assert api.requests[-1] == {"model": "picky", "messages": MESSAGES, "max_completion_tokens": 50, "seed": 3,
                                "reasoning_effort": "none"}
    assert backend.dropped == {"temperature", "response_format"}
    assert sum("does not accept" in r.message for r in caplog.records) == 3
    # later calls go out in the adapted form straight away, and the cache key is unaffected
    backend.complete(MESSAGES, temperature=0.7, max_tokens=50, seed=4)
    assert len(api.requests) == 5 and "temperature" not in api.requests[-1]
    assert backend.fingerprint() == before
    # a model that accepts everything is left alone
    api.reject = set()
    plain = backends.OpenAICompatBackend("easy", base_url=url, api_key_env=None)
    assert plain.complete(MESSAGES, temperature=0.0).temperature_sent is True and plain.dropped == set()
    assert backends.OpenAICompatBackend("cold", base_url=url, api_key_env=None,
                                        send_temperature=False).complete(MESSAGES).temperature_sent is False


def test_backend_does_not_swallow_other_errors(picky):
    url, api = picky
    waits = []
    api.reject = {"reasoning_effort"}                                # not something we may silently drop
    backend = backends.OpenAICompatBackend("picky", base_url=url, api_key_env=None,
                                           extra_body={"reasoning_effort": "none"})
    with pytest.raises(backends.BackendHTTPError, match="reasoning_effort") as err:
        backend.complete(MESSAGES)
    assert err.value.status == 400 and len(api.requests) == 1
    api.reject, api.status, api.requests = set(), 401, []
    with pytest.raises(backends.BackendHTTPError, match="Incorrect API key"):
        backend.complete(MESSAGES)
    assert len(api.requests) == 1                                    # a wrong key is not retried
    api.status, api.quota, api.requests = None, True, []
    with pytest.raises(backends.BackendError, match="no credit left"):
        backend.complete(MESSAGES)
    assert len(api.requests) == 1                                    # nor is an empty account
    with pytest.raises(backends.BackendHTTPError):
        backends.post_json(url + "/chat/completions", {"x": 1}, sleep=waits.append)
    assert waits == []


@pytest.mark.parametrize("body,expected", [
    ('{"error": {"message": "Unsupported parameter: \'temperature\' is not supported with this model.", '
     '"param": "temperature", "code": "unsupported_parameter"}}', "temperature"),
    ('{"error": {"message": "Unsupported value: \'temperature\' does not support 0 with this model. Only the '
     'default (1) value is supported.", "param": "temperature", "code": "unsupported_value"}}', "temperature"),
    ('{"error": {"message": "Unsupported parameter: \'max_tokens\' is not supported with this model. Use '
     '\'max_completion_tokens\' instead.", "param": "max_tokens", "code": "unsupported_parameter"}}', "max_tokens"),
    ('{"error": {"message": "Invalid parameter: \'response_format\' of type \'json_object\' is not supported '
     'with this model.", "param": "response_format", "code": null}}', "response_format"),
    ('{"error": {"message": "Unrecognized request argument supplied: seed", "param": null, "code": null}}', None),
    ('{"error": {"message": "Unknown parameter: \'seed\'.", "param": null, "code": "unknown_parameter"}}', "seed"),
    ('{"error": {"message": "Invalid value for \'messages\': expected an array.", "param": "messages"}}', None),
    ('{"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}', None),
    ("<html>502 Bad Gateway</html>", None),
    ('{"error": "model overloaded"}', None),
])
def test_unsupported_parameter_is_read_from_the_error_reply(body, expected):
    assert backends.unsupported_parameter(body) == expected


# ------------------------------------------------------------------ what p-hat is when temperature is ignored

class _NoTemperature(backends.ChatBackend):
    """A model that only runs at its default temperature: every reply is a fresh draw."""
    kind = "mock"

    def __init__(self, honoured=False):
        super().__init__("mock")
        self.honoured = honoured
        self.calls = 0

    def complete(self, messages, *, temperature=0.0, max_tokens=200, seed=None, json_mode=True):
        self.calls += 1
        keys = keys_in_prompt(messages)
        probs = np.random.default_rng(seed).dirichlet(np.ones(len(keys)))
        text = json.dumps({k: float(p) for k, p in zip(keys, probs)})
        return backends.Completion(text=text, temperature_sent=self.honoured)


def test_p_hat_uses_every_reply_when_the_model_ignores_temperature(tmp_path, small_problem, caplog):
    prob = small_problem
    source = ProblemPromptSource(prob)
    spec = PersonaSpec(n_samples=4, seed=2)
    cells = np.arange(15)
    path = str(tmp_path / "calls.jsonl")
    cold = PersonaRunner(prob, source, _NoTemperature(), spec, CallCache(path))
    with caplog.at_level(logging.WARNING, logger="sim.simulators.persona"):
        out = cold.run("cold", cells=cells)
    assert out.meta["temperature_honoured"] is False and "all replies" in out.meta["p_hat"]
    assert cold.counts["default_temperature"] == cold.counts["calls"] == 15 * 5
    assert any("default temperature" in r.message for r in caplog.records)
    # p-hat = mean of the "greedy" reply and the four samples (all are draws from the same law)
    again = PersonaRunner(prob, source, _NoTemperature(), spec, CallCache(None))
    greedy, _ = again._ask(0, "greedy0", spec.temperature, spec.seed)
    np.testing.assert_allclose(out.probs[0], (greedy + out.members[:, 0].sum(0)) / 5)
    assert not np.allclose(out.probs[0], greedy)
    # a rerun from the cache gives the same output even though no request is made
    resumed_backend = _NoTemperature(honoured=True)
    resumed = PersonaRunner(prob, source, resumed_backend, spec, CallCache(path)).run("cold", cells=cells)
    assert resumed_backend.calls == 0 and resumed.meta["temperature_honoured"] is False
    np.testing.assert_allclose(resumed.probs, out.probs)
    # when the temperature is honoured, p-hat is the temperature-0 reply, as specified
    warm = PersonaRunner(prob, source, _NoTemperature(honoured=True), spec, CallCache(None))
    out_warm = warm.run("warm", cells=cells)
    np.testing.assert_allclose(out_warm.probs[0], greedy)
    assert out_warm.meta["temperature_honoured"] is True and warm.counts["default_temperature"] == 0


# ------------------------------------------------------------------ configuration

def test_api_model_is_gpt_6_luna_with_reasoning_switched_off(root_dir):
    cfg = config.load_config(root=root_dir)
    model = cfg["llm"]["models"]["gpt6_luna"]
    assert model["model"] == "gpt-6-luna" and model["backend"] == "openai_compat"
    assert model["extra_body"] == {"reasoning_effort": "none"}            # the model's own default is "medium"
    assert model["max_tokens_field"] == "max_completion_tokens" and model["cutoff"] == "2026-05"
    assert "gpt54_nano" not in cfg["llm"]["models"]
    for sim_id in ("r1a_gpt", "r1b_gpt", "r2_gpt"):
        assert cfg["simulators"][sim_id]["model"] == "gpt6_luna"
    assert cfg["recall_probe"]["models"] == ["llama31_8b", "gpt6_luna"]
    backend = backends.make_backend({k: v for k, v in model.items() if k != "max_tokens"})
    assert backend._payload(MESSAGES, 0.0, 400, 1, True) == {
        "model": "gpt-6-luna", "messages": MESSAGES, "max_completion_tokens": 400, "temperature": 0.0, "seed": 1,
        "response_format": {"type": "json_object"}, "reasoning_effort": "none"}


def test_s2_configuration_is_consistent(root_dir):
    cfg = config.load_config(os.path.join(root_dir, "configs", "s2.yaml"), root=root_dir)
    assert cfg["paths"]["results_dir"] == "results/s2"
    for section in ("rq1", "rq2", "rq3"):
        assert all(s in cfg["simulators"] for s in cfg[section]["simulators"]), section
        assert {"r2_gpt", "r2_llama"} <= set(cfg[section]["simulators"])
    assert all(a in cfg["rq1"]["simulators"] and b in cfg["rq1"]["simulators"] for a, b, _ in cfg["rq1"]["contrasts"])
    assert cfg["testbed"]["aasb"]["context_mode"] == "outside"


# ------------------------------------------------------------------ command line

SMALL = {
    "paths": {"results_dir": "results", "simulated_dir": "simulated", "cache_dir": "cache"},
    "testbed": {"name": "synthetic",
                "synthetic": {"group_sizes": [8, 7, 6], "n_dec": 5, "response_rate": 0.8, "seed": 2}},
    "simulators": {"mock_named": {"kind": "persona", "variant": "named", "model": "mock"},
                   "noise": {"kind": "synthetic", "mode": "iid", "e": 0.2, "seed": 1, "n_members": 3}},
    "llm": {"persona": {"n_samples": 1}},
    "rq1": {"simulators": ["mock_named", "noise", "r0_prior"], "cv_folds": 3, "n_boot": 20, "contrasts": []},
    "rq2": {"simulators": ["noise"], "reps": 50},
    "rq3": {"simulators": ["noise"], "rhos": [0.5], "reps": 3, "n_boot": 20},
    "synthetic": {"n_sim_seeds": 1, "rhos": [0.5], "reps": 2, "n_members": 2,
                  "scenarios": [{"name": "structured", "mode": "structured", "groups": [0], "k": 2, "e_s": [0.9]}]},
    "sensitivity": {"simulators": ["noise"], "rhos": [0.5], "reps": 2, "variants": [{"name": "baseline", "set": {}}]},
    "figures": {"formats": ["png"], "dpi": 60},
}


@pytest.fixture()
def project(tmp_path, root_dir, monkeypatch):
    os.makedirs(tmp_path / "configs")
    with open(os.path.join(root_dir, "configs", "default.yaml"), encoding="utf-8") as f:
        (tmp_path / "configs" / "default.yaml").write_text(f.read(), encoding="utf-8")
    (tmp_path / "configs" / "small.yaml").write_text(yaml.safe_dump(SMALL), encoding="utf-8")
    monkeypatch.setenv("SIM_ROOT", str(tmp_path))
    return tmp_path, str(tmp_path / "configs" / "small.yaml")


def test_one_command_runs_the_pipeline_and_keeps_a_log(project, capsys):
    root, small = project
    assert cli.main(["status", "-c", small]) == 0
    before = capsys.readouterr().out
    assert "mock_named" in before and "False" in before and "cached model replies: none" in before
    assert cli.main(["simulate", "-c", small, "-m", "mock_named"]) == 0
    assert cli.main(["all", "-c", small]) == 0
    out = capsys.readouterr().out
    assert "all done: describe, rq1, rq2, rq3, synthetic, sensitivity, report" in out
    results = root / "results" / "synthetic"
    for step in ("describe", "rq1", "rq2", "rq3", "synthetic", "sensitivity"):
        assert (results / step / "manifest.json").exists(), step
    assert (results / "paper" / "fig2_goodhart_curves.png").exists()
    logs = sorted(glob.glob(str(results / "logs" / "all-*.log")))
    assert len(logs) == 1
    text = open(logs[0], encoding="utf-8").read()
    assert "step 1/7: describe" in text and "step 7/7: report" in text and "rq3 done in" in text
    assert glob.glob(str(results / "logs" / "simulate-*.log"))
    # status now reports what exists
    assert cli.main(["status", "-c", small]) == 0
    after = capsys.readouterr().out
    assert "{'mock': " in after and "rq3" in after and "files" in after
    info = cli.status(config.load_config(small, root=str(root)))
    assert info["cached_replies"]["mock"] > 0 and set(cli.STEPS[:-1]) <= set(info["results"])
    persona = next(s for s in info["simulators"] if s["simulator"] == "mock_named")
    assert persona["precomputed"] is True and persona["is_mock"] is True and persona["temperature_honoured"] is True


def test_selected_steps_run_in_pipeline_order(project, capsys):
    root, small = project
    assert cli.main(["all", "-c", small, "--steps", "rq2", "describe"]) == 0
    assert "all done: describe, rq2" in capsys.readouterr().out
    results = root / "results" / "synthetic"
    assert (results / "rq2" / "pressure_curve.csv").exists() and not (results / "rq3").exists()
    with pytest.raises(SystemExit):
        cli.main(["all", "-c", small, "--steps", "rq9"])


def test_a_failing_command_exits_with_1_and_logs_the_reason(project, capsys):
    root, small = project
    code = cli.main(["rq2", "-c", small, "--set", "rq2.simulators=[mock_named]"])     # not simulated yet
    assert code == 1
    logs = glob.glob(str(root / "results" / "synthetic" / "logs" / "rq2-*.log"))
    assert len(logs) == 1
    text = open(logs[0], encoding="utf-8").read()
    assert "rq2 failed" in text and "MissingSimulatorOutput" in text
    assert "python -m sim simulate --simulator mock_named" in text
    # a dry run and read-only commands leave no log behind
    assert cli.main(["simulate", "-c", small, "-m", "mock_named", "--dry-run"]) == 0
    assert cli.main(["show-config", "-c", small]) == 0
    capsys.readouterr()
    assert not glob.glob(str(root / "results" / "synthetic" / "logs" / "simulate-*.log"))
    assert logging.getLogger().handlers and not any(
        isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers)
