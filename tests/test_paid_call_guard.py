"""Tests for the paid-call guard, fatal API errors (401/402/403), and run-validity rules."""

import time
from pathlib import Path
from typing import Any, List

import pytest

import sca.eval.runner as runner_mod
from sca.config import ExperimentConfig, SystemSpec
from sca.eval.runner import EvaluationRunner, RunResult, run_single_simulation
from sca.eval.validity import BatchHaltedError, check_run_validity
from sca.llm.client import (
    BaseLLMAdapter,
    FatalAPIError,
    LLMClient,
    PaidCallsDisabledError,
)
from sca.sim.environment import EchelonState
from sca.systems.single_agent import SingleAgentSystem
from sca.systems.base import get_system
from sca.data.scenarios import generate_scenarios
from sca.config import load_sim_config

REAL_MODEL = "glm-isoquant"  # openai_compatible entry in configs/models.yaml


class HTTPStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeNetworkAdapter(BaseLLMAdapter):
    """Stands in for a real provider adapter; records every 'network' call."""

    def __init__(self, error: Exception = None):
        super().__init__()
        self.calls: List[str] = []
        self.error = error

    def generate_raw(self, prompt: str, **kwargs: Any):
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return "hello", 5, 5, 0


@pytest.fixture
def real_client(tmp_path: Path, monkeypatch):
    """LLMClient routed to a fake network adapter (no real HTTP ever happens)."""
    client = LLMClient(mock_mode=False, cache_dir=tmp_path / "cache", db_path=tmp_path / "t.duckdb")
    adapter = FakeNetworkAdapter()
    monkeypatch.setattr(client, "get_adapter", lambda *_a, **_k: adapter)
    client._fake_adapter = adapter
    return client


def _gen(client: LLMClient, **kw):
    return client.generate("ping", model=REAL_MODEL, log_to_db=False, max_retries=3, **kw)


# ---------------------------------------------------------------------------
# Paid-call guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, "", "0", "true", "yes"])
def test_real_call_refused_unless_allow_paid_calls_is_1(real_client, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ALLOW_PAID_CALLS", raising=False)
    else:
        monkeypatch.setenv("ALLOW_PAID_CALLS", value)
    with pytest.raises(PaidCallsDisabledError):
        _gen(real_client, use_cache=False)
    assert real_client._fake_adapter.calls == []


def test_real_call_allowed_with_flag(real_client, monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_CALLS", "1")
    resp = _gen(real_client, use_cache=False)
    assert resp.content == "hello"
    assert len(real_client._fake_adapter.calls) == 1


def test_mock_calls_work_without_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_PAID_CALLS", raising=False)
    client = LLMClient(mock_mode=True, cache_dir=tmp_path / "cache", db_path=tmp_path / "t.duckdb")
    resp = client.generate("ping", model=REAL_MODEL, use_cache=False, log_to_db=False)
    assert resp.content


def test_cached_calls_work_without_flag(real_client, monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_CALLS", "1")
    _gen(real_client, use_cache=True)  # populate cache
    monkeypatch.delenv("ALLOW_PAID_CALLS")
    resp = _gen(real_client, use_cache=True)
    assert resp.cached is True
    assert len(real_client._fake_adapter.calls) == 1


# ---------------------------------------------------------------------------
# HTTP 401 / 402 / 403: stop immediately, no retry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [401, 402, 403])
def test_fatal_status_not_retried(real_client, monkeypatch, status):
    monkeypatch.setenv("ALLOW_PAID_CALLS", "1")
    real_client._fake_adapter.error = HTTPStatusError(status)
    with pytest.raises(FatalAPIError) as exc_info:
        _gen(real_client, use_cache=False)
    assert exc_info.value.status_code == status
    assert len(real_client._fake_adapter.calls) == 1


def test_retryable_status_still_retried(real_client, monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_CALLS", "1")
    monkeypatch.setattr("sca.llm.client.time.sleep", lambda *_: None)
    real_client._fake_adapter.error = HTTPStatusError(503)
    with pytest.raises(HTTPStatusError):
        _gen(real_client, use_cache=False)
    assert len(real_client._fake_adapter.calls) == 4


def _state() -> EchelonState:
    return EchelonState(
        echelon="retailer", step=1, inventory=150.0, backlog=0.0, pipeline=50.0,
        incoming_shipments=25.0, current_demand=100.0, last_order=100.0, lead_time=2,
        history_window=[],
    )


@pytest.mark.parametrize("exc", [FatalAPIError("HTTP 402", status_code=402), PaidCallsDisabledError("no")])
def test_system_does_not_fall_back_on_fatal_errors(exc):
    class RaisingClient:
        config = None

        def generate(self, *a, **k):
            raise exc

    system = SingleAgentSystem(name="single_agent", llm_client=RaisingClient(), db_path=False)
    system.log_to_db = False
    with pytest.raises(type(exc)):
        system.decide_orders({"retailer": _state()})
    assert system.fallback_count == 0


# ---------------------------------------------------------------------------
# Run validity
# ---------------------------------------------------------------------------
def test_check_run_validity_rules():
    assert check_run_validity({"decisions": 100, "fallbacks": 5, "llm_calls": 100}) is None
    assert "fallback rate" in check_run_validity({"decisions": 100, "fallbacks": 6, "llm_calls": 106})
    assert "fewer than decisions" in check_run_validity({"decisions": 100, "fallbacks": 0, "llm_calls": 99})


def test_system_counts_reset_per_run(tmp_path):
    client = LLMClient(mock_mode=True, db_path=tmp_path / "t.duckdb")
    system = SingleAgentSystem(name="single_agent", llm_client=client, db_path=False)
    system.log_to_db = False
    system.decide_orders({"retailer": _state()})
    assert system.llm_stats()["decisions"] == 1
    assert system.llm_stats()["llm_calls"] >= 1
    system.reset()
    assert system.llm_stats() == {"decisions": 0, "fallbacks": 0, "llm_calls": 0}


def test_run_marked_invalid_on_high_fallback_rate(monkeypatch):
    def fake_get_system(system_id):
        sys_ = get_system(system_id)
        sys_.llm_stats = lambda: {"decisions": 100, "fallbacks": 10, "llm_calls": 110}
        return sys_

    monkeypatch.setattr(runner_mod, "get_system", fake_get_system)
    scenario = generate_scenarios()["stable"]
    res = run_single_simulation(
        runner_mod.RunSpec(system_id="baseline_s_S", system_type="heuristic", scenario_name="stable", seed=100),
        scenario,
        load_sim_config().model_dump(),
        "t",
    )
    assert res.status == "invalid"
    assert "fallback rate" in res.error


def test_run_aborted_on_fatal_error(monkeypatch):
    def fake_get_system(system_id):
        sys_ = get_system(system_id)

        def boom(_states):
            raise FatalAPIError("HTTP 401", status_code=401)

        sys_.decide_orders = boom
        return sys_

    monkeypatch.setattr(runner_mod, "get_system", fake_get_system)
    res = run_single_simulation(
        runner_mod.RunSpec(system_id="baseline_s_S", system_type="heuristic", scenario_name="stable", seed=100),
        generate_scenarios()["stable"],
        load_sim_config().model_dump(),
        "t",
    )
    assert res.status == "aborted"


@pytest.mark.parametrize("bad_status", ["invalid", "aborted"])
def test_batch_halts_on_bad_run(tmp_path, monkeypatch, bad_status):
    db = tmp_path / "b.duckdb"
    cfg = ExperimentConfig(
        experiment_name="halt_test",
        db_path=db,
        output_dir=tmp_path / "results",
        systems=[SystemSpec(id="single_agent", type="agentic", enabled=True)],
        scenarios=["stable"],
        seeds=[100, 101, 102, 103],
        concurrency={"max_workers_non_llm": 1, "max_workers_llm": 1},
    )
    monkeypatch.setattr(runner_mod, "verify_config_matches_manifest", lambda *_a, **_k: (True, "h", "h"))
    executed = []

    def fake_run(spec, scenario, sim_cfg, exp_name, max_fallback_rate=0.05):
        executed.append(spec.run_id)
        if len(executed) > 1:
            time.sleep(0.3)  # later runs are slow, so queued ones can still be cancelled
        return RunResult(spec=spec, status=bad_status, experiment_name=exp_name, error="bad")

    monkeypatch.setattr(runner_mod, "run_single_simulation", fake_run)
    runner = EvaluationRunner(experiment_config=cfg, db_path=db)
    with pytest.raises(BatchHaltedError):
        runner.run_grid()
    assert len(executed) <= 2
