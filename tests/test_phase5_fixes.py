"""Regression tests for the bugs found in docs/diagnosis_phase5.md.

1. System 3 request failure: truncated/empty output must surface as TruncatedOutputError, without a
   silent second request, without a pointless retry, and with its token cost recorded.
2. System 6 vs System 2c plumbing: S6's tool target is the 2c tool chain; ordering it reproduces 2c;
   validation keeps the clipped order; LLM failures count as fallbacks; db_path=False writes nothing.
3. One shared safe-range function for the prompts and validate_order.
"""

import random
from types import SimpleNamespace

import numpy as np
import pytest
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion

from sca.baselines.smoothed_out import load_frozen_2c_params
from sca.config import get_config, load_models_config
from sca.data.scenarios import generate_scenarios
from sca.llm.client import OpenAICompatibleAdapter, TruncatedOutputError, check_completion
from sca.sim.environment import ECHELONS, BeerGameEnv, EchelonState
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import get_system
from sca.systems.orchestrated import OrchestratedSystem
from sca.systems.single_agent import SingleAgentSystem
from sca.tools import ValidateOrderInput, safe_order_bounds, validate_order


def _completion(content, finish_reason, prompt_tokens=900, completion_tokens=256):
    return ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 0, "model": "glm-5.3-flash",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    })


def _state(echelon="retailer", last_order=100.0, step=1):
    return EchelonState(
        echelon=echelon, step=step, inventory=150.0, backlog=0.0, pipeline=50.0,
        incoming_shipments=25.0, current_demand=100.0, last_order=last_order, lead_time=2,
        history_window=[],
    )


# ---------------------------------------------------------------------------
# 1. System 3 request failure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("content,finish", [(None, "length"), ("", "stop"), ('{"order": 1', "length")])
def test_truncated_or_empty_output_raises_with_tokens(content, finish):
    with pytest.raises(TruncatedOutputError) as exc_info:
        check_completion(_completion(content, finish), "glm-5.3-flash")
    assert exc_info.value.prompt_tokens == 900
    assert exc_info.value.completion_tokens == 256


def test_valid_completion_passes():
    assert check_completion(_completion('{"order": 1}', "stop"), "m")[0] == '{"order": 1}'


def test_length_stop_in_parse_does_not_send_second_request(monkeypatch):
    """v4.1 bug: parse() hit the length limit, the adapter silently re-sent via create() and returned ''."""
    monkeypatch.setenv("ISOQUANT_API_KEY", "test-key-not-real")
    adapter = OpenAICompatibleAdapter(load_models_config().get_model("glm-isoquant"))
    created = []

    def parse(**_kw):
        raise LengthFinishReasonError(completion=_completion(None, "length"))

    adapter.client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: created.append(kw))),
    )
    from sca.systems.llm_base import OrderDecision

    with pytest.raises(TruncatedOutputError):
        adapter.generate_raw("prompt", response_model=OrderDecision)
    assert created == []


def test_s3_truncation_falls_back_once_and_records_cost(monkeypatch):
    calls = []

    class TruncatingClient:
        config = None

        def generate(self, *a, **k):
            calls.append(1)
            raise TruncatedOutputError("empty", finish_reason="length", prompt_tokens=900, completion_tokens=256)

    system = SingleAgentSystem(name="single_agent", llm_client=TruncatingClient(), db_path=False,
                               config={"model": "glm-isoquant"})
    logged = {}
    monkeypatch.setattr(system, "_log_decision_to_duckdb", lambda **kw: logged.update(kw))
    system.decide_orders({"retailer": _state()})

    assert len(calls) == 1  # no retry with the same token budget
    assert logged["fallback"] is True
    assert logged["error_type"] == "TruncatedOutputError"
    assert logged["completion_tokens"] == 256
    assert logged["cost_usd"] > 0
    assert system.llm_stats() == {"decisions": 1, "fallbacks": 1, "llm_calls": 1}


# ---------------------------------------------------------------------------
# 3. Shared safe range
# ---------------------------------------------------------------------------
def test_validate_order_clips_exactly_into_safe_order_bounds():
    rng = random.Random(0)
    for _ in range(300):
        kw = dict(
            last_order=rng.choice([None, 0.0, rng.uniform(0, 400)]),
            forecast_demand=rng.choice([None, 0.0, rng.uniform(0, 400)]),
            rate_limit_delta=rng.choice([None, rng.uniform(0, 1)]),
            forecast_cap_multiple=rng.choice([None, rng.uniform(1, 3)]),
            capacity_limit=rng.choice([None, 300.0]),
            max_order=rng.choice([None, 500.0]),
        )
        proposed = rng.uniform(-50, 900)
        low, high = safe_order_bounds(**kw)
        out = validate_order(ValidateOrderInput(proposed_order=proposed, **kw), db_path=False)
        assert out.validated_order == pytest.approx(round(min(max(proposed, low), high), 2), abs=0.011)


def test_s3_prompt_safe_range_is_the_validated_range():
    system = SingleAgentSystem(name="single_agent", llm_client=SimpleNamespace(config=None), db_path=False)
    state = _state(last_order=100.0)
    tools = system.compute_deterministic_tools(state, "retailer")
    low, high = safe_order_bounds(last_order=100.0, forecast_demand=tools["forecast"], max_order=500.0)
    assert tools["safe_range"] == (round(low, 1), round(high, 1))
    # Uses the configured delta, not a hard-coded 0.5
    delta = get_config().sim.four_echelon.rate_limit_delta
    assert low == pytest.approx(max(0.0, 100.0 - delta * max(100.0, tools["forecast"])), abs=0.06)


def test_s6_validation_enforces_prompt_range_and_keeps_clipped_order():
    system = OrchestratedSystem(config={"db_path": False}, db_path=False)
    orch = system.orchestrators["retailer"]
    state = _state(last_order=100.0)
    bounds = orch.safe_bounds(state, forecast=100.0, capacity_limit=500.0)
    out = validate_order(ValidateOrderInput(proposed_order=10_000.0, inventory_position=0.0,
                                            max_order=500.0, **bounds["limits"]), db_path=False)
    assert out.validated_order == pytest.approx(round(bounds["high"], 2))

    # Inside S6: an out-of-range order is clipped to the bound, not replaced by last_order
    orch.run_negotiation = lambda **kw: (10_000.0, 1, 1, "negotiation_compromise", 0)
    orch.detect_conflicts = lambda *a, **k: SimpleNamespace(has_conflict=True, conflict_type="x", details={})
    orders = system.decide_orders({"retailer": state})
    rec = system.decision_history[-1]
    assert orders["retailer"] < 10_000.0
    assert orders["retailer"] != state.last_order
    assert rec.fallback_used is False


# ---------------------------------------------------------------------------
# 2. System 6 vs System 2c plumbing
# ---------------------------------------------------------------------------
def _run(policy, scenario, seed):
    env = BeerGameEnv(config=get_config().sim, scenario=scenario, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)
    policy.calibrate(scenario)
    policy.reset(scenario)
    for _ in range(40):
        states = env.step(policy.decide_orders(states))
    m = compute_simulation_metrics(env, warmup_steps=5)
    return m.total_cost, m.end_to_end_bullwhip


@pytest.mark.parametrize("scenario_name", ["stable", "seasonal", "shock"])
@pytest.mark.parametrize("seed", [100, 101, 102])
def test_s6_ordering_tool_target_matches_2c(scenario_name, seed):
    scenario = generate_scenarios()[scenario_name]
    cost_2c, bw_2c = _run(get_system("baseline_2c", config={**load_frozen_2c_params(), "db_path": False}),
                          scenario, seed)
    cost_6, bw_6 = _run(OrchestratedSystem(config={"order_source": "tool_target", "db_path": False}, db_path=False),
                        scenario, seed)
    assert cost_6 == pytest.approx(cost_2c, rel=0.05)
    assert bw_6 == pytest.approx(bw_2c, rel=0.05)


def test_s6_llm_failure_in_negotiation_counts_as_fallback():
    class FailingClient:
        config = None

        def generate(self, *a, **k):
            raise ValueError("empty content")

    scenario = generate_scenarios()["shock"]
    env = BeerGameEnv(config=get_config().sim, scenario=scenario, seed=101, steady_state_init=True)
    states = env.reset(seed=101)
    system = OrchestratedSystem(llm_client=FailingClient(), config={"db_path": False}, db_path=False)
    system.calibrate(scenario)
    system.reset(scenario)
    system.decide_orders(states)
    negotiated = [r for r in system.decision_history if r.rounds_used > 0]
    assert negotiated, "shock seed 101 step 0 negotiates (see diagnosis_phase5.md)"
    assert all(r.fallback_used for r in negotiated)
    assert system.total_fallbacks == len(negotiated)


def test_s6_reset_clears_per_run_state():
    system = OrchestratedSystem(config={"db_path": False}, db_path=False)
    system.total_fallbacks = 3
    system.target_policy.last_orders = {e: 999.0 for e in ECHELONS}
    scenario = generate_scenarios()["stable"]
    system.reset(scenario)
    assert system.total_fallbacks == 0
    assert system.target_policy.last_orders == {e: float(scenario.demand[0]) for e in ECHELONS}


def test_s6_db_path_false_writes_no_database(tmp_path, monkeypatch):
    system = OrchestratedSystem(config={"db_path": False}, db_path=False)
    monkeypatch.chdir(tmp_path)
    system.decide_orders({"retailer": _state()})
    assert list(tmp_path.iterdir()) == []
    assert np.isfinite(system.decision_history[-1].final_order)
