"""Tests for agent design v5 (docs/agent_design_v5.md)."""

import json

import pytest
from pydantic import ValidationError

from sca.config import get_config, load_agent_v5_config
from sca.data.scenarios import generate_scenarios
from sca.llm.client import LLMClient
from sca.sim.environment import ECHELONS, BeerGameEnv, EchelonState
from sca.systems.base import get_system
from sca.systems.v5 import (
    ACTIONS,
    ActionDecision,
    RegimeTracker,
    V5_SYSTEM_IDS,
    basic_reference,
    compute_order,
    nearest_action,
    summarize_decisions,
)
from sca.tools import safe_order_bounds

LOG_FIELDS = {"action", "llm_regime", "detector_regime", "reference", "multiplier", "order_pre_clip",
              "order_post_clip", "band_low", "band_high", "prompt_version"}


@pytest.fixture(scope="module")
def scenarios():
    return generate_scenarios()


def _run(system_id, scenario, seed=102, steps=40, mode="detector_copy", config=None):
    system = get_system(system_id, config={"db_path": False, "seed": seed, **(config or {})})
    if mode is not None:
        system.llm_client.mock.configure_v5_mode(mode, seed=seed)
    system.calibrate(scenario)
    system.reset(scenario)
    env = BeerGameEnv(config=get_config().sim, scenario=scenario, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)
    system.attach_env(env)
    for _ in range(steps):
        states = env.step(system.decide_orders(states))
    return system, env


def _state(**kw):
    base = dict(echelon="retailer", step=3, inventory=150.0, backlog=0.0, pipeline=50.0, incoming_shipments=25.0,
                current_demand=100.0, last_order=100.0, lead_time=2, history_window=[])
    return EchelonState(**{**base, **kw})


# --- output schema -----------------------------------------------------------
def test_action_decision_schema():
    ok = ActionDecision.model_validate({"regime": "shock", "action": "boost_large", "rationale": "Demand jumped."})
    assert ok.action == "boost_large"
    for bad in (
        {"regime": "shock", "action": "order_more", "rationale": "x"},
        {"regime": "spike", "action": "follow", "rationale": "x"},
        {"regime": "stable", "action": "follow", "rationale": " ".join(["w"] * 21)},
    ):
        with pytest.raises(ValidationError):
            ActionDecision.model_validate(bad)


# --- deterministic pieces ----------------------------------------------------
def test_basic_reference_formula():
    out = basic_reference([90, 100, 110, 120, 130, 140], inventory_position=200.0, window=5, horizon=4.0,
                          mean_lead_time=3.0)
    mean_w = 120.0
    assert out["trailing_mean"] == pytest.approx(mean_w)
    assert out["s_basic"] == pytest.approx(mean_w * 3.0)
    assert out["reference"] == pytest.approx(mean_w + (mean_w * 3.0 - 200.0) / 4.0)
    assert basic_reference([10.0], inventory_position=10_000.0, window=5, horizon=4.0, mean_lead_time=3.0)[
        "reference"] == 0.0


def test_compute_order_clips_into_band():
    assert compute_order(100.0, 1.8, 50.0, 150.0) == (pytest.approx(180.0), 150.0)
    assert compute_order(100.0, 0.9, 95.0, 150.0) == (pytest.approx(90.0), 95.0)
    assert compute_order(100.0, 1.2, 50.0, 150.0) == (pytest.approx(120.0), pytest.approx(120.0))


def test_multipliers_come_from_config():
    cfg = load_agent_v5_config()
    system = get_system("single_agent_v5", config={"db_path": False})
    assert system.multipliers == cfg.multipliers
    assert cfg.multipliers["follow"] == 1.0
    assert len(set(cfg.multipliers.values())) == 4  # all four actions distinct


def test_nearest_action():
    m = {"follow": 1.0, "boost_small": 1.2, "boost_large": 1.8, "trim": 0.9}
    assert nearest_action(100.0, 100.0, m) == "follow"
    assert nearest_action(185.0, 100.0, m) == "boost_large"
    assert nearest_action(88.0, 100.0, m) == "trim"


def test_regime_tracker_phases():
    tracker = RegimeTracker(load_agent_v5_config())
    phases = [tracker.update("retailer", d)["phase"] for d in [100.0] * 15 + [200.0] * 3 + [100.0] * 20]
    assert "shock" in phases and "after_shock" in phases
    first_shock = phases.index("shock")
    assert phases.index("after_shock") > first_shock
    assert phases[-1] == "stable"


# --- band: shared function, widened only by the detector ----------------------
def test_band_is_shared_safe_order_bounds():
    system = get_system("single_agent_v5", config={"db_path": False})
    band = system._band("retailer", _state(), demand_estimate=100.0, phase="stable")
    low, high = safe_order_bounds(last_order=100.0, forecast_demand=100.0, rate_limit_delta=system.cfg.band.delta,
                                  forecast_cap_multiple=system.cfg.band.forecast_cap_multiple, max_order=500.0)
    assert (band["low"], band["high"]) == (low, high)


def test_band_widens_only_from_detector_phase_and_only_for_s5_s6():
    cfg = load_agent_v5_config()
    wide_cfg = cfg.model_copy(update={"band": cfg.band.model_copy(update={"delta_shock": 1.5})})
    s3 = get_system("single_agent_v5", config={"db_path": False}, v5_config=wide_cfg)
    s5 = get_system("multi_agent_tools_v5", config={"db_path": False}, v5_config=wide_cfg)
    st = _state(last_order=100.0)
    assert s5._band("retailer", st, 100.0, "shock")["delta"] == 1.5
    assert s5._band("retailer", st, 100.0, "stable")["delta"] == cfg.band.delta
    assert s3._band("retailer", st, 100.0, "shock")["delta"] == cfg.band.delta


def test_llm_regime_never_widens_band(scenarios):
    # Mock always says "stable"/follow vs. detector_copy: the band depends only on the detector
    a, _ = _run("multi_agent_tools_v5", scenarios["shock"], mode="always_follow")
    b, _ = _run("multi_agent_tools_v5", scenarios["shock"], mode="detector_copy")
    widened_a = [r["band_widened"] for r in a.decision_log]
    widened_b = [r["band_widened"] for r in b.decision_log]
    phases = [r["detector_phase"] == "shock" for r in a.decision_log]
    assert widened_a == phases
    assert widened_a == widened_b


# --- systems -----------------------------------------------------------------
@pytest.mark.parametrize("system_id", V5_SYSTEM_IDS)
def test_same_action_space_and_logging(system_id, scenarios):
    system, _ = _run(system_id, scenarios["shock"], steps=10)
    rows = system.decision_log
    assert len(rows) == 10 * len(ECHELONS)
    assert LOG_FIELDS <= set(rows[0])
    assert all(r["prompt_version"] == "v5.0" for r in rows)
    assert all(r["action"] in ACTIONS for r in rows)
    for r in rows:
        assert r["band_low"] - 1e-9 <= r["order_post_clip"] <= r["band_high"] + 1e-9
        assert r["order_pre_clip"] == pytest.approx(r["reference"] * r["multiplier"], abs=1e-3)
    assert system.llm_stats() == {"decisions": 40, "fallbacks": 0, "llm_calls": 40}


def test_systems_differ_only_in_information_tools_orchestration(scenarios):
    prompts = {}
    for sid in V5_SYSTEM_IDS:
        system = get_system(sid, config={"db_path": False})
        system.llm_client.mock.configure_v5_mode("always_follow")
        sc = scenarios["stable"]
        system.calibrate(sc)
        system.reset(sc)
        env = BeerGameEnv(config=get_config().sim, scenario=sc, seed=100, steady_state_init=True)
        system.decide_orders(env.reset(seed=100))
        prompts[sid] = system.llm_client.mock.call_history[0]["prompt"]
    assert "ALL ECHELONS" not in prompts["single_agent_v5"]
    assert "ALL ECHELONS" in prompts["multi_agent_info_v5"]
    assert "TOOL OUTPUTS" not in prompts["multi_agent_info_v5"]
    assert "TOOL OUTPUTS" in prompts["multi_agent_tools_v5"]
    assert "SPECIALIST PROPOSALS" not in prompts["multi_agent_tools_v5"]
    assert "SPECIALIST PROPOSALS" in prompts["orchestrated_v5"]
    # S3/S4 prompts carry no regime flag
    assert "Regime detector" not in prompts["single_agent_v5"]
    for p in prompts.values():
        assert '"action": "follow" | "boost_small" | "boost_large" | "trim"' in p


def test_s5_reference_is_smoothed_order_up_to_tool(scenarios):
    system, _ = _run("multi_agent_tools_v5", scenarios["stable"], steps=3)
    assert system.parameters["reference"] == "smoothed_order_up_to"
    assert get_system("single_agent_v5", config={"db_path": False}).parameters["reference"] == "basic"


# --- MockLLM modes -----------------------------------------------------------
@pytest.mark.parametrize("mode,expect_fallback", [
    ("valid", False), ("always_follow", False), ("detector_copy", False),
    ("malformed", True), ("bad_action", True),
])
def test_mock_modes(mode, expect_fallback, scenarios):
    system, _ = _run("single_agent_v5", scenarios["shock"], steps=10, mode=mode)
    s = summarize_decisions(system.decision_log)
    assert (s["fallback_rate"] == 1.0) if expect_fallback else (s["fallback_rate"] == 0.0)
    assert system.llm_stats()["llm_calls"] == system.llm_stats()["decisions"]
    if mode == "always_follow":
        assert s["action_mix"]["follow"] == 1.0
    if mode == "detector_copy":
        assert s["regime_accuracy"] == 1.0
    if expect_fallback:
        assert all(r["order_post_clip"] <= r["band_high"] + 1e-9 for r in system.decision_log)


def test_mock_context_never_reaches_cache_key(tmp_path):
    client = LLMClient(mock_mode=True, cache_dir=tmp_path, db_path=tmp_path / "t.duckdb")
    client.mock.configure_v5_mode("always_follow")
    k = []
    for ctx in ({"detector_regime": "stable"}, {"detector_regime": "shock"}):
        client.generate("p", response_model=ActionDecision, use_cache=True, log_to_db=False, mock_context=ctx)
        k.append(len(list(tmp_path.glob("*.json"))))
    assert k == [1, 1]  # second call hit the same cache entry


# --- scripted choosers -------------------------------------------------------
def test_scripted_choosers(scenarios):
    follow, _ = _run("single_agent_v5", scenarios["shock"], mode=None, config={"chooser": "follow"})
    assert {r["action"] for r in follow.decision_log} == {"follow"}
    assert follow.llm_stats()["decisions"] == 0  # scripted runs are not LLM runs
    mapped, _ = _run("single_agent_v5", scenarios["shock"], mode=None, config={"chooser": "detector_map"})
    assert {"boost_large", "trim"} <= {r["action"] for r in mapped.decision_log}
    oracle, _ = _run("single_agent_v5", scenarios["shock"], mode=None, config={"chooser": "oracle"})
    assert len(oracle.decision_log) == 160


def test_reset_clears_per_run_state(scenarios):
    system, _ = _run("orchestrated_v5", scenarios["shock"], steps=5)
    system.reset(scenarios["shock"])
    assert system.decision_log == [] and system.llm_stats() == {"decisions": 0, "fallbacks": 0, "llm_calls": 0}
    assert all(v == [] for v in system.received.values())
    assert all(v == [] for v in system.tracker.history.values())


def test_db_logging_writes_v5_decisions(tmp_path, scenarios):
    import duckdb

    db = tmp_path / "v5.duckdb"
    system = get_system("single_agent_v5", config={"seed": 1}, db_path=db)
    system.llm_client.mock.configure_v5_mode("detector_copy")
    sc = scenarios["stable"]
    system.calibrate(sc)
    system.reset(sc)
    env = BeerGameEnv(config=get_config().sim, scenario=sc, seed=1, steady_state_init=True)
    system.decide_orders(env.reset(seed=1))
    rows = duckdb.connect(str(db), read_only=True).execute(
        "select action, llm_regime, detector_regime, reference, multiplier, order_pre_clip, order_post_clip, "
        "band_low, band_high, prompt_version from v5_decisions").fetchall()
    assert len(rows) == 4 and all(r[-1] == "v5.0" for r in rows)
    assert json.dumps(rows, default=str)
