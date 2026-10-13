"""Unit tests for tool modes (deterministic vs llm_function_calling), prompt v3, and fallback rules.

Verifies:
1. Config option tool_mode: deterministic (default) vs llm_function_calling.
2. Deterministic mode:
   - Deterministic analytical tools executed by code.
   - LLM sees tool outputs (forecast, EOQ, smoothed order-up-to target, safe range).
   - Prompt v3 formatting: fixed instructions first, variable state last, last 5 orders,
     forecast, safe range, order stability objective, concise rationale (<40 words), JSON only.
   - Prompt version recorded in every row (e.g. single_agent_v3, multi_agent_info_v3, orchestrated_v3).
   - In deterministic mode, tool-choice and parameter accuracy in reports are marked
     "n/a (tools deterministic)", replaced with recommendation adoption rate, mean deviation,
     and override share.
3. LLM Function Calling mode:
   - Tool requests executed, evaluated against deterministic expected-tool rule.
   - Computes numeric tool-choice accuracy and parameter accuracy.
4. Fallback Rule:
   - Invalid output -> retry once -> fallback to previous order (state.last_order), counted and logged.
5. Decision log demonstrations for both modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import duckdb
import pytest

from sca.config import FourEchelonConfig
from sca.db import get_duckdb_connection
from sca.eval.behaviour import compute_behavioural_metrics, generate_behaviour_summary_table
from sca.llm.client import LLMClient
from sca.sim.environment import EchelonHistoryRecord, EchelonState
from sca.systems.llm_base import OrderDecision
from sca.systems.multi_agent_tools import MultiAgentToolsSystem
from sca.systems.single_agent import SingleAgentSystem


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Provide a fresh initialized DuckDB database file for isolation."""
    db_file = tmp_path / "test_tool_modes.duckdb"
    conn = get_duckdb_connection(db_file)
    conn.close()
    return db_file


def make_test_state(
    echelon: str = "retailer",
    step: int = 5,
    inventory: float = 120.0,
    backlog: float = 0.0,
    pipeline: float = 40.0,
    current_demand: float = 100.0,
    last_order: float = 95.0,
    lead_time: int = 2,
    order_history: List[float] = None,
) -> EchelonState:
    """Create an EchelonState with populated history for prompt v3 testing."""
    if order_history is None:
        order_history = [90.0, 92.0, 94.0, 95.0, 95.0]

    history = [
        EchelonHistoryRecord(
            step=i + 1,
            demand=95.0 + (i % 3),
            order_placed=ord_val,
            inventory=120.0,
            backlog=0.0,
            incoming_shipment=95.0,
            outgoing_shipment=95.0,
            cost=50.0,
        )
        for i, ord_val in enumerate(order_history)
    ]

    return EchelonState(
        echelon=echelon,
        step=step,
        inventory=inventory,
        backlog=backlog,
        pipeline=pipeline,
        incoming_shipments=25.0,
        current_demand=current_demand,
        last_order=last_order,
        lead_time=lead_time,
        history_window=history,
    )


# ===========================================================================
# 1. Configuration Verification
# ===========================================================================
def test_config_tool_mode_default_and_validation():
    """Verify tool_mode defaults to 'deterministic' and validates allowed values."""
    cfg = FourEchelonConfig()
    assert cfg.tool_mode == "deterministic"

    cfg_fc = FourEchelonConfig(tool_mode="llm_function_calling")
    assert cfg_fc.tool_mode == "llm_function_calling"

    with pytest.raises(ValueError, match="Invalid tool_mode"):
        FourEchelonConfig(tool_mode="invalid_mode")


# ===========================================================================
# 2. Deterministic Mode & Prompt v3 Verification
# ===========================================================================
def test_deterministic_mode_prompt_v3_structure(temp_db: Path):
    """Verify Prompt v3 structure: fixed instructions first, state last, last 5 orders, forecast, safe range."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    client.mock.set_response(
        "CURRENT LOCAL STATE",
        OrderDecision(quantity=100.0, reasoning="Order matches forecast and preserves inventory stability"),
    )

    system = SingleAgentSystem(
        name="single_agent",
        llm_client=client,
        db_path=temp_db,
        run_id="run_det_prompt_v3",
        tool_mode="deterministic",
    )

    state = make_test_state(echelon="retailer", step=5, last_order=95.0, order_history=[91.0, 92.0, 93.0, 94.0, 95.0])
    orders = system.decide_orders({"retailer": state})
    assert orders["retailer"] == 100.0

    # Retrieve the exact prompt sent to LLM from history
    assert len(client.mock.call_history) > 0
    prompt_text = client.mock.call_history[-1]["prompt"]

    # 1. Fixed instructions first
    assert "OPERATIONAL OBJECTIVE:" in prompt_text
    assert "DECISION CONSTRAINTS & OUTPUT FORMAT:" in prompt_text
    assert "Stable orders are part of the objective" in prompt_text
    assert "Keep the rationale strictly under 40 words" in prompt_text
    assert "Output JSON only" in prompt_text
    assert "deterministic tools plus LLM decision and negotiation" in prompt_text

    # 2. Variable state last (appears after fixed instructions)
    assert "CURRENT LOCAL STATE:" in prompt_text
    assert prompt_text.index("OPERATIONAL OBJECTIVE:") < prompt_text.index("CURRENT LOCAL STATE:")
    assert "Last 5 Orders: [91.0, 92.0, 93.0, 94.0, 95.0]" in prompt_text
    assert "Demand Forecast (t+1):" in prompt_text
    assert "Safe Order Range:" in prompt_text
    assert "Smoothed Order-Up-To Target (Tool Recommendation):" in prompt_text

    # 3. Prompt version recorded in agent_calls
    conn = duckdb.connect(str(temp_db))
    row = conn.execute(
        """
        SELECT prompt_version, tool_recommendation, order_quantity
        FROM agent_calls
        WHERE run_id = 'run_det_prompt_v3' AND echelon = 'retailer'
        """
    ).fetchone()
    conn.close()

    assert row is not None
    p_ver, rec, qty = row
    assert p_ver == "single_agent_v3"
    assert rec is not None
    assert qty == 100.0


def test_deterministic_mode_adoption_and_reports(temp_db: Path):
    """Verify recommendation adoption rate, mean deviation, and override metrics in deterministic mode."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    system = SingleAgentSystem(
        name="single_agent",
        llm_client=client,
        db_path=temp_db,
        run_id="run_det_adoption",
        tool_mode="deterministic",
    )

    # Step 1: Decision within 5% of recommendation (Adoption)
    # Compute deterministic recommendation
    state1 = make_test_state(echelon="retailer", step=1, last_order=100.0)
    tools1 = system.compute_deterministic_tools(state1, "retailer")
    rec1 = tools1["recommendation"]
    client.mock.set_response(
        "CURRENT LOCAL STATE",
        OrderDecision(quantity=rec1, reasoning="Adopting tool recommendation directly"),
    )
    orders1 = system.decide_orders({"retailer": state1})
    assert orders1["retailer"] == rec1

    # Step 2: Decision overriding recommendation (> 5% deviation)
    state2 = make_test_state(echelon="retailer", step=2, last_order=100.0)
    tools2 = system.compute_deterministic_tools(state2, "retailer")
    rec2 = tools2["recommendation"]
    override_qty = rec2 + 50.0  # definitely > 5%
    client.mock.set_response(
        "CURRENT LOCAL STATE",
        OrderDecision(quantity=override_qty, reasoning="Overriding recommendation due to anticipated spike"),
    )
    orders2 = system.decide_orders({"retailer": state2})
    assert orders2["retailer"] == override_qty

    # Register run metadata
    conn = duckdb.connect(str(temp_db))
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (run_id, experiment_name, status, config, metadata)
        VALUES ('run_det_adoption', 'exp_det', 'completed', ?, ?)
        """,
        [
            json.dumps({"system": "single_agent", "scenario": "stable", "tool_mode": "deterministic"}),
            json.dumps({"system": "single_agent", "tool_mode": "deterministic"}),
        ],
    )
    conn.close()

    # Compute behavioural metrics
    bm = compute_behavioural_metrics(temp_db, "run_det_adoption")
    assert bm.tool_mode == "deterministic"
    # Tool accuracy marked n/a in deterministic mode
    assert bm.tool_choice_accuracy is None
    assert bm.parameter_accuracy is None

    # Recommendation metrics: 1 adopted, 1 overridden => 50% adoption, 50% override
    assert bm.recommendation_adoption_rate == 0.5
    assert bm.recommendation_override_share == 0.5
    assert bm.mean_recommendation_deviation > 0.0

    # Summary table formatting check
    table = generate_behaviour_summary_table([bm])
    assert "n/a (tools deterministic)" in table
    assert "50.0%" in table


# ===========================================================================
# 3. LLM Function Calling Mode Verification
# ===========================================================================
def test_function_calling_mode_metrics(temp_db: Path):
    """Verify llm_function_calling mode records tool calls and computes accuracy against expected-tool rule."""
    system = MultiAgentToolsSystem(
        name="multi_agent_tools",
        config={"aggregation_rule": "median", "tool_mode": "llm_function_calling"},
        db_path=temp_db,
        run_id="run_fc_normal",
        tool_mode="llm_function_calling",
    )

    states = {
        "retailer": make_test_state(echelon="retailer", step=1),
        "wholesaler": make_test_state(echelon="wholesaler", step=1),
        "distributor": make_test_state(echelon="distributor", step=1),
        "manufacturer": make_test_state(echelon="manufacturer", step=1),
    }

    orders = system.decide_orders(states)
    assert len(orders) == 4

    # Verify tool calls recorded in DuckDB
    conn = duckdb.connect(str(temp_db))
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (run_id, experiment_name, status, config, metadata)
        VALUES ('run_fc_normal', 'exp_fc', 'completed', ?, ?)
        """,
        [
            json.dumps({"system": "multi_agent_tools", "scenario": "stable", "tool_mode": "llm_function_calling"}),
            json.dumps({"system": "multi_agent_tools", "tool_mode": "llm_function_calling"}),
        ],
    )
    conn.close()

    bm = compute_behavioural_metrics(temp_db, "run_fc_normal")
    assert bm.tool_mode == "llm_function_calling"
    assert bm.tool_choice_accuracy == 1.0
    assert bm.parameter_accuracy == 1.0

    table = generate_behaviour_summary_table([bm])
    assert "100.0%" in table
    assert "n/a (tools deterministic)" not in table


# ===========================================================================
# 4. Fallback Rule Verification
# ===========================================================================
def test_fallback_sequence_invalid_then_retry_success(temp_db: Path):
    """Verify fallback sequence: invalid output -> retry once -> success on retry."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    client.mock.set_sequential_responses(
        "retailer",
        [
            "INVALID_NON_JSON_OUTPUT_GARBAGE",
            json.dumps({"quantity": 105.0, "reasoning": "Recovered valid order on retry"}),
        ],
    )

    system = SingleAgentSystem(
        name="single_agent",
        llm_client=client,
        db_path=temp_db,
        run_id="run_retry_success",
        tool_mode="deterministic",
    )

    state = make_test_state(echelon="retailer", step=1, last_order=90.0)
    orders = system.decide_orders({"retailer": state})

    assert orders["retailer"] == 105.0

    conn = duckdb.connect(str(temp_db))
    row = conn.execute(
        """
        SELECT retry_count, fallback, order_quantity
        FROM agent_calls
        WHERE run_id = 'run_retry_success' AND echelon = 'retailer'
        """
    ).fetchone()
    conn.close()

    assert row is not None
    retries, fallback, qty = row
    assert retries == 1
    assert fallback is False
    assert qty == 105.0


def test_fallback_sequence_double_invalid_falls_back_to_previous_order(temp_db: Path):
    """Verify fallback rule: invalid output -> retry once -> previous order, counted and logged."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    client.mock.set_sequential_responses(
        "retailer",
        [
            "INVALID_JSON_ATTEMPT_1",
            "INVALID_JSON_ATTEMPT_2",
        ],
    )

    system = SingleAgentSystem(
        name="single_agent",
        llm_client=client,
        db_path=temp_db,
        run_id="run_double_fallback",
        tool_mode="deterministic",
    )

    previous_order = 88.0
    state = make_test_state(echelon="retailer", step=2, last_order=previous_order)
    orders = system.decide_orders({"retailer": state})

    # Assert fallback to previous order
    assert orders["retailer"] == previous_order

    # Verify DuckDB records fallback=True, retry_count=1, and metric counted
    conn = duckdb.connect(str(temp_db))
    call_row = conn.execute(
        """
        SELECT retry_count, fallback, order_quantity, prompt_version
        FROM agent_calls
        WHERE run_id = 'run_double_fallback' AND echelon = 'retailer'
        """
    ).fetchone()

    fb_metrics = conn.execute(
        """
        SELECT count(*), sum(metric_value)
        FROM metrics
        WHERE run_id = 'run_double_fallback' AND metric_name = 'fallback_count'
        """
    ).fetchone()
    conn.close()

    assert call_row is not None
    retries, fallback, qty, p_ver = call_row
    assert retries == 1
    assert fallback is True
    assert qty == previous_order
    assert p_ver == "single_agent_v3"

    assert fb_metrics is not None
    assert fb_metrics[0] >= 1
    assert fb_metrics[1] >= 1.0


# ===========================================================================
# 5. Decision Log Demonstrations
# ===========================================================================
def test_decision_logs_for_both_modes(temp_db: Path, capsys: pytest.CaptureFixture):
    """Execute and display formatted decision logs for deterministic and llm_function_calling modes."""
    # 1. Deterministic Mode Run
    client_det = LLMClient(mock_mode=True, db_path=temp_db)
    client_det.mock.set_response(
        "CURRENT LOCAL STATE",
        OrderDecision(quantity=98.0, reasoning="Balanced replenishment aligned with deterministic forecast"),
    )
    sys_det = SingleAgentSystem(
        name="single_agent",
        llm_client=client_det,
        db_path=temp_db,
        run_id="demo_log_deterministic",
        tool_mode="deterministic",
    )
    state_det = make_test_state(echelon="retailer", step=3, last_order=95.0)
    sys_det.decide_orders({"retailer": state_det})

    # Retrieve deterministic decision log from DuckDB
    conn = duckdb.connect(str(temp_db))
    log_det = conn.execute(
        """
        SELECT system_name, echelon, step_number, prompt_version,
               tool_recommendation, order_quantity, reasoning, retry_count, fallback
        FROM agent_calls
        WHERE run_id = 'demo_log_deterministic' AND echelon = 'retailer'
        """
    ).fetchone()

    # 2. LLM Function Calling Mode Run
    sys_fc = MultiAgentToolsSystem(
        name="multi_agent_tools",
        config={"aggregation_rule": "median", "tool_mode": "llm_function_calling"},
        db_path=temp_db,
        run_id="demo_log_function_calling",
        tool_mode="llm_function_calling",
    )
    state_fc = make_test_state(echelon="retailer", step=3, last_order=95.0)
    orders_fc = sys_fc.decide_orders({"retailer": state_fc})

    # Retrieve function calling tool logs from DuckDB
    logs_fc = conn.execute(
        """
        SELECT tool_name, expected_tool, is_correct_tool, is_parameter_valid
        FROM tool_calls
        WHERE run_id = 'demo_log_function_calling'
        ORDER BY created_at ASC
        """
    ).fetchall()
    conn.close()

    print("\n" + "=" * 80)
    print("DECISION LOG 1: DETERMINISTIC MODE ('deterministic tools plus LLM decision and negotiation')")
    print("=" * 80)
    print(f"System:           {log_det[0]}")
    print(f"Echelon:          {log_det[1]}")
    print(f"Step:             {log_det[2]}")
    print(f"Prompt Version:   {log_det[3]}")
    print(f"Tool Recommendation: {log_det[4]:.1f}")
    print(f"Decided Order:    {log_det[5]:.1f}")
    print(f"LLM Reasoning:    \"{log_det[6]}\"")
    print(f"Retries / Fallback: {log_det[7]} / {log_det[8]}")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("DECISION LOG 2: LLM FUNCTION CALLING MODE ('llm_function_calling')")
    print("=" * 80)
    print(f"System:           {sys_fc.name}")
    print("Echelon:          retailer")
    print("Step:             3")
    print(f"Decided Order:    {orders_fc['retailer']:.1f}")
    print("Tool Execution Trace:")
    for idx, (t_name, exp_t, is_corr, is_val) in enumerate(logs_fc, start=1):
        print(f"  [{idx}] Invoked Tool: '{t_name:<16}' | Expected: '{exp_t:<16}' | Correct: {is_corr} | Valid Param: {is_val}")
    print("=" * 80 + "\n")

    captured = capsys.readouterr()
    assert "DECISION LOG 1: DETERMINISTIC MODE" in captured.out
    assert "DECISION LOG 2: LLM FUNCTION CALLING MODE" in captured.out
