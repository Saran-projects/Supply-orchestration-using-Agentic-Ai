"""Unit tests for Behavioural Evaluation Metrics.

Tests each metric on small synthetic DuckDB logs with exact known answers:
1. Tool-choice accuracy (actual vs expected_tool label)
2. Parameter accuracy (tool inputs vs true observed state within tolerance)
3. Instruction-following rate (valid, in-range outputs on the first attempt)
4. Programmatic groundedness (regex number extraction vs true state numbers)
5. Fallback and error rate
6. Negotiation efficiency (rounds, messages, conflict share)
7. Resource consumption (tokens, cost in USD from models.yaml with cache hits = $0, latency)
8. End-to-end compute_behavioural_metrics and DuckDB persistence to metrics table
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from sca.db import get_duckdb_connection
from sca.eval.behaviour import (
    BehaviouralMetrics,
    compute_behavioural_metrics,
    compute_fallback_and_error_rate,
    compute_groundedness,
    compute_instruction_following_rate,
    compute_negotiation_efficiency,
    compute_parameter_accuracy,
    compute_resource_consumption,
    compute_tool_choice_accuracy,
    extract_numbers_from_text,
    generate_behaviour_summary_table,
    is_number_grounded,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Provide a fresh initialized DuckDB database file for synthetic testing."""
    db_file = tmp_path / "synthetic_eval.duckdb"
    conn = get_duckdb_connection(db_file)
    conn.close()
    return db_file


def _insert_run(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    system: str = "single_agent",
    scenario: str = "stable",
    model: str = "glm-isoquant",
    seed: int = 42,
) -> None:
    """Helper to insert a runs record."""
    cfg = json.dumps({"system": system, "scenario": scenario, "model": model, "seed": seed})
    conn.execute(
        "INSERT INTO runs (run_id, experiment_name, status, config) VALUES (?, 'test_exp', 'completed', ?)",
        [run_id, cfg],
    )


# ---------------------------------------------------------------------------
# 1. Tool-Choice Accuracy
# ---------------------------------------------------------------------------
def test_tool_choice_accuracy(temp_db: Path) -> None:
    """Test tool-choice accuracy: actual vs expected_tool label (3 of 4 correct -> 0.75)."""
    conn = get_duckdb_connection(temp_db)
    run_id = "run_tool_choice"
    _insert_run(conn, run_id)

    # Insert 4 tool calls: 3 correct, 1 wrong
    tool_data = [
        ("tc_1", "forecast_demand", "forecast_demand", True),
        ("tc_2", "forecast_demand", "forecast_demand", True),
        ("tc_3", "eoq_quantity", "forecast_demand", False),  # Wrong tool called
        ("tc_4", "ss_policy_order", "ss_policy_order", True),
    ]
    for tid, tool, expected, is_correct in tool_data:
        conn.execute(
            """
            INSERT INTO tool_calls (
                tool_call_id, run_id, tool_name, expected_tool, is_correct_tool
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [tid, run_id, tool, expected, is_correct],
        )

    acc, total = compute_tool_choice_accuracy(conn, run_id)
    assert total == 4
    assert acc == pytest.approx(0.75, abs=1e-4)

    # Empty run returns None, 0
    empty_acc, empty_tot = compute_tool_choice_accuracy(conn, "non_existent_run")
    assert empty_acc is None
    assert empty_tot == 0

    conn.close()


# ---------------------------------------------------------------------------
# 2. Parameter Accuracy
# ---------------------------------------------------------------------------
def test_parameter_accuracy(temp_db: Path) -> None:
    """Test parameter accuracy: tool inputs vs true observed state within tolerance."""
    conn = get_duckdb_connection(temp_db)
    run_id = "run_param_acc"
    _insert_run(conn, run_id)

    # Step 1: retailer inventory is 100.0, demand is 50.0
    step_info = json.dumps({
        "retailer": {"inventory": 100.0, "backlog": 0.0, "pipeline": 20.0, "demand": 50.0}
    })
    conn.execute(
        "INSERT INTO steps (step_id, run_id, step_number, inventory_level, info) VALUES ('s1', ?, 1, 100, ?)",
        [run_id, step_info],
    )

    # 4 Tool Calls:
    # 1. Exact inventory (100.0) -> Valid & accurate
    # 2. Within 5% tolerance (104.0 vs 100.0, diff 4.0 <= 5.0) -> Accurate
    # 3. Beyond tolerance (200.0 vs 100.0) -> Inaccurate
    # 4. Schema validation failure (is_parameter_valid = False) -> Inaccurate
    tool_cases = [
        ("tc_1", "ss_policy_order", json.dumps({"inventory": 100.0}), True, False),
        ("tc_2", "ss_policy_order", json.dumps({"inventory": 104.0}), True, False),
        ("tc_3", "ss_policy_order", json.dumps({"inventory": 200.0}), True, False),
        ("tc_4", "ss_policy_order", json.dumps({"inventory": 100.0}), False, False),
    ]
    for tid, tname, in_args, is_valid, is_err in tool_cases:
        conn.execute(
            """
            INSERT INTO tool_calls (
                tool_call_id, run_id, step_id, tool_name, input_arguments,
                is_parameter_valid, is_error
            ) VALUES (?, ?, 's1', ?, ?, ?, ?)
            """,
            [tid, run_id, tname, in_args, is_valid, is_err],
        )

    acc, total = compute_parameter_accuracy(conn, run_id, tolerance=0.05)
    assert total == 4
    assert acc == pytest.approx(0.50, abs=1e-4)  # 2 of 4 accurate

    conn.close()


# ---------------------------------------------------------------------------
# 3. Instruction-Following Rate
# ---------------------------------------------------------------------------
def test_instruction_following_rate(temp_db: Path) -> None:
    """Test instruction-following rate: valid, in-range outputs on first attempt."""
    conn = get_duckdb_connection(temp_db)
    run_id = "run_instruction_following"
    _insert_run(conn, run_id)

    # 4 Agent calls:
    # 1. First attempt valid, in-range (retry=0, fallback=False, order=100) -> Valid
    # 2. Succeeded only after retry (retry=1) -> Invalid on first attempt
    # 3. Fallback used (fallback=True) -> Invalid
    # 4. Out of capacity (order=9999.0 > max 500) -> Invalid
    calls = [
        ("c1", 0, True, False, 100.0),
        ("c2", 1, True, False, 100.0),
        ("c3", 0, True, True, 100.0),
        ("c4", 0, True, False, 9999.0),
    ]
    for cid, retries, ok, fb, qty in calls:
        conn.execute(
            """
            INSERT INTO agent_calls (
                call_id, run_id, agent_name, retry_count, structured_output_ok,
                fallback, order_quantity
            ) VALUES (?, ?, 'retailer_agent', ?, ?, ?, ?)
            """,
            [cid, run_id, retries, ok, fb, qty],
        )

    rate, total = compute_instruction_following_rate(conn, run_id, max_capacity=500.0)
    assert total == 4
    assert rate == pytest.approx(0.25, abs=1e-4)  # 1 of 4 valid

    conn.close()


# ---------------------------------------------------------------------------
# 4. Programmatic Groundedness Evaluator
# ---------------------------------------------------------------------------
def test_extract_numbers_and_groundedness_helper() -> None:
    """Verify regex extraction and numerical tolerance matching."""
    text = "Current inventory is 150.0, backlog is 0, lead time is 2 days. Replenishing 120.5 units."
    nums = extract_numbers_from_text(text)
    assert nums == [150.0, 0.0, 2.0, 120.5]

    candidates = {150.0, 0.0, 2.0}
    assert is_number_grounded(150.0, candidates) is True
    assert is_number_grounded(151.0, candidates, abs_tol=1.0) is True
    assert is_number_grounded(120.5, candidates) is False


def test_groundedness_from_db(temp_db: Path) -> None:
    """Test groundedness score computation from agent reasoning text."""
    conn = get_duckdb_connection(temp_db)
    run_id = "run_groundedness"
    _insert_run(conn, run_id)

    step_info = json.dumps({
        "retailer": {
            "inventory": 150.0,
            "backlog": 0.0,
            "pipeline": 50.0,
            "incoming": 25.0,
            "last_order": 100.0,
        }
    })
    conn.execute(
        "INSERT INTO steps (step_id, run_id, step_number, inventory_level, info) VALUES ('s1', ?, 1, 150, ?)",
        [run_id, step_info],
    )

    # Reasoning contains 3 numbers: 150 (grounded), 50 (grounded), 9999 (hallucinated)
    conn.execute(
        """
        INSERT INTO agent_calls (
            call_id, run_id, step_id, step_number, agent_name, echelon, reasoning
        ) VALUES (
            'c1', ?, 's1', 1, 'retailer_agent', 'retailer',
            'We hold 150 units in stock and have 50 on order pipeline. I speculate target 9999 units.'
        )
        """,
        [run_id],
    )

    score, grounded_cnt, total_nums = compute_groundedness(conn, run_id)
    assert total_nums == 3
    assert grounded_cnt == 2
    assert score == pytest.approx(2.0 / 3.0, abs=1e-4)

    conn.close()


# ---------------------------------------------------------------------------
# 5. Fallback and Error Rate
# ---------------------------------------------------------------------------
def test_fallback_and_error_rate(temp_db: Path) -> None:
    """Test fallback and execution error rates computation."""
    conn = get_duckdb_connection(temp_db)
    run_id = "run_fb_err"
    _insert_run(conn, run_id)

    # 4 Agent calls: 1 fallback, 1 error
    calls = [
        ("c1", False, None),
        ("c2", True, None),  # Fallback
        ("c3", False, "JSONDecodeError"),  # Error
        ("c4", False, None),
    ]
    for cid, fb, err in calls:
        conn.execute(
            """
            INSERT INTO agent_calls (call_id, run_id, agent_name, fallback, error_type)
            VALUES (?, ?, 'retailer_agent', ?, ?)
            """,
            [cid, run_id, fb, err],
        )

    # 1 Tool call: 0 error
    conn.execute(
        """
        INSERT INTO tool_calls (tool_call_id, run_id, tool_name, is_error)
        VALUES ('tc_1', ?, 'forecast_demand', FALSE)
        """,
        [run_id],
    )

    fb_rate, err_rate = compute_fallback_and_error_rate(conn, run_id)
    assert fb_rate == pytest.approx(0.25, abs=1e-4)  # 1 fallback out of 4 decisions
    assert err_rate == pytest.approx(1.0 / 5.0, abs=1e-4)  # 1 error out of 5 total calls (4 agent + 1 tool)

    conn.close()


# ---------------------------------------------------------------------------
# 6. Negotiation Efficiency
# ---------------------------------------------------------------------------
def test_negotiation_efficiency(temp_db: Path) -> None:
    """Test negotiation rounds, messages, and conflict share."""
    conn = get_duckdb_connection(temp_db)
    run_id = "run_negotiation"
    _insert_run(conn, run_id, system="orchestrated")

    # 2 decisions:
    # Decision 1: conflict, 2 rounds, 4 messages
    # Decision 2: no conflict, 0 rounds, 0 messages
    decisions = [
        ("d1", 1, "retailer", True, "capacity_limit", 2, 4, "negotiation", 100.0),
        ("d2", 1, "wholesaler", False, None, 0, 0, "consensus", 80.0),
    ]
    for did, s_num, ech, has_c, c_type, r_used, m_ex, res, final_ord in decisions:
        conn.execute(
            """
            INSERT INTO orchestrated_decisions (
                decision_id, run_id, step_number, echelon, has_conflict,
                conflict_type, rounds_used, messages_exchanged, resolution_method, final_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [did, run_id, s_num, ech, has_c, c_type, r_used, m_ex, res, final_ord],
        )

    avg_rounds, avg_msgs, conflict_share = compute_negotiation_efficiency(conn, run_id)
    assert avg_rounds == pytest.approx(1.0, abs=1e-4)  # (2 + 0) / 2
    assert avg_msgs == pytest.approx(2.0, abs=1e-4)  # (4 + 0) / 2
    assert conflict_share == pytest.approx(0.5, abs=1e-4)  # 1 of 2 had conflict

    conn.close()


# ---------------------------------------------------------------------------
# 7. Resource Consumption and Pricing from models.yaml
# ---------------------------------------------------------------------------
def test_resource_consumption_and_cache_hits(temp_db: Path) -> None:
    """Test token summation, cost in USD from models.yaml (cache hits cost $0), and latency."""
    conn = get_duckdb_connection(temp_db)
    run_id = "run_resources"
    _insert_run(conn, run_id, model="glm-isoquant")

    # glm-isoquant pricing in configs/models.yaml:
    # price_per_1k_input: 0.000154, price_per_1k_output: 0.00044
    #
    # Call 1: 1000 prompt, 500 completion -> cost = 1*0.000154 + 0.5*0.00044 = 0.000374
    # Call 2: 1000 prompt, 500 completion, CACHE HIT -> cost MUST BE 0.0
    # Call 3: 2000 prompt, 1000 completion -> cost = 2*0.000154 + 1*0.00044 = 0.000748
    calls = [
        ("c1", "glm-isoquant", 1000, 500, 100.0, False, "retailer", 1),
        ("c2", "glm-isoquant", 1000, 500, 5.0, True, "wholesaler", 1),
        ("c3", "glm-isoquant", 2000, 1000, 200.0, False, "distributor", 1),
    ]
    for cid, model, p_tok, c_tok, lat, cache_hit, ech, s_num in calls:
        conn.execute(
            """
            INSERT INTO agent_calls (
                call_id, run_id, model_name, agent_name, prompt_tokens, completion_tokens,
                latency_ms, cache_hit, echelon, step_number
            ) VALUES (?, ?, ?, 'agent', ?, ?, ?, ?, ?, ?)
            """,
            [cid, run_id, model, p_tok, c_tok, lat, cache_hit, ech, s_num],
        )

    avg_tok, avg_cost, avg_lat, cache_rate = compute_resource_consumption(conn, run_id)

    # Total tokens: 1500 + 1500 + 3000 = 6000
    # 3 distinct decisions -> avg_tokens = 2000.0
    assert avg_tok == pytest.approx(2000.0, abs=1e-2)

    # Total cost: 0.000374 + 0.000000 + 0.000748 = 0.001122 USD
    # Avg cost: 0.001122 / 3 = 0.000374 USD
    assert avg_cost == pytest.approx(0.000374, abs=1e-6)

    # Total latency: 100 + 5 + 200 = 305 ms -> Avg latency = 305 / 3 = 101.67 ms
    assert avg_lat == pytest.approx(101.67, abs=1e-2)

    # 1 of 3 cache hits -> 0.3333
    assert cache_rate == pytest.approx(0.3333, abs=1e-4)

    conn.close()


# ---------------------------------------------------------------------------
# 8. Full End-to-End Pipeline & DuckDB Metrics Table Persistence
# ---------------------------------------------------------------------------
def test_full_pipeline_persistence_and_summary_table(temp_db: Path) -> None:
    """Test compute_behavioural_metrics persists to DuckDB and table renders correctly."""
    conn = get_duckdb_connection(temp_db)
    run_id = "run_full_eval"
    _insert_run(conn, run_id, system="multi_agent_tools", scenario="stable", model="mock", seed=101)

    # Step state
    conn.execute(
        """
        INSERT INTO steps (step_id, run_id, step_number, inventory_level, info)
        VALUES ('s1', ?, 1, 100, ?)
        """,
        [run_id, json.dumps({"retailer": {"inventory": 100.0, "demand": 50.0}})],
    )

    # Agent call
    conn.execute(
        """
        INSERT INTO agent_calls (
            call_id, run_id, step_id, step_number, agent_name, echelon,
            prompt_tokens, completion_tokens, latency_ms, cache_hit,
            retry_count, structured_output_ok, fallback, order_quantity, reasoning
        ) VALUES (
            'c1', ?, 's1', 1, 'retailer_agent', 'retailer',
            200, 50, 45.0, FALSE, 0, TRUE, FALSE, 100.0,
            'Current inventory 100 units satisfies downstream demand 50 units.'
        )
        """,
        [run_id],
    )

    # Tool call
    conn.execute(
        """
        INSERT INTO tool_calls (
            tool_call_id, run_id, step_id, tool_name, expected_tool,
            is_correct_tool, is_parameter_valid, input_arguments
        ) VALUES (
            'tc_1', ?, 's1', 'forecast_demand', 'forecast_demand',
            TRUE, TRUE, ?
        )
        """,
        [run_id, json.dumps({"recent_history": [50.0], "horizon": 1})],
    )
    conn.close()

    # Compute and persist
    metrics: BehaviouralMetrics = compute_behavioural_metrics(temp_db, run_id, write_to_db=True)

    assert metrics.system == "multi_agent_tools"
    assert metrics.scenario == "stable"
    assert metrics.seed == 101
    assert metrics.tool_choice_accuracy == 1.0
    assert metrics.parameter_accuracy == 1.0
    assert metrics.instruction_following_rate == 1.0
    assert metrics.groundedness_score == 1.0
    assert metrics.fallback_rate == 0.0

    # Verify DuckDB persistence
    conn = get_duckdb_connection(temp_db, read_only=True)

    # Check behavioural_metrics table
    bm_row = conn.execute("SELECT * FROM behavioural_metrics WHERE run_id = ?", [run_id]).fetchone()
    assert bm_row is not None

    # Check metrics table (category = 'behaviour')
    met_rows = conn.execute(
        "SELECT metric_name, metric_value FROM metrics WHERE run_id = ? AND category = 'behaviour'",
        [run_id],
    ).fetchall()
    assert len(met_rows) >= 10
    met_names = {r[0] for r in met_rows}
    assert "behaviour_tool_choice_accuracy" in met_names
    assert "behaviour_groundedness_score" in met_names
    assert "behaviour_instruction_following_rate" in met_names

    conn.close()

    # Verify summary table formatting
    table_str = generate_behaviour_summary_table([metrics])
    assert "multi_agent_tools" in table_str
    assert "stable" in table_str
    assert "100.0%" in table_str
