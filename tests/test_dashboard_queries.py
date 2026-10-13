"""Unit tests for Dashboard DuckDB query functions and component helpers.

Uses a temporary fixture database with synthetic telemetry rows to ensure
completely isolated, deterministic, and offline test execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from sca.dashboard.components import (
    export_transcript_to_markdown,
    export_transcript_to_slide_html,
)
from sca.dashboard.queries import (
    check_message_groundedness,
    get_aggregate_negotiation_analytics,
    get_available_runs,
    get_baselines_20seeds_summary,
    get_conflicting_steps_summary,
    get_cost_and_reconciliation_data,
    get_decision_details,
    get_glm_real_comparison_metrics,
    get_mock_vs_glm_plumbing_metrics,
    get_scenario_metadata,
    get_shock_baseline_demand_vs_orders,
    get_step_comparison_s3_vs_s6,
    load_demand_series_sample,
)
from sca.db import init_schema


@pytest.fixture
def fixture_db_path(tmp_path: Path) -> Path:
    """Create a temporary DuckDB database populated with realistic synthetic records."""
    db_file = tmp_path / "test_dashboard.duckdb"
    conn = duckdb.connect(str(db_file))
    init_schema(conn)

    # Add data_source column to runs, steps, agent_calls
    conn.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS data_source VARCHAR")
    conn.execute("ALTER TABLE agent_calls ADD COLUMN IF NOT EXISTS data_source VARCHAR")
    conn.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS data_source VARCHAR")

    # 1. Insert Runs
    runs_data = [
        ("baseline_s_S_shock_seed_42", "4_echelon_baseline", "completed", json.dumps({"system": "baseline_s_S", "seed": 42}), "baseline"),
        ("baseline_eoq_shock_seed_42", "4_echelon_baseline", "completed", json.dumps({"system": "baseline_eoq", "seed": 42}), "baseline"),
        ("s3_mock_shock_seed_42", "mock_benchmark", "completed", json.dumps({"system": "single_agent", "model": "mock", "seed": 42}), "mock_llm"),
        ("s6_mock_shock_seed_42", "mock_benchmark", "completed", json.dumps({"system": "orchestrated", "model": "mock", "seed": 42}), "mock_llm"),
        ("s3_glm_isoquant_shock_seed_42", "glm_real_benchmark", "completed", json.dumps({"system": "single_agent", "model": "glm-isoquant", "seed": 42}), "glm_real"),
        ("s6_glm_isoquant_shock_seed_42", "glm_real_benchmark", "completed", json.dumps({"system": "orchestrated", "model": "glm-isoquant", "seed": 42}), "glm_real"),
    ]
    for r_id, exp_name, status, cfg_json, ds in runs_data:
        conn.execute(
            "INSERT INTO runs (run_id, experiment_name, status, config, data_source) VALUES (?, ?, ?, ?, ?)",
            [r_id, exp_name, status, cfg_json, ds],
        )

    # 2. Insert Steps
    for r_id, _, _, _, ds in runs_data:
        for s_idx in range(5):
            conn.execute(
                """
                INSERT INTO steps (step_id, run_id, step_number, inventory_level, backlog_units, demand_units, action_taken, info, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    f"step_{r_id}_{s_idx}",
                    r_id,
                    s_idx + 1,
                    200 - s_idx * 10,
                    s_idx * 5,
                    100 + s_idx * 2,
                    json.dumps({"retailer": 105.0, "wholesaler": 100.0, "distributor": 95.0, "manufacturer": 90.0}),
                    json.dumps({
                        "retailer": {"inventory": 200.0, "backlog": 0.0, "pipeline": 50.0},
                        "current_demand": 100.0,
                    }),
                    ds,
                ],
            )

    # 3. Insert Orchestrated Decisions
    for r_id in ["s6_mock_shock_seed_42", "s6_glm_isoquant_shock_seed_42"]:
        for s_num in range(3):
            for ech in ["retailer", "wholesaler", "distributor", "manufacturer"]:
                has_conf = (s_num == 0 or s_num == 2)
                c_type = "forecast_deviation,proposals_disagree" if has_conf else "none"
                r_used = 2 if has_conf else 0
                conn.execute(
                    """
                    INSERT INTO orchestrated_decisions (
                        decision_id, run_id, step_number, echelon, has_conflict, conflict_type,
                        rounds_used, messages_exchanged, resolution_method, final_order,
                        proposals_json, fallback_used, latency_ms, total_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        f"dec_{r_id}_{s_num}_{ech}",
                        r_id,
                        s_num,
                        ech,
                        has_conf,
                        c_type,
                        r_used,
                        r_used * 2,
                        "negotiation_compromise" if has_conf else "no_conflict_consensus",
                        102.5,
                        json.dumps({"demand": 105.0, "inventory": 95.0, "procurement": 100.0}),
                        False,
                        250.0,
                        500,
                    ],
                )

    # 4. Insert Messages
    conn.execute(
        """
        INSERT INTO messages (message_id, run_id, step_id, sender, receiver, role, content, metadata)
        VALUES (?, 's6_glm_isoquant_shock_seed_42', 'step_0', 'orchestrator_retailer', 'mediator_retailer', 'system',
                'Resolve conflict between proposals. Current Inventory is 200.0 and Demand is 100.0.',
                ?)
        """,
        ["msg_01", json.dumps({"round": 1, "echelon": "retailer"})],
    )
    conn.execute(
        """
        INSERT INTO messages (message_id, run_id, step_id, sender, receiver, role, content, metadata)
        VALUES (?, 's6_glm_isoquant_shock_seed_42', 'step_0', 'mediator_retailer', 'orchestrator_retailer', 'assistant',
                'Concession proposed: Order 102.5 units to cover inventory gap.',
                ?)
        """,
        ["msg_02", json.dumps({"round": 1, "revised_qty": 102.5})],
    )

    # 5. Insert Tool Calls
    conn.execute(
        """
        INSERT INTO tool_calls (
            tool_call_id, run_id, step_id, agent_name, tool_name,
            input_arguments, output_result, is_error, is_correct_tool, is_parameter_valid, duration_ms
        ) VALUES (
            'tc_01', 's6_glm_isoquant_shock_seed_42', 'step_0', 'demand_retailer', 'forecast_demand',
            '{"recent_history": [100.0], "horizon": 1}', '{"point_forecast": 100.0}', FALSE, TRUE, TRUE, 12.5
        )
        """
    )
    conn.execute(
        """
        INSERT INTO tool_calls (
            tool_call_id, run_id, step_id, agent_name, tool_name,
            input_arguments, output_result, is_error, is_correct_tool, is_parameter_valid, duration_ms
        ) VALUES (
            'tc_02', 's6_glm_isoquant_shock_seed_42', 'step_0', 'orchestrator_retailer', 'validate_order',
            '{"proposed_order": 102.5}', '{"validated_order": 102.5, "is_valid": true}', FALSE, TRUE, TRUE, 5.0
        )
        """
    )

    # 6. Insert Agent Calls
    for r_id in ["s3_glm_isoquant_shock_seed_42", "s6_glm_isoquant_shock_seed_42"]:
        for s_idx in range(3):
            conn.execute(
                """
                INSERT INTO agent_calls (
                    call_id, run_id, step_id, agent_name, model_name, provider,
                    prompt_tokens, completion_tokens, cached_tokens, total_cost, cost_usd,
                    latency_ms, retry_count, structured_output_ok, error_type, cache_hit,
                    echelon, step_number, fallback, order_quantity, reasoning, data_source
                ) VALUES (?, ?, ?, 'test_agent', 'glm-isoquant', 'openai_compatible',
                          800, 200, 0, 0.0003, 0.0003, 500.0, 0, TRUE, NULL, FALSE,
                          'retailer', ?, FALSE, 102.5, 'Inventory is 200.0 with demand 100.0', 'glm_real')
                """,
                [f"call_{r_id}_{s_idx}", r_id, f"step_{s_idx}", s_idx],
            )

    # 7. Insert Metrics & Behavioural Metrics
    conn.execute(
        """
        INSERT INTO metrics (metric_id, run_id, metric_name, metric_value, category)
        VALUES
            ('m_01', 'baseline_s_S_shock_seed_42', 'total_cost', 15420.0, 'financial'),
            ('m_02', 'baseline_s_S_shock_seed_42', 'bullwhip_ratio_retailer', 1.85, 'stability'),
            ('m_03', 'baseline_s_S_shock_seed_42', 'bullwhip_ratio_wholesaler', 2.10, 'stability'),
            ('m_04', 'baseline_s_S_shock_seed_42', 'bullwhip_ratio_distributor', 2.45, 'stability'),
            ('m_05', 'baseline_s_S_shock_seed_42', 'bullwhip_ratio_manufacturer', 2.90, 'stability'),
            ('m_06', 'baseline_s_S_shock_seed_42', 'fill_rate_retailer', 0.965, 'service'),
            ('m_07', 'baseline_s_S_shock_seed_42', 'stockout_steps_retailer', 2.0, 'service')
        """
    )
    conn.execute(
        """
        INSERT INTO behavioural_metrics (
            metric_id, run_id, system_name, scenario_name, model_name, seed,
            tool_choice_accuracy, parameter_accuracy, instruction_following_rate,
            groundedness_score, fallback_rate, error_rate, avg_rounds, avg_messages,
            conflict_share, avg_tokens, avg_cost_usd, avg_latency_ms
        ) VALUES (
            'bm_01', 's6_glm_isoquant_shock_seed_42', 'orchestrated', 'shock', 'glm-isoquant', 42,
            1.0, 1.0, 1.0, 0.85, 0.0, 0.0, 1.33, 2.66, 0.66, 1500.0, 0.0008, 450.0
        )
        """
    )

    conn.close()
    return db_file


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
def test_get_available_runs(fixture_db_path: Path):
    runs = get_available_runs(fixture_db_path)
    assert len(runs) == 6
    run_ids = [r["run_id"] for r in runs]
    assert "s3_glm_isoquant_shock_seed_42" in run_ids
    assert "s6_glm_isoquant_shock_seed_42" in run_ids
    assert "baseline_s_S_shock_seed_42" in run_ids


def test_get_scenario_metadata():
    meta = get_scenario_metadata()
    assert meta["total_records"] == 180519
    assert "stable" in meta["scenarios"]
    assert "seasonal" in meta["scenarios"]
    assert "shock" in meta["scenarios"]
    assert meta["scenarios"]["shock"]["cv"] == 0.226


def test_get_baselines_20seeds_summary(fixture_db_path: Path):
    df = get_baselines_20seeds_summary(fixture_db_path)
    assert not df.empty
    assert "Policy" in df.columns
    assert "Total Cost ($)" in df.columns
    assert "(S,s)" in df["Policy"].values


def test_get_shock_baseline_demand_vs_orders(fixture_db_path: Path):
    df = get_shock_baseline_demand_vs_orders(fixture_db_path, seed=42)
    assert not df.empty
    assert "step_number" in df.columns
    assert "retailer_order" in df.columns


def test_get_glm_real_comparison_metrics(fixture_db_path: Path):
    df = get_glm_real_comparison_metrics(fixture_db_path, seed=42)
    assert not df.empty
    assert "Run ID" in df.columns
    assert "Data Source" in df.columns
    assert "Instruction Following" in df.columns
    assert "Groundedness" in df.columns

    # Verify baseline rows have "n/a" for LLM fields
    baseline_row = df[df["Run ID"] == "baseline_s_S_shock_seed_42"].iloc[0]
    assert baseline_row["Avg Rounds"] == "n/a"
    assert baseline_row["Total Tok"] == "n/a"


def test_get_decision_details(fixture_db_path: Path):
    dec = get_decision_details(fixture_db_path, "s6_glm_isoquant_shock_seed_42", step_number=0, echelon="retailer")
    assert dec["run_id"] == "s6_glm_isoquant_shock_seed_42"
    assert dec["has_conflict"] is True
    assert dec["final_order"] == 102.5
    assert len(dec["messages"]) >= 1
    assert len(dec["tools"]) >= 1
    assert dec["state"]["inventory"] == 200.0


def test_get_conflicting_steps_summary(fixture_db_path: Path):
    conflicts = get_conflicting_steps_summary(fixture_db_path, "s6_glm_isoquant_shock_seed_42", echelon="retailer")
    assert len(conflicts) >= 1
    assert conflicts[0]["step_number"] in (0, 2)
    assert "Conflict" in conflicts[0]["label"]


def test_check_message_groundedness():
    state = {"inventory": 200.0, "current_demand": 100.0, "backlog": 0.0}
    text = "Current inventory is 200.0 units while demand is 100.0 units, but we need 999.0 units."
    checks = check_message_groundedness(text, state, tolerance=0.05)
    assert len(checks) == 3
    # 200.0 and 100.0 are grounded; 999.0 is ungrounded
    nums = {c["number"]: c["grounded"] for c in checks}
    assert nums[200.0] is True
    assert nums[100.0] is True
    assert nums[999.0] is False


def test_get_step_comparison_s3_vs_s6(fixture_db_path: Path):
    comp = get_step_comparison_s3_vs_s6(
        fixture_db_path,
        s3_run_id="s3_glm_isoquant_shock_seed_42",
        s6_run_id="s6_glm_isoquant_shock_seed_42",
        step_number=0,
        echelon="retailer",
    )
    assert comp["step_number"] == 0
    assert "s3" in comp and "s6" in comp
    assert comp["s6"]["order"] == 102.5


def test_get_aggregate_negotiation_analytics(fixture_db_path: Path):
    agg = get_aggregate_negotiation_analytics(fixture_db_path, "s6_glm_isoquant_shock_seed_42")
    assert agg["total_decisions"] >= 1
    assert "rounds_distribution" in agg
    assert "conflict_shares" in agg


def test_get_mock_vs_glm_plumbing_metrics(fixture_db_path: Path):
    df = get_mock_vs_glm_plumbing_metrics(fixture_db_path)
    assert not df.empty
    assert "runner_type" in df.columns
    assert "Fallback Rate" in df.columns


def test_get_cost_and_reconciliation_data(fixture_db_path: Path):
    data = get_cost_and_reconciliation_data(fixture_db_path)
    assert "console_billed_usd" in data
    assert "all_time_logged_usd" in data
    assert "token_measurements" in data
    assert "planned_grid" in data


def test_export_transcript_generators(fixture_db_path: Path):
    dec = get_decision_details(fixture_db_path, "s6_glm_isoquant_shock_seed_42", step_number=0, echelon="retailer")
    md = export_transcript_to_markdown(dec)
    assert "# SupplyChain Multi-Agent Decision Transcript" in md
    assert "Retailer" in md

    html = export_transcript_to_slide_html(dec)
    assert "<div style=" in html
    assert "Order: 102.5" in html


def test_load_demand_series_sample():
    df = load_demand_series_sample()
    assert not df.empty
    assert "order_date" in df.columns or "date" in df.columns
    assert "order_item_quantity" in df.columns or "demand" in df.columns
    assert len(df) > 50


def test_get_decision_details_single_agent(fixture_db_path: Path):
    dec = get_decision_details(fixture_db_path, "s3_glm_isoquant_shock_seed_42", step_number=0, echelon="retailer")
    assert dec["run_id"] == "s3_glm_isoquant_shock_seed_42"
    assert dec["step_number"] == 0
    assert dec["echelon"] == "retailer"
    assert "single_agent" in dec["proposals"]
    assert dec["final_order"] == 102.5
    assert len(dec["messages"]) >= 1
    assert "Inventory is 200.0" in dec["messages"][0]["content"]

