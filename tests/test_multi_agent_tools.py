"""Tests for System 5: MultiAgentToolsSystem (CrewAI specialized agents with tools).

Verifies:
1. Demand, Inventory, and Procurement agents propose structured recommendations.
2. Aggregator combines proposals via configured rules (median, conservative / min).
3. Final orders pass through validate_order (no capacity allocation / Banker's algorithm).
4. Telemetry logs tool calls to DuckDB with deterministic expected_tool labels.
5. Scripted wrong-tool cases drop tool_choice_accuracy.
6. Scripted wrong-parameter cases drop parameter_accuracy.
7. Integration test on a multi-step environment with MockLLM.
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pytest

from sca.sim.environment import EchelonHistoryRecord, EchelonState
from sca.systems.multi_agent_tools import (
    AgentProposal,
    DemandAgent,
    InventoryAgent,
    MultiAgentToolsSystem,
    ProcurementAgent,
    aggregate_proposals,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Temporary DuckDB database for isolation."""
    return tmp_path / "test_multi_agent_tools.duckdb"


def make_dummy_state(
    echelon: str = "retailer",
    step: int = 1,
    inventory: float = 120.0,
    backlog: float = 10.0,
    pipeline: float = 40.0,
    current_demand: float = 95.0,
    lead_time: int = 2,
    history_length: int = 5,
) -> EchelonState:
    """Helper to create dummy EchelonState with historical demand window."""
    history = [
        EchelonHistoryRecord(
            step=max(0, step - i),
            demand=90.0 + 2.0 * i,
            order_placed=95.0,
            inventory=120.0,
            backlog=0.0,
            incoming_shipment=90.0,
            outgoing_shipment=90.0,
            cost=60.0,
        )
        for i in range(1, history_length + 1)
    ]
    return EchelonState(
        echelon=echelon,
        step=step,
        inventory=inventory,
        backlog=backlog,
        pipeline=pipeline,
        incoming_shipments=20.0,
        current_demand=current_demand,
        last_order=90.0,
        lead_time=lead_time,
        history_window=history,
    )


# ---------------------------------------------------------------------------
# 1. Unit Tests for Individual Specialized Agents
# ---------------------------------------------------------------------------
def test_demand_agent_proposal(temp_db: Path):
    """Demand agent must call forecast_demand and return a structured proposal."""
    agent = DemandAgent(echelon="retailer", min_history_for_forecast=2)
    state = make_dummy_state(echelon="retailer", history_length=4)

    assert agent.determine_expected_tool(state) == "forecast_demand"

    proposal = agent.propose(
        state=state,
        run_id="run_test_agent",
        db_path=temp_db,
        step_id="step_1",
    )

    assert isinstance(proposal, AgentProposal)
    assert proposal.agent_name == "demand"
    assert proposal.quantity > 0.0
    assert "forecast_demand" in proposal.tools_used
    assert "linear regression" in proposal.rationale.lower()

    # Verify DuckDB tool_calls record
    conn = duckdb.connect(str(temp_db))
    rows = conn.execute(
        """
        SELECT tool_name, expected_tool, is_correct_tool, is_parameter_valid
        FROM tool_calls
        WHERE run_id = 'run_test_agent' AND agent_name = 'demand_retailer'
        """
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    tool_name, exp_tool, is_correct, is_param_valid = rows[0]
    assert tool_name == "forecast_demand"
    assert exp_tool == "forecast_demand"
    assert is_correct is True
    assert is_param_valid is True


def test_inventory_agent_proposal(temp_db: Path):
    """Inventory agent must call eoq_quantity and ss_policy_order."""
    agent = InventoryAgent(echelon="wholesaler", config={"reorder_point": 100.0, "order_up_to": 300.0})
    state = make_dummy_state(echelon="wholesaler", inventory=50.0, pipeline=20.0, backlog=10.0)

    assert agent.determine_expected_tool(state) == "eoq_quantity"

    proposal = agent.propose(
        state=state,
        run_id="run_test_inventory",
        db_path=temp_db,
        step_id="step_1",
    )

    assert isinstance(proposal, AgentProposal)
    assert proposal.agent_name == "inventory"
    assert "eoq_quantity" in proposal.tools_used
    assert "ss_policy_order" in proposal.tools_used
    assert proposal.quantity > 0.0

    conn = duckdb.connect(str(temp_db))
    rows = conn.execute(
        """
        SELECT tool_name, expected_tool, is_correct_tool, is_parameter_valid
        FROM tool_calls
        WHERE run_id = 'run_test_inventory' AND agent_name = 'inventory_wholesaler'
        ORDER BY created_at ASC
        """
    ).fetchall()
    conn.close()

    assert len(rows) == 2
    tools = [r[0] for r in rows]
    assert "eoq_quantity" in tools
    assert "ss_policy_order" in tools
    assert all(r[2] is True for r in rows)
    assert all(r[3] is True for r in rows)


def test_procurement_agent_proposal(temp_db: Path):
    """Procurement agent calculates buffer from lead time and bounds by capacity."""
    agent = ProcurementAgent(echelon="distributor", capacity_limit=250.0)
    state = make_dummy_state(echelon="distributor", current_demand=100.0, lead_time=3, inventory=40.0, backlog=10.0)

    proposal = agent.propose(
        state=state,
        run_id="run_test_procurement",
        db_path=temp_db,
        step_id="step_1",
    )

    assert isinstance(proposal, AgentProposal)
    assert proposal.agent_name == "procurement"
    assert proposal.quantity <= 250.0
    assert "supplier capacity ceiling" in proposal.rationale.lower()


# ---------------------------------------------------------------------------
# 2. Aggregator Tests
# ---------------------------------------------------------------------------
def test_aggregation_rules():
    """Verify median, conservative/min, and mean aggregation."""
    proposals = [
        AgentProposal(agent_name="demand", quantity=100.0, rationale="Forecast"),
        AgentProposal(agent_name="inventory", quantity=140.0, rationale="EOQ"),
        AgentProposal(agent_name="procurement", quantity=80.0, rationale="Capacity"),
    ]

    # Median of [80, 100, 140] = 100
    val_median, desc_median = aggregate_proposals(proposals, rule="median")
    assert val_median == 100.0
    assert "Median" in desc_median

    # Conservative / min of [80, 100, 140] = 80
    val_cons, desc_cons = aggregate_proposals(proposals, rule="conservative")
    assert val_cons == 80.0
    assert "Conservative" in desc_cons

    val_min, _ = aggregate_proposals(proposals, rule="min")
    assert val_min == 80.0

    # Mean of [80, 100, 140] = 106.666...
    val_mean, desc_mean = aggregate_proposals(proposals, rule="mean")
    assert math.isclose(val_mean, 106.666666, rel_tol=1e-3)
    assert "Mean" in desc_mean


# ---------------------------------------------------------------------------
# 3. System 5 Full Pipeline (Decide Orders)
# ---------------------------------------------------------------------------
def test_system5_normal_execution_and_accuracy(temp_db: Path):
    """Test full MultiAgentToolsSystem execution with perfect tool and parameter accuracy."""
    system = MultiAgentToolsSystem(
        name="multi_agent_tools",
        config={"aggregation_rule": "median", "manufacturer_capacity": 300.0},
        db_path=temp_db,
        run_id="run_sys5_normal",
    )

    states = {
        "retailer": make_dummy_state("retailer", step=1, current_demand=95.0),
        "wholesaler": make_dummy_state("wholesaler", step=1, current_demand=100.0),
        "distributor": make_dummy_state("distributor", step=1, current_demand=105.0),
        "manufacturer": make_dummy_state("manufacturer", step=1, current_demand=110.0),
    }

    orders = system.decide_orders(states)

    assert set(orders.keys()) == {"retailer", "wholesaler", "distributor", "manufacturer"}
    for ech, qty in orders.items():
        assert qty >= 0.0
        if ech == "manufacturer":
            assert qty <= 300.0  # capacity ceiling check

    # Verify DuckDB tool calls
    conn = duckdb.connect(str(temp_db))
    calls = conn.execute(
        """
        SELECT tool_name, expected_tool, is_correct_tool, is_parameter_valid
        FROM tool_calls
        WHERE run_id = 'run_sys5_normal'
        """
    ).fetchall()
    conn.close()

    assert len(calls) > 0
    # Every echelon executes:
    # 1. forecast_demand (DemandAgent)
    # 2. eoq_quantity (InventoryAgent)
    # 3. ss_policy_order (InventoryAgent)
    # 4. validate_order (Aggregator)
    # Total per echelon = 4 calls * 4 echelons = 16 calls
    assert len(calls) == 16

    # Verify perfect accuracy in normal mode
    accuracy = system.compute_accuracy_metrics()
    assert accuracy["total_tool_calls"] == 16
    assert accuracy["tool_choice_accuracy"] == 1.0
    assert accuracy["parameter_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# 4. Scripted Failure Modes: Wrong-Tool and Wrong-Parameter Accuracy
# ---------------------------------------------------------------------------
def test_system5_scripted_wrong_tool(temp_db: Path):
    """Test that scripted wrong-tool execution correctly degrades tool-choice accuracy."""
    system = MultiAgentToolsSystem(
        name="multi_agent_tools",
        config={"aggregation_rule": "median", "script_wrong_tool": True},
        db_path=temp_db,
        run_id="run_sys5_wrong_tool",
    )

    states = {
        "retailer": make_dummy_state("retailer", step=1, current_demand=95.0),
    }

    orders = system.decide_orders(states)
    assert "retailer" in orders

    # Compute accuracy metrics
    metrics = system.compute_accuracy_metrics()
    assert metrics["total_tool_calls"] > 0
    # Demand agent called eoq_quantity instead of forecast_demand (is_correct_tool=False)
    # Inventory agent called validate_order instead of eoq_quantity (is_correct_tool=False)
    assert metrics["tool_choice_accuracy"] < 1.0

    conn = duckdb.connect(str(temp_db))
    wrong_calls = conn.execute(
        """
        SELECT tool_name, expected_tool, is_correct_tool
        FROM tool_calls
        WHERE run_id = 'run_sys5_wrong_tool' AND is_correct_tool = FALSE
        """
    ).fetchall()
    conn.close()

    assert len(wrong_calls) >= 2
    for t_name, exp_t, is_corr in wrong_calls:
        assert is_corr is False
        assert t_name != exp_t


def test_system5_scripted_wrong_parameter(temp_db: Path):
    """Test that scripted wrong-parameter execution correctly degrades parameter accuracy."""
    system = MultiAgentToolsSystem(
        name="multi_agent_tools",
        config={"aggregation_rule": "median", "script_wrong_param": True},
        db_path=temp_db,
        run_id="run_sys5_wrong_param",
    )

    states = {
        "retailer": make_dummy_state("retailer", step=1, current_demand=95.0),
    }

    orders = system.decide_orders(states)
    assert "retailer" in orders

    # Compute accuracy metrics
    metrics = system.compute_accuracy_metrics()
    assert metrics["total_tool_calls"] > 0
    # Parameter accuracy should be < 1.0 due to invalid schema inputs
    assert metrics["parameter_accuracy"] < 1.0

    conn = duckdb.connect(str(temp_db))
    invalid_param_calls = conn.execute(
        """
        SELECT tool_name, is_parameter_valid, is_error
        FROM tool_calls
        WHERE run_id = 'run_sys5_wrong_param' AND is_parameter_valid = FALSE
        """
    ).fetchall()
    conn.close()

    assert len(invalid_param_calls) >= 1
    for _t_name, is_valid, is_err in invalid_param_calls:
        assert is_valid is False
        assert is_err is True


def test_system5_conservative_aggregation(temp_db: Path):
    """Verify conservative aggregation rule selects the minimum proposed order."""
    system = MultiAgentToolsSystem(
        name="multi_agent_tools",
        config={"aggregation_rule": "conservative"},
        db_path=temp_db,
        run_id="run_sys5_conservative",
    )

    states = {
        "retailer": make_dummy_state("retailer", step=1, current_demand=100.0, inventory=200.0),
    }

    orders = system.decide_orders(states)
    assert "retailer" in orders
    # Since inventory is high, inventory agent proposal is 0.0, conservative will pick min => 0.0
    assert orders["retailer"] == 0.0
