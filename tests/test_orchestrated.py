"""Tests for System 6: OrchestratedSystem and OrchestratorAgent.

Verifies:
1. Conflict detection rules:
   - capacity exceeded
   - forecast deviation
   - proposal disagreement
2. No-conflict shortcut makes zero negotiation calls (cost/latency saving).
3. The 2-round negotiation cap is never exceeded.
4. Validation failure triggers deterministic fallback and counts it.
5. Ablation flags (enable_negotiation, enable_validation, enable_conflict_skip) alter behavior.
6. Multi-step execution logs all decisions, messages, and metrics to DuckDB.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from sca.llm.client import LLMClient
from sca.sim.environment import EchelonHistoryRecord, EchelonState
from sca.systems.multi_agent_tools import AgentProposal
from sca.systems.orchestrated import (
    NegotiationResponse,
    OrchestratedSystem,
    OrchestratorAgent,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Temporary isolated DuckDB file."""
    return tmp_path / "test_orchestrated.duckdb"


def make_dummy_state(
    echelon: str = "retailer",
    step: int = 1,
    inventory: float = 120.0,
    backlog: float = 10.0,
    pipeline: float = 40.0,
    current_demand: float = 95.0,
    last_order: float = 90.0,
    lead_time: int = 2,
    history_length: int = 4,
) -> EchelonState:
    """Helper to create dummy EchelonState."""
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
        last_order=last_order,
        lead_time=lead_time,
        history_window=history,
    )


# ---------------------------------------------------------------------------
# 1. Conflict Detection Rules
# ---------------------------------------------------------------------------
def test_conflict_detection_capacity_exceeded():
    """Rule 1: Requested quantity exceeds supplier capacity."""
    agent = OrchestratorAgent(
        echelon="manufacturer",
        config={"forecast_deviation_threshold": 0.25, "proposals_disagree_threshold": 0.30},
    )
    capacity_limit = 250.0

    proposals = {
        "demand": AgentProposal(agent_name="demand", quantity=100.0, rationale="Normal forecast"),
        "inventory": AgentProposal(agent_name="inventory", quantity=320.0, rationale="Exceeds capacity"),
        "procurement": AgentProposal(agent_name="procurement", quantity=250.0, rationale="Capped"),
    }

    res = agent.detect_conflicts(proposals, capacity_limit=capacity_limit)
    assert res.has_conflict is True
    assert "capacity_exceeded" in res.conflict_type
    assert "inventory" in res.details["capacity_exceeded"]["exceeded_agents"]


def test_conflict_detection_forecast_deviation():
    """Rule 2: Proposal deviates > X% from demand forecast."""
    agent = OrchestratorAgent(
        echelon="retailer",
        config={"forecast_deviation_threshold": 0.20, "proposals_disagree_threshold": 0.50},
    )
    capacity_limit = 500.0

    # Demand forecast = 100.0, inventory is 135.0 (35% deviation > 20% threshold)
    proposals = {
        "demand": AgentProposal(agent_name="demand", quantity=100.0, rationale="Forecast"),
        "inventory": AgentProposal(agent_name="inventory", quantity=135.0, rationale="EOQ push"),
        "procurement": AgentProposal(agent_name="procurement", quantity=105.0, rationale="Available"),
    }

    res = agent.detect_conflicts(proposals, capacity_limit=capacity_limit)
    assert res.has_conflict is True
    assert "forecast_deviation" in res.conflict_type
    assert "inventory" in res.details["forecast_deviation"]["deviations"]


def test_conflict_detection_proposals_disagree():
    """Rule 3: Proposals disagree by more than Y% spread."""
    agent = OrchestratorAgent(
        echelon="retailer",
        config={"forecast_deviation_threshold": 0.50, "proposals_disagree_threshold": 0.25},
    )
    capacity_limit = 500.0

    # Range: 80.0 to 140.0. Spread = (140 - 80) / 140 = 0.428 > 0.25
    proposals = {
        "demand": AgentProposal(agent_name="demand", quantity=110.0, rationale="Forecast"),
        "inventory": AgentProposal(agent_name="inventory", quantity=140.0, rationale="High batch"),
        "procurement": AgentProposal(agent_name="procurement", quantity=80.0, rationale="Low pipeline"),
    }

    res = agent.detect_conflicts(proposals, capacity_limit=capacity_limit)
    assert res.has_conflict is True
    assert "proposals_disagree" in res.conflict_type
    assert res.details["proposals_disagree"]["spread"] > 0.25


def test_conflict_detection_no_conflict():
    """No conflict when all proposals are close and within capacity."""
    agent = OrchestratorAgent(
        echelon="retailer",
        config={"forecast_deviation_threshold": 0.25, "proposals_disagree_threshold": 0.30},
    )
    capacity_limit = 500.0

    proposals = {
        "demand": AgentProposal(agent_name="demand", quantity=100.0, rationale="Forecast"),
        "inventory": AgentProposal(agent_name="inventory", quantity=105.0, rationale="EOQ"),
        "procurement": AgentProposal(agent_name="procurement", quantity=98.0, rationale="Capacity"),
    }

    res = agent.detect_conflicts(proposals, capacity_limit=capacity_limit)
    assert res.has_conflict is False
    assert res.conflict_type == "none"


# ---------------------------------------------------------------------------
# 2. No-Conflict Shortcut: Zero Negotiation Calls
# ---------------------------------------------------------------------------
def test_no_conflict_shortcut_makes_zero_negotiation_calls(temp_db: Path):
    """When there is no conflict, negotiation is bypassed completely, saving LLM calls."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    system = OrchestratedSystem(
        name="orchestrated",
        config={
            "enable_conflict_skip": True,
            "forecast_deviation_threshold": 5.0,  # Wide thresholds so no conflict triggers
            "proposals_disagree_threshold": 5.0,
        },
        llm_client=client,
        db_path=temp_db,
        run_id="run_shortcut_test",
    )

    state = make_dummy_state("retailer", step=1, current_demand=95.0)
    orders = system.decide_orders({"retailer": state})

    assert "retailer" in orders
    rec = system.decision_history[0]
    assert rec.has_conflict is False
    assert rec.resolution_method == "no_conflict_consensus"
    assert rec.rounds_used == 0
    assert rec.messages_exchanged == 0
    assert rec.total_tokens == 0

    # Verify zero messages in DuckDB messages table
    conn = duckdb.connect(str(temp_db))
    msg_count = conn.execute(
        "SELECT count(*) FROM messages WHERE run_id = 'run_shortcut_test'"
    ).fetchone()[0]
    conn.close()
    assert msg_count == 0


# ---------------------------------------------------------------------------
# 3. 2-Round Cap is Strictly Enforced
# ---------------------------------------------------------------------------
def test_negotiation_2_round_cap_never_exceeded(temp_db: Path):
    """Negotiation must never exceed max_rounds = 2, even when disagreement persists."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    # Configure mock to return conflicting responses without complete convergence
    client.mock.set_response(
        "CURRENT NEGOTIATION CONTEXT",
        NegotiationResponse(
            agent_name="mediator",
            revised_quantity=180.0,
            rationale="Small concession",
            concession_made=True,
        ),
    )

    system = OrchestratedSystem(
        name="orchestrated",
        config={
            "max_rounds": 10,  # Explicitly attempt to set higher, must be clamped to 2
            "forecast_deviation_threshold": 0.01,  # Force conflict
            "proposals_disagree_threshold": 0.01,  # Force conflict
        },
        llm_client=client,
        db_path=temp_db,
        run_id="run_cap_test",
    )

    state = make_dummy_state("wholesaler", step=1, current_demand=100.0)
    _ = system.decide_orders({"wholesaler": state})

    rec = system.decision_history[0]
    assert rec.has_conflict is True
    assert rec.rounds_used <= 2
    assert rec.rounds_used > 0
    assert rec.messages_exchanged <= 4  # max 2 rounds * 2 messages per round

    conn = duckdb.connect(str(temp_db))
    rows = conn.execute(
        "SELECT rounds_used, messages_exchanged FROM orchestrated_decisions WHERE run_id = 'run_cap_test'"
    ).fetchone()
    conn.close()

    assert rows[0] <= 2


# ---------------------------------------------------------------------------
# 4. Validation Failure and Deterministic Fallback
# ---------------------------------------------------------------------------
def test_validation_fallback_works_and_counts(temp_db: Path):
    """Validation failure triggers deterministic fallback to last_order and increments counter."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    expected_fallback_order = 88.0

    system = OrchestratedSystem(
        name="orchestrated",
        config={
            "enable_validation": True,
            "force_validation_failure": True,  # Test hook to simulate validation rejection
        },
        llm_client=client,
        db_path=temp_db,
        run_id="run_fallback_test",
    )

    state = make_dummy_state("distributor", step=1, last_order=expected_fallback_order)
    orders = system.decide_orders({"distributor": state})

    # Order must match state.last_order via fallback
    assert orders["distributor"] == expected_fallback_order
    rec = system.decision_history[0]
    assert rec.fallback_used is True
    assert system.total_fallbacks == 1

    # Verify DuckDB metrics recorded fallback_count = 1.0
    conn = duckdb.connect(str(temp_db))
    fb_metric = conn.execute(
        """
        SELECT metric_value FROM metrics
        WHERE run_id = 'run_fallback_test' AND metric_name = 'fallback_count'
        """
    ).fetchone()
    conn.close()

    assert fb_metric is not None
    assert fb_metric[0] == 1.0


# ---------------------------------------------------------------------------
# 5. Ablation Flags Behavior
# ---------------------------------------------------------------------------
def test_ablation_negotiation_off(temp_db: Path):
    """Ablation: With enable_negotiation=False, conflict does not trigger LLM calls."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    system = OrchestratedSystem(
        name="orchestrated",
        config={
            "enable_negotiation": False,
            "forecast_deviation_threshold": 0.01,  # Force conflict
            "proposals_disagree_threshold": 0.01,
        },
        llm_client=client,
        db_path=temp_db,
        run_id="run_ablation_neg_off",
    )

    state = make_dummy_state("retailer", step=1)
    _ = system.decide_orders({"retailer": state})

    rec = system.decision_history[0]
    assert rec.has_conflict is True
    assert rec.resolution_method == "ablation_no_negotiation"
    assert rec.rounds_used == 0
    assert rec.messages_exchanged == 0


def test_ablation_conflict_skip_off(temp_db: Path):
    """Ablation: With enable_conflict_skip=False, negotiation runs even without conflict."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    system = OrchestratedSystem(
        name="orchestrated",
        config={
            "enable_conflict_skip": False,  # Never skip
            "forecast_deviation_threshold": 5.0,  # No conflict triggers
            "proposals_disagree_threshold": 5.0,
        },
        llm_client=client,
        db_path=temp_db,
        run_id="run_ablation_no_skip",
    )

    state = make_dummy_state("retailer", step=1)
    _ = system.decide_orders({"retailer": state})

    rec = system.decision_history[0]
    assert rec.has_conflict is False
    # Even without conflict, negotiation ran because skip was disabled
    assert rec.rounds_used >= 1
    assert rec.messages_exchanged >= 2


def test_ablation_validation_off(temp_db: Path):
    """Ablation: With enable_validation=False, validate_order is bypassed."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    system = OrchestratedSystem(
        name="orchestrated",
        config={
            "enable_validation": False,
            "force_validation_failure": True,  # Would fail if validation were enabled
        },
        llm_client=client,
        db_path=temp_db,
        run_id="run_ablation_val_off",
    )

    state = make_dummy_state("retailer", step=1)
    _ = system.decide_orders({"retailer": state})

    rec = system.decision_history[0]
    assert rec.fallback_used is False  # Fallback was not triggered because validation is off
