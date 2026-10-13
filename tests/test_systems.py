"""Unit tests for System 3 (single_agent) and System 4 (multi_agent_info).

Tests:
1. Valid output path (clean JSON OrderDecision, retry=0, fallback=False)
2. Malformed output then retry success (retry=1, fallback=False, order applied)
3. Malformed twice then fallback (retry=1, fallback=True, order=state.last_order, DuckDB metric logged)
4. Out-of-range output then retry success
5. Out-of-range twice then fallback
6. Determinism per seed (identical trajectories for identical seeds, different for different seeds)
7. Multi-agent shared visibility verification
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from sca.llm.client import LLMClient
from sca.sim.environment import BeerGameEnv, EchelonState
from sca.systems.multi_agent_info import MultiAgentInfoSystem
from sca.systems.single_agent import OrderDecision, SingleAgentSystem


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Temporary DuckDB file path for testing agent_calls and metrics."""
    return tmp_path / "test_systems.duckdb"


def make_dummy_state(echelon: str = "retailer", step: int = 1, last_order: float = 100.0) -> EchelonState:
    """Construct an EchelonState instance for testing."""
    return EchelonState(
        echelon=echelon,
        step=step,
        inventory=150.0,
        backlog=0.0,
        pipeline=50.0,
        incoming_shipments=25.0,
        current_demand=100.0,
        last_order=last_order,
        lead_time=2,
        history_window=[],
    )


# ---------------------------------------------------------------------------
# 1. Valid Output Path
# ---------------------------------------------------------------------------
def test_valid_output_path(temp_db: Path):
    """Verify that valid structured JSON yields retry_count=0, fallback=False, and logs correctly."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    client.mock.set_response(
        "CURRENT LOCAL STATE",
        OrderDecision(quantity=110.0, reasoning="Normal inventory replenishment"),
    )

    system = SingleAgentSystem(
        name="single_agent",
        llm_client=client,
        db_path=temp_db,
        run_id="run_valid_test",
    )

    state = make_dummy_state("retailer", step=1, last_order=90.0)
    orders = system.decide_orders({"retailer": state})

    assert orders["retailer"] == 110.0

    # Verify DuckDB agent_calls
    conn = duckdb.connect(str(temp_db))
    rows = conn.execute(
        """
        SELECT system_name, echelon, step_number, prompt_version,
               retry_count, fallback, order_quantity, structured_output_ok
        FROM agent_calls
        WHERE run_id = 'run_valid_test' AND echelon = 'retailer'
        """
    ).fetchall()
    assert len(rows) == 1
    sys_name, ech, step_num, p_ver, retries, fallback, qty, struct_ok = rows[0]
    assert sys_name == "single_agent"
    assert ech == "retailer"
    assert step_num == 1
    assert p_ver == "single_agent_v3"
    assert retries == 0
    assert fallback is False
    assert qty == 110.0
    assert struct_ok is True

    # Verify DuckDB metrics has no fallback rows
    fallback_metrics = conn.execute(
        "SELECT count(*) FROM metrics WHERE run_id = 'run_valid_test' AND metric_name = 'fallback_count'"
    ).fetchone()[0]
    assert fallback_metrics == 0
    conn.close()


# ---------------------------------------------------------------------------
# 2. Malformed Output Then Retry Success
# ---------------------------------------------------------------------------
def test_malformed_output_then_retry_success(temp_db: Path):
    """Verify that an initial malformed output followed by valid output succeeds with retry_count=1."""
    client = LLMClient(mock_mode=True, db_path=temp_db)

    # Sequence: first response is broken JSON, second (retry) is valid
    client.mock.set_sequential_responses(
        "retailer",
        [
            "INVALID_RAW_JSON_OUTPUT_{{{",
            json.dumps({"quantity": 135.0, "reasoning": "Recovered valid order on retry"}),
        ],
    )

    system = SingleAgentSystem(
        name="single_agent",
        llm_client=client,
        db_path=temp_db,
        run_id="run_retry_success_test",
    )

    state = make_dummy_state("retailer", step=2, last_order=95.0)
    orders = system.decide_orders({"retailer": state})

    assert orders["retailer"] == 135.0

    conn = duckdb.connect(str(temp_db))
    rows = conn.execute(
        """
        SELECT retry_count, fallback, order_quantity, structured_output_ok
        FROM agent_calls
        WHERE run_id = 'run_retry_success_test' AND echelon = 'retailer'
        """
    ).fetchall()
    assert len(rows) == 1
    retries, fallback, qty, struct_ok = rows[0]
    assert retries == 1
    assert fallback is False
    assert qty == 135.0
    assert struct_ok is True

    fallback_metrics = conn.execute(
        "SELECT count(*) FROM metrics WHERE run_id = 'run_retry_success_test' AND metric_name = 'fallback_count'"
    ).fetchone()[0]
    assert fallback_metrics == 0
    conn.close()


# ---------------------------------------------------------------------------
# 3. Malformed Twice Then Fallback
# ---------------------------------------------------------------------------
def test_malformed_twice_then_fallback(temp_db: Path):
    """Verify that two consecutive malformed outputs trigger fallback to state.last_order and log to DuckDB."""
    client = LLMClient(mock_mode=True, db_path=temp_db)

    # Sequence: both initial attempt and retry return malformed outputs
    client.mock.set_sequential_responses(
        "retailer",
        [
            "MALFORMED_OUTPUT_ATTEMPT_1",
            "MALFORMED_OUTPUT_ATTEMPT_2",
        ],
    )

    expected_last_order = 88.0
    system = SingleAgentSystem(
        name="single_agent",
        llm_client=client,
        db_path=temp_db,
        run_id="run_double_fail_test",
    )

    state = make_dummy_state("retailer", step=3, last_order=expected_last_order)
    orders = system.decide_orders({"retailer": state})

    # Should fall back to state.last_order
    assert orders["retailer"] == expected_last_order

    conn = duckdb.connect(str(temp_db))
    rows = conn.execute(
        """
        SELECT retry_count, fallback, order_quantity, structured_output_ok
        FROM agent_calls
        WHERE run_id = 'run_double_fail_test' AND echelon = 'retailer'
        """
    ).fetchall()
    assert len(rows) == 1
    retries, fallback, qty, struct_ok = rows[0]
    assert retries == 1
    assert fallback is True
    assert qty == expected_last_order
    assert struct_ok is False

    # Check metrics table records the fallback
    metric_rows = conn.execute(
        """
        SELECT metric_name, metric_value, category
        FROM metrics
        WHERE run_id = 'run_double_fail_test' AND metric_name = 'fallback_count'
        """
    ).fetchall()
    assert len(metric_rows) == 1
    m_name, m_val, cat = metric_rows[0]
    assert m_name == "fallback_count"
    assert m_val == 1.0
    assert cat == "reliability"
    conn.close()


# ---------------------------------------------------------------------------
# 4. Out-of-Range Output Then Retry Success
# ---------------------------------------------------------------------------
def test_out_of_range_then_retry_success(temp_db: Path):
    """Verify that out-of-range output triggers retry and succeeds when retry returns valid quantity."""
    client = LLMClient(mock_mode=True, db_path=temp_db)

    # Sequence: attempt 1 exceeds max capacity, attempt 2 provides valid order
    client.mock.set_sequential_responses(
        "retailer",
        [
            json.dumps({"quantity": 9999.0, "reasoning": "Excessive out-of-range order"}),
            json.dumps({"quantity": 75.0, "reasoning": "Corrected in-range order"}),
        ],
    )

    system = SingleAgentSystem(
        name="single_agent",
        llm_client=client,
        db_path=temp_db,
        run_id="run_range_retry_success",
    )

    state = make_dummy_state("retailer", step=4, last_order=60.0)
    orders = system.decide_orders({"retailer": state})

    assert orders["retailer"] == 75.0

    conn = duckdb.connect(str(temp_db))
    row = conn.execute(
        """
        SELECT retry_count, fallback, order_quantity
        FROM agent_calls
        WHERE run_id = 'run_range_retry_success' AND echelon = 'retailer'
        """
    ).fetchone()
    assert row[0] == 1
    assert row[1] is False
    assert row[2] == 75.0
    conn.close()


# ---------------------------------------------------------------------------
# 5. Out-of-Range Twice Then Fallback
# ---------------------------------------------------------------------------
def test_out_of_range_twice_then_fallback(temp_db: Path):
    """Verify that consecutive out-of-range outputs trigger fallback to state.last_order."""
    client = LLMClient(mock_mode=True, db_path=temp_db)

    client.mock.set_sequential_responses(
        "manufacturer",
        [
            json.dumps({"quantity": 10000.0, "reasoning": "First out-of-range"}),
            json.dumps({"quantity": -50.0, "reasoning": "Second out-of-range (negative)"}),
        ],
    )

    expected_last_order = 125.0
    system = MultiAgentInfoSystem(
        name="multi_agent_info",
        llm_client=client,
        db_path=temp_db,
        run_id="run_range_fallback_test",
    )

    state = make_dummy_state("manufacturer", step=5, last_order=expected_last_order)
    orders = system.decide_orders({"manufacturer": state})

    assert orders["manufacturer"] == expected_last_order

    conn = duckdb.connect(str(temp_db))
    row = conn.execute(
        """
        SELECT retry_count, fallback, order_quantity
        FROM agent_calls
        WHERE run_id = 'run_range_fallback_test' AND echelon = 'manufacturer'
        """
    ).fetchone()
    assert row[0] == 1
    assert row[1] is True
    assert row[2] == expected_last_order

    fallback_count = conn.execute(
        "SELECT count(*) FROM metrics WHERE run_id = 'run_range_fallback_test' AND metric_name = 'fallback_count'"
    ).fetchone()[0]
    assert fallback_count == 1
    conn.close()


# ---------------------------------------------------------------------------
# 6. Determinism Per Seed
# ---------------------------------------------------------------------------
def test_determinism_per_seed(temp_db: Path):
    """Verify that MockLLM scripted policy produces identical decisions given the same seed."""
    env1 = BeerGameEnv(seed=42)
    env1.reset(seed=42)

    client1 = LLMClient(mock_mode=True, db_path=temp_db)
    client1.mock.configure_scripted_policy(
        enabled=True,
        seed=12345,
        base_order=100.0,
        noise_std=15.0,
        max_order=300.0,
    )
    system1 = SingleAgentSystem(
        name="single_agent",
        llm_client=client1,
        db_path=temp_db,
        config={"seed": 12345},
    )

    traj1 = []
    for _ in range(5):
        states = {e: env1.observe(e) for e in ("retailer", "wholesaler", "distributor", "manufacturer")}
        orders = system1.decide_orders(states)
        traj1.append(dict(orders))
        env1.step(orders)

    # Second run with exact same seed
    env2 = BeerGameEnv(seed=42)
    env2.reset(seed=42)

    client2 = LLMClient(mock_mode=True, db_path=temp_db)
    client2.mock.configure_scripted_policy(
        enabled=True,
        seed=12345,
        base_order=100.0,
        noise_std=15.0,
        max_order=300.0,
    )
    system2 = SingleAgentSystem(
        name="single_agent",
        llm_client=client2,
        db_path=temp_db,
        config={"seed": 12345},
    )

    traj2 = []
    for _ in range(5):
        states = {e: env2.observe(e) for e in ("retailer", "wholesaler", "distributor", "manufacturer")}
        orders = system2.decide_orders(states)
        traj2.append(dict(orders))
        env2.step(orders)

    assert traj1 == traj2, "Runs with identical seeds must produce identical ordering trajectories"

    # Third run with different seed must differ
    env3 = BeerGameEnv(seed=42)
    env3.reset(seed=42)
    client3 = LLMClient(mock_mode=True, db_path=temp_db)
    client3.mock.configure_scripted_policy(
        enabled=True,
        seed=99999,
        base_order=100.0,
        noise_std=15.0,
        max_order=300.0,
    )
    system3 = SingleAgentSystem(
        name="single_agent",
        llm_client=client3,
        db_path=temp_db,
        config={"seed": 99999},
    )
    traj3 = []
    for _ in range(5):
        states = {e: env3.observe(e) for e in ("retailer", "wholesaler", "distributor", "manufacturer")}
        orders = system3.decide_orders(states)
        traj3.append(dict(orders))
        env3.step(orders)

    assert traj1 != traj3, "Runs with different seeds should produce different ordering trajectories"


# ---------------------------------------------------------------------------
# 7. Multi-Agent Shared Information Sharing
# ---------------------------------------------------------------------------
def test_multi_agent_info_shares_data(temp_db: Path):
    """Verify that MultiAgentInfoSystem embeds cross-echelon state in the prompt."""
    client = LLMClient(mock_mode=True, db_path=temp_db)
    system = MultiAgentInfoSystem(
        name="multi_agent_info",
        llm_client=client,
        db_path=temp_db,
    )

    states = {
        "retailer": make_dummy_state("retailer", step=1),
        "wholesaler": make_dummy_state("wholesaler", step=1),
        "distributor": make_dummy_state("distributor", step=1),
        "manufacturer": make_dummy_state("manufacturer", step=1),
    }

    system.decide_orders(states)

    # Check mock history prompts
    assert len(client.mock.call_history) >= 4
    for call in client.mock.call_history:
        prompt_text = call["prompt"]
        assert "SHARED SUPPLY CHAIN INFORMATION:" in prompt_text
        assert "Retailer:" in prompt_text
        assert "Wholesaler:" in prompt_text
        assert "Distributor:" in prompt_text
        assert "Manufacturer:" in prompt_text
