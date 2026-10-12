"""Unit tests for supply chain tools against hand-computed values and CrewAI integration."""

import math

import duckdb
import pytest

from sca.tools import (
    ALL_CREWAI_TOOLS,
    EoqQuantityInput,
    EoqQuantityOutput,
    EoqQuantityTool,
    ForecastDemandInput,
    ForecastDemandOutput,
    ForecastDemandTool,
    SsPolicyOrderInput,
    SsPolicyOrderOutput,
    SsPolicyOrderTool,
    ValidateOrderInput,
    ValidateOrderOutput,
    ValidateOrderTool,
    eoq_quantity,
    forecast_demand,
    ss_policy_order,
    validate_order,
)


@pytest.fixture
def temp_db(tmp_path):
    """Temporary DuckDB path for tool testing."""
    return tmp_path / "tools_test.duckdb"


# ---------------------------------------------------------------------------
# 1. EOQ Tool Hand-Computed Tests
# ---------------------------------------------------------------------------
def test_eoq_quantity_hand_computed(temp_db):
    """Test EOQ against hand-computed values:

    Given:
      mean_demand (D) = 100.0 units/day
      order_fixed_cost (K) = 100.0 USD
      holding_cost_per_unit (h) = 0.50 USD/unit/day

    Hand Calculations:
      EOQ = sqrt(2 * 100 * 100 / 0.50) = sqrt(40000) = 200.0 units
      order_frequency_days = 200 / 100 = 2.0 days
      annual_orders = 365 / 2.0 = 182.5 orders/year
      cycle_holding_cost = (200 / 2) * 0.50 = 50.0 USD
      ordering_cost = 100.0 USD
      total_daily_cost = (100 * 100 / 200) + (200 * 0.50 / 2) = 50.0 + 50.0 = 100.0 USD/day
    """
    inp = EoqQuantityInput(
        mean_demand=100.0,
        order_fixed_cost=100.0,
        holding_cost_per_unit=0.50,
    )
    res: EoqQuantityOutput = eoq_quantity(inp, db_path=temp_db)

    assert res.eoq == pytest.approx(200.0)
    assert res.order_frequency_days == pytest.approx(2.0)
    assert res.annual_orders == pytest.approx(182.5)
    assert res.cycle_holding_cost == pytest.approx(50.0)
    assert res.ordering_cost == pytest.approx(100.0)
    assert res.total_daily_cost == pytest.approx(100.0)


def test_eoq_quantity_second_case(temp_db):
    """Test second hand-computed EOQ case:

    Given:
      D = 450, K = 50, h = 1.0
      EOQ = sqrt(2 * 450 * 50 / 1.0) = sqrt(45000) = 212.132
    """
    inp = EoqQuantityInput(
        mean_demand=450.0,
        order_fixed_cost=50.0,
        holding_cost_per_unit=1.0,
    )
    res = eoq_quantity(inp, db_path=temp_db)
    assert res.eoq == pytest.approx(math.sqrt(45000.0), rel=1e-2)


# ---------------------------------------------------------------------------
# 2. (s, S) Policy Order Hand-Computed Tests
# ---------------------------------------------------------------------------
def test_ss_policy_order_hand_computed(temp_db):
    """Test (s, S) policy against hand-computed values:

    Case 1: IP < s (Reorder Triggered)
      inventory = 100.0
      pipeline = 50.0
      backlog = 20.0
      reorder_point (s) = 200.0
      order_up_to (S) = 500.0

      IP = inventory + pipeline - backlog = 100 + 50 - 20 = 130.0
      Since IP (130) < s (200):
      order_quantity = S - IP = 500.0 - 130.0 = 370.0
      reorder_triggered = True
    """
    inp1 = SsPolicyOrderInput(
        inventory=100.0,
        pipeline=50.0,
        backlog=20.0,
        reorder_point=200.0,
        order_up_to=500.0,
    )
    res1: SsPolicyOrderOutput = ss_policy_order(inp1, db_path=temp_db)

    assert res1.inventory_position == pytest.approx(130.0)
    assert res1.order_quantity == pytest.approx(370.0)
    assert res1.reorder_triggered is True

    # Case 2: IP >= s (No Reorder)
    # inventory = 250.0, pipeline = 0.0, backlog = 0.0 -> IP = 250.0 >= 200.0
    # order_quantity = 0.0, reorder_triggered = False
    inp2 = SsPolicyOrderInput(
        inventory=250.0,
        pipeline=0.0,
        backlog=0.0,
        reorder_point=200.0,
        order_up_to=500.0,
    )
    res2 = ss_policy_order(inp2, db_path=temp_db)

    assert res2.inventory_position == pytest.approx(250.0)
    assert res2.order_quantity == 0.0
    assert res2.reorder_triggered is False


def test_ss_policy_invalid_levels():
    """Verify S < s raises validation error."""
    with pytest.raises(ValueError, match="order_up_to .* must be greater"):
        SsPolicyOrderInput(
            inventory=100.0,
            reorder_point=300.0,
            order_up_to=200.0,  # S < s invalid
        )


# ---------------------------------------------------------------------------
# 3. Validate Order Hand-Computed Tests
# ---------------------------------------------------------------------------
def test_validate_order_hand_computed(temp_db):
    """Test order validation against boundary constraints:

    Case 1: Capacity limit violation
      proposed_order = 450.0, capacity_limit = 300.0 -> clamped to 300.0, is_valid = False
    """
    inp1 = ValidateOrderInput(
        proposed_order=450.0,
        capacity_limit=300.0,
        min_order=0.0,
    )
    res1: ValidateOrderOutput = validate_order(inp1, db_path=temp_db)
    assert res1.validated_order == pytest.approx(300.0)
    assert res1.is_valid is False
    assert res1.adjustments_made is True
    assert len(res1.violations) == 1
    assert "capacity limit" in res1.violations[0]

    # Case 2: Below minimum allowable order
    # proposed_order = -25.0, min_order = 0.0 -> clamped to 0.0, is_valid = False
    inp2 = ValidateOrderInput(
        proposed_order=-25.0,
        min_order=0.0,
    )
    res2 = validate_order(inp2, db_path=temp_db)
    assert res2.validated_order == 0.0
    assert res2.is_valid is False
    assert res2.adjustments_made is True

    # Case 3: Compliant order within bounds
    # proposed_order = 180.0, capacity_limit = 300.0, min_order = 0.0 -> 180.0, is_valid = True
    inp3 = ValidateOrderInput(
        proposed_order=180.0,
        capacity_limit=300.0,
        min_order=0.0,
    )
    res3 = validate_order(inp3, db_path=temp_db)
    assert res3.validated_order == 180.0
    assert res3.is_valid is True
    assert res3.adjustments_made is False
    assert len(res3.violations) == 0


# ---------------------------------------------------------------------------
# 4. Forecast Demand Tool Tests
# ---------------------------------------------------------------------------
def test_forecast_demand_tool(temp_db):
    """Test demand forecast tool on flat and trending historical sequences."""
    inp = ForecastDemandInput(
        recent_history=[100.0, 100.0, 100.0, 100.0, 100.0],
        horizon=3,
        model_type="linear",
    )
    res: ForecastDemandOutput = forecast_demand(inp, db_path=temp_db)

    assert res.horizon == 3
    assert len(res.forecast_series) == 3
    assert res.point_forecast == pytest.approx(100.0, abs=1.0)
    assert res.mean_forecast == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# 5. DuckDB Execution Logging Verification
# ---------------------------------------------------------------------------
def test_tool_calls_logged_to_duckdb(temp_db):
    """Verify each tool invocation writes a structured record to DuckDB."""
    # Execute tools
    eoq_quantity(
        EoqQuantityInput(mean_demand=120.0, order_fixed_cost=80.0, holding_cost_per_unit=0.40),
        db_path=temp_db,
    )
    ss_policy_order(
        SsPolicyOrderInput(inventory=50.0, reorder_point=150.0, order_up_to=400.0),
        db_path=temp_db,
    )
    validate_order(
        ValidateOrderInput(proposed_order=250.0, capacity_limit=300.0),
        db_path=temp_db,
    )

    conn = duckdb.connect(str(temp_db), read_only=True)
    rows = conn.execute("SELECT tool_name, is_error FROM tool_calls").fetchall()
    conn.close()

    assert len(rows) == 3
    tool_names = [r[0] for r in rows]
    assert "eoq_quantity" in tool_names
    assert "ss_policy_order" in tool_names
    assert "validate_order" in tool_names
    assert all(r[1] is False for r in rows)


# ---------------------------------------------------------------------------
# 6. CrewAI Tools Integration Tests
# ---------------------------------------------------------------------------
def test_crewai_tool_classes():
    """Verify tools can be run via CrewAI BaseTool interface."""
    assert len(ALL_CREWAI_TOOLS) >= 4

    eoq_tool = EoqQuantityTool()
    eoq_json = eoq_tool.run(mean_demand=100.0, order_fixed_cost=100.0, holding_cost_per_unit=0.50)
    assert "eoq" in eoq_json
    assert "200.0" in eoq_json

    ss_tool = SsPolicyOrderTool()
    ss_json = ss_tool.run(inventory=100.0, pipeline=0.0, backlog=0.0, reorder_point=200.0, order_up_to=500.0)
    assert "order_quantity" in ss_json
    assert "400.0" in ss_json

    val_tool = ValidateOrderTool()
    val_json = val_tool.run(proposed_order=500.0, capacity_limit=300.0)
    assert "validated_order" in val_json
    assert "300.0" in val_json

    f_tool = ForecastDemandTool()
    f_json = f_tool.run(recent_history=[90.0, 95.0, 100.0], horizon=2)
    assert "point_forecast" in f_json

    from sca.tools import SmoothedOrderUpToTool
    sm_tool = SmoothedOrderUpToTool()
    sm_json = sm_tool.run(
        current_demand=100.0,
        inventory=200.0,
        pipeline=50.0,
        backlog=0.0,
        last_order=100.0,
        lead_time_mean=2.0,
        alpha=0.2,
    )
    assert "order_quantity" in sm_json


def test_smoothed_order_up_to_hand_computed(temp_db):
    """Test smoothed_order_up_to calculations."""
    from sca.tools import SmoothedOrderUpToInput, smoothed_order_up_to

    inp = SmoothedOrderUpToInput(
        current_demand=100.0,
        inventory=150.0,
        pipeline=50.0,
        backlog=0.0,
        last_order=100.0,
        lead_time_mean=2.0,
        lead_time_std=0.0,
        demand_std=10.0,
        safety_stock_factor=1.65,
        alpha=0.5,
    )
    res = smoothed_order_up_to(inp, db_path=temp_db)
    # L=2, demand_std=10, var_ltd = 2 * 100 = 200, sigma_ltd = sqrt(200) = 14.142
    # SS = 1.65 * 14.142 = 23.33
    # Target S = 2 * 100 + 23.33 = 223.33
    # IP = 150 + 50 - 0 = 200
    # raw_order = 223.33 - 200 = 23.33
    # smoothed_order = 0.5 * 23.33 + 0.5 * 100 = 11.67 + 50 = 61.67
    assert res.smoothed_demand == pytest.approx(100.0)
    assert res.inventory_position == pytest.approx(200.0)
    assert res.order_quantity == pytest.approx(61.67, abs=0.1)


def test_validate_order_rate_limiting_and_cap(temp_db):
    """Test validate_order rate limiting and forecast-relative cap clipping."""
    # Test rate limit upper clip
    val_upper = validate_order(
        ValidateOrderInput(
            proposed_order=200.0,
            last_order=100.0,
            rate_limit_delta=0.5,  # max allowed = 100 + 0.5*100 = 150.0
        ),
        db_path=temp_db,
    )
    assert val_upper.validated_order == 150.0
    assert val_upper.adjustments_made is True
    assert any("Rate limit intervention" in v for v in val_upper.interventions)

    # Test rate limit lower clip
    val_lower = validate_order(
        ValidateOrderInput(
            proposed_order=20.0,
            last_order=100.0,
            rate_limit_delta=0.5,  # min allowed = 100 - 0.5*100 = 50.0
        ),
        db_path=temp_db,
    )
    assert val_lower.validated_order == 50.0
    assert val_lower.adjustments_made is True

    # Test forecast-relative cap
    val_cap = validate_order(
        ValidateOrderInput(
            proposed_order=300.0,
            forecast_demand=100.0,
            forecast_cap_multiple=2.0,  # max allowed = 200.0
        ),
        db_path=temp_db,
    )
    assert val_cap.validated_order == 200.0
    assert val_cap.adjustments_made is True
    assert any("Forecast cap intervention" in v for v in val_cap.interventions)

