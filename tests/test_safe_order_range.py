"""Unit tests for safe_order_range and new pure tools (smoothed_forecast, order_up_to_target, smoothed_order, regime_detector)."""

import pytest

from sca.tools import (
    OrderUpToTargetInput,
    RegimeDetectorInput,
    SmoothedForecastInput,
    SmoothedOrderInput,
    ValidateOrderInput,
    order_up_to_target,
    regime_detector,
    safe_order_range,
    smoothed_forecast,
    smoothed_order,
    validate_order,
)


def test_safe_order_range_never_collapses_at_zero():
    """Verify that safe_order_range never collapses to [0, 0] when o_prev = 0."""
    # When previous order is 0 and forecast is 100 with delta = 0.5:
    low, high = safe_order_range(o_prev=0.0, d_hat=100.0, delta=0.5)
    assert low == 0.0
    assert high == 50.0
    assert high > low, "Safe range collapsed when o_prev = 0!"

    # Across varying positive forecasts and deltas:
    for d in [10.0, 50.0, 100.0, 250.0]:
        for delta in [0.2, 0.5, 0.8]:
            l, h = safe_order_range(o_prev=0.0, d_hat=d, delta=delta)
            assert l == 0.0
            assert h == pytest.approx(delta * d)
            assert h > 0.0, f"Range collapsed for d={d}, delta={delta}"


def test_safe_order_range_positive_orders():
    """Verify safe_order_range behavior when previous order is positive."""
    # When o_prev = 100 and d_hat = 100 with delta = 0.5:
    low, high = safe_order_range(o_prev=100.0, d_hat=100.0, delta=0.5)
    assert low == pytest.approx(50.0)
    assert high == pytest.approx(150.0)

    # When o_prev = 200 and d_hat = 100: scale = max(200, 100) = 200
    low2, high2 = safe_order_range(o_prev=200.0, d_hat=100.0, delta=0.5)
    assert low2 == pytest.approx(100.0)
    assert high2 == pytest.approx(300.0)


def test_validate_order_using_safe_order_range(tmp_path):
    """Verify validate_order rate limiting uses safe_order_range and does not clamp to 0 when last_order=0."""
    temp_db = tmp_path / "val_test.duckdb"

    # Proposed order is 40.0, last_order is 0.0, forecast is 100.0, delta is 0.5 (safe range [0, 50])
    res = validate_order(
        ValidateOrderInput(
            proposed_order=40.0,
            last_order=0.0,
            forecast_demand=100.0,
            rate_limit_delta=0.5,
        ),
        db_path=temp_db,
    )
    assert res.validated_order == 40.0
    assert res.is_valid is True
    assert not res.adjustments_made

    # Proposed order is 80.0, should be clipped to upper safe limit 50.0
    res_clipped = validate_order(
        ValidateOrderInput(
            proposed_order=80.0,
            last_order=0.0,
            forecast_demand=100.0,
            rate_limit_delta=0.5,
        ),
        db_path=temp_db,
    )
    assert res_clipped.validated_order == 50.0
    assert res_clipped.adjustments_made is True
    assert any("Rate limit intervention" in inv for inv in res_clipped.interventions)


def test_smoothed_forecast_tool(tmp_path):
    """Verify pure smoothed_forecast tool."""
    temp_db = tmp_path / "sm_fc_test.duckdb"
    res1 = smoothed_forecast(
        SmoothedForecastInput(current_demand=120.0, previous_smoothed_demand=None, alpha=0.8),
        db_path=temp_db,
    )
    assert res1.smoothed_demand == 120.0

    res2 = smoothed_forecast(
        SmoothedForecastInput(current_demand=150.0, previous_smoothed_demand=100.0, alpha=0.6),
        db_path=temp_db,
    )
    # 0.6 * 150 + 0.4 * 100 = 90 + 40 = 130
    assert res2.smoothed_demand == pytest.approx(130.0)


def test_order_up_to_target_tool(tmp_path):
    """Verify pure order_up_to_target tool."""
    temp_db = tmp_path / "out_target_test.duckdb"
    res = order_up_to_target(
        OrderUpToTargetInput(
            forecast_demand=100.0,
            lead_time_mean=3.0,
            lead_time_std=1.0,
            demand_std=20.0,
            z=2.0,
            include_lead_time_variance=True,
        ),
        db_path=temp_db,
    )
    # var_ltd = 3.0 * (20^2) + (100^2) * (1.0^2) = 1200 + 10000 = 11200
    # sigma_ltd = sqrt(11200) = 105.83
    # safety_stock = 2.0 * 105.83 = 211.66
    # target = 300 + 211.66 = 511.66
    assert res.order_up_to_target == pytest.approx(511.66, abs=0.2)
    assert res.expected_lead_time_demand == pytest.approx(300.0)


def test_smoothed_order_tool(tmp_path):
    """Verify pure smoothed_order tool."""
    temp_db = tmp_path / "sm_order_test.duckdb"
    res = smoothed_order(
        SmoothedOrderInput(
            order_up_to_target=500.0,
            inventory_position=400.0,
            last_order=80.0,
            beta=0.7,
        ),
        db_path=temp_db,
    )
    # raw_order = 500 - 400 = 100.0
    # smoothed_order = 0.7 * 100 + 0.3 * 80 = 70 + 24 = 94.0
    assert res.raw_order == pytest.approx(100.0)
    assert res.smoothed_order == pytest.approx(94.0)


def test_regime_detector_tool(tmp_path):
    """Verify pure regime_detector tool across stable, trend, and shock series."""
    temp_db = tmp_path / "regime_test.duckdb"

    # Stable series around 100
    stable_hist = [100.0 + (i % 3) for i in range(20)]
    res_stable = regime_detector(
        RegimeDetectorInput(demand_history=stable_hist),
        db_path=temp_db,
    )
    assert res_stable.regime == "stable"

    # Shock series with sudden level jump from 100 to 180 in recent 5 steps
    shock_hist = [100.0] * 15 + [180.0] * 5
    res_shock = regime_detector(
        RegimeDetectorInput(demand_history=shock_hist),
        db_path=temp_db,
    )
    assert res_shock.regime == "shock"
    assert res_shock.confidence >= 0.5
