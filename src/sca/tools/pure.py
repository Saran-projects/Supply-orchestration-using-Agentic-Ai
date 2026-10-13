"""Pure Python tool implementations for supply chain decision-making with DuckDB execution logging."""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sca.config import get_config
from sca.db import get_duckdb_connection
from sca.forecast.models import get_forecaster
from sca.tools.schemas import (
    EoqQuantityInput,
    EoqQuantityOutput,
    ForecastDemandInput,
    ForecastDemandOutput,
    OrderUpToTargetInput,
    OrderUpToTargetOutput,
    RegimeDetectorInput,
    RegimeDetectorOutput,
    SmoothedForecastInput,
    SmoothedForecastOutput,
    SmoothedOrderInput,
    SmoothedOrderOutput,
    SmoothedOrderUpToInput,
    SmoothedOrderUpToOutput,
    SsPolicyOrderInput,
    SsPolicyOrderOutput,
    ValidateOrderInput,
    ValidateOrderOutput,
)

logger = logging.getLogger(__name__)


def log_tool_to_duckdb(
    tool_name: str,
    input_args: Dict[str, Any],
    output_result: Dict[str, Any],
    duration_ms: float,
    is_error: bool = False,
    error_message: Optional[str] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
    expected_tool: Optional[str] = None,
    is_correct_tool: Optional[bool] = None,
    is_parameter_valid: Optional[bool] = None,
) -> str:
    """Log tool call execution details into DuckDB tool_calls table."""
    effective_run_id = run_id or "standalone_tool_session"
    effective_agent = agent_name or "supply_chain_agent"
    tool_call_id = f"tool_{uuid.uuid4().hex[:12]}"
    eff_correct = (
        is_correct_tool
        if is_correct_tool is not None
        else ((tool_name == expected_tool) if expected_tool else None)
    )
    eff_param_valid = (
        is_parameter_valid
        if is_parameter_valid is not None
        else (not is_error)
    )

    if db_path is False or db_path == "disabled" or db_path == "":
        return tool_call_id

    target_db = db_path
    if target_db is None:
        try:
            target_db = get_config().experiment.db_path
        except Exception:
            target_db = Path("results/sca_metrics.duckdb")

    try:
        conn = get_duckdb_connection(target_db)
        # Ensure parent run row exists to satisfy foreign key constraint
        conn.execute(
            """
            INSERT OR IGNORE INTO runs (run_id, experiment_name, status, metadata)
            VALUES (?, 'tool_execution', 'active', ?)
            """,
            [effective_run_id, json.dumps({"source": "tool_call"})],
        )

        conn.execute(
            """
            INSERT INTO tool_calls (
                tool_call_id, call_id, run_id, step_id, agent_name,
                tool_name, input_arguments, output_result, is_error,
                error_message, duration_ms, expected_tool, is_correct_tool, is_parameter_valid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                tool_call_id,
                None,
                effective_run_id,
                step_id,
                effective_agent,
                tool_name,
                json.dumps(input_args),
                json.dumps(output_result),
                is_error,
                error_message,
                round(duration_ms, 2),
                expected_tool,
                eff_correct,
                eff_param_valid,
            ],
        )
        conn.close()
    except Exception as exc:
        logger.warning("Could not log tool call '%s' to DuckDB (%s): %s", tool_name, target_db, exc)

    return tool_call_id


# ---------------------------------------------------------------------------
# 1. forecast_demand
# ---------------------------------------------------------------------------
def forecast_demand(
    params: Union[ForecastDemandInput, Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_tool: Optional[str] = None,
    is_correct_tool: Optional[bool] = None,
    is_parameter_valid: Optional[bool] = None,
    **kwargs: Any,
) -> ForecastDemandOutput:
    """Forecast future demand based on recent historical demand series."""
    start_time = time.time()
    inp_dict = (
        params.model_dump()
        if hasattr(params, "model_dump")
        else (dict(params) if isinstance(params, dict) else {"raw": str(params)})
    )
    is_error = False
    error_msg = None
    output: Optional[ForecastDemandOutput] = None

    try:
        inp = params if isinstance(params, ForecastDemandInput) else ForecastDemandInput.model_validate(params)
        inp_dict = inp.model_dump()
        hist = np.asarray(inp.recent_history, dtype=float)
        horizon = inp.horizon

        if len(hist) < 2:
            # Fallback for single observation
            single_val = float(hist[0])
            series = [single_val] * horizon
        elif len(hist) <= 14:
            # Short history: fit linear trend
            x = np.arange(len(hist), dtype=float)
            slope, intercept = np.polyfit(x, hist, 1)
            future_x = np.arange(len(hist), len(hist) + horizon, dtype=float)
            series = [float(max(0.0, slope * fx + intercept)) for fx in future_x]
        else:
            try:
                # Full lag-feature model
                forecaster = get_forecaster(model_type=inp.model_type)
                forecaster.fit(hist)
                series = [float(p) for p in forecaster.predict_horizon(hist, horizon=horizon)]
            except Exception:
                x = np.arange(len(hist), dtype=float)
                slope, intercept = np.polyfit(x, hist, 1)
                future_x = np.arange(len(hist), len(hist) + horizon, dtype=float)
                series = [float(max(0.0, slope * fx + intercept)) for fx in future_x]

        point_forecast = round(series[0], 2)
        mean_forecast = round(float(np.mean(series)), 2)
        series_rounded = [round(val, 2) for val in series]

        output = ForecastDemandOutput(
            point_forecast=point_forecast,
            forecast_series=series_rounded,
            mean_forecast=mean_forecast,
            horizon=horizon,
            model_used=inp.model_type,
        )

    except Exception as e:
        is_error = True
        error_msg = str(e)
        logger.exception("Error in forecast_demand: %s", e)
        raise e
    finally:
        elapsed_ms = (time.time() - start_time) * 1000.0
        out_dict = output.model_dump() if output else {}
        log_tool_to_duckdb(
            tool_name="forecast_demand",
            input_args=inp_dict,
            output_result=out_dict,
            duration_ms=elapsed_ms,
            is_error=is_error,
            error_message=error_msg,
            run_id=run_id,
            step_id=step_id,
            agent_name=agent_name,
            db_path=db_path,
            expected_tool=expected_tool,
            is_correct_tool=is_correct_tool,
            is_parameter_valid=is_parameter_valid,
        )

    return output


# ---------------------------------------------------------------------------
# 2. eoq_quantity
# ---------------------------------------------------------------------------
def eoq_quantity(
    params: Union[EoqQuantityInput, Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_tool: Optional[str] = None,
    is_correct_tool: Optional[bool] = None,
    is_parameter_valid: Optional[bool] = None,
    **kwargs: Any,
) -> EoqQuantityOutput:
    """Calculate Economic Order Quantity (EOQ) using Wilson's formula."""
    start_time = time.time()
    inp_dict = (
        params.model_dump()
        if hasattr(params, "model_dump")
        else (dict(params) if isinstance(params, dict) else {"raw": str(params)})
    )
    is_error = False
    error_msg = None
    output: Optional[EoqQuantityOutput] = None

    try:
        inp = params if isinstance(params, EoqQuantityInput) else EoqQuantityInput.model_validate(params)
        inp_dict = inp.model_dump()
        mu = inp.mean_demand
        K = inp.order_fixed_cost
        h = inp.holding_cost_per_unit

        # Wilson's classic EOQ formula: sqrt(2 * D * K / h)
        eoq_val = math.sqrt((2.0 * mu * K) / h)
        order_freq = eoq_val / mu
        annual_orders = 365.0 / max(0.01, order_freq)
        cycle_holding_cost = (eoq_val / 2.0) * h
        ordering_cost = K
        total_daily_cost = (mu * K / eoq_val) + (eoq_val * h / 2.0)

        output = EoqQuantityOutput(
            eoq=round(eoq_val, 2),
            order_frequency_days=round(order_freq, 2),
            annual_orders=round(annual_orders, 2),
            cycle_holding_cost=round(cycle_holding_cost, 2),
            ordering_cost=round(ordering_cost, 2),
            total_daily_cost=round(total_daily_cost, 2),
        )

    except Exception as e:
        is_error = True
        error_msg = str(e)
        logger.exception("Error in eoq_quantity: %s", e)
        raise e
    finally:
        elapsed_ms = (time.time() - start_time) * 1000.0
        out_dict = output.model_dump() if output else {}
        log_tool_to_duckdb(
            tool_name="eoq_quantity",
            input_args=inp_dict,
            output_result=out_dict,
            duration_ms=elapsed_ms,
            is_error=is_error,
            error_message=error_msg,
            run_id=run_id,
            step_id=step_id,
            agent_name=agent_name,
            db_path=db_path,
            expected_tool=expected_tool,
            is_correct_tool=is_correct_tool,
            is_parameter_valid=is_parameter_valid,
        )

    return output


# ---------------------------------------------------------------------------
# 3. ss_policy_order
# ---------------------------------------------------------------------------
def ss_policy_order(
    params: Union[SsPolicyOrderInput, Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_tool: Optional[str] = None,
    is_correct_tool: Optional[bool] = None,
    is_parameter_valid: Optional[bool] = None,
    **kwargs: Any,
) -> SsPolicyOrderOutput:
    """Calculate order quantity under (s, S) min-max policy: order S - IP if IP < s else 0."""
    start_time = time.time()
    inp_dict = (
        params.model_dump()
        if hasattr(params, "model_dump")
        else (dict(params) if isinstance(params, dict) else {"raw": str(params)})
    )
    is_error = False
    error_msg = None
    output: Optional[SsPolicyOrderOutput] = None

    try:
        inp = params if isinstance(params, SsPolicyOrderInput) else SsPolicyOrderInput.model_validate(params)
        inp_dict = inp.model_dump()
        # Inventory position: on-hand + pipeline - backlog
        ip = inp.inventory + inp.pipeline - inp.backlog
        s = inp.reorder_point
        S = inp.order_up_to

        if ip < s:
            order_qty = max(0.0, S - ip)
            triggered = True
        else:
            order_qty = 0.0
            triggered = False

        output = SsPolicyOrderOutput(
            order_quantity=round(order_qty, 2),
            inventory_position=round(ip, 2),
            reorder_triggered=triggered,
            reorder_point=round(s, 2),
            order_up_to=round(S, 2),
        )

    except Exception as e:
        is_error = True
        error_msg = str(e)
        logger.exception("Error in ss_policy_order: %s", e)
        raise e
    finally:
        elapsed_ms = (time.time() - start_time) * 1000.0
        out_dict = output.model_dump() if output else {}
        log_tool_to_duckdb(
            tool_name="ss_policy_order",
            input_args=inp_dict,
            output_result=out_dict,
            duration_ms=elapsed_ms,
            is_error=is_error,
            error_message=error_message if (error_msg is not None and (error_message := error_msg)) else error_msg,
            run_id=run_id,
            step_id=step_id,
            agent_name=agent_name,
            db_path=db_path,
            expected_tool=expected_tool,
            is_correct_tool=is_correct_tool,
            is_parameter_valid=is_parameter_valid,
        )

    return output


# ---------------------------------------------------------------------------
# 3b. safe_order_range
# ---------------------------------------------------------------------------
def safe_order_range(
    o_prev: float,
    d_hat: float,
    delta: float,
) -> Tuple[float, float]:
    """Calculate safe order range around previous order scaled by max(o_prev, d_hat).

    [max(0.0, o_prev - delta * max(o_prev, d_hat)), o_prev + delta * max(o_prev, d_hat)]
    Guarantees the safe order range never collapses when o_prev == 0 (as long as d_hat > 0).
    """
    scale = max(float(o_prev), float(d_hat))
    low = max(0.0, float(o_prev) - float(delta) * scale)
    high = float(o_prev) + float(delta) * scale
    return (float(low), float(high))


# ---------------------------------------------------------------------------
# 4. validate_order
# ---------------------------------------------------------------------------
def validate_order(
    params: Union[ValidateOrderInput, Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_tool: Optional[str] = None,
    is_correct_tool: Optional[bool] = None,
    is_parameter_valid: Optional[bool] = None,
    **kwargs: Any,
) -> ValidateOrderOutput:
    """Validate and constrain proposed order against capacity, inventory position, and bounds."""
    start_time = time.time()
    inp_dict = (
        params.model_dump()
        if hasattr(params, "model_dump")
        else (dict(params) if isinstance(params, dict) else {"raw": str(params)})
    )
    is_error = False
    error_msg = None
    output: Optional[ValidateOrderOutput] = None

    try:
        inp = params if isinstance(params, ValidateOrderInput) else ValidateOrderInput.model_validate(params)
        inp_dict = inp.model_dump()
        orig = float(inp.proposed_order)
        val = orig
        violations: List[str] = []
        adjustments_made = False

        # Resolve config defaults for delta and cap multiple if not provided in input
        cfg_fe = None
        try:
            cfg_fe = get_config().sim.four_echelon
        except Exception:
            pass

        eff_delta = (
            inp.rate_limit_delta
            if inp.rate_limit_delta is not None
            else (getattr(cfg_fe, "rate_limit_delta", None) if cfg_fe else None)
        )
        eff_cap_mult = (
            inp.forecast_cap_multiple
            if inp.forecast_cap_multiple is not None
            else (getattr(cfg_fe, "forecast_cap_multiple", None) if cfg_fe else None)
        )

        interventions: List[str] = []

        # 1. Non-negative / minimum order check (Hard bound lower: min_order >= 0)
        min_bound = max(0.0, float(inp.min_order))
        if val < min_bound:
            msg = f"Proposed order {val:.1f} is below minimum allowed threshold {min_bound:.1f}."
            violations.append(msg)
            interventions.append(msg)
            logger.info("validate_order clip: %.2f -> %.2f (%s)", val, min_bound, msg)
            val = min_bound
            adjustments_made = True

        # 2. Rate limiting check (Smoothing band via shared function safe_order_range)
        if eff_delta is not None and inp.last_order is not None:
            ref_d = (
                inp.forecast_demand
                if (inp.forecast_demand is not None and inp.forecast_demand > 0)
                else max(inp.last_order, 1.0)
            )
            min_allowed, max_allowed = safe_order_range(inp.last_order, ref_d, eff_delta)
            if val > max_allowed:
                msg = (
                    f"Rate limit intervention: proposed order {val:.1f} clipped to upper limit "
                    f"{max_allowed:.1f} (delta={eff_delta:.2f}, last_order={inp.last_order:.1f})."
                )
                violations.append(msg)
                interventions.append(msg)
                logger.info("validate_order clip: %.2f -> %.2f (%s)", val, max_allowed, msg)
                val = max_allowed
                adjustments_made = True
            elif val < min_allowed:
                msg = (
                    f"Rate limit intervention: proposed order {val:.1f} clipped to lower limit "
                    f"{min_allowed:.1f} (delta={eff_delta:.2f}, last_order={inp.last_order:.1f})."
                )
                violations.append(msg)
                interventions.append(msg)
                logger.info("validate_order clip: %.2f -> %.2f (%s)", val, min_allowed, msg)
                val = min_allowed
                adjustments_made = True

        # 3. Forecast-relative cap check
        if eff_cap_mult is not None and inp.forecast_demand is not None and inp.forecast_demand > 0:
            cap_val = eff_cap_mult * inp.forecast_demand
            if val > cap_val:
                msg = (
                    f"Forecast cap intervention: proposed order {val:.1f} clipped to {cap_val:.1f} "
                    f"(cap_multiple={eff_cap_mult:.1f}, forecast={inp.forecast_demand:.1f})."
                )
                violations.append(msg)
                interventions.append(msg)
                logger.info("validate_order clip: %.2f -> %.2f (%s)", val, cap_val, msg)
                val = cap_val
                adjustments_made = True

        # 4. Capacity limit constraint (Hard bound upper: 0..capacity)
        if inp.capacity_limit is not None and val > inp.capacity_limit:
            msg = f"Capacity intervention: proposed order {val:.1f} exceeds facility capacity limit of {inp.capacity_limit:.1f} units."
            violations.append(msg)
            interventions.append(msg)
            logger.info("validate_order clip: %.2f -> %.2f (%s)", val, inp.capacity_limit, msg)
            val = float(inp.capacity_limit)
            adjustments_made = True

        # 5. Maximum order ceiling check
        if inp.max_order is not None and val > inp.max_order:
            msg = f"Proposed order {val:.1f} exceeds maximum order policy threshold of {inp.max_order:.1f} units."
            violations.append(msg)
            interventions.append(msg)
            logger.info("validate_order clip: %.2f -> %.2f (%s)", val, inp.max_order, msg)
            val = float(inp.max_order)
            adjustments_made = True

        # 6. Reasonableness check against extreme positive inventory position
        if inp.inventory_position > 5000.0 and val > 0:
            violations.append(
                f"Warning: placing order ({val:.1f}) while inventory position ({inp.inventory_position:.1f}) is exceptionally high."
            )

        is_valid = (len(violations) == 0)

        output = ValidateOrderOutput(
            validated_order=round(val, 2),
            original_order=round(orig, 2),
            is_valid=is_valid,
            adjustments_made=adjustments_made,
            violations=violations,
            inventory_position=round(inp.inventory_position, 2),
            interventions=interventions,
        )

    except Exception as e:
        is_error = True
        error_msg = str(e)
        logger.exception("Error in validate_order: %s", e)
        raise e
    finally:
        elapsed_ms = (time.time() - start_time) * 1000.0
        out_dict = output.model_dump() if output else {}
        log_tool_to_duckdb(
            tool_name="validate_order",
            input_args=inp_dict,
            output_result=out_dict,
            duration_ms=elapsed_ms,
            is_error=is_error,
            error_message=error_msg,
            run_id=run_id,
            step_id=step_id,
            agent_name=agent_name,
            db_path=db_path,
            expected_tool=expected_tool,
            is_correct_tool=is_correct_tool,
            is_parameter_valid=is_parameter_valid,
        )

    return output


# ---------------------------------------------------------------------------
# 5. smoothed_order_up_to
# ---------------------------------------------------------------------------
def smoothed_order_up_to(
    params: Union[SmoothedOrderUpToInput, Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_tool: Optional[str] = None,
    is_correct_tool: Optional[bool] = None,
    is_parameter_valid: Optional[bool] = None,
    **kwargs: Any,
) -> SmoothedOrderUpToOutput:
    """Calculate replenishment order using exponentially smoothed demand forecast, order-up-to target with safety stock, and order smoothing."""
    start_time = time.time()
    inp_dict = (
        params.model_dump()
        if hasattr(params, "model_dump")
        else (dict(params) if isinstance(params, dict) else {"raw": str(params)})
    )
    is_error = False
    error_msg = None
    output: Optional[SmoothedOrderUpToOutput] = None

    try:
        inp = params if isinstance(params, SmoothedOrderUpToInput) else SmoothedOrderUpToInput.model_validate(params)
        inp_dict = inp.model_dump()

        # 1. Exponentially smoothed demand forecast (alpha)
        alpha = inp.alpha
        beta = inp.beta if inp.beta is not None else alpha

        if inp.previous_smoothed_demand is not None:
            sm_demand = (alpha * inp.current_demand) + ((1.0 - alpha) * inp.previous_smoothed_demand)
        else:
            sm_demand = inp.current_demand
        sm_demand = max(0.0, sm_demand)

        # 2. Dynamic order-up-to target S with safety stock (z, lead time mean & optional lead time variance)
        L = max(0.0, float(inp.lead_time_mean))
        sigma_L = max(0.0, float(inp.lead_time_std)) if inp.include_lead_time_variance else 0.0
        sigma_d = max(0.0, float(inp.demand_std))
        effective_z = inp.z if inp.z is not None else inp.safety_stock_factor

        var_ltd = (L * (sigma_d ** 2)) + ((sm_demand ** 2) * (sigma_L ** 2))
        sigma_ltd = math.sqrt(max(1.0, var_ltd))
        safety_stock = effective_z * sigma_ltd
        order_up_to_target = (L * sm_demand) + safety_stock

        # 3. Inventory position and raw deficit
        ip = inp.inventory + inp.pipeline - inp.backlog
        raw_order = max(0.0, order_up_to_target - ip)

        # 4. Order smoothing with beta
        smoothed_order = (beta * raw_order) + ((1.0 - beta) * inp.last_order)
        smoothed_order = max(0.0, smoothed_order)

        output = SmoothedOrderUpToOutput(
            order_quantity=round(smoothed_order, 2),
            raw_order_quantity=round(raw_order, 2),
            order_up_to_target=round(order_up_to_target, 2),
            smoothed_demand=round(sm_demand, 2),
            safety_stock=round(safety_stock, 2),
            inventory_position=round(ip, 2),
            alpha=round(alpha, 4),
            beta=round(beta, 4),
            z=round(effective_z, 4),
            include_lead_time_variance=inp.include_lead_time_variance,
        )

    except Exception as e:
        is_error = True
        error_msg = str(e)
        logger.exception("Error in smoothed_order_up_to: %s", e)
        raise e
    finally:
        elapsed_ms = (time.time() - start_time) * 1000.0
        out_dict = output.model_dump() if output else {}
        log_tool_to_duckdb(
            tool_name="smoothed_order_up_to",
            input_args=inp_dict,
            output_result=out_dict,
            duration_ms=elapsed_ms,
            is_error=is_error,
            error_message=error_msg,
            run_id=run_id,
            step_id=step_id,
            agent_name=agent_name,
            db_path=db_path,
            expected_tool=expected_tool or "smoothed_order_up_to",
            is_correct_tool=is_correct_tool,
            is_parameter_valid=is_parameter_valid,
        )

    return output


# ---------------------------------------------------------------------------
# 6. smoothed_forecast
# ---------------------------------------------------------------------------
def smoothed_forecast(
    params: Union[SmoothedForecastInput, Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_tool: Optional[str] = None,
    is_correct_tool: Optional[bool] = None,
    is_parameter_valid: Optional[bool] = None,
    **kwargs: Any,
) -> SmoothedForecastOutput:
    """Calculate exponential smoothing demand forecast with parameter alpha."""
    start_time = time.time()
    inp = params if isinstance(params, SmoothedForecastInput) else SmoothedForecastInput.model_validate(params)
    inp_dict = inp.model_dump()
    is_error = False
    error_msg = None
    output: Optional[SmoothedForecastOutput] = None

    try:
        alpha = float(inp.alpha)
        if inp.previous_smoothed_demand is not None:
            sm = (alpha * float(inp.current_demand)) + ((1.0 - alpha) * float(inp.previous_smoothed_demand))
        else:
            sm = float(inp.current_demand)
        sm = max(0.0, sm)

        output = SmoothedForecastOutput(
            smoothed_demand=round(sm, 2),
            current_demand=round(float(inp.current_demand), 2),
            alpha=round(alpha, 4),
        )
    except Exception as e:
        is_error = True
        error_msg = str(e)
        logger.exception("Error in smoothed_forecast: %s", e)
        raise e
    finally:
        elapsed_ms = (time.time() - start_time) * 1000.0
        out_dict = output.model_dump() if output else {}
        log_tool_to_duckdb(
            tool_name="smoothed_forecast",
            input_args=inp_dict,
            output_result=out_dict,
            duration_ms=elapsed_ms,
            is_error=is_error,
            error_message=error_msg,
            run_id=run_id,
            step_id=step_id,
            agent_name=agent_name,
            db_path=db_path,
            expected_tool=expected_tool or "smoothed_forecast",
            is_correct_tool=is_correct_tool,
            is_parameter_valid=is_parameter_valid,
        )

    return output


# ---------------------------------------------------------------------------
# 7. order_up_to_target
# ---------------------------------------------------------------------------
def order_up_to_target(
    params: Union[OrderUpToTargetInput, Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_tool: Optional[str] = None,
    is_correct_tool: Optional[bool] = None,
    is_parameter_valid: Optional[bool] = None,
    **kwargs: Any,
) -> OrderUpToTargetOutput:
    """Calculate base-stock target level S = L * d_hat + SS using lead-time demand variance."""
    start_time = time.time()
    inp = params if isinstance(params, OrderUpToTargetInput) else OrderUpToTargetInput.model_validate(params)
    inp_dict = inp.model_dump()
    is_error = False
    error_msg = None
    output: Optional[OrderUpToTargetOutput] = None

    try:
        L = max(0.0, float(inp.lead_time_mean))
        sigma_L = max(0.0, float(inp.lead_time_std)) if inp.include_lead_time_variance else 0.0
        sigma_d = max(0.0, float(inp.demand_std))
        d_hat = max(0.0, float(inp.forecast_demand))
        z = float(inp.z)

        var_ltd = (L * (sigma_d ** 2)) + ((d_hat ** 2) * (sigma_L ** 2))
        sigma_ltd = math.sqrt(max(1.0, var_ltd))
        safety_stock = z * sigma_ltd
        ltd_mean = L * d_hat
        target_S = ltd_mean + safety_stock

        output = OrderUpToTargetOutput(
            order_up_to_target=round(target_S, 2),
            safety_stock=round(safety_stock, 2),
            sigma_ltd=round(sigma_ltd, 2),
            expected_lead_time_demand=round(ltd_mean, 2),
        )
    except Exception as e:
        is_error = True
        error_msg = str(e)
        logger.exception("Error in order_up_to_target: %s", e)
        raise e
    finally:
        elapsed_ms = (time.time() - start_time) * 1000.0
        out_dict = output.model_dump() if output else {}
        log_tool_to_duckdb(
            tool_name="order_up_to_target",
            input_args=inp_dict,
            output_result=out_dict,
            duration_ms=elapsed_ms,
            is_error=is_error,
            error_message=error_msg,
            run_id=run_id,
            step_id=step_id,
            agent_name=agent_name,
            db_path=db_path,
            expected_tool=expected_tool or "order_up_to_target",
            is_correct_tool=is_correct_tool,
            is_parameter_valid=is_parameter_valid,
        )

    return output


# ---------------------------------------------------------------------------
# 8. smoothed_order
# ---------------------------------------------------------------------------
def smoothed_order(
    params: Union[SmoothedOrderInput, Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_tool: Optional[str] = None,
    is_correct_tool: Optional[bool] = None,
    is_parameter_valid: Optional[bool] = None,
    **kwargs: Any,
) -> SmoothedOrderOutput:
    """Calculate smoothed order with parameter beta: beta * raw + (1 - beta) * last_order."""
    start_time = time.time()
    inp = params if isinstance(params, SmoothedOrderInput) else SmoothedOrderInput.model_validate(params)
    inp_dict = inp.model_dump()
    is_error = False
    error_msg = None
    output: Optional[SmoothedOrderOutput] = None

    try:
        raw_order = max(0.0, float(inp.order_up_to_target) - float(inp.inventory_position))
        beta = float(inp.beta)
        sm_order = (beta * raw_order) + ((1.0 - beta) * float(inp.last_order))
        sm_order = max(0.0, sm_order)

        output = SmoothedOrderOutput(
            smoothed_order=round(sm_order, 2),
            raw_order=round(raw_order, 2),
            beta=round(beta, 4),
            inventory_position=round(float(inp.inventory_position), 2),
        )
    except Exception as e:
        is_error = True
        error_msg = str(e)
        logger.exception("Error in smoothed_order: %s", e)
        raise e
    finally:
        elapsed_ms = (time.time() - start_time) * 1000.0
        out_dict = output.model_dump() if output else {}
        log_tool_to_duckdb(
            tool_name="smoothed_order",
            input_args=inp_dict,
            output_result=out_dict,
            duration_ms=elapsed_ms,
            is_error=is_error,
            error_message=error_msg,
            run_id=run_id,
            step_id=step_id,
            agent_name=agent_name,
            db_path=db_path,
            expected_tool=expected_tool or "smoothed_order",
            is_correct_tool=is_correct_tool,
            is_parameter_valid=is_parameter_valid,
        )

    return output


# ---------------------------------------------------------------------------
# 9. regime_detector
# ---------------------------------------------------------------------------
def regime_detector(
    params: Union[RegimeDetectorInput, Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    expected_tool: Optional[str] = None,
    is_correct_tool: Optional[bool] = None,
    is_parameter_valid: Optional[bool] = None,
    **kwargs: Any,
) -> RegimeDetectorOutput:
    """Detect demand regime: 'stable' | 'trend' | 'shock' using short/long mean ratio and CUSUM."""
    start_time = time.time()
    inp = params if isinstance(params, RegimeDetectorInput) else RegimeDetectorInput.model_validate(params)
    inp_dict = inp.model_dump()
    is_error = False
    error_msg = None
    output: Optional[RegimeDetectorOutput] = None

    try:
        hist = [float(x) for x in inp.demand_history]
        n = len(hist)
        sw = inp.short_window
        lw = min(inp.long_window, n)

        if n < sw:
            m = float(np.mean(hist)) if n > 0 else 100.0
            output = RegimeDetectorOutput(
                regime="stable",
                confidence=1.0,
                short_mean=round(m, 2),
                long_mean=round(m, 2),
                ratio=1.0,
                details={"status": "insufficient_history", "samples": n},
            )
        else:
            short_mean = float(np.mean(hist[-sw:]))
            long_mean = float(np.mean(hist[-lw:]))
            long_safe = max(long_mean, 1e-4)
            ratio = short_mean / long_safe

            std_w = max(float(np.std(hist[-lw:])), 1.0)
            cusum_z = abs(short_mean - long_mean) / (std_w / math.sqrt(sw))

            if ratio >= inp.shock_threshold or ratio <= (1.0 / inp.shock_threshold):
                regime = "shock"
                confidence = min(1.0, max(0.5, abs(ratio - 1.0) / (inp.shock_threshold - 1.0)))
            elif ratio >= inp.trend_threshold or ratio <= (1.0 / inp.trend_threshold):
                regime = "trend"
                confidence = min(1.0, max(0.5, abs(ratio - 1.0) / (inp.trend_threshold - 1.0)))
            else:
                regime = "stable"
                confidence = min(1.0, max(0.5, 1.0 - abs(ratio - 1.0)))

            output = RegimeDetectorOutput(
                regime=regime,
                confidence=round(confidence, 4),
                short_mean=round(short_mean, 2),
                long_mean=round(long_mean, 2),
                ratio=round(ratio, 4),
                details={
                    "cusum_z_score": round(cusum_z, 2),
                    "history_length": n,
                    "short_window": sw,
                    "long_window": lw,
                },
            )
    except Exception as e:
        is_error = True
        error_msg = str(e)
        logger.exception("Error in regime_detector: %s", e)
        raise e
    finally:
        elapsed_ms = (time.time() - start_time) * 1000.0
        out_dict = output.model_dump() if output else {}
        log_tool_to_duckdb(
            tool_name="regime_detector",
            input_args=inp_dict,
            output_result=out_dict,
            duration_ms=elapsed_ms,
            is_error=is_error,
            error_message=error_msg,
            run_id=run_id,
            step_id=step_id,
            agent_name=agent_name,
            db_path=db_path,
            expected_tool=expected_tool or "regime_detector",
            is_correct_tool=is_correct_tool,
            is_parameter_valid=is_parameter_valid,
        )

    return output

