"""Pydantic input and output schemas for supply chain tools."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# 1. Forecast Demand
# ---------------------------------------------------------------------------
class ForecastDemandInput(BaseModel):
    """Input parameters for demand forecasting."""

    recent_history: List[float] = Field(
        ...,
        min_length=1,
        description="Historical demand observations in chronological order (e.g. last 14-30 steps).",
    )
    horizon: int = Field(
        default=1,
        ge=1,
        description="Number of future time steps to forecast.",
    )
    model_type: str = Field(
        default="linear",
        description="Model architecture to use: 'linear' (Linear Regression) or 'gradient_boosting'.",
    )


class ForecastDemandOutput(BaseModel):
    """Forecasted demand outputs."""

    point_forecast: float = Field(
        description="Immediate next-step (t+1) forecasted demand."
    )
    forecast_series: List[float] = Field(
        description="Forecasted demand values over the requested horizon."
    )
    mean_forecast: float = Field(
        description="Mean forecasted demand over the requested horizon."
    )
    horizon: int = Field(
        description="Requested forecast horizon steps."
    )
    model_used: str = Field(
        description="Forecasting model used ('linear' or 'gradient_boosting')."
    )


# ---------------------------------------------------------------------------
# 2. Economic Order Quantity (EOQ)
# ---------------------------------------------------------------------------
class EoqQuantityInput(BaseModel):
    """Input parameters for Economic Order Quantity (EOQ) calculation."""

    mean_demand: float = Field(
        ...,
        gt=0.0,
        description="Mean demand rate (units per period/day).",
    )
    order_fixed_cost: float = Field(
        default=100.0,
        ge=0.0,
        description="Fixed order placement setup cost ($/order).",
    )
    holding_cost_per_unit: float = Field(
        default=0.50,
        gt=0.0,
        description="Holding cost per unit per period ($/unit/day).",
    )


class EoqQuantityOutput(BaseModel):
    """Calculated Economic Order Quantity and cost metrics."""

    eoq: float = Field(
        description="Optimal batch order quantity (Wilson's EOQ formula)."
    )
    order_frequency_days: float = Field(
        description="Average interval between orders in days: EOQ / mean_demand."
    )
    annual_orders: float = Field(
        description="Estimated number of replenishment orders per 365-day year."
    )
    cycle_holding_cost: float = Field(
        description="Cycle inventory holding cost per replenishment cycle: (EOQ / 2) * holding_cost."
    )
    ordering_cost: float = Field(
        description="Fixed ordering cost per replenishment cycle."
    )
    total_daily_cost: float = Field(
        description="Total daily inventory cost: (mean_demand * K / EOQ) + (EOQ * h / 2)."
    )


# ---------------------------------------------------------------------------
# 3. (s, S) Policy Order
# ---------------------------------------------------------------------------
class SsPolicyOrderInput(BaseModel):
    """Input parameters for (s, S) min-max inventory replenishment order decision."""

    inventory: float = Field(
        ...,
        ge=0.0,
        description="Current on-hand physical inventory.",
    )
    pipeline: float = Field(
        default=0.0,
        ge=0.0,
        description="Units currently on order in transit pipeline.",
    )
    backlog: float = Field(
        default=0.0,
        ge=0.0,
        description="Current unfulfilled customer/downstream backlog.",
    )
    reorder_point: float = Field(
        ...,
        ge=0.0,
        description="Reorder point threshold s. Order is placed when IP < s.",
    )
    order_up_to: float = Field(
        ...,
        ge=0.0,
        description="Order-up-to target level S. When triggered, order = S - IP.",
    )

    @model_validator(mode="after")
    def validate_levels(self) -> "SsPolicyOrderInput":
        if self.order_up_to < self.reorder_point:
            raise ValueError(
                f"order_up_to (S={self.order_up_to}) must be greater than or equal to reorder_point (s={self.reorder_point})."
            )
        return self


class SsPolicyOrderOutput(BaseModel):
    """Replenishment order decision under (s, S) policy."""

    order_quantity: float = Field(
        description="Units to order upstream (S - IP if IP < s else 0.0)."
    )
    inventory_position: float = Field(
        description="Current inventory position IP = inventory + pipeline - backlog."
    )
    reorder_triggered: bool = Field(
        description="True if inventory position fell below reorder point s."
    )
    reorder_point: float = Field(
        description="Configured reorder point s."
    )
    order_up_to: float = Field(
        description="Configured order-up-to level S."
    )


# ---------------------------------------------------------------------------
# 4. Validate Order
# ---------------------------------------------------------------------------
class ValidateOrderInput(BaseModel):
    """Input parameters to validate and constrain proposed orders."""

    proposed_order: float = Field(
        ...,
        description="Proposed order quantity to validate.",
    )
    inventory_position: float = Field(
        default=0.0,
        description="Current inventory position (inventory + pipeline - backlog).",
    )
    capacity_limit: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Production or throughput capacity limit (e.g. manufacturer capacity 300.0).",
    )
    max_order: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum allowable order ceiling.",
    )
    min_order: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum allowable order quantity (default 0.0).",
    )
    echelon: Optional[str] = Field(
        default=None,
        description="Echelon role ('retailer', 'wholesaler', 'distributor', 'manufacturer').",
    )
    last_order: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Order placed at previous step (for rate limiting).",
    )
    forecast_demand: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Forecasted demand for upcoming step (for forecast-relative cap).",
    )
    rate_limit_delta: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum allowable fractional order change relative to last_order (rate limiting delta).",
    )
    forecast_cap_multiple: Optional[float] = Field(
        default=None,
        ge=1.0,
        description="Maximum multiple of forecasted demand allowed (forecast-relative cap).",
    )


class ValidateOrderOutput(BaseModel):
    """Result of order validation and boundary clipping."""

    validated_order: float = Field(
        description="Final constrained order quantity after range checks."
    )
    original_order: float = Field(
        description="Original unadjusted proposed order quantity."
    )
    is_valid: bool = Field(
        description="True if original proposed order complied with all constraints without alteration."
    )
    adjustments_made: bool = Field(
        description="True if the proposed order quantity was modified/clamped."
    )
    violations: List[str] = Field(
        default_factory=list,
        description="List of detected boundary constraint violations or adjustment reasons.",
    )
    inventory_position: float = Field(
        description="Inventory position at time of order validation."
    )
    interventions: List[str] = Field(
        default_factory=list,
        description="List of clipping interventions applied (rate limiting, forecast cap, capacity limit).",
    )


# ---------------------------------------------------------------------------
# 5. Smoothed Order-Up-To
# ---------------------------------------------------------------------------
class SmoothedOrderUpToInput(BaseModel):
    """Input parameters for smoothed order-up-to replenishment policy."""

    current_demand: float = Field(
        ...,
        ge=0.0,
        description="Latest observed demand for this echelon.",
    )
    inventory: float = Field(
        ...,
        ge=0.0,
        description="Current on-hand inventory.",
    )
    pipeline: float = Field(
        default=0.0,
        ge=0.0,
        description="Units in transit pipeline.",
    )
    backlog: float = Field(
        default=0.0,
        ge=0.0,
        description="Current backlog.",
    )
    last_order: float = Field(
        default=0.0,
        ge=0.0,
        description="Order placed at previous step (for order smoothing).",
    )
    lead_time_mean: float = Field(
        default=2.0,
        ge=0.0,
        description="Mean lead time in periods.",
    )
    lead_time_std: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of lead time.",
    )
    demand_std: float = Field(
        default=20.0,
        ge=0.0,
        description="Standard deviation of demand.",
    )
    safety_stock_factor: float = Field(
        default=1.65,
        ge=0.0,
        description="Safety stock z-factor (1.65 for 95% service level).",
    )
    z: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Explicit safety stock factor z (overrides safety_stock_factor if specified).",
    )
    include_lead_time_variance: bool = Field(
        default=True,
        description="Whether to include lead-time variance in safety stock calculation.",
    )
    previous_smoothed_demand: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Previous period exponentially smoothed demand forecast.",
    )
    alpha: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Exponential demand forecast smoothing factor alpha in [0, 1].",
    )
    beta: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Order smoothing factor beta in [0, 1]. If None, defaults to alpha.",
    )


class SmoothedOrderUpToOutput(BaseModel):
    """Output results from smoothed order-up-to policy."""

    order_quantity: float = Field(
        description="Final smoothed order quantity to place upstream."
    )
    raw_order_quantity: float = Field(
        description="Unsmoothed order quantity max(0, S - IP)."
    )
    order_up_to_target: float = Field(
        description="Order-up-to target level S = L * smoothed_demand + SS."
    )
    smoothed_demand: float = Field(
        description="Exponentially smoothed demand forecast."
    )
    safety_stock: float = Field(
        description="Calculated safety stock buffer."
    )
    inventory_position: float = Field(
        description="Current inventory position IP = inventory + pipeline - backlog."
    )
    alpha: float = Field(
        description="Forecast smoothing parameter alpha utilized."
    )
    beta: float = Field(
        description="Order smoothing parameter beta utilized."
    )
    z: float = Field(
        description="Safety factor z utilized."
    )
    include_lead_time_variance: bool = Field(
        description="Whether lead-time variance was included in safety stock."
    )


# ---------------------------------------------------------------------------
# 6. Smoothed Forecast
# ---------------------------------------------------------------------------
class SmoothedForecastInput(BaseModel):
    """Input parameters for exponential smoothing demand forecast."""

    current_demand: float = Field(
        ...,
        ge=0.0,
        description="Latest realized demand observation.",
    )
    previous_smoothed_demand: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Previous smoothed demand forecast.",
    )
    alpha: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Exponential smoothing weight alpha in [0, 1].",
    )


class SmoothedForecastOutput(BaseModel):
    """Output from exponential smoothing forecast."""

    smoothed_demand: float = Field(
        description="Exponentially smoothed demand estimate."
    )
    current_demand: float = Field(
        description="Observed current demand."
    )
    alpha: float = Field(
        description="Smoothing factor alpha used."
    )


# ---------------------------------------------------------------------------
# 7. Order-Up-To Target
# ---------------------------------------------------------------------------
class OrderUpToTargetInput(BaseModel):
    """Input parameters for order-up-to target S calculation."""

    forecast_demand: float = Field(
        ...,
        ge=0.0,
        description="Forecast demand rate (e.g. smoothed demand).",
    )
    lead_time_mean: float = Field(
        ...,
        ge=0.0,
        description="Mean lead time in periods.",
    )
    lead_time_std: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of lead time.",
    )
    demand_std: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of demand.",
    )
    z: float = Field(
        default=2.0,
        ge=0.0,
        description="Safety factor z.",
    )
    include_lead_time_variance: bool = Field(
        default=True,
        description="Whether to include lead-time variance in safety stock calculation.",
    )


class OrderUpToTargetOutput(BaseModel):
    """Calculated base-stock target level and safety buffer."""

    order_up_to_target: float = Field(
        description="Calculated base-stock target level S = L * d_hat + SS."
    )
    safety_stock: float = Field(
        description="Calculated safety stock buffer SS."
    )
    sigma_ltd: float = Field(
        description="Standard deviation of lead-time demand."
    )
    expected_lead_time_demand: float = Field(
        description="Expected lead-time demand L * forecast_demand."
    )


# ---------------------------------------------------------------------------
# 8. Smoothed Order
# ---------------------------------------------------------------------------
class SmoothedOrderInput(BaseModel):
    """Input parameters for order smoothing with beta."""

    order_up_to_target: float = Field(
        ...,
        ge=0.0,
        description="Target inventory position level S.",
    )
    inventory_position: float = Field(
        ...,
        description="Current inventory position IP = inventory + pipeline - backlog.",
    )
    last_order: float = Field(
        ...,
        ge=0.0,
        description="Order placed at previous step.",
    )
    beta: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Order smoothing factor beta in [0, 1].",
    )


class SmoothedOrderOutput(BaseModel):
    """Output from smoothed ordering rule."""

    smoothed_order: float = Field(
        description="Final smoothed order quantity beta * raw + (1 - beta) * last_order."
    )
    raw_order: float = Field(
        description="Raw unsmoothed order deficit max(0, S - IP)."
    )
    beta: float = Field(
        description="Order smoothing factor beta used."
    )
    inventory_position: float = Field(
        description="Inventory position used."
    )


# ---------------------------------------------------------------------------
# 9. Regime Detector
# ---------------------------------------------------------------------------
class RegimeDetectorInput(BaseModel):
    """Input parameters for demand regime detection."""

    demand_history: List[float] = Field(
        ...,
        min_length=1,
        description="Historical observed demand series in chronological order.",
    )
    short_window: int = Field(
        default=5,
        gt=0,
        description="Short-term moving average window length.",
    )
    long_window: int = Field(
        default=15,
        gt=0,
        description="Long-term moving average window length.",
    )
    shock_threshold: float = Field(
        default=1.30,
        ge=1.0,
        description="Short/long mean ratio threshold to trigger shock regime.",
    )
    trend_threshold: float = Field(
        default=1.15,
        ge=1.0,
        description="Short/long mean ratio threshold to trigger trend regime.",
    )


class RegimeDetectorOutput(BaseModel):
    """Classification output from demand regime detector."""

    regime: str = Field(
        description="Detected regime: 'stable' | 'trend' | 'shock'."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score in [0.0, 1.0].",
    )
    short_mean: float = Field(
        description="Short-term window mean demand."
    )
    long_mean: float = Field(
        description="Long-term window mean demand."
    )
    ratio: float = Field(
        description="Ratio of short-term mean to long-term mean."
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed diagnostics and CUSUM statistics.",
    )

