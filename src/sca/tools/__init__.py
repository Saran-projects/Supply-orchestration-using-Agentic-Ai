"""Executable tools for supply chain agents: forecasting, EOQ, (s, S) ordering, and validation."""

from sca.tools.crew import (
    ALL_CREWAI_TOOLS,
    EoqQuantityTool,
    ForecastDemandTool,
    SmoothedOrderUpToTool,
    SsPolicyOrderTool,
    ValidateOrderTool,
    eoq_quantity_tool,
    forecast_demand_tool,
    smoothed_order_up_to_tool,
    ss_policy_order_tool,
    validate_order_tool,
)
from sca.tools.pure import (
    eoq_quantity,
    forecast_demand,
    log_tool_to_duckdb,
    smoothed_order_up_to,
    ss_policy_order,
    validate_order,
)
from sca.tools.schemas import (
    EoqQuantityInput,
    EoqQuantityOutput,
    ForecastDemandInput,
    ForecastDemandOutput,
    SmoothedOrderUpToInput,
    SmoothedOrderUpToOutput,
    SsPolicyOrderInput,
    SsPolicyOrderOutput,
    ValidateOrderInput,
    ValidateOrderOutput,
)

__all__ = [
    # Schemas
    "ForecastDemandInput",
    "ForecastDemandOutput",
    "EoqQuantityInput",
    "EoqQuantityOutput",
    "SsPolicyOrderInput",
    "SsPolicyOrderOutput",
    "ValidateOrderInput",
    "ValidateOrderOutput",
    "SmoothedOrderUpToInput",
    "SmoothedOrderUpToOutput",
    # Pure Functions
    "forecast_demand",
    "eoq_quantity",
    "ss_policy_order",
    "validate_order",
    "smoothed_order_up_to",
    "log_tool_to_duckdb",
    # CrewAI Tools
    "ForecastDemandTool",
    "EoqQuantityTool",
    "SsPolicyOrderTool",
    "ValidateOrderTool",
    "SmoothedOrderUpToTool",
    "forecast_demand_tool",
    "eoq_quantity_tool",
    "ss_policy_order_tool",
    "validate_order_tool",
    "smoothed_order_up_to_tool",
    "ALL_CREWAI_TOOLS",
]
