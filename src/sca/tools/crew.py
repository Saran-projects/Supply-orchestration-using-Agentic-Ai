"""CrewAI tool wrappers for supply chain decision-making."""

from __future__ import annotations

from typing import Any, List, Type

from crewai.tools import BaseTool
from pydantic import BaseModel

from sca.tools.pure import (
    eoq_quantity,
    forecast_demand,
    smoothed_order_up_to,
    ss_policy_order,
    validate_order,
)
from sca.tools.schemas import (
    EoqQuantityInput,
    ForecastDemandInput,
    SmoothedOrderUpToInput,
    SsPolicyOrderInput,
    ValidateOrderInput,
)


class ForecastDemandTool(BaseTool):
    """CrewAI tool for demand forecasting using time-series lag models."""

    name: str = "forecast_demand"
    description: str = (
        "Forecast future demand using recent historical demand series. "
        "Supports linear regression and gradient boosting models."
    )
    args_schema: Type[BaseModel] = ForecastDemandInput

    def _run(self, **kwargs: Any) -> str:
        inp = ForecastDemandInput.model_validate(kwargs)
        result = forecast_demand(inp)
        return result.model_dump_json()


class EoqQuantityTool(BaseTool):
    """CrewAI tool for Economic Order Quantity (EOQ) batch sizing."""

    name: str = "eoq_quantity"
    description: str = (
        "Calculate optimal replenishment batch size (EOQ), order frequency, "
        "and holding/ordering cost breakdown using Wilson's formula."
    )
    args_schema: Type[BaseModel] = EoqQuantityInput

    def _run(self, **kwargs: Any) -> str:
        inp = EoqQuantityInput.model_validate(kwargs)
        result = eoq_quantity(inp)
        return result.model_dump_json()


class SsPolicyOrderTool(BaseTool):
    """CrewAI tool for (s, S) min-max inventory replenishment ordering."""

    name: str = "ss_policy_order"
    description: str = (
        "Calculate replenishment order under (s, S) policy: computes inventory position "
        "(on-hand + pipeline - backlog) and orders up to S when position falls below s."
    )
    args_schema: Type[BaseModel] = SsPolicyOrderInput

    def _run(self, **kwargs: Any) -> str:
        inp = SsPolicyOrderInput.model_validate(kwargs)
        result = ss_policy_order(inp)
        return result.model_dump_json()


class ValidateOrderTool(BaseTool):
    """CrewAI tool for validating proposed orders against bounds and capacity limits."""

    name: str = "validate_order"
    description: str = (
        "Validate and constrain proposed order quantities against production capacity limits, "
        "inventory position reasonableness, and non-negative minimums."
    )
    args_schema: Type[BaseModel] = ValidateOrderInput

    def _run(self, **kwargs: Any) -> str:
        inp = ValidateOrderInput.model_validate(kwargs)
        result = validate_order(inp)
        return result.model_dump_json()


class SmoothedOrderUpToTool(BaseTool):
    """CrewAI tool for smoothed order-up-to policy calculation."""

    name: str = "smoothed_order_up_to"
    description: str = (
        "Calculate replenishment order using exponentially smoothed demand forecast, "
        "order-up-to target with safety stock, and order smoothing."
    )
    args_schema: Type[BaseModel] = SmoothedOrderUpToInput

    def _run(self, **kwargs: Any) -> str:
        inp = SmoothedOrderUpToInput.model_validate(kwargs)
        result = smoothed_order_up_to(inp)
        return result.model_dump_json()


# Singletons ready for agent instantiation
forecast_demand_tool = ForecastDemandTool()
eoq_quantity_tool = EoqQuantityTool()
ss_policy_order_tool = SsPolicyOrderTool()
validate_order_tool = ValidateOrderTool()
smoothed_order_up_to_tool = SmoothedOrderUpToTool()

ALL_CREWAI_TOOLS: List[BaseTool] = [
    forecast_demand_tool,
    eoq_quantity_tool,
    ss_policy_order_tool,
    validate_order_tool,
    smoothed_order_up_to_tool,
]
