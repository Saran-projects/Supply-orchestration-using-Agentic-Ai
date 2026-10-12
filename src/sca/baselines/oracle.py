"""Clairvoyant / Oracle Inventory Policy Baseline.

Acts as a theoretical upper bound benchmark by looking ahead directly at the
known demand path to order exactly what will be consumed after lead time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from sca.data.scenarios import Scenario
from sca.sim.environment import ECHELONS, EchelonState
from sca.systems.base import OrderingSystem, register_system

logger = logging.getLogger(__name__)


@register_system(["baseline_oracle", "OraclePolicy", "oracle", "clairvoyant"])
class OraclePolicy(OrderingSystem):
    """Clairvoyant benchmark policy with perfect future demand visibility."""

    def __init__(
        self,
        name: str = "baseline_oracle",
        config: Optional[Dict[str, Any]] = None,
        lead_time_offset: int = 4,
    ):
        super().__init__(name=name, config=config)
        self.lt_offset: int = int(self.config.get("lead_time_offset", lead_time_offset))
        self.future_demand: List[float] = []
        self.parameters = {
            "policy": "oracle",
            "lead_time_offset": self.lt_offset,
        }

    def calibrate(self, scenario: Scenario, config: Optional[Dict[str, Any]] = None) -> None:
        """Cache future demand from scenario."""
        cfg = config or self.config
        if "lead_time_offset" in cfg:
            self.lt_offset = int(cfg["lead_time_offset"])
        if hasattr(scenario, "demand"):
            self.future_demand = [float(x) for x in scenario.demand]
        self.calibrated = True

    def reset(self, scenario: Optional[Scenario] = None) -> None:
        """Reset internal system state at the start of a simulation run."""
        if scenario is not None and hasattr(scenario, "demand"):
            self.future_demand = [float(x) for x in scenario.demand]
        self.parameters["lead_time_offset"] = self.lt_offset

    def set_demand(self, demand_array: Union[List[float], np.ndarray]) -> None:
        """Directly inject realization demand array for seed-specific simulation."""
        self.future_demand = [float(x) for x in demand_array]

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Place orders anticipating future demand after lead time."""
        step = states["retailer"].step if "retailer" in states else 0
        target_idx = step + self.lt_offset
        if target_idx < len(self.future_demand):
            order_qty = float(self.future_demand[target_idx])
        elif len(self.future_demand) > 0:
            order_qty = float(self.future_demand[-1])
        else:
            order_qty = 100.0

        return {e: order_qty for e in ECHELONS}
