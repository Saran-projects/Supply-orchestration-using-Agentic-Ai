"""Constant Order Policy Baseline (System 0).

Orders a constant quantity equal to the training-period mean demand at every step,
completely ignoring current inventory, backlog, or pipeline state inputs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from sca.data.scenarios import Scenario
from sca.sim.environment import ECHELONS, EchelonState
from sca.systems.base import OrderingSystem, register_system

logger = logging.getLogger(__name__)


@register_system([
    "baseline_constant",
    "baseline_0",
    "system_0",
    "system0",
    "constant",
    "s0",
])
class ConstantOrderPolicy(OrderingSystem):
    """System 0: Constant order equal to training-period mean demand."""

    def __init__(self, name: str = "baseline_constant", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.mean_demand: float = 100.0
        self.parameters = {
            "policy": "constant",
            "mean_demand": self.mean_demand,
        }

    def calibrate(self, scenario: Scenario, config: Optional[Dict[str, Any]] = None) -> None:
        """Derive constant order quantity from training history mean demand."""
        train_data = scenario.train_demand
        if len(train_data) == 0:
            self.mean_demand = 100.0
        else:
            self.mean_demand = float(np.mean(train_data))

        self.parameters = {
            "policy": "constant",
            "mean_demand": round(self.mean_demand, 2),
        }
        self.calibrated = True
        logger.info("Calibrated %s on '%s': constant_order=%.2f", self.name, scenario.name, self.mean_demand)

    def reset(self, scenario: Optional[Scenario] = None) -> None:
        """Reset internal system state at start of simulation run."""
        pass

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Place constant orders equal to training-period mean demand for all echelons."""
        return {e: float(self.mean_demand) for e in ECHELONS}
