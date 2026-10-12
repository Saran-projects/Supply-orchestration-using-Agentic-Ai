"""Base-Stock (Order-Up-To S) Inventory Policy Baseline.

Orders up to a target base-stock level S at every step:
    IP = inventory + pipeline - backlog
    order = max(0, S - IP)
Where S = L * mu_D + z * sigma_LTD.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np

from sca.data.scenarios import Scenario
from sca.sim.environment import ECHELONS, EchelonState
from sca.systems.base import OrderingSystem, register_system

logger = logging.getLogger(__name__)


@register_system(["baseline_base_stock", "BaseStockPolicy", "base_stock"])
class BaseStockPolicy(OrderingSystem):
    """Pure base-stock order-up-to replenishment policy for all 4 echelons."""

    def __init__(
        self,
        name: str = "baseline_base_stock",
        config: Optional[Dict[str, Any]] = None,
        z: float = 2.0,
    ):
        super().__init__(name=name, config=config)
        self.z: float = float(self.config.get("z", self.config.get("safety_stock_factor", z)))
        self.S: float = 0.0
        self.S_levels: Dict[str, float] = {}
        self.outstanding_orders: Dict[str, float] = {e: 0.0 for e in ECHELONS}
        self._init_defaults()

    def _init_defaults(self):
        """Set default parameters before scenario calibration."""
        default_S = float(self.config.get("base_stock_level", self.config.get("order_up_to", 686.56)))
        self.S = default_S
        self.S_levels = {e: default_S for e in ECHELONS}
        self.parameters = {
            "policy": "base_stock",
            "z": self.z,
            "S": self.S,
            "S_levels": dict(self.S_levels),
        }

    def calibrate(self, scenario: Scenario, config: Optional[Dict[str, Any]] = None) -> None:
        """Derive target base-stock level S from scenario training history."""
        cfg = config or self.config
        if "z" in cfg:
            self.z = float(cfg["z"])
        elif "safety_stock_factor" in cfg:
            self.z = float(cfg["safety_stock_factor"])

        train_data = scenario.train_demand
        if len(train_data) == 0:
            mu_d = 100.0
            sigma_d = 20.0
        else:
            mu_d = float(np.mean(train_data))
            sigma_d = float(np.std(train_data))

        if hasattr(scenario, "lead_time_sampler") and scenario.lead_time_sampler:
            L = float(scenario.lead_time_sampler.mean)
            sigma_L = float(scenario.lead_time_sampler.std)
        else:
            L = 3.52
            sigma_L = 1.62

        var_ltd = (L * (sigma_d**2)) + ((mu_d**2) * (sigma_L**2))
        sigma_ltd = math.sqrt(max(1.0, var_ltd))
        safety_stock = self.z * sigma_ltd
        self.S = float((L * mu_d) + safety_stock)

        # Allow echelon overrides if specified
        for e in ECHELONS:
            override = cfg.get(f"{e}_S", cfg.get("base_stock_level", None))
            self.S_levels[e] = float(override if override is not None else self.S)

        self.parameters = {
            "policy": "base_stock",
            "z": self.z,
            "mu_d": round(mu_d, 2),
            "sigma_d": round(sigma_d, 2),
            "lead_time_mean": round(L, 2),
            "lead_time_std": round(sigma_L, 2),
            "sigma_ltd": round(sigma_ltd, 2),
            "safety_stock": round(safety_stock, 2),
            "S": round(self.S, 2),
            "S_levels": {k: round(v, 2) for k, v in self.S_levels.items()},
        }
        self.calibrated = True
        logger.info(
            "Calibrated %s on '%s': z=%.2f, mu_d=%.1f, L=%.2f, S=%.2f",
            self.name,
            scenario.name,
            self.z,
            mu_d,
            L,
            self.S,
        )

    def reset(self, scenario: Optional[Scenario] = None) -> None:
        """Reset internal system state at the start of a simulation run."""
        self.outstanding_orders = {e: 0.0 for e in ECHELONS}
        if scenario is not None and not self.calibrated:
            self.calibrate(scenario)

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Decide orders for all echelons using the base-stock rule."""
        if not hasattr(self, "outstanding_orders"):
            self.outstanding_orders = {e: 0.0 for e in ECHELONS}

        orders = {}
        for e in ECHELONS:
            st = states[e]
            self.outstanding_orders[e] = max(0.0, self.outstanding_orders[e] - st.incoming_shipments)
            on_order = max(st.pipeline, self.outstanding_orders[e])
            ip = st.inventory + on_order - st.backlog
            target_S = self.S_levels.get(e, self.S)
            order_qty = max(0.0, target_S - ip)
            orders[e] = float(order_qty)
            self.outstanding_orders[e] += order_qty

        return orders
