"""(s, S) Inventory Policy Baseline.

Orders up to S whenever the inventory position (inventory + pipeline - backlog)
falls below the reorder point s:
    IP = inventory + pipeline - backlog
    order = (S - IP) if IP < s else 0
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


@register_system(["baseline_s_S", "SsPolicy", "s_S", "ss"])
class SsPolicy(OrderingSystem):
    """(s, S) min-max inventory replenishment policy for all 4 echelons."""

    def __init__(self, name: str = "baseline_s_S", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.s_levels: Dict[str, float] = {}
        self.S_levels: Dict[str, float] = {}
        self._init_defaults()

    def _init_defaults(self):
        """Set default fallback parameters if not yet calibrated."""
        for e in ECHELONS:
            self.s_levels[e] = float(self.config.get("reorder_point", 150.0))
            self.S_levels[e] = float(self.config.get("order_up_to", 350.0))
        self.parameters = {
            "policy": "s_S",
            "s_levels": dict(self.s_levels),
            "S_levels": dict(self.S_levels),
        }

    def calibrate(self, scenario: Scenario, config: Optional[Dict[str, Any]] = None) -> None:
        """Derive (s, S) parameters from scenario training history and cost model.

        Calculates:
          mu_d, sigma_d = mean and std of scenario.train_demand
          L, sigma_L    = mean and std of lead-time distribution
          sigma_LTD     = sqrt(L * sigma_d^2 + mu_d^2 * sigma_L^2)
          SS (safety)   = z * sigma_LTD
          s (reorder)   = L * mu_d + SS
          EOQ (batch)   = sqrt(2 * mu_d * K / h)
          S (order-up)  = s + EOQ
        """
        cfg = config or self.config

        train_data = scenario.train_demand
        if len(train_data) == 0:
            mu_d = 100.0
            sigma_d = 20.0
        else:
            mu_d = float(np.mean(train_data))
            sigma_d = float(np.std(train_data))

        # Lead time stats from sampler
        L = float(scenario.lead_time_sampler.mean)
        sigma_L = float(scenario.lead_time_sampler.std)

        # Cost parameters
        K = float(cfg.get("order_placement_fixed_cost", 100.0))
        h = float(cfg.get("holding_cost_per_unit", 0.50))
        z = float(cfg.get("safety_stock_factor", 1.65))  # Default 95% service level

        # Lead-time demand variance & safety stock
        var_ltd = (L * (sigma_d ** 2)) + ((mu_d ** 2) * (sigma_L ** 2))
        sigma_ltd = math.sqrt(max(1.0, var_ltd))
        safety_stock = z * sigma_ltd

        # Reorder point s and order-up-to S with tuning factors
        s_factor = float(cfg.get("s_factor", cfg.get("s_multiplier", 1.0)))
        S_factor = float(cfg.get("S_factor", cfg.get("S_multiplier", 1.0)))

        s_calc = ((L * mu_d) + safety_stock) * s_factor
        eoq = math.sqrt((2.0 * max(1.0, mu_d) * K) / max(0.01, h))
        S_calc = s_calc + (eoq * S_factor)

        # Calibrate per echelon (with optional config overrides)
        for e in ECHELONS:
            s_override = cfg.get(f"{e}_s", cfg.get("s", None))
            S_override = cfg.get(f"{e}_S", cfg.get("S", None))

            self.s_levels[e] = float(s_override if s_override is not None else s_calc)
            self.S_levels[e] = float(S_override if S_override is not None else S_calc)

        self.parameters = {
            "policy": "s_S",
            "mu_d": round(mu_d, 2),
            "sigma_d": round(sigma_d, 2),
            "lead_time_mean": round(L, 2),
            "lead_time_std": round(sigma_L, 2),
            "safety_stock": round(safety_stock, 2),
            "eoq": round(eoq, 2),
            "s_factor": s_factor,
            "S_factor": S_factor,
            "s_levels": {k: round(v, 2) for k, v in self.s_levels.items()},
            "S_levels": {k: round(v, 2) for k, v in self.S_levels.items()},
        }
        self.calibrated = True
        logger.info(
            "Calibrated %s on '%s': mu=%.1f, sigma=%.1f, s=%.1f (x%.2f), S=%.1f (x%.2f), EOQ=%.1f",
            self.name,
            scenario.name,
            mu_d,
            sigma_d,
            s_calc,
            s_factor,
            S_calc,
            S_factor,
            eoq,
        )

    def reset(self, scenario: Optional[Scenario] = None) -> None:
        """Reset internal system state at the start of a simulation run."""
        self.outstanding_orders = {e: 0.0 for e in ECHELONS}

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Decide orders for all echelons using the (s, S) rule."""
        if not hasattr(self, "outstanding_orders"):
            self.outstanding_orders = {e: 0.0 for e in ECHELONS}

        orders = {}
        for e in ECHELONS:
            st = states[e]
            # Account for newly received incoming shipments
            self.outstanding_orders[e] = max(0.0, self.outstanding_orders[e] - st.incoming_shipments)
            on_order = max(st.pipeline, self.outstanding_orders[e])

            # Inventory position: on-hand + on-order (pipeline) - backlog
            ip = st.inventory + on_order - st.backlog
            s = self.s_levels[e]
            S = self.S_levels[e]

            if ip < s:
                order_qty = max(0.0, S - ip)
            else:
                order_qty = 0.0

            orders[e] = order_qty
            self.outstanding_orders[e] += order_qty

        return orders
