"""Economic Order Quantity (EOQ / ROP) Inventory Policy Baseline.

Orders a batch quantity Q (derived via Wilson's EOQ formula) whenever the inventory
position (inventory + pipeline - backlog) falls below the reorder point (ROP):
    IP = inventory + pipeline - backlog
    order = batches * EOQ if IP < ROP else 0
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


@register_system(["baseline_eoq", "EoqPolicy", "eoq"])
class EoqPolicy(OrderingSystem):
    """(ROP, EOQ) inventory replenishment policy for all 4 echelons."""

    def __init__(self, name: str = "baseline_eoq", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.rop_levels: Dict[str, float] = {}
        self.eoq_levels: Dict[str, float] = {}
        self.allow_multi_batch: bool = True
        self._init_defaults()

    def _init_defaults(self):
        """Set default fallback parameters before scenario calibration."""
        for e in ECHELONS:
            self.rop_levels[e] = float(self.config.get("reorder_point", 150.0))
            self.eoq_levels[e] = float(self.config.get("eoq_quantity", 200.0))
        self.allow_multi_batch = bool(self.config.get("allow_multi_batch", True))
        self.parameters = {
            "policy": "EOQ",
            "rop_levels": dict(self.rop_levels),
            "eoq_levels": dict(self.eoq_levels),
            "allow_multi_batch": self.allow_multi_batch,
        }

    def calibrate(self, scenario: Scenario, config: Optional[Dict[str, Any]] = None) -> None:
        """Derive (ROP, EOQ) parameters from scenario training history and cost model.

        Calculates:
          mu_d, sigma_d = mean and std of scenario.train_demand
          L, sigma_L    = mean and std of lead-time distribution
          sigma_LTD     = sqrt(L * sigma_d^2 + mu_d^2 * sigma_L^2)
          SS (safety)   = z * sigma_LTD
          ROP (reorder) = L * mu_d + SS
          EOQ (batch)   = sqrt(2 * mu_d * K / h)
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
        var_ltd = (L * (sigma_d**2)) + ((mu_d**2) * (sigma_L**2))
        sigma_ltd = math.sqrt(max(1.0, var_ltd))
        safety_stock = z * sigma_ltd

        # Reorder point ROP and batch order quantity EOQ with multipliers
        rop_mult = float(cfg.get("reorder_point_factor", cfg.get("rop_multiplier", 1.0)))
        eoq_mult = float(cfg.get("order_quantity_multiplier", cfg.get("eoq_multiplier", 1.0)))

        rop_calc = ((L * mu_d) + safety_stock) * rop_mult
        eoq_calc = math.sqrt((2.0 * max(1.0, mu_d) * K) / max(0.01, h)) * eoq_mult

        self.allow_multi_batch = bool(cfg.get("allow_multi_batch", True))

        # Calibrate per echelon (with optional config overrides)
        for e in ECHELONS:
            rop_override = cfg.get(f"{e}_rop", cfg.get(f"{e}_s", cfg.get("reorder_point", None)))
            eoq_override = cfg.get(f"{e}_eoq", cfg.get(f"{e}_Q", cfg.get("eoq_quantity", None)))

            self.rop_levels[e] = float(rop_override if rop_override is not None else rop_calc)
            self.eoq_levels[e] = float(eoq_override if eoq_override is not None else eoq_calc)

        self.parameters = {
            "policy": "EOQ",
            "mu_d": round(mu_d, 2),
            "sigma_d": round(sigma_d, 2),
            "lead_time_mean": round(L, 2),
            "lead_time_std": round(sigma_L, 2),
            "safety_stock": round(safety_stock, 2),
            "eoq": round(eoq_calc, 2),
            "reorder_point_factor": rop_mult,
            "order_quantity_multiplier": eoq_mult,
            "rop_levels": {k: round(v, 2) for k, v in self.rop_levels.items()},
            "eoq_levels": {k: round(v, 2) for k, v in self.eoq_levels.items()},
            "allow_multi_batch": self.allow_multi_batch,
        }
        self.calibrated = True
        logger.info(
            "Calibrated %s on '%s': mu=%.1f, sigma=%.1f, ROP=%.1f (x%.2f), EOQ=%.1f (x%.2f)",
            self.name,
            scenario.name,
            mu_d,
            sigma_d,
            rop_calc,
            rop_mult,
            eoq_calc,
            eoq_mult,
        )

    def reset(self, scenario: Optional[Scenario] = None) -> None:
        """Reset internal system state at the start of a simulation run."""
        self.outstanding_orders = {e: 0.0 for e in ECHELONS}

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Decide orders for all echelons using the (ROP, EOQ) rule.

        When IP < ROP, place an order for EOQ units (or multiple batches if
        the deficit exceeds EOQ and allow_multi_batch is True).
        """
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
            rop = self.rop_levels[e]
            eoq = self.eoq_levels[e]

            if ip < rop:
                if self.allow_multi_batch:
                    deficit = rop - ip
                    batches = max(1, math.ceil(deficit / max(1.0, eoq)))
                    order_qty = float(batches * eoq)
                else:
                    order_qty = float(eoq)
            else:
                order_qty = 0.0

            orders[e] = order_qty
            self.outstanding_orders[e] += order_qty

        return orders
