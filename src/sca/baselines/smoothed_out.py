"""Smoothed Order-Up-To Inventory Policy Baseline (System 2b).

Implements an adaptive order-up-to replenishment policy without LLM calls:
1. Exponentially smoothed demand forecast.
2. Dynamic order-up-to target S = L * smoothed_demand + SS with safety stock.
3. Order smoothing: O = alpha * (S - IP) + (1 - alpha) * last_order.
4. Validation via validate_order with rate limiting (delta) and forecast-relative cap.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from sca.config import get_config
from sca.data.scenarios import Scenario
from sca.sim.environment import ECHELONS, EchelonState
from sca.systems.base import OrderingSystem, register_system
from sca.tools import (
    SmoothedOrderUpToInput,
    SmoothedOrderUpToOutput,
    ValidateOrderInput,
    ValidateOrderOutput,
    smoothed_order_up_to,
    validate_order,
)

logger = logging.getLogger(__name__)


@register_system([
    "baseline_smoothed_order_up_to",
    "baseline_2b",
    "system_2b",
    "system2b",
    "smoothed_order_up_to",
    "s2b",
])
class SmoothedOrderUpToPolicy(OrderingSystem):
    """System 2b: Smoothed order-up-to replenishment baseline without LLM."""

    def __init__(
        self,
        name: str = "baseline_smoothed_order_up_to",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        cfg_fe = None
        try:
            cfg_fe = get_config().sim.four_echelon
        except Exception:
            pass

        self.alpha: float = float(
            self.config.get("alpha", getattr(cfg_fe, "alpha", 0.8) if cfg_fe else 0.8)
        )
        self.beta: float = float(
            self.config.get(
                "beta",
                getattr(cfg_fe, "beta", None) if (cfg_fe and getattr(cfg_fe, "beta", None) is not None) else self.alpha,
            )
        )
        self.z: float = float(
            self.config.get(
                "z",
                self.config.get(
                    "safety_stock_factor",
                    getattr(cfg_fe, "safety_factor_z", 1.65) if cfg_fe else 1.65,
                ),
            )
        )
        self.include_lead_time_variance: bool = bool(
            self.config.get(
                "include_lead_time_variance",
                getattr(cfg_fe, "include_lead_time_variance", True) if cfg_fe else True,
            )
        )
        self.delta: float = float(
            self.config.get(
                "delta",
                self.config.get(
                    "rate_limit_delta",
                    getattr(cfg_fe, "rate_limit_delta", 0.5) if cfg_fe else 0.5,
                ),
            )
        )
        self.forecast_cap_multiple: float = float(
            self.config.get(
                "forecast_cap_multiple",
                getattr(cfg_fe, "forecast_cap_multiple", 2.0) if cfg_fe else 2.0,
            )
        )
        self.safety_stock_factor: float = self.z

        # Baseline parameters
        self.mu_d: float = 100.0
        self.demand_std: float = 20.0
        self.lead_time_mean: float = 2.0
        self.lead_time_std: float = 0.8

        self.smoothed_demands: Dict[str, float] = {}
        self.last_orders: Dict[str, float] = {}
        self._init_defaults()

    def _init_defaults(self):
        """Initialize default state tracking."""
        for e in ECHELONS:
            self.smoothed_demands[e] = self.mu_d
            self.last_orders[e] = self.mu_d

        self.parameters = {
            "policy": "smoothed_order_up_to",
            "alpha": self.alpha,
            "beta": self.beta,
            "z": self.z,
            "include_lead_time_variance": self.include_lead_time_variance,
            "delta": self.delta,
            "forecast_cap_multiple": self.forecast_cap_multiple,
            "safety_stock_factor": self.z,
            "mu_d": self.mu_d,
            "demand_std": self.demand_std,
            "lead_time_mean": self.lead_time_mean,
            "lead_time_std": self.lead_time_std,
        }

    def calibrate(self, scenario: Scenario, config: Optional[Dict[str, Any]] = None) -> None:
        """Derive policy parameters from scenario training history and config."""
        cfg = config or self.config

        train_data = scenario.train_demand
        if len(train_data) == 0:
            self.mu_d = 100.0
            self.demand_std = 20.0
        else:
            self.mu_d = float(np.mean(train_data))
            self.demand_std = float(np.std(train_data))

        # Lead time distribution stats
        if hasattr(scenario, "lead_time_sampler") and scenario.lead_time_sampler:
            self.lead_time_mean = float(scenario.lead_time_sampler.mean)
            self.lead_time_std = float(scenario.lead_time_sampler.std)
        else:
            self.lead_time_mean = 2.0
            self.lead_time_std = 0.8

        # Overrides from config if provided
        if "alpha" in cfg:
            self.alpha = float(cfg["alpha"])
        if "beta" in cfg:
            self.beta = float(cfg["beta"])
        elif "alpha" in cfg and "beta" not in cfg and self.beta == self.alpha:
            self.beta = self.alpha
        if "z" in cfg:
            self.z = float(cfg["z"])
        elif "safety_stock_factor" in cfg:
            self.z = float(cfg["safety_stock_factor"])
        self.safety_stock_factor = self.z

        if "include_lead_time_variance" in cfg:
            self.include_lead_time_variance = bool(cfg["include_lead_time_variance"])
        if "delta" in cfg:
            self.delta = float(cfg["delta"])
        elif "rate_limit_delta" in cfg:
            self.delta = float(cfg["rate_limit_delta"])
        if "forecast_cap_multiple" in cfg:
            self.forecast_cap_multiple = float(cfg["forecast_cap_multiple"])

        effective_lt_std = self.lead_time_std if self.include_lead_time_variance else 0.0

        self.parameters = {
            "policy": "smoothed_order_up_to",
            "alpha": self.alpha,
            "beta": self.beta,
            "z": self.z,
            "include_lead_time_variance": self.include_lead_time_variance,
            "delta": self.delta,
            "forecast_cap_multiple": self.forecast_cap_multiple,
            "safety_stock_factor": self.z,
            "mu_d": round(self.mu_d, 2),
            "demand_std": round(self.demand_std, 2),
            "lead_time_mean": round(self.lead_time_mean, 2),
            "lead_time_std": round(self.lead_time_std, 2),
            "effective_lead_time_std": round(effective_lt_std, 2),
        }
        self.calibrated = True
        logger.info(
            "Calibrated %s on '%s': alpha=%.2f, beta=%.2f, z=%.2f, mu_d=%.1f, L=%.2f, std_L=%.2f (lt_var=%s, eff_std_L=%.2f), delta=%.2f, cap_mult=%.1f",
            self.name,
            scenario.name,
            self.alpha,
            self.beta,
            self.z,
            self.mu_d,
            self.lead_time_mean,
            self.lead_time_std,
            self.include_lead_time_variance,
            effective_lt_std,
            self.delta,
            self.forecast_cap_multiple,
        )

    def reset(self, scenario: Optional[Scenario] = None) -> None:
        """Reset internal system state at start of simulation run."""
        first_demand = self.mu_d
        if scenario is not None and len(scenario.demand) > 0:
            first_demand = float(scenario.demand[0])

        self.smoothed_demands = {e: first_demand for e in ECHELONS}
        self.last_orders = {e: first_demand for e in ECHELONS}

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Decide orders for all echelons using smoothed order-up-to and validation."""
        orders: Dict[str, float] = {}

        for e in ECHELONS:
            st = states[e]
            prev_smooth = self.smoothed_demands.get(e, st.current_demand)
            last_ord = self.last_orders.get(e, st.current_demand)

            # 1. Calculate smoothed order-up-to
            inp_tool = SmoothedOrderUpToInput(
                current_demand=float(st.current_demand),
                inventory=float(st.inventory),
                pipeline=float(st.pipeline),
                backlog=float(st.backlog),
                last_order=float(last_ord),
                lead_time_mean=self.lead_time_mean,
                lead_time_std=self.lead_time_std,
                demand_std=self.demand_std,
                safety_stock_factor=self.safety_stock_factor,
                z=self.z,
                include_lead_time_variance=self.include_lead_time_variance,
                previous_smoothed_demand=float(prev_smooth),
                alpha=self.alpha,
                beta=self.beta,
            )
            tool_db = self.config.get("db_path", False)
            out_tool: SmoothedOrderUpToOutput = smoothed_order_up_to(inp_tool, db_path=tool_db)
            self.smoothed_demands[e] = out_tool.smoothed_demand

            # 2. Validate order with rate limiting & forecast cap
            cap_limit = 300.0 if e == "manufacturer" else None
            val_inp = ValidateOrderInput(
                proposed_order=out_tool.order_quantity,
                inventory_position=out_tool.inventory_position,
                min_order=0.0,
                capacity_limit=cap_limit,
                echelon=e,
                last_order=last_ord,
                forecast_demand=out_tool.smoothed_demand,
                rate_limit_delta=self.delta,
                forecast_cap_multiple=self.forecast_cap_multiple,
            )
            val_out: ValidateOrderOutput = validate_order(val_inp, db_path=tool_db)

            final_order = float(val_out.validated_order)
            self.last_orders[e] = final_order
            orders[e] = final_order

        return orders
