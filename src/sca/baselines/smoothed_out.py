"""Smoothed Order-Up-To Inventory Policy Baseline (System 2b).

Implements an adaptive order-up-to replenishment policy without LLM calls:
1. Exponentially smoothed demand forecast.
2. Dynamic order-up-to target S = L * smoothed_demand + SS with safety stock.
3. Order smoothing: O = alpha * (S - IP) + (1 - alpha) * last_order.
4. Validation via validate_order with rate limiting (delta) and forecast-relative cap.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sca.config import get_config
from sca.data.scenarios import Scenario
from sca.sim.environment import ECHELONS, EchelonState
from sca.systems.base import OrderingSystem, register_system
from sca.tools import (
    RegimeDetectorInput,
    RegimeDetectorOutput,
    SmoothedOrderUpToInput,
    SmoothedOrderUpToOutput,
    ValidateOrderInput,
    ValidateOrderOutput,
    regime_detector,
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
        self.info_sharing: bool = bool(self.config.get("info_sharing", False))
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
            "info_sharing": self.info_sharing,
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

        if "info_sharing" in cfg:
            self.info_sharing = bool(cfg["info_sharing"])

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
            "info_sharing": self.info_sharing,
            "mu_d": round(self.mu_d, 2),
            "demand_std": round(self.demand_std, 2),
            "lead_time_mean": round(self.lead_time_mean, 2),
            "lead_time_std": round(self.lead_time_std, 2),
            "effective_lead_time_std": round(effective_lt_std, 2),
        }
        self.calibrated = True
        logger.info(
            "Calibrated %s on '%s': alpha=%.2f, beta=%.2f, z=%.2f, mu_d=%.1f, L=%.2f, std_L=%.2f (lt_var=%s, eff_std_L=%.2f), delta=%.2f, cap_mult=%.1f, info_sharing=%s",
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
            self.info_sharing,
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
        cust_demand = None
        if self.info_sharing and "retailer" in states:
            cust_demand = float(states["retailer"].current_demand)

        for e in ECHELONS:
            st = states[e]
            d_observed = cust_demand if (self.info_sharing and cust_demand is not None) else float(st.current_demand)
            prev_smooth = self.smoothed_demands.get(e, d_observed)
            last_ord = self.last_orders.get(e, d_observed)

            # 1. Calculate smoothed order-up-to
            inp_tool = SmoothedOrderUpToInput(
                current_demand=float(d_observed),
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


def load_frozen_2c_params(path: str | Path = "results/baselines_frozen.json") -> Dict[str, Any]:
    """Tuned System 2c parameters from the frozen baselines file ({} if unavailable)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f)["tuned_policies"]["system_2c_adaptive_smoothed_out"]["parameters"])
    except (OSError, KeyError, ValueError):
        return {}


@register_system([
    "baseline_smoothed_order_up_to_adaptive",
    "baseline_2c",
    "system_2c",
    "system2c",
    "s2c",
])
class System2cPolicy(SmoothedOrderUpToPolicy):
    """System 2c: System 2b plus demand regime detector.

    In shock regime: raises beta and widens delta for a fixed number of steps,
    then returns to steady-state values. Supports info-sharing flag where each
    echelon forecasts from end-customer demand instead of its neighbour's orders.
    """

    def __init__(
        self,
        name: str = "baseline_2c",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.beta_shock: float = float(self.config.get("beta_shock", min(1.0, self.beta * 1.5)))
        self.delta_shock: float = float(self.config.get("delta_shock", min(1.0, self.delta * 1.5)))
        self.shock_duration: int = int(self.config.get("shock_duration", 6))
        self.short_window: int = int(self.config.get("short_window", 5))
        self.long_window: int = int(self.config.get("long_window", 15))
        self.shock_threshold: float = float(self.config.get("shock_threshold", 1.30))
        self.trend_threshold: float = float(self.config.get("trend_threshold", 1.15))

        self.demand_histories: Dict[str, List[float]] = {e: [] for e in ECHELONS}
        self.shock_countdowns: Dict[str, int] = {e: 0 for e in ECHELONS}
        self.detected_regimes: Dict[str, str] = {e: "stable" for e in ECHELONS}

    def calibrate(self, scenario: Scenario, config: Optional[Dict[str, Any]] = None) -> None:
        super().calibrate(scenario, config)
        cfg = config or self.config
        if "beta_shock" in cfg:
            self.beta_shock = float(cfg["beta_shock"])
        else:
            self.beta_shock = min(1.0, self.beta * 1.5)
        if "delta_shock" in cfg:
            self.delta_shock = float(cfg["delta_shock"])
        else:
            self.delta_shock = min(1.0, self.delta * 1.5)
        if "shock_duration" in cfg:
            self.shock_duration = int(cfg["shock_duration"])
        if "short_window" in cfg:
            self.short_window = int(cfg["short_window"])
        if "long_window" in cfg:
            self.long_window = int(cfg["long_window"])
        if "shock_threshold" in cfg:
            self.shock_threshold = float(cfg["shock_threshold"])
        if "trend_threshold" in cfg:
            self.trend_threshold = float(cfg["trend_threshold"])

        self.parameters.update({
            "policy": "system_2c_adaptive",
            "beta_shock": self.beta_shock,
            "delta_shock": self.delta_shock,
            "shock_duration": self.shock_duration,
            "short_window": self.short_window,
            "long_window": self.long_window,
            "shock_threshold": self.shock_threshold,
            "trend_threshold": self.trend_threshold,
        })

    def reset(self, scenario: Optional[Scenario] = None) -> None:
        super().reset(scenario)
        first_d = self.mu_d
        if scenario is not None and len(scenario.demand) > 0:
            first_d = float(scenario.demand[0])
        self.demand_histories = {e: [first_d] for e in ECHELONS}
        self.shock_countdowns = {e: 0 for e in ECHELONS}
        self.detected_regimes = {e: "stable" for e in ECHELONS}

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        orders: Dict[str, float] = {}
        cust_demand = None
        if self.info_sharing and "retailer" in states:
            cust_demand = float(states["retailer"].current_demand)

        tool_db = self.config.get("db_path", False)

        for e in ECHELONS:
            st = states[e]
            d_observed = cust_demand if (self.info_sharing and cust_demand is not None) else float(st.current_demand)
            self.demand_histories[e].append(d_observed)

            # Detect regime using regime_detector tool
            reg_out: RegimeDetectorOutput = regime_detector(
                RegimeDetectorInput(
                    demand_history=self.demand_histories[e],
                    short_window=self.short_window,
                    long_window=self.long_window,
                    shock_threshold=self.shock_threshold,
                    trend_threshold=self.trend_threshold,
                ),
                db_path=tool_db,
            )
            self.detected_regimes[e] = reg_out.regime

            # If shock detected, activate shock countdown
            if reg_out.regime == "shock":
                self.shock_countdowns[e] = self.shock_duration

            # Resolve effective beta and delta
            if self.shock_countdowns[e] > 0:
                eff_beta = self.beta_shock
                eff_delta = self.delta_shock
                self.shock_countdowns[e] -= 1
            else:
                eff_beta = self.beta
                eff_delta = self.delta

            prev_smooth = self.smoothed_demands.get(e, d_observed)
            last_ord = self.last_orders.get(e, d_observed)

            # 1. Calculate smoothed order-up-to with eff_beta
            inp_tool = SmoothedOrderUpToInput(
                current_demand=float(d_observed),
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
                beta=eff_beta,
            )
            out_tool: SmoothedOrderUpToOutput = smoothed_order_up_to(inp_tool, db_path=tool_db)
            self.smoothed_demands[e] = out_tool.smoothed_demand

            # 2. Validate order with eff_delta
            cap_limit = 300.0 if e == "manufacturer" else None
            val_inp = ValidateOrderInput(
                proposed_order=out_tool.order_quantity,
                inventory_position=out_tool.inventory_position,
                min_order=0.0,
                capacity_limit=cap_limit,
                echelon=e,
                last_order=last_ord,
                forecast_demand=out_tool.smoothed_demand,
                rate_limit_delta=eff_delta,
                forecast_cap_multiple=self.forecast_cap_multiple,
            )
            val_out: ValidateOrderOutput = validate_order(val_inp, db_path=tool_db)

            final_order = float(val_out.validated_order)
            self.last_orders[e] = final_order
            orders[e] = final_order

        return orders

