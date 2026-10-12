"""Metrics module for 4-echelon supply chain simulation: costs, bullwhip ratio, stockout steps, and fill rate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np

if TYPE_CHECKING:
    from sca.sim.environment import BeerGameEnv

logger = logging.getLogger(__name__)


@dataclass
class SimulationMetrics:
    """Comprehensive performance and operational metrics for 4-echelon simulation."""

    total_cost: float
    cost_by_echelon: Dict[str, float]
    holding_cost_by_echelon: Dict[str, float]
    backlog_cost_by_echelon: Dict[str, float]
    bullwhip_ratio: Dict[str, float]
    stockout_steps: Dict[str, int]
    fill_rate: Dict[str, float]
    service_level: Dict[str, float]
    mean_inventory: Dict[str, float]
    mean_backlog: Dict[str, float]
    total_steps: int
    order_variance_by_echelon: Dict[str, float] = field(default_factory=dict)
    demand_variance_by_echelon: Dict[str, float] = field(default_factory=dict)
    end_to_end_bullwhip: float = 1.0
    cost_if_constraints_met: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Format metrics into a clear markdown table with n/a for NaNs."""
        e2e_str = "n/a" if np.isnan(self.end_to_end_bullwhip) else (
            "inf" if np.isinf(self.end_to_end_bullwhip) else f"{self.end_to_end_bullwhip:.3f}"
        )
        c_constr_str = "n/a" if (self.cost_if_constraints_met is None or np.isnan(self.cost_if_constraints_met)) else f"${self.cost_if_constraints_met:,.2f}"

        lines = [
            "# Simulation Performance & Bullwhip Metrics",
            f"- **Total System Cost:** ${self.total_cost:,.2f}",
            f"- **Cost If Constraints Met:** {c_constr_str}",
            f"- **Total Evaluated Steps:** {self.total_steps} days",
            f"- **End-to-End Bullwhip Ratio (Var(Mfr Orders) / Var(Customer Demand)):** {e2e_str}",
            "",
            "| Echelon | Total Cost ($) | Holding Cost ($) | Backlog Cost ($) | Bullwhip Ratio | Order Variance | Demand Variance | Fill Rate (%) | Stockout Steps | Service Level (%) |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for e in ["retailer", "wholesaler", "distributor", "manufacturer"]:
            c = self.cost_by_echelon.get(e, 0.0)
            hc = self.holding_cost_by_echelon.get(e, 0.0)
            bc = self.backlog_cost_by_echelon.get(e, 0.0)
            bw = self.bullwhip_ratio.get(e, float("nan"))
            bw_str = "n/a" if np.isnan(bw) else ("inf" if np.isinf(bw) else f"{bw:.3f}")
            ov = self.order_variance_by_echelon.get(e, 0.0)
            dv = self.demand_variance_by_echelon.get(e, 0.0)
            fr = self.fill_rate.get(e, float("nan"))
            fr_str = "n/a" if np.isnan(fr) else f"{fr * 100:.2f}%"
            so = self.stockout_steps.get(e, 0)
            sl = self.service_level.get(e, float("nan"))
            sl_str = "n/a" if np.isnan(sl) else f"{sl * 100:.1f}%"
            lines.append(
                f"| **{e.capitalize()}** | ${c:,.2f} | ${hc:,.2f} | ${bc:,.2f} | {bw_str} | {ov:,.2f} | {dv:,.2f} | {fr_str} | {so} | {sl_str} |"
            )
        return "\n".join(lines)


def calculate_bullwhip_ratio(orders: Union[List[float], np.ndarray], demand: Union[List[float], np.ndarray]) -> float:
    """Calculate the bullwhip ratio: Var(orders) / Var(demand).

    Handles edge cases:
    - If Var(orders) == 0 and Var(demand) == 0: 0/0 is undefined -> returns np.nan.
    - If Var(orders) > 0 and Var(demand) == 0: returns float('inf').
    - Otherwise: Var(orders) / Var(demand).
    """
    ord_arr = np.array(orders, dtype=float)
    dem_arr = np.array(demand, dtype=float)

    if len(ord_arr) < 2 or len(dem_arr) < 2:
        return float("nan")

    var_orders = float(np.var(ord_arr, ddof=1))
    var_demand = float(np.var(dem_arr, ddof=1))

    if np.isclose(var_demand, 0.0):
        # If demand has zero variance
        if np.isclose(var_orders, 0.0):
            logger.info("Bullwhip ratio undefined (0/0): both order variance and demand variance are zero; returning NaN")
            return float("nan")
        logger.warning(
            "Bullwhip ratio undefined: demand variance is zero (Var(orders)=%.4f, Var(demand)=0.0); returning NaN",
            var_orders,
        )
        return float("nan")

    return float(var_orders / var_demand)


def calculate_fill_rate(shipped: Union[List[float], np.ndarray], demand: Union[List[float], np.ndarray]) -> float:
    """Calculate fill rate: sum(shipped) / sum(demand).
    If sum(demand) == 0 and sum(shipped) == 0, 0/0 is undefined -> returns np.nan.
    """
    tot_shipped = float(np.sum(shipped))
    tot_demand = float(np.sum(demand))

    if tot_demand <= 0.0:
        if tot_shipped <= 0.0:
            return float("nan")  # 0/0 undefined ratio -> NaN
        return 1.0
    return float(min(1.0, max(0.0, tot_shipped / tot_demand)))


def compute_cost_if_constraints_met(
    total_cost: float,
    fill_rate: float,
    service_level: float,
    mean_bullwhip: Optional[float] = None,
    bullwhip_ceiling: Optional[float] = None,
    min_fill_rate: float = 0.99,
    min_service_level: float = 0.85,
) -> float:
    """Compute constrained cost: returns total_cost if all constraints are satisfied, else NaN.

    Benchmark v2 Constraints:
    - Fill Rate >= 99% (0.99)
    - Step Service Level >= 85% (0.85)
    - Mean Bullwhip <= bullwhip_ceiling (if specified)
    """
    if np.isnan(fill_rate) or fill_rate < min_fill_rate:
        return float("nan")
    if np.isnan(service_level) or service_level < min_service_level:
        return float("nan")
    if bullwhip_ceiling is not None and mean_bullwhip is not None:
        if not np.isnan(mean_bullwhip) and mean_bullwhip > bullwhip_ceiling:
            return float("nan")
    return float(total_cost)


def compute_simulation_metrics(
    env: BeerGameEnv,
    warmup_steps: Optional[int] = None,
) -> SimulationMetrics:
    """Extract and compute all analytical metrics from a completed or in-progress BeerGameEnv.

    Args:
        env: The simulation environment instance.
        warmup_steps: Initial steps to discard (warm-up period). If None, defaults to
                      env.cfg.warmup_steps (default 5 steps from config).
    """
    echelons = ("retailer", "wholesaler", "distributor", "manufacturer")

    cost_by_echelon = {}
    holding_cost_by_echelon = {}
    backlog_cost_by_echelon = {}
    bullwhip_ratio = {}
    order_variance_by_echelon = {}
    demand_variance_by_echelon = {}
    stockout_steps = {}
    fill_rate = {}
    service_level = {}
    mean_inventory = {}
    mean_backlog = {}

    raw_steps = len(env.history["retailer"])
    if warmup_steps is None:
        warmup_steps = getattr(env.cfg, "warmup_steps", getattr(env.sim_cfg, "warmup_steps", 5))

    # Apply warmup only when history strictly exceeds warmup_steps
    eval_start = warmup_steps if (raw_steps > warmup_steps and warmup_steps > 0) else 0
    total_steps = raw_steps - eval_start

    for e in echelons:
        hist = env.history[e]
        if not hist or len(hist) <= eval_start:
            cost_by_echelon[e] = 0.0
            holding_cost_by_echelon[e] = 0.0
            backlog_cost_by_echelon[e] = 0.0
            bullwhip_ratio[e] = 1.0
            order_variance_by_echelon[e] = 0.0
            demand_variance_by_echelon[e] = 0.0
            stockout_steps[e] = 0
            fill_rate[e] = 1.0
            service_level[e] = 1.0
            mean_inventory[e] = 0.0
            mean_backlog[e] = 0.0
            continue

        eval_hist = hist[eval_start:]
        demands = np.array([r.demand for r in eval_hist], dtype=float)
        orders = np.array([r.order_placed for r in eval_hist], dtype=float)
        inventories = np.array([r.inventory for r in eval_hist], dtype=float)
        backlogs = np.array([r.backlog for r in eval_hist], dtype=float)
        shipped = np.array([r.outgoing_shipment for r in eval_hist], dtype=float)

        # Holding & Backlog costs
        h_costs = env.cfg.holding_cost_per_unit * inventories
        b_costs = env.cfg.backlog_cost_per_unit * backlogs
        tot_h_cost = float(np.sum(h_costs))
        tot_b_cost = float(np.sum(b_costs))

        holding_cost_by_echelon[e] = tot_h_cost
        backlog_cost_by_echelon[e] = tot_b_cost
        cost_by_echelon[e] = tot_h_cost + tot_b_cost

        # Absolute variances and bullwhip ratio: Var(orders) / Var(demand)
        ov = float(np.var(orders, ddof=1)) if len(orders) > 1 else 0.0
        dv = float(np.var(demands, ddof=1)) if len(demands) > 1 else 0.0
        order_variance_by_echelon[e] = ov
        demand_variance_by_echelon[e] = dv
        bullwhip_ratio[e] = calculate_bullwhip_ratio(orders, demands)

        # Stockout steps: steps where backlog > 0
        stockout_steps[e] = int(np.sum(backlogs > 1e-4))

        # Fill rate: cumulative shipped / cumulative demand
        fill_rate[e] = calculate_fill_rate(shipped, demands)

        # Service level: proportion of steps with zero backlog
        service_level[e] = float(np.mean(backlogs <= 1e-4))

        mean_inventory[e] = float(np.mean(inventories))
        mean_backlog[e] = float(np.mean(backlogs))

    total_cost = sum(cost_by_echelon.values())

    # End-to-end bullwhip: Var(manufacturer orders) / Var(customer demand)
    cust_demand_var = demand_variance_by_echelon.get("retailer", 0.0)
    mfr_order_var = order_variance_by_echelon.get("manufacturer", 0.0)
    if np.isclose(cust_demand_var, 0.0):
        if np.isclose(mfr_order_var, 0.0):
            logger.info("End-to-end bullwhip undefined (0/0): both mfr order variance and customer demand variance are zero; returning NaN")
            end_to_end_bullwhip = float("nan")
        else:
            logger.warning(
                "End-to-end bullwhip undefined: customer demand variance is zero (Var(mfr_orders)=%.4f); returning NaN",
                mfr_order_var,
            )
            end_to_end_bullwhip = float("nan")
    else:
        end_to_end_bullwhip = float(mfr_order_var / cust_demand_var)

    # Benchmark v2 Constrained Cost: cost if constraints met (fill rate >= 99%, service level >= 85%)
    ret_fr = fill_rate.get("retailer", float("nan"))
    ret_sl = service_level.get("retailer", float("nan"))
    cost_if_constraints_met = compute_cost_if_constraints_met(
        total_cost=total_cost,
        fill_rate=ret_fr,
        service_level=ret_sl,
    )

    return SimulationMetrics(
        total_cost=total_cost,
        cost_by_echelon=cost_by_echelon,
        holding_cost_by_echelon=holding_cost_by_echelon,
        backlog_cost_by_echelon=backlog_cost_by_echelon,
        bullwhip_ratio=bullwhip_ratio,
        stockout_steps=stockout_steps,
        fill_rate=fill_rate,
        service_level=service_level,
        mean_inventory=mean_inventory,
        mean_backlog=mean_backlog,
        total_steps=total_steps,
        order_variance_by_echelon=order_variance_by_echelon,
        demand_variance_by_echelon=demand_variance_by_echelon,
        end_to_end_bullwhip=end_to_end_bullwhip,
        cost_if_constraints_met=cost_if_constraints_met,
    )
