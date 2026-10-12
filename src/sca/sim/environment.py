"""4-Echelon Serial Supply Chain Simulation Environment (Beer-Game style).

Echelons in downstream-to-upstream order:
  Customer -> Retailer -> Wholesaler -> Distributor -> Manufacturer -> Factory

State per echelon: inventory, backlog, pipeline, incoming_shipments, last_order, lead_time.
Dynamics per step:
  1. Receive shipments due
  2. Receive demand/orders from downstream
  3. Ship what inventory allows (backlog the rest)
  4. Place new order upstream (with manufacturer capacity constraint)
  5. In-transit pipeline with stochastic lead times & late delivery probability
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from sca.config import FourEchelonConfig, SimConfig, get_config
from sca.data.scenarios import LeadTimeSampler, Scenario

logger = logging.getLogger(__name__)

ECHELONS = ("retailer", "wholesaler", "distributor", "manufacturer")
UPSTREAM_MAP = {
    "retailer": "wholesaler",
    "wholesaler": "distributor",
    "distributor": "manufacturer",
    "manufacturer": None,  # Upstream is factory/raw materials
}
DOWNSTREAM_MAP = {
    "retailer": None,      # Downstream is external customer
    "wholesaler": "retailer",
    "distributor": "wholesaler",
    "manufacturer": "distributor",
}


class InTransitShipment(BaseModel):
    """Shipment moving through the transportation pipeline."""

    shipment_id: int
    quantity: float = Field(ge=0.0)
    dispatch_step: int
    arrival_step: int
    lead_time: int
    is_late: bool = False

    @property
    def remaining_steps(self) -> int:
        return max(0, self.arrival_step - self.dispatch_step)


class EchelonHistoryRecord(BaseModel):
    """Historical state and action record for a single step."""

    step: int
    demand: float
    order_placed: float
    inventory: float
    backlog: float
    incoming_shipment: float
    outgoing_shipment: float
    cost: float


class EchelonState(BaseModel):
    """Pydantic state model for an individual echelon in the supply chain."""

    echelon: str
    step: int
    inventory: float = Field(default=0.0, ge=0.0, description="On-hand inventory (never negative)")
    backlog: float = Field(default=0.0, ge=0.0, description="Unfulfilled orders (never negative)")
    pipeline: float = Field(default=0.0, ge=0.0, description="Total units currently on order in transit")
    pipeline_shipments: List[InTransitShipment] = Field(default_factory=list)
    incoming_shipments: float = Field(default=0.0, ge=0.0, description="Units received at start of this step")
    current_demand: float = Field(default=0.0, ge=0.0, description="Demand received from downstream to fulfill")
    last_order: float = Field(default=0.0, ge=0.0, description="Last order placed upstream")
    lead_time: int = Field(default=2, ge=0, description="Lead time of last dispatched/received shipment")
    holding_cost: float = Field(default=0.0, ge=0.0)
    backlog_cost: float = Field(default=0.0, ge=0.0)
    step_cost: float = Field(default=0.0, ge=0.0)
    history_window: List[EchelonHistoryRecord] = Field(default_factory=list)


class BeerGameEnv:
    """4-Echelon Serial Supply Chain Simulation Environment."""

    def __init__(
        self,
        config: Optional[Union[SimConfig, FourEchelonConfig]] = None,
        scenario: Optional[Scenario] = None,
        demand_array: Optional[Union[List[float], np.ndarray]] = None,
        lead_time_sampler: Optional[LeadTimeSampler] = None,
        late_delivery_probability: Optional[float] = None,
        seed: Optional[int] = 42,
        steady_state_init: Optional[bool] = None,
    ):
        if isinstance(config, SimConfig):
            self.cfg = config.four_echelon
            self.sim_cfg = config
        elif isinstance(config, FourEchelonConfig):
            self.cfg = config
            self.sim_cfg = None
        else:
            self.sim_cfg = get_config().sim
            self.cfg = self.sim_cfg.four_echelon

        self.scenario = scenario
        self.steady_state_init = (
            steady_state_init
            if steady_state_init is not None
            else getattr(self.cfg, "steady_state_init", False)
        )

        # Demand series: scenario demand > custom demand array > fallback constant
        if scenario is not None:
            use_indep = (
                getattr(self.cfg, "independent_demand_paths", None)
                if self.cfg
                else getattr(self.sim_cfg, "independent_demand_paths", True)
            )
            if use_indep and hasattr(scenario, "get_demand_for_seed"):
                self.external_demand = np.array(
                    scenario.get_demand_for_seed(seed if seed is not None else 42), dtype=float
                )
            else:
                self.external_demand = np.array(scenario.demand, dtype=float)

            self.lead_time_sampler = lead_time_sampler or scenario.lead_time_sampler
            self.late_delivery_prob = (
                late_delivery_probability
                if late_delivery_probability is not None
                else scenario.late_delivery_probability
            )
        elif demand_array is not None:
            self.external_demand = np.array(demand_array, dtype=float)
            self.lead_time_sampler = lead_time_sampler or LeadTimeSampler(seed=seed)
            self.late_delivery_prob = late_delivery_probability if late_delivery_probability is not None else 0.0
        else:
            self.external_demand = np.full(self.sim_cfg.time_horizon_days if self.sim_cfg else 90, 100.0)
            self.lead_time_sampler = lead_time_sampler or LeadTimeSampler(seed=seed)
            self.late_delivery_prob = late_delivery_probability if late_delivery_probability is not None else 0.0

        self.initial_seed = seed
        self.rng = np.random.default_rng(seed)

        # Enforce common horizon from config (default 40 steps)
        horizon_limit = (
            getattr(self.cfg, "horizon_steps", None)
            or getattr(self.sim_cfg, "time_horizon_days", None)
            if (self.cfg or self.sim_cfg)
            else 40
        )
        if horizon_limit is not None and len(self.external_demand) > horizon_limit:
            self.external_demand = self.external_demand[:horizon_limit]

        # State tracking
        self.step_idx = 0
        self.max_steps = len(self.external_demand)
        self.shipment_counter = 0

        # Physical states
        self.inventory: Dict[str, float] = {}
        self.backlog: Dict[str, float] = {}
        self.pipelines: Dict[str, List[InTransitShipment]] = {}
        self.incoming_shipments: Dict[str, float] = {}
        self.current_demand: Dict[str, float] = {}
        self.last_order: Dict[str, float] = {}
        self.last_lead_time: Dict[str, int] = {}
        self.step_costs: Dict[str, float] = {}
        self.history: Dict[str, List[EchelonHistoryRecord]] = {}

        # Conservation tracking
        self.cumulative_customer_demand = 0.0
        self.cumulative_customer_delivered = 0.0
        self.cumulative_factory_produced = 0.0
        self.initial_total_units = 0.0

        self.reset(seed=self.initial_seed)

    def reset(self, seed: Optional[int] = None) -> Dict[str, EchelonState]:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.initial_seed = seed
            if self.scenario is not None:
                use_indep = (
                    getattr(self.cfg, "independent_demand_paths", None)
                    if self.cfg
                    else getattr(self.sim_cfg, "independent_demand_paths", True)
                )
                if use_indep and hasattr(self.scenario, "get_demand_for_seed"):
                    self.external_demand = np.array(
                        self.scenario.get_demand_for_seed(seed), dtype=float
                    )
                self.lead_time_sampler = LeadTimeSampler(seed=seed)

        self.step_idx = 0
        self.shipment_counter = 0

        init_inv = float(self.cfg.initial_inventory)
        init_backlog = float(self.cfg.initial_backlog)

        self.inventory = {e: init_inv for e in ECHELONS}
        self.backlog = {e: init_backlog for e in ECHELONS}
        self.pipelines = {e: [] for e in ECHELONS}
        self.incoming_shipments = {e: 0.0 for e in ECHELONS}
        self.last_lead_time = {e: self.cfg.default_lead_time for e in ECHELONS}
        self.step_costs = {e: 0.0 for e in ECHELONS}
        self.history = {e: [] for e in ECHELONS}

        # Step 0 demand
        first_demand = float(self.external_demand[0]) if len(self.external_demand) > 0 else 100.0
        self.current_demand = {
            "retailer": first_demand,
            "wholesaler": first_demand,
            "distributor": first_demand,
            "manufacturer": first_demand,
        }

        # Steady state pipeline initialization
        if self.steady_state_init:
            lt_mean = getattr(self.lead_time_sampler, "mean", float(self.cfg.default_lead_time))
            base_lt = max(1, int(round(lt_mean)))
            for e in ECHELONS:
                for k in range(1, base_lt + 1):
                    self.shipment_counter += 1
                    self.pipelines[e].append(
                        InTransitShipment(
                            shipment_id=self.shipment_counter,
                            quantity=first_demand,
                            dispatch_step=k - base_lt,
                            arrival_step=k,
                            lead_time=base_lt,
                            is_late=False,
                        )
                    )
                self.last_order[e] = first_demand
        else:
            self.last_order = {e: 0.0 for e in ECHELONS}

        # Conservation tracking
        self.cumulative_customer_demand = first_demand
        self.cumulative_customer_delivered = 0.0
        self.cumulative_factory_produced = 0.0
        self.initial_total_units = sum(self.inventory.values()) + self.total_pipeline

        return {e: self.observe(e) for e in ECHELONS}

    def observe(self, echelon: str) -> EchelonState:
        """Get the observation state for a specific echelon."""
        if echelon not in ECHELONS:
            raise ValueError(f"Unknown echelon: '{echelon}'. Expected one of {ECHELONS}")

        inv = self.inventory[echelon]
        back = self.backlog[echelon]
        pipe_shipments = list(self.pipelines[echelon])
        pipe_total = sum(s.quantity for s in pipe_shipments)
        inc = self.incoming_shipments[echelon]
        dem = self.current_demand[echelon]
        last_ord = self.last_order[echelon]
        lt = self.last_lead_time[echelon]

        h_cost = float(self.cfg.holding_cost_per_unit * inv)
        b_cost = float(self.cfg.backlog_cost_per_unit * back)
        s_cost = h_cost + b_cost

        # Extract history window
        hist_window = self.history[echelon][-self.cfg.history_window :]

        return EchelonState(
            echelon=echelon,
            step=self.step_idx,
            inventory=inv,
            backlog=back,
            pipeline=pipe_total,
            pipeline_shipments=pipe_shipments,
            incoming_shipments=inc,
            current_demand=dem,
            last_order=last_ord,
            lead_time=lt,
            holding_cost=h_cost,
            backlog_cost=b_cost,
            step_cost=s_cost,
            history_window=hist_window,
        )

    def _sample_lead_time(self) -> Tuple[int, bool]:
        """Sample transit lead time and evaluate late delivery delay."""
        raw_lt = int(self.lead_time_sampler.sample(1))
        # Enforce non-negative integer lead time
        base_lt = max(0, raw_lt)

        is_late = bool(self.rng.random() < self.late_delivery_prob)
        delay = self.cfg.late_delivery_delay if is_late else 0
        final_lt = base_lt + delay
        return final_lt, is_late

    def step(self, orders: Dict[str, float]) -> Dict[str, EchelonState]:
        """Execute one simulation step across all four echelons.

        Dynamics:
        1. Ship what inventory allows to downstream (backlog remaining).
        2. Put outgoing shipments into upstream/downstream pipeline with stochastic lead times.
        3. Factory produces for Manufacturer up to manufacturer_capacity limit.
        4. Advance step index and receive shipments due at next step.
        5. Update downstream demand for next step.
        """
        # Parse and sanitize order quantities (never negative)
        sanitized_orders = {
            e: max(0.0, float(orders.get(e, 0.0))) for e in ECHELONS
        }

        # Manufacturer production order is bounded by capacity from config
        factory_order = min(sanitized_orders["manufacturer"], float(self.cfg.manufacturer_capacity))
        sanitized_orders["manufacturer"] = factory_order

        # -------------------------------------------------------------
        # 1. Ship what inventory allows (backlog the rest)
        # -------------------------------------------------------------
        shipments_sent: Dict[str, float] = {}

        # Retailer -> External Customer
        ret_demand = self.backlog["retailer"] + self.current_demand["retailer"]
        ship_ret = min(self.inventory["retailer"], ret_demand)
        self.inventory["retailer"] -= ship_ret
        self.backlog["retailer"] = ret_demand - ship_ret
        shipments_sent["retailer"] = ship_ret
        self.cumulative_customer_delivered += ship_ret

        # Wholesaler -> Retailer
        who_demand = self.backlog["wholesaler"] + self.current_demand["wholesaler"]
        ship_who = min(self.inventory["wholesaler"], who_demand)
        self.inventory["wholesaler"] -= ship_who
        self.backlog["wholesaler"] = who_demand - ship_who
        shipments_sent["wholesaler"] = ship_who

        # Distributor -> Wholesaler
        dis_demand = self.backlog["distributor"] + self.current_demand["distributor"]
        ship_dis = min(self.inventory["distributor"], dis_demand)
        self.inventory["distributor"] -= ship_dis
        self.backlog["distributor"] = dis_demand - ship_dis
        shipments_sent["distributor"] = ship_dis

        # Manufacturer -> Distributor
        man_demand = self.backlog["manufacturer"] + self.current_demand["manufacturer"]
        ship_man = min(self.inventory["manufacturer"], man_demand)
        self.inventory["manufacturer"] -= ship_man
        self.backlog["manufacturer"] = man_demand - ship_man
        shipments_sent["manufacturer"] = ship_man

        # Factory -> Manufacturer (unlimited raw materials, produced quantity = factory_order)
        ship_factory = factory_order
        self.cumulative_factory_produced += ship_factory

        # -------------------------------------------------------------
        # 2. Pipeline insertion with stochastic lead times
        # -------------------------------------------------------------
        # Inbound to Retailer (from Wholesaler)
        if ship_who > 0:
            lt, late = self._sample_lead_time()
            self.shipment_counter += 1
            self.pipelines["retailer"].append(
                InTransitShipment(
                    shipment_id=self.shipment_counter,
                    quantity=ship_who,
                    dispatch_step=self.step_idx,
                    arrival_step=self.step_idx + max(1, lt),
                    lead_time=lt,
                    is_late=late,
                )
            )
            self.last_lead_time["retailer"] = lt

        # Inbound to Wholesaler (from Distributor)
        if ship_dis > 0:
            lt, late = self._sample_lead_time()
            self.shipment_counter += 1
            self.pipelines["wholesaler"].append(
                InTransitShipment(
                    shipment_id=self.shipment_counter,
                    quantity=ship_dis,
                    dispatch_step=self.step_idx,
                    arrival_step=self.step_idx + max(1, lt),
                    lead_time=lt,
                    is_late=late,
                )
            )
            self.last_lead_time["wholesaler"] = lt

        # Inbound to Distributor (from Manufacturer)
        if ship_man > 0:
            lt, late = self._sample_lead_time()
            self.shipment_counter += 1
            self.pipelines["distributor"].append(
                InTransitShipment(
                    shipment_id=self.shipment_counter,
                    quantity=ship_man,
                    dispatch_step=self.step_idx,
                    arrival_step=self.step_idx + max(1, lt),
                    lead_time=lt,
                    is_late=late,
                )
            )
            self.last_lead_time["distributor"] = lt

        # Inbound to Manufacturer (from Factory production)
        if ship_factory > 0:
            lt, late = self._sample_lead_time()
            self.shipment_counter += 1
            self.pipelines["manufacturer"].append(
                InTransitShipment(
                    shipment_id=self.shipment_counter,
                    quantity=ship_factory,
                    dispatch_step=self.step_idx,
                    arrival_step=self.step_idx + max(1, lt),
                    lead_time=lt,
                    is_late=late,
                )
            )
            self.last_lead_time["manufacturer"] = lt

        # -------------------------------------------------------------
        # 3. Compute costs and record step history
        # -------------------------------------------------------------
        for e in ECHELONS:
            inv = self.inventory[e]
            back = self.backlog[e]
            cost = (self.cfg.holding_cost_per_unit * inv) + (self.cfg.backlog_cost_per_unit * back)
            self.step_costs[e] = cost

            self.history[e].append(
                EchelonHistoryRecord(
                    step=self.step_idx,
                    demand=self.current_demand[e],
                    order_placed=sanitized_orders[e],
                    inventory=inv,
                    backlog=back,
                    incoming_shipment=self.incoming_shipments[e],
                    outgoing_shipment=shipments_sent[e],
                    cost=cost,
                )
            )
            self.last_order[e] = sanitized_orders[e]

        # -------------------------------------------------------------
        # 4. Advance time step and deliver shipments due at step + 1
        # -------------------------------------------------------------
        self.step_idx += 1

        for e in ECHELONS:
            due_shipments = []
            remaining_pipeline = []
            for s in self.pipelines[e]:
                if s.arrival_step <= self.step_idx:
                    due_shipments.append(s)
                else:
                    remaining_pipeline.append(s)

            arrived_qty = sum(s.quantity for s in due_shipments)
            self.inventory[e] += arrived_qty
            self.incoming_shipments[e] = arrived_qty
            self.pipelines[e] = remaining_pipeline

        # -------------------------------------------------------------
        # 5. Update incoming demand for next step
        # -------------------------------------------------------------
        # Retailer sees external customer demand
        if self.step_idx < len(self.external_demand):
            next_ext_demand = float(self.external_demand[self.step_idx])
        else:
            next_ext_demand = 0.0

        self.current_demand["retailer"] = next_ext_demand
        self.cumulative_customer_demand += next_ext_demand

        # Other echelons see only downstream orders placed at previous step
        self.current_demand["wholesaler"] = sanitized_orders["retailer"]
        self.current_demand["distributor"] = sanitized_orders["wholesaler"]
        self.current_demand["manufacturer"] = sanitized_orders["distributor"]

        return {e: self.observe(e) for e in ECHELONS}

    @property
    def total_inventory(self) -> float:
        """Total on-hand inventory across all 4 echelons."""
        return sum(self.inventory.values())

    @property
    def total_backlog(self) -> float:
        """Total backlog across all 4 echelons."""
        return sum(self.backlog.values())

    @property
    def total_pipeline(self) -> float:
        """Total units in transit across all 4 pipelines."""
        return sum(sum(s.quantity for s in p) for p in self.pipelines.values())

    def check_unit_conservation(self, tolerance: float = 1e-4) -> bool:
        """Verify conservation of physical units in system.

        Total units currently in system (inventory + pipeline) + cumulative units
        delivered to customer MUST exactly equal initial units + cumulative factory production.
        """
        current_system_units = self.total_inventory + self.total_pipeline
        expected_system_units = (self.initial_total_units + self.cumulative_factory_produced) - self.cumulative_customer_delivered
        diff = abs(current_system_units - expected_system_units)
        is_conserved = diff <= tolerance
        if not is_conserved:
            logger.error(
                "Unit conservation violated at step %d! Current=%f, Expected=%f, Diff=%f",
                self.step_idx,
                current_system_units,
                expected_system_units,
                diff,
            )
        return is_conserved
