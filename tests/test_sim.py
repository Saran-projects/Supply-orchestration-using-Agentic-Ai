"""Tests for 4-echelon serial Beer-Game supply chain simulation and metrics.

Verifies:
1. Conservation of physical units across the entire system.
2. Determinism per random seed.
3. Invariant: backlog and inventory are never negative.
4. Metrics matches exact hand-computed examples.
5. Manufacturer production capacity constraint.
6. Observation isolation (Retailer sees external demand, others see downstream orders).
"""

import numpy as np
import pytest

from sca.config import FourEchelonConfig
from sca.data.scenarios import LeadTimeSampler
from sca.sim import (
    ECHELONS,
    BeerGameEnv,
    EchelonState,
    calculate_bullwhip_ratio,
    calculate_fill_rate,
    compute_simulation_metrics,
)


class TestBeerGameUnitConservation:
    """Test unit conservation law across the 4-echelon supply chain."""

    def test_unit_conservation_constant_orders(self):
        env = BeerGameEnv(seed=101)
        for t in range(40):
            orders = {e: 80.0 for e in ECHELONS}
            env.step(orders)
            assert env.check_unit_conservation(), f"Conservation failed at step {t}"

    def test_unit_conservation_stochastic_orders_and_lead_times(self):
        rng = np.random.default_rng(2024)
        env = BeerGameEnv(seed=2024)

        for t in range(60):
            # Highly erratic random orders across echelons
            orders = {
                e: float(rng.integers(0, 350)) for e in ECHELONS
            }
            env.step(orders)
            assert env.check_unit_conservation(), f"Conservation failed at step {t} with random orders"


class TestBeerGameDeterminism:
    """Verify that identical random seeds produce strictly bit-identical trajectories."""

    def test_determinism_per_seed(self):
        seed = 777
        env1 = BeerGameEnv(seed=seed)
        env2 = BeerGameEnv(seed=seed)

        rng = np.random.default_rng(999)
        test_orders = [
            {e: float(rng.integers(20, 200)) for e in ECHELONS}
            for _ in range(25)
        ]

        for step_idx, ords in enumerate(test_orders):
            obs1 = env1.step(ords)
            obs2 = env2.step(ords)

            for e in ECHELONS:
                assert obs1[e].inventory == obs2[e].inventory, f"Inventory mismatch at step {step_idx} echelon {e}"
                assert obs1[e].backlog == obs2[e].backlog, f"Backlog mismatch at step {step_idx} echelon {e}"
                assert obs1[e].pipeline == obs2[e].pipeline, f"Pipeline mismatch at step {step_idx} echelon {e}"
                assert obs1[e].incoming_shipments == obs2[e].incoming_shipments
                assert obs1[e].current_demand == obs2[e].current_demand
                assert obs1[e].step_cost == obs2[e].step_cost


class TestNonNegativityInvariants:
    """Verify that inventory and backlog are never negative under extreme conditions."""

    def test_inventory_and_backlog_never_negative(self):
        cfg = FourEchelonConfig(
            initial_inventory=5.0,  # Extremely low inventory
            initial_backlog=0.0,
            manufacturer_capacity=150.0,
        )
        env = BeerGameEnv(config=cfg, seed=54321)
        rng = np.random.default_rng(54321)

        for step_idx in range(50):
            # Alternate between zero orders and massive demand spikes
            orders = {
                e: float(rng.choice([0.0, 10.0, 500.0])) for e in ECHELONS
            }
            obs = env.step(orders)

            for e in ECHELONS:
                state = obs[e]
                assert state.inventory >= 0.0, f"Negative inventory ({state.inventory}) at step {step_idx} in {e}"
                assert state.backlog >= 0.0, f"Negative backlog ({state.backlog}) at step {step_idx} in {e}"
                assert state.pipeline >= 0.0, f"Negative pipeline ({state.pipeline}) at step {step_idx} in {e}"


class TestMetricsHandComputedExamples:
    """Validate metrics on exact hand-calculated traces."""

    def test_hand_computed_3_step_trace(self):
        """Hand calculation trace:
        Parameters:
          initial_inv = 10.0, initial_backlog = 0.0
          holding_cost = 1.0, backlog_cost = 2.0
          fixed_lead_time = 1, late_prob = 0.0
          demand = [5.0, 15.0]
        Actions:
          Step 0 orders = all 10.0
          Step 1 orders = all 15.0

        Step 0:
          Retailer ships 5, inv=5, back=0, cost=5.0
          Wholesaler ships 5, inv=5, back=0, cost=5.0
          Distributor ships 5, inv=5, back=0, cost=5.0
          Manufacturer ships 5, inv=5, back=0, cost=5.0
          Total step 0 cost = 20.0

        Step 1:
          Retailer receives 5 from Wholesaler -> inv=10. Demand=15.
          Ships 10, inv=0, backlog=5, cost = 0*1.0 + 5*2.0 = 10.0
          Wholesaler receives 5 from Distributor -> inv=10. Demand=10.
          Ships 10, inv=0, backlog=0, cost = 0.0
          Distributor receives 5 from Manufacturer -> inv=10. Demand=10.
          Ships 10, inv=0, backlog=0, cost = 0.0
          Manufacturer receives 10 from Factory -> inv=15. Demand=10.
          Ships 10, inv=5, backlog=0, cost = 5*1.0 = 5.0
          Total step 1 cost = 15.0

        Cumulative:
          Total Cost = 20.0 + 15.0 = 35.0
          Retailer Cost = 5.0 + 10.0 = 15.0
          Wholesaler Cost = 5.0 + 0.0 = 5.0
          Distributor Cost = 5.0 + 0.0 = 5.0
          Manufacturer Cost = 5.0 + 5.0 = 10.0
          Retailer Fill Rate = 15 shipped / 20 demand = 0.75 (75%)
          Retailer Stockout Steps = 1 (step 1 backlog=5)
        """
        cfg = FourEchelonConfig(
            initial_inventory=10.0,
            initial_backlog=0.0,
            holding_cost_per_unit=1.0,
            backlog_cost_per_unit=2.0,
            manufacturer_capacity=50.0,
        )
        fixed_lt_sampler = LeadTimeSampler(days=np.array([1]), probabilities=np.array([1.0]))

        env = BeerGameEnv(
            config=cfg,
            demand_array=[5.0, 15.0],
            lead_time_sampler=fixed_lt_sampler,
            late_delivery_probability=0.0,
            seed=42,
        )

        # Step 0
        env.step({"retailer": 10.0, "wholesaler": 10.0, "distributor": 10.0, "manufacturer": 10.0})
        # Step 1
        env.step({"retailer": 15.0, "wholesaler": 15.0, "distributor": 15.0, "manufacturer": 15.0})

        metrics = compute_simulation_metrics(env)

        assert metrics.total_cost == 35.0
        assert metrics.cost_by_echelon["retailer"] == 15.0
        assert metrics.cost_by_echelon["wholesaler"] == 5.0
        assert metrics.cost_by_echelon["distributor"] == 5.0
        assert metrics.cost_by_echelon["manufacturer"] == 10.0
        assert metrics.fill_rate["retailer"] == 0.75
        assert metrics.stockout_steps["retailer"] == 1
        assert metrics.stockout_steps["wholesaler"] == 0
        assert metrics.stockout_steps["distributor"] == 0
        assert metrics.stockout_steps["manufacturer"] == 0

    def test_hand_computed_bullwhip_formula(self):
        """Orders: [10, 30] -> mean=20, variance = ((-10)^2 + (+10)^2)/1 = 200.
        Demand: [10, 20] -> mean=15, variance = ((-5)^2 + (+5)^2)/1 = 50.
        Bullwhip ratio = 200 / 50 = 4.0.
        """
        orders = [10.0, 30.0]
        demand = [10.0, 20.0]
        bw = calculate_bullwhip_ratio(orders, demand)
        assert bw == pytest.approx(4.0)

    def test_hand_computed_fill_rate_formula(self):
        shipped = [15.0, 25.0]  # sum = 40
        demand = [20.0, 30.0]   # sum = 50
        fr = calculate_fill_rate(shipped, demand)
        assert fr == pytest.approx(0.80)


class TestManufacturerCapacityAndObservationIsolation:
    """Verify manufacturer capacity clamping and observation scope."""

    def test_manufacturer_capacity_clamping(self):
        cfg = FourEchelonConfig(
            initial_inventory=50.0,
            manufacturer_capacity=100.0,  # Cap at 100
        )
        env = BeerGameEnv(config=cfg, seed=123)

        # Submit an excessive order of 500 for manufacturer
        env.step({"retailer": 50, "wholesaler": 50, "distributor": 50, "manufacturer": 500})

        # Manufacturer's last order recorded should be clamped to 100
        man_state = env.observe("manufacturer")
        assert man_state.last_order == 100.0

        # Manufacturer inbound pipeline shipment should equal 100.0
        assert len(env.pipelines["manufacturer"]) == 1
        assert env.pipelines["manufacturer"][0].quantity == 100.0

    def test_observation_isolation(self):
        ext_demands = [73.0, 85.0]
        env = BeerGameEnv(demand_array=ext_demands, seed=42)

        # Step 0
        r_state = env.observe("retailer")
        assert r_state.current_demand == 73.0  # Retailer sees external demand

        # Place distinct orders
        env.step({"retailer": 90.0, "wholesaler": 110.0, "distributor": 130.0, "manufacturer": 150.0})

        # At Step 1:
        # Retailer sees external demand 85.0
        assert env.observe("retailer").current_demand == 85.0
        # Wholesaler sees ONLY Retailer's order (90.0)
        assert env.observe("wholesaler").current_demand == 90.0
        # Distributor sees ONLY Wholesaler's order (110.0)
        assert env.observe("distributor").current_demand == 110.0
        # Manufacturer sees ONLY Distributor's order (130.0)
        assert env.observe("manufacturer").current_demand == 130.0

    def test_history_window_length(self):
        cfg = FourEchelonConfig(history_window=5)
        env = BeerGameEnv(config=cfg, seed=42)

        for _ in range(12):
            env.step({e: 100.0 for e in ECHELONS})

        # History window in observe() should be capped at 5 records
        r_state = env.observe("retailer")
        assert len(r_state.history_window) == 5
        assert isinstance(r_state, EchelonState)
