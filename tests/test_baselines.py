"""Unit tests for baseline replenishment policies (s, S) and EOQ."""

import numpy as np
import pytest

from sca.baselines import EoqPolicy, SsPolicy
from sca.data.scenarios import LeadTimeSampler, Scenario
from sca.sim.environment import ECHELONS, BeerGameEnv, EchelonState
from sca.systems.base import get_system, list_systems


@pytest.fixture
def dummy_scenario():
    """Create a minimal synthetic Scenario for calibration tests."""
    train_dates = [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(100)]
    sc_dates = [np.datetime64("2020-04-10") + np.timedelta64(i, "D") for i in range(30)]
    train_dem = np.full(100, 100.0)
    sc_dem = np.full(30, 100.0)
    sampler = LeadTimeSampler(seed=42)

    return Scenario(
        name="test_scenario",
        demand=sc_dem,
        raw_demand=sc_dem,
        dates=sc_dates,
        train_demand=train_dem,
        train_raw_demand=train_dem,
        train_dates=train_dates,
        lead_time_sampler=sampler,
        late_delivery_probability=0.1,
        scale_factor=1.0,
    )


def test_registry_contains_baselines():
    """Verify both policies are registered with expected aliases."""
    systems = list_systems()
    assert "baseline_s_s" in systems or "baseline_s_S" in systems
    assert "baseline_eoq" in systems

    s_pol = get_system("baseline_s_S")
    assert isinstance(s_pol, SsPolicy)

    eoq_pol = get_system("baseline_eoq")
    assert isinstance(eoq_pol, EoqPolicy)


def test_ss_policy_decisions():
    """Verify (s, S) orders up to S when IP < s, and orders 0 when IP >= s."""
    policy = SsPolicy(config={"reorder_point": 150.0, "order_up_to": 350.0})
    policy.reset()

    # Case 1: IP = 100 (below s=150) -> Order = 350 - 100 = 250
    states_low = {
        e: EchelonState(
            echelon=e,
            step=0,
            inventory=100.0,
            backlog=0.0,
            pipeline=0.0,
            incoming_shipments=0.0,
        )
        for e in ECHELONS
    }
    orders_low = policy.decide_orders(states_low)
    for e in ECHELONS:
        assert orders_low[e] == pytest.approx(250.0)

    # Reset for Case 2
    policy.reset()
    # Case 2: IP = 200 (above s=150) -> Order = 0
    states_high = {
        e: EchelonState(
            echelon=e,
            step=0,
            inventory=200.0,
            backlog=0.0,
            pipeline=0.0,
            incoming_shipments=0.0,
        )
        for e in ECHELONS
    }
    orders_high = policy.decide_orders(states_high)
    for e in ECHELONS:
        assert orders_high[e] == 0.0


def test_ss_policy_calibration(dummy_scenario):
    """Verify (s, S) calibration derives expected statistical levels."""
    policy = SsPolicy()
    assert not policy.calibrated
    policy.calibrate(dummy_scenario)
    assert policy.calibrated

    params = policy.get_parameters()
    assert "s_levels" in params
    assert "S_levels" in params
    assert params["mu_d"] == pytest.approx(100.0)
    for e in ECHELONS:
        assert policy.S_levels[e] > policy.s_levels[e]


def test_eoq_policy_decisions():
    """Verify EOQ orders batch quantity when IP < ROP, and 0 when IP >= ROP."""
    policy = EoqPolicy(config={"reorder_point": 150.0, "eoq_quantity": 200.0})
    policy.reset()

    # Case 1: IP = 100 (below ROP=150, deficit=50 <= 200) -> Order = 1 batch = 200
    states_low = {
        e: EchelonState(
            echelon=e,
            step=0,
            inventory=100.0,
            backlog=0.0,
            pipeline=0.0,
            incoming_shipments=0.0,
        )
        for e in ECHELONS
    }
    orders_low = policy.decide_orders(states_low)
    for e in ECHELONS:
        assert orders_low[e] == pytest.approx(200.0)

    # Reset for Case 2
    policy.reset()
    # Case 2: IP = 200 (above ROP=150) -> Order = 0
    states_high = {
        e: EchelonState(
            echelon=e,
            step=0,
            inventory=200.0,
            backlog=0.0,
            pipeline=0.0,
            incoming_shipments=0.0,
        )
        for e in ECHELONS
    }
    orders_high = policy.decide_orders(states_high)
    for e in ECHELONS:
        assert orders_high[e] == 0.0


def test_eoq_policy_calibration(dummy_scenario):
    """Verify EOQ calibration calculates ROP and EOQ without error."""
    policy = EoqPolicy()
    assert not policy.calibrated
    policy.calibrate(dummy_scenario)
    assert policy.calibrated

    params = policy.get_parameters()
    assert "rop_levels" in params
    assert "eoq_levels" in params
    assert params["eoq"] > 0
    for e in ECHELONS:
        assert policy.rop_levels[e] > 0
        assert policy.eoq_levels[e] > 0


def test_policies_run_in_environment(dummy_scenario):
    """Verify all policies step through a simulated episode without violating conservation."""
    from sca.baselines.smoothed_out import SmoothedOrderUpToPolicy

    for PolicyCls in [SsPolicy, EoqPolicy, SmoothedOrderUpToPolicy]:
        pol = PolicyCls()
        pol.calibrate(dummy_scenario)
        env = BeerGameEnv(scenario=dummy_scenario, seed=123)
        states = env.reset(seed=123)
        pol.reset(scenario=dummy_scenario)

        for _ in range(len(dummy_scenario.demand)):
            orders = pol.decide_orders(states)
            states = env.step(orders)
            assert env.check_unit_conservation()


def test_smoothed_order_up_to_policy(dummy_scenario):
    """Verify System 2b policy can be instantiated from registry and runs."""
    from sca.systems.base import get_system
    pol = get_system("baseline_smoothed_order_up_to", config={"alpha": 0.4, "delta": 0.5})
    pol.calibrate(dummy_scenario)
    assert pol.calibrated
    assert pol.alpha == 0.4
    assert pol.delta == 0.5

    env = BeerGameEnv(scenario=dummy_scenario, seed=42)
    states = env.reset(seed=42)
    pol.reset(scenario=dummy_scenario)
    orders = pol.decide_orders(states)
    assert len(orders) == 4
    for e in orders:
        assert orders[e] >= 0.0

