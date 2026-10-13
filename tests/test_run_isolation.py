"""Run isolation and state leak prevention tests.

Verifies that for every registered system, executing a simulation on a target
(scenario, seed) in isolation produces identical metrics when executed as the
last item in a batch of runs using the same system instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from sca.data.scenarios import generate_scenarios
from sca.sim.environment import BeerGameEnv
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import get_system

REGISTERED_SYSTEM_IDS = [
    "baseline_constant",
    "baseline_s_S",
    "baseline_eoq",
    "baseline_smoothed_order_up_to",
    "baseline_base_stock",
    "baseline_oracle",
    "single_agent",
    "multi_agent_info",
    "multi_agent_tools",
    "orchestrated",
]

TARGET_SEED = 104
BATCH_SEEDS = [100, 101, 102, 103, 104]
TEST_STEPS = 10
WARMUP = 2


@pytest.fixture(scope="module")
def shared_scenario():
    """Load standard seasonal scenario for isolation verification."""
    scenarios = generate_scenarios()
    return scenarios["seasonal"]


@pytest.mark.parametrize("system_id", REGISTERED_SYSTEM_IDS)
def test_system_run_isolation(system_id: str, shared_scenario):
    """Assert metrics on target seed match whether run in isolation or after preceding seeds."""
    sc = shared_scenario
    sys_cfg = {"db_path": False}

    # 1. Run target seed in isolation
    system_iso = get_system(system_id, config=sys_cfg)
    system_iso.calibrate(sc, config=sys_cfg)
    system_iso.reset(sc)

    env_iso = BeerGameEnv(scenario=sc, seed=TARGET_SEED, steady_state_init=True)
    states_iso = env_iso.reset(seed=TARGET_SEED)

    for _ in range(TEST_STEPS):
        orders = system_iso.decide_orders(states_iso)
        states_iso = env_iso.step(orders)

    metrics_iso = compute_simulation_metrics(env_iso, warmup_steps=WARMUP)

    # 2. Re-run as the last item of a batch of seeds using ONE reused system instance
    system_batch = get_system(system_id, config=sys_cfg)
    system_batch.calibrate(sc, config=sys_cfg)

    metrics_batch = None
    for seed in BATCH_SEEDS:
        # Crucial requirement: system.reset() and env.reset() called per run
        system_batch.reset(sc)
        env_batch = BeerGameEnv(scenario=sc, seed=seed, steady_state_init=True)
        states_batch = env_batch.reset(seed=seed)

        for _ in range(TEST_STEPS):
            orders = system_batch.decide_orders(states_batch)
            states_batch = env_batch.step(orders)

        if seed == TARGET_SEED:
            metrics_batch = compute_simulation_metrics(env_batch, warmup_steps=WARMUP)

    assert metrics_batch is not None, "Batch run did not evaluate target seed"

    # 3. Assert exact metrics equivalence
    assert metrics_iso.total_cost == pytest.approx(metrics_batch.total_cost, rel=1e-5), (
        f"Cost mismatch for {system_id}: isolated={metrics_iso.total_cost}, batch={metrics_batch.total_cost}"
    )
    assert metrics_iso.fill_rate["retailer"] == pytest.approx(
        metrics_batch.fill_rate["retailer"], rel=1e-5
    ), f"Fill rate mismatch for {system_id}"
    assert metrics_iso.service_level["retailer"] == pytest.approx(
        metrics_batch.service_level["retailer"], rel=1e-5
    ), f"Service level mismatch for {system_id}"

    # Bullwhip check (handling possible NaN)
    e2e_iso = metrics_iso.end_to_end_bullwhip
    e2e_bat = metrics_batch.end_to_end_bullwhip
    if np.isnan(e2e_iso):
        assert np.isnan(e2e_bat), f"Bullwhip NaN mismatch for {system_id}"
    else:
        assert e2e_iso == pytest.approx(e2e_bat, rel=1e-5), f"End-to-end bullwhip mismatch for {system_id}"
