"""Pipeline Benchmark Execution for Systems 3 to 6 using MockLLM.

Simulates:
- System 3: SingleAgentSystem
- System 4: MultiAgentInfoSystem
- System 5: MultiAgentToolsSystem
- System 6: OrchestratedSystem
on the 'stable' scenario for 15 steps each using MockLLM.

Computes and logs all behavioural metrics into DuckDB:
1. Tool-choice accuracy
2. Parameter accuracy
3. Instruction-following rate
4. Groundedness (programmatic regex vs true state)
5. Fallback and error rate
6. Negotiation efficiency (rounds, messages, conflict share)
7. Tokens, cost in USD (from models.yaml; cache hits = $0), latency
8. Persistence to DuckDB `metrics` and `behavioural_metrics` tables
9. Formatted summary benchmark table
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "mock-key-for-offline-simulation"

from sca.data.scenarios import generate_scenarios
from sca.db import get_duckdb_connection
from sca.eval.behaviour import (
    BehaviouralMetrics,
    compute_behavioural_metrics,
    generate_behaviour_summary_table,
)
from sca.llm.client import LLMClient
from sca.sim.environment import ECHELONS, BeerGameEnv
from sca.systems.multi_agent_info import MultiAgentInfoSystem
from sca.systems.multi_agent_tools import MultiAgentToolsSystem
from sca.systems.orchestrated import OrchestratedSystem
from sca.systems.single_agent import SingleAgentSystem


def _log_step_to_db(
    conn,
    run_id: str,
    step_num: int,
    env: BeerGameEnv,
    orders: Dict[str, float],
) -> None:
    """Helper to log simulation step state to DuckDB steps table."""
    info_dict = {
        e: {
            "inventory": float(env.inventory[e]),
            "backlog": float(env.backlog[e]),
            "pipeline": float(sum(s.quantity for s in env.pipelines[e])),
            "incoming": float(env.incoming_shipments[e]),
            "last_order": float(env.last_order[e]),
            "cost": float(env.step_costs[e]),
            "demand": float(env.external_demand[step_num]) if e == "retailer" and step_num < len(env.external_demand) else float(orders.get(e, 100.0)),
        }
        for e in ECHELONS
    }

    conn.execute(
        """
        INSERT OR REPLACE INTO steps (
            step_id, run_id, step_number, inventory_level, backlog_units, demand_units, action_taken, info
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            f"{run_id}_s{step_num}",
            run_id,
            step_num,
            int(round(env.total_inventory)),
            int(round(env.total_backlog)),
            int(round(env.external_demand[step_num])) if step_num < len(env.external_demand) else 0,
            json.dumps({e: float(orders.get(e, 0.0)) for e in orders}),
            json.dumps(info_dict),
        ],
    )


def run_behaviour_benchmark(steps_count: int = 15) -> List[BehaviouralMetrics]:
    """Execute end-to-end simulation of Systems 3-6 and evaluate behavioural metrics."""
    db_path = Path("results/behaviour_benchmark.duckdb")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    scenarios = generate_scenarios()
    scenario = scenarios["stable"]
    seed = 42

    print("=" * 110)
    print("SCA BEHAVIOURAL BENCHMARK: SYSTEMS 3 TO 6 (MockLLM)")
    print(f"Scenario: {scenario.name} | Steps per system: {steps_count} | Seed: {seed}")
    print("=" * 110)

    evaluated_metrics: List[BehaviouralMetrics] = []

    # -----------------------------------------------------------------------
    # System 3: SingleAgentSystem
    # -----------------------------------------------------------------------
    print("\n[1/4] Running System 3: SingleAgentSystem...")
    run_id_s3 = "system3_single_agent_stable_seed_42"
    conn = get_duckdb_connection(db_path)
    conn.execute(
        "INSERT INTO runs (run_id, experiment_name, status, config) VALUES (?, 'benchmark', 'completed', ?)",
        [run_id_s3, json.dumps({"system": "single_agent", "scenario": "stable", "model": "mock", "seed": seed})],
    )
    conn.close()

    client_s3 = LLMClient(mock_mode=True, db_path=db_path)
    client_s3.mock.configure_scripted_policy(
        enabled=True,
        seed=101,
        failure_rate_malformed=0.05,
        failure_rate_out_of_range=0.05,
        base_order=100.0,
        noise_std=10.0,
        max_order=300.0,
    )
    env_s3 = BeerGameEnv(scenario=scenario, seed=seed)
    env_s3.reset(seed=seed)
    sys3 = SingleAgentSystem(
        name="single_agent",
        llm_client=client_s3,
        db_path=db_path,
        run_id=run_id_s3,
        config={"seed": seed, "system": "single_agent", "scenario": "stable", "model": "mock"},
    )

    conn = get_duckdb_connection(db_path)
    for t in range(steps_count):
        states = {e: env_s3.observe(e) for e in ECHELONS}
        orders = sys3.decide_orders(states)
        _log_step_to_db(conn, run_id_s3, t, env_s3, orders)
        env_s3.step(orders)
    conn.close()

    m3 = compute_behavioural_metrics(db_path, run_id_s3, write_to_db=True)
    evaluated_metrics.append(m3)
    print("  -> System 3 completed and evaluated.")

    # -----------------------------------------------------------------------
    # System 4: MultiAgentInfoSystem
    # -----------------------------------------------------------------------
    print("\n[2/4] Running System 4: MultiAgentInfoSystem...")
    run_id_s4 = "system4_multi_agent_info_stable_seed_42"
    conn = get_duckdb_connection(db_path)
    conn.execute(
        "INSERT INTO runs (run_id, experiment_name, status, config) VALUES (?, 'benchmark', 'completed', ?)",
        [run_id_s4, json.dumps({"system": "multi_agent_info", "scenario": "stable", "model": "mock", "seed": seed})],
    )
    conn.close()

    client_s4 = LLMClient(mock_mode=True, db_path=db_path)
    client_s4.mock.configure_scripted_policy(
        enabled=True,
        seed=202,
        failure_rate_malformed=0.05,
        failure_rate_out_of_range=0.05,
        base_order=100.0,
        noise_std=10.0,
        max_order=300.0,
    )
    env_s4 = BeerGameEnv(scenario=scenario, seed=seed)
    env_s4.reset(seed=seed)
    sys4 = MultiAgentInfoSystem(
        name="multi_agent_info",
        llm_client=client_s4,
        db_path=db_path,
        run_id=run_id_s4,
        config={"seed": seed, "system": "multi_agent_info", "scenario": "stable", "model": "mock"},
    )

    conn = get_duckdb_connection(db_path)
    for t in range(steps_count):
        states = {e: env_s4.observe(e) for e in ECHELONS}
        orders = sys4.decide_orders(states)
        _log_step_to_db(conn, run_id_s4, t, env_s4, orders)
        env_s4.step(orders)
    conn.close()

    m4 = compute_behavioural_metrics(db_path, run_id_s4, write_to_db=True)
    evaluated_metrics.append(m4)
    print("  -> System 4 completed and evaluated.")

    # -----------------------------------------------------------------------
    # System 5: MultiAgentToolsSystem
    # -----------------------------------------------------------------------
    print("\n[3/4] Running System 5: MultiAgentToolsSystem...")
    run_id_s5 = "system5_multi_agent_tools_stable_seed_42"
    conn = get_duckdb_connection(db_path)
    conn.execute(
        "INSERT INTO runs (run_id, experiment_name, status, config) VALUES (?, 'benchmark', 'completed', ?)",
        [run_id_s5, json.dumps({"system": "multi_agent_tools", "scenario": "stable", "model": "mock", "seed": seed})],
    )
    conn.close()

    client_s5 = LLMClient(mock_mode=True, db_path=db_path)
    env_s5 = BeerGameEnv(scenario=scenario, seed=seed)
    env_s5.reset(seed=seed)
    sys5 = MultiAgentToolsSystem(
        name="multi_agent_tools",
        config={
            "seed": seed,
            "system": "multi_agent_tools",
            "scenario": "stable",
            "model": "mock",
            "aggregation_rule": "median",
            "order_fixed_cost": 100.0,
            "holding_cost_per_unit": 0.50,
            "reorder_point": 120.0,
            "order_up_to": 300.0,
            "manufacturer_capacity": 300.0,
        },
        llm_client=client_s5,
        run_id=run_id_s5,
        db_path=db_path,
    )

    conn = get_duckdb_connection(db_path)
    for t in range(steps_count):
        states = {e: env_s5.observe(e) for e in ECHELONS}
        orders = sys5.decide_orders(states)
        _log_step_to_db(conn, run_id_s5, t, env_s5, orders)
        env_s5.step(orders)
    conn.close()

    m5 = compute_behavioural_metrics(db_path, run_id_s5, write_to_db=True)
    evaluated_metrics.append(m5)
    print("  -> System 5 completed and evaluated.")

    # -----------------------------------------------------------------------
    # System 6: OrchestratedSystem
    # -----------------------------------------------------------------------
    print("\n[4/4] Running System 6: OrchestratedSystem...")
    run_id_s6 = "system6_orchestrated_stable_seed_42"
    conn = get_duckdb_connection(db_path)
    conn.execute(
        "INSERT INTO runs (run_id, experiment_name, status, config) VALUES (?, 'benchmark', 'completed', ?)",
        [run_id_s6, json.dumps({"system": "orchestrated", "scenario": "stable", "model": "mock", "seed": seed})],
    )
    conn.close()

    client_s6 = LLMClient(mock_mode=True, db_path=db_path)
    env_s6 = BeerGameEnv(scenario=scenario, seed=seed)
    env_s6.reset(seed=seed)
    sys6 = OrchestratedSystem(
        name="orchestrated",
        config={
            "seed": seed,
            "system": "orchestrated",
            "scenario": "stable",
            "model": "mock",
            "enable_negotiation": True,
            "enable_validation": True,
            "enable_conflict_skip": True,
            "forecast_deviation_threshold": 0.25,
            "proposals_disagree_threshold": 0.30,
            "max_rounds": 2,
            "manufacturer_capacity": 300.0,
        },
        llm_client=client_s6,
        run_id=run_id_s6,
        db_path=db_path,
    )

    conn = get_duckdb_connection(db_path)
    for t in range(steps_count):
        states = {e: env_s6.observe(e) for e in ECHELONS}
        orders = sys6.decide_orders(states)
        _log_step_to_db(conn, run_id_s6, t, env_s6, orders)
        env_s6.step(orders)
    conn.close()

    m6 = compute_behavioural_metrics(db_path, run_id_s6, write_to_db=True)
    evaluated_metrics.append(m6)
    print("  -> System 6 completed and evaluated.")

    # -----------------------------------------------------------------------
    # Print Benchmark Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("BEHAVIOURAL EVALUATION SUMMARY TABLE (DuckDB Persisted)")
    print("=" * 110)
    table_str = generate_behaviour_summary_table(evaluated_metrics)
    print(table_str)
    print("=" * 110)

    # Verify metrics table persistence
    conn = get_duckdb_connection(db_path, read_only=True)
    total_metrics_rows = conn.execute("SELECT count(*) FROM metrics WHERE category = 'behaviour'").fetchone()[0]
    total_bm_rows = conn.execute("SELECT count(*) FROM behavioural_metrics").fetchone()[0]
    conn.close()
    print(f"\nVerification: {total_bm_rows} runs recorded in behavioural_metrics, {total_metrics_rows} granular metric rows in metrics table.")

    return evaluated_metrics


if __name__ == "__main__":
    run_behaviour_benchmark(steps_count=15)
