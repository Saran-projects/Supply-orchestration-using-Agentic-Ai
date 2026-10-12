"""Run System 3 (single_agent) and System 4 (multi_agent_info) on a 30-step scenario with MockLLM."""

from pathlib import Path

import duckdb

from sca.data.scenarios import generate_scenarios
from sca.llm.client import LLMClient
from sca.sim.environment import BeerGameEnv
from sca.systems.multi_agent_info import MultiAgentInfoSystem
from sca.systems.single_agent import SingleAgentSystem


def run_demo():
    db_path = Path("results/demo_systems.duckdb")
    if db_path.exists():
        db_path.unlink()

    print("=" * 80)
    print("1. Loading Scenario: stable")
    print("=" * 80)
    scenarios = generate_scenarios()
    scenario = scenarios["stable"]
    print(f"Scenario Name: {scenario.name}")
    print(f"Scenario Length: {len(scenario.demand)} steps, Mean Demand: {scenario.demand.mean():.2f}")

    # -------------------------------------------------------------
    # 2. Run System 3: SingleAgentSystem
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. Running System 3: SingleAgentSystem for 30 steps with MockLLM")
    print("=" * 80)
    env_s3 = BeerGameEnv(scenario=scenario, seed=42)
    env_s3.reset(seed=42)

    client_s3 = LLMClient(mock_mode=True, db_path=db_path)
    # Enable scripted policy with failure rates to exercise retries, fallbacks, and metrics
    client_s3.mock.configure_scripted_policy(
        enabled=True,
        seed=101,
        failure_rate_malformed=0.12,
        failure_rate_out_of_range=0.08,
        base_order=100.0,
        noise_std=12.0,
        max_order=300.0,
    )

    sys3 = SingleAgentSystem(
        name="single_agent",
        llm_client=client_s3,
        db_path=db_path,
        run_id="run_system3_demo",
        config={"seed": 101},
    )

    for _step_idx in range(30):
        states = {e: env_s3.observe(e) for e in ("retailer", "wholesaler", "distributor", "manufacturer")}
        orders = sys3.decide_orders(states)
        env_s3.step(orders)

    print("System 3 simulation completed (30 steps, 120 decisions logged).")

    # -------------------------------------------------------------
    # 3. Run System 4: MultiAgentInfoSystem
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. Running System 4: MultiAgentInfoSystem for 30 steps with MockLLM")
    print("=" * 80)
    env_s4 = BeerGameEnv(scenario=scenario, seed=42)
    env_s4.reset(seed=42)

    client_s4 = LLMClient(mock_mode=True, db_path=db_path)
    client_s4.mock.configure_scripted_policy(
        enabled=True,
        seed=202,
        failure_rate_malformed=0.12,
        failure_rate_out_of_range=0.08,
        base_order=100.0,
        noise_std=12.0,
        max_order=300.0,
    )

    sys4 = MultiAgentInfoSystem(
        name="multi_agent_info",
        llm_client=client_s4,
        db_path=db_path,
        run_id="run_system4_demo",
        config={"seed": 202},
    )

    for _step_idx in range(30):
        states = {e: env_s4.observe(e) for e in ("retailer", "wholesaler", "distributor", "manufacturer")}
        orders = sys4.decide_orders(states)
        env_s4.step(orders)

    print("System 4 simulation completed (30 steps, 120 decisions logged).")

    # -------------------------------------------------------------
    # 4. Display DuckDB Telemetry Rows
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. DuckDB Agent Calls Summary & Sample Rows")
    print("=" * 80)
    conn = duckdb.connect(str(db_path))

    summary_df = conn.execute(
        """
        SELECT system_name,
               count(*) AS total_decisions,
               sum(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) AS retried_decisions,
               sum(CASE WHEN fallback THEN 1 ELSE 0 END) AS fallback_decisions,
               round(avg(latency_ms), 2) AS avg_latency_ms,
               round(avg(prompt_tokens), 1) AS avg_in_tokens,
               round(avg(completion_tokens), 1) AS avg_out_tokens
        FROM agent_calls
        GROUP BY system_name
        ORDER BY system_name
        """
    ).fetchdf()
    print("\n--- Aggregate Decision Statistics by System ---")
    print(summary_df.to_string(index=False))

    # Sample rows across steps for both systems
    sample_df = conn.execute(
        """
        SELECT system_name AS system,
               echelon,
               step_number AS step,
               prompt_version,
               prompt_tokens AS in_tok,
               completion_tokens AS out_tok,
               round(latency_ms, 1) AS latency_ms,
               retry_count AS retries,
               fallback,
               round(order_quantity, 1) AS "order"
        FROM agent_calls
        WHERE step_number IN (0, 1, 2, 14, 29)
        ORDER BY system_name, step_number, echelon
        """
    ).fetchdf()

    print("\n--- Sample Decision Rows (Steps 0, 1, 2, 14, 29) ---")
    print(sample_df.to_string(index=False))

    fallback_rows = conn.execute(
        """
        SELECT system_name, echelon, step_number, retry_count, fallback, order_quantity
        FROM agent_calls
        WHERE fallback = true
        """
    ).fetchall()
    print(f"\n--- Total Fallbacks Counted in agent_calls: {len(fallback_rows)} ---")
    for r in fallback_rows:
        print(f"  System={r[0]}, Echelon={r[1]}, Step={r[2]}, Retries={r[3]}, Fallback={r[4]}, FinalOrder={r[5]:.1f}")

    metrics_df = conn.execute(
        """
        SELECT run_id, step_number, metric_name, metric_value, metadata
        FROM metrics
        WHERE metric_name = 'fallback_count'
        """
    ).fetchdf()
    print(f"\n--- DuckDB metrics Table Fallback Records: {len(metrics_df)} ---")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    run_demo()
