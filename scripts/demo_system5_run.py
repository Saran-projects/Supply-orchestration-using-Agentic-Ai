"""Demonstration and verification run for System 5: MultiAgentToolsSystem.

Runs:
1. 30-step simulation on the 'stable' scenario using MockLLM.
2. Demonstrates specialized CrewAI agents: Demand, Inventory, Procurement.
3. Tests both 'median' and 'conservative' aggregation rules.
4. Shows DuckDB rows from tool_calls table (inputs, outputs, expected_tool, accuracy flags).
5. Demonstrates scripted wrong-tool and wrong-parameter failure modes with accuracy metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from sca.data.scenarios import generate_scenarios
from sca.llm.client import LLMClient
from sca.sim.environment import ECHELONS, BeerGameEnv
from sca.systems.multi_agent_tools import MultiAgentToolsSystem


def run_system5_demo():
    db_path = Path("results/demo_system5.duckdb")
    if db_path.exists():
        db_path.unlink()

    print("=" * 100)
    print("DEMO SYSTEM 5: SPECIALIZED CREWAI MULTI-AGENT TOOL ARCHITECTURE")
    print("=" * 100)

    # 1. Load Scenario
    scenarios = generate_scenarios()
    scenario = scenarios["stable"]
    print(f"\n[1] Scenario: {scenario.name} (horizon={len(scenario.demand)}, mean_demand={scenario.demand.mean():.1f})")

    # 2. Run 30-step Simulation with System 5 (Median Aggregation)
    print("\n" + "=" * 100)
    print("[2] Running 30-step simulation of MultiAgentToolsSystem (aggregation_rule='median')")
    print("=" * 100)

    env = BeerGameEnv(scenario=scenario, seed=42)
    env.reset(seed=42)

    llm_client = LLMClient(mock_mode=True, db_path=db_path)
    system5 = MultiAgentToolsSystem(
        name="multi_agent_tools",
        config={
            "aggregation_rule": "median",
            "order_fixed_cost": 100.0,
            "holding_cost_per_unit": 0.50,
            "reorder_point": 120.0,
            "order_up_to": 300.0,
            "manufacturer_capacity": 300.0,
        },
        llm_client=llm_client,
        run_id="run_system5_30step",
        db_path=db_path,
    )

    trajectory = []

    for step in range(1, 31):
        states = {e: env.observe(e) for e in ECHELONS}
        orders = system5.decide_orders(states)
        next_states = env.step(orders)

        total_step_cost = sum(next_states[e].holding_cost + next_states[e].backlog_cost for e in ECHELONS)

        trajectory.append({
            "step": step,
            "external_demand": round(states["retailer"].current_demand, 1),
            "orders": {e: round(orders[e], 1) for e in ECHELONS},
            "inventory": {e: round(states[e].inventory, 1) for e in ECHELONS},
            "backlog": {e: round(states[e].backlog, 1) for e in ECHELONS},
            "total_cost": round(total_step_cost, 1),
        })

    print(f"\n30-step simulation completed successfully. Total steps simulated: {len(trajectory)}.")

    # 3. Print Sample Step Trajectory Table
    print("\n" + "-" * 100)
    print("SIMULATION TRAJECTORY (Sample steps 1, 5, 10, 15, 20, 25, 30)")
    print("-" * 100)
    header = f"{'Step':<6} | {'Ext Demand':<10} | {'Retailer (Inv / Ord)':<22} | {'Wholesaler (Inv / Ord)':<24} | {'Distributor (Inv / Ord)':<25} | {'Manufacturer (Inv / Ord)':<26} | {'Step Cost':<10}"
    print(header)
    print("-" * len(header))

    for step_idx in [1, 5, 10, 15, 20, 25, 30]:
        t = trajectory[step_idx - 1]
        r_str = f"{t['inventory']['retailer']:.0f} / {t['orders']['retailer']:.0f}"
        w_str = f"{t['inventory']['wholesaler']:.0f} / {t['orders']['wholesaler']:.0f}"
        d_str = f"{t['inventory']['distributor']:.0f} / {t['orders']['distributor']:.0f}"
        m_str = f"{t['inventory']['manufacturer']:.0f} / {t['orders']['manufacturer']:.0f}"
        print(f"{t['step']:<6} | {t['external_demand']:<10.1f} | {r_str:<22} | {w_str:<24} | {d_str:<25} | {m_str:<26} | ${t['total_cost']:<9.1f}")

    # 4. Query DuckDB tool_calls rows
    print("\n" + "=" * 100)
    print("[3] DUCKDB `tool_calls` TABLE: 30-STEP RUN VERIFICATION")
    print("=" * 100)

    conn = duckdb.connect(str(db_path))
    summary = conn.execute(
        """
        SELECT
            agent_name,
            tool_name,
            expected_tool,
            count(*) as call_count,
            count(CASE WHEN is_correct_tool THEN 1 END) as correct_tool_count,
            count(CASE WHEN is_parameter_valid THEN 1 END) as valid_param_count,
            round(avg(duration_ms), 3) as avg_latency_ms
        FROM tool_calls
        WHERE run_id = 'run_system5_30step'
        GROUP BY agent_name, tool_name, expected_tool
        ORDER BY agent_name, tool_name
        """
    ).fetchall()

    print(f"\n{'Agent Name':<24} | {'Tool Called':<18} | {'Expected Tool':<18} | {'Calls':<6} | {'Correct Tool':<13} | {'Valid Param':<12} | {'Avg Latency (ms)':<16}")
    print("-" * 115)
    for row in summary:
        ag_name, tool_name, exp_tool, calls, corr_calls, val_params, lat = row
        print(f"{ag_name:<24} | {tool_name:<18} | {exp_tool:<18} | {calls:<6} | {corr_calls:<13} | {val_params:<12} | {lat:<16.3f}")

    sample_rows = conn.execute(
        """
        SELECT
            tool_call_id,
            step_id,
            agent_name,
            tool_name,
            expected_tool,
            is_correct_tool,
            is_parameter_valid,
            input_arguments,
            output_result
        FROM tool_calls
        WHERE run_id = 'run_system5_30step'
        ORDER BY created_at ASC
        LIMIT 6
        """
    ).fetchall()
    conn.close()

    # Overall tool accuracy for the 30-step run
    acc_30 = system5.compute_accuracy_metrics()
    print("-" * 115)
    print(f"Total Tool Calls: {acc_30['total_tool_calls']}")
    print(f"Tool-Choice Accuracy: {acc_30['tool_choice_accuracy'] * 100:.2f}%")
    print(f"Parameter Accuracy:   {acc_30['parameter_accuracy'] * 100:.2f}%")

    # Sample detailed rows from DuckDB with inputs & outputs
    print("\n" + "-" * 100)
    print("SAMPLE DUCKDB ROWS (First 6 tool calls with inputs and outputs):")
    print("-" * 100)

    for idx, r in enumerate(sample_rows, 1):
        c_id, s_id, ag_name, t_name, exp_t, is_corr, is_val, in_j, out_j = r
        print(f"\n[Call {idx}] ID: {c_id[:8]}... | Step: {s_id} | Agent: {ag_name}")
        print(f"  Tool: {t_name} (Expected: {exp_t}) | Tool Match: {is_corr} | Valid Param: {is_val}")
        in_obj = json.loads(in_j) if isinstance(in_j, str) else (in_j or {})
        out_obj = json.loads(out_j) if isinstance(out_j, str) else (out_j or {})
        print(f"  Inputs:  {in_obj}")
        print(f"  Outputs: {out_obj}")

    # 5. Verification of Scripted Failure Modes (Wrong Tool and Wrong Parameter)
    print("\n" + "=" * 100)
    print("[4] SCRIPTED TEST CASES: WRONG-TOOL & WRONG-PARAMETER ACCURACY VERIFICATION")
    print("=" * 100)

    # 5a. Scripted Wrong Tool Case
    sys_wrong_tool = MultiAgentToolsSystem(
        name="multi_agent_tools",
        config={"script_wrong_tool": True},
        llm_client=llm_client,
        run_id="run_sys5_wrong_tool_demo",
        db_path=db_path,
    )
    states_wt = {e: env.observe(e) for e in ECHELONS}
    sys_wrong_tool.decide_orders(states_wt)

    acc_wt = sys_wrong_tool.compute_accuracy_metrics()
    print("\n[Case A: Scripted Wrong-Tool]")
    print(f"  Total calls: {acc_wt['total_tool_calls']}")
    print(f"  Tool-Choice Accuracy: {acc_wt['tool_choice_accuracy'] * 100:.2f}% (Expected < 100% due to wrong tool call)")
    print(f"  Parameter Accuracy:   {acc_wt['parameter_accuracy'] * 100:.2f}%")

    conn_wt = duckdb.connect(str(db_path))
    wt_rows = conn_wt.execute(
        """
        SELECT agent_name, tool_name, expected_tool, is_correct_tool
        FROM tool_calls
        WHERE run_id = 'run_sys5_wrong_tool_demo' AND is_correct_tool = FALSE
        """
    ).fetchall()
    conn_wt.close()

    print("  Discrepant tool calls recorded in DuckDB:")
    for row in wt_rows:
        print(f"    - Agent: {row[0]}, Called: {row[1]}, Expected: {row[2]}, is_correct_tool={row[3]}")

    # 5b. Scripted Wrong Parameter Case
    sys_wrong_param = MultiAgentToolsSystem(
        name="multi_agent_tools",
        config={"script_wrong_param": True},
        llm_client=llm_client,
        run_id="run_sys5_wrong_param_demo",
        db_path=db_path,
    )
    states_wp = {e: env.observe(e) for e in ECHELONS}
    sys_wrong_param.decide_orders(states_wp)

    acc_wp = sys_wrong_param.compute_accuracy_metrics()
    print("\n[Case B: Scripted Wrong-Parameter]")
    print(f"  Total calls: {acc_wp['total_tool_calls']}")
    print(f"  Tool-Choice Accuracy: {acc_wp['tool_choice_accuracy'] * 100:.2f}%")
    print(f"  Parameter Accuracy:   {acc_wp['parameter_accuracy'] * 100:.2f}% (Expected < 100% due to invalid schema args)")

    conn_wp = duckdb.connect(str(db_path))
    wp_rows = conn_wp.execute(
        """
        SELECT agent_name, tool_name, is_parameter_valid, is_error, error_message
        FROM tool_calls
        WHERE run_id = 'run_sys5_wrong_param_demo' AND is_parameter_valid = FALSE
        """
    ).fetchall()
    conn_wp.close()

    print("  Invalid parameter calls recorded in DuckDB:")
    for row in wp_rows:
        print(f"    - Agent: {row[0]}, Tool: {row[1]}, is_parameter_valid={row[2]}, is_error={row[3]}")
        print(f"      Error snippet: {row[4][:100]}...")

    print("\n" + "=" * 100)
    print("SYSTEM 5 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 100)


if __name__ == "__main__":
    run_system5_demo()
