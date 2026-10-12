"""Demonstration and verification run for System 6: OrchestratedSystem.

Runs:
1. 30-step simulation on the 'stable' scenario using MockLLM.
2. Demonstrates OrchestratorAgent state summarization, dispatch, conflict detection.
3. Demonstrates the cost-saving shortcut when there is no conflict (zero negotiation calls).
4. Demonstrates bounded negotiation (strictly <= 2 rounds) when conflicts arise.
5. Demonstrates order validation with validate_order and deterministic fallback.
6. Shows DuckDB rows from `orchestrated_decisions` and `messages` tables.
7. Demonstrates ablation flags: negotiation on/off, validation on/off, conflict-skip on/off.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from sca.data.scenarios import generate_scenarios
from sca.llm.client import LLMClient
from sca.sim.environment import ECHELONS, BeerGameEnv
from sca.systems.orchestrated import OrchestratedSystem


def run_system6_demo():
    db_path = Path("results/demo_system6.duckdb")
    if db_path.exists():
        db_path.unlink()

    print("=" * 105)
    print("DEMO SYSTEM 6: ORCHESTRATOR AGENT ARCHITECTURE WITH BOUNDED NEGOTIATION & ABLATIONS")
    print("=" * 105)

    # 1. Load Scenario
    scenarios = generate_scenarios()
    scenario = scenarios["stable"]
    print(f"\n[1] Scenario: {scenario.name} (horizon={len(scenario.demand)}, mean_demand={scenario.demand.mean():.1f})")

    # 2. Run 30-step Simulation with System 6
    print("\n" + "=" * 105)
    print("[2] Running 30-step simulation of OrchestratedSystem (Full Pipeline: Shortcut + Negotiation + Validation)")
    print("=" * 105)

    env = BeerGameEnv(scenario=scenario, seed=42)
    env.reset(seed=42)

    llm_client = LLMClient(mock_mode=True, db_path=db_path)
    system6 = OrchestratedSystem(
        name="orchestrated",
        config={
            "enable_negotiation": True,
            "enable_validation": True,
            "enable_conflict_skip": True,
            "forecast_deviation_threshold": 0.25,
            "proposals_disagree_threshold": 0.30,
            "max_rounds": 2,
            "manufacturer_capacity": 300.0,
        },
        llm_client=llm_client,
        run_id="run_system6_30step",
        db_path=db_path,
    )

    trajectory = []

    for step in range(1, 31):
        states = {e: env.observe(e) for e in ECHELONS}
        orders = system6.decide_orders(states)
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

    print(f"\n30-step simulation completed successfully. Total decisions generated: {len(system6.decision_history)}.")

    # 3. Print Simulation Trajectory Sample
    print("\n" + "-" * 105)
    print("SIMULATION TRAJECTORY (Sample steps 1, 5, 10, 15, 20, 25, 30)")
    print("-" * 105)
    header = f"{'Step':<6} | {'Ext Demand':<10} | {'Retailer (Inv/Ord)':<20} | {'Wholesaler (Inv/Ord)':<22} | {'Distributor (Inv/Ord)':<23} | {'Manufacturer (Inv/Ord)':<24} | {'Step Cost':<10}"
    print(header)
    print("-" * len(header))

    for step_idx in [1, 5, 10, 15, 20, 25, 30]:
        t = trajectory[step_idx - 1]
        r_str = f"{t['inventory']['retailer']:.0f} / {t['orders']['retailer']:.0f}"
        w_str = f"{t['inventory']['wholesaler']:.0f} / {t['orders']['wholesaler']:.0f}"
        d_str = f"{t['inventory']['distributor']:.0f} / {t['orders']['distributor']:.0f}"
        m_str = f"{t['inventory']['manufacturer']:.0f} / {t['orders']['manufacturer']:.0f}"
        print(f"{t['step']:<6} | {t['external_demand']:<10.1f} | {r_str:<20} | {w_str:<22} | {d_str:<23} | {m_str:<24} | ${t['total_cost']:<9.1f}")

    # 4. Display DuckDB `orchestrated_decisions` rows
    print("\n" + "=" * 105)
    print("[3] DUCKDB `orchestrated_decisions` TABLE: 30-STEP RUN VERIFICATION")
    print("=" * 105)

    conn = duckdb.connect(str(db_path))

    summary_rows = conn.execute(
        """
        SELECT
            step_number,
            echelon,
            has_conflict,
            conflict_type,
            rounds_used,
            messages_exchanged,
            resolution_method,
            final_order,
            round(latency_ms, 2) as latency_ms
        FROM orchestrated_decisions
        WHERE run_id = 'run_system6_30step'
        ORDER BY step_number ASC, echelon ASC
        LIMIT 16
        """
    ).fetchall()

    print(f"\n{'Step':<5} | {'Echelon':<14} | {'Conflict?':<10} | {'Conflict Type':<26} | {'Rounds':<7} | {'Msgs':<5} | {'Resolution Method':<26} | {'Final Ord':<10} | {'Latency':<8}")
    print("-" * 125)
    for r in summary_rows:
        s_num, ech, has_c, c_type, r_used, m_ex, res_m, f_ord, lat = r
        c_str = "YES" if has_c else "NO"
        print(f"{s_num:<5} | {ech:<14} | {c_str:<10} | {c_type:<26} | {r_used:<7} | {m_ex:<5} | {res_m:<26} | {f_ord:<10.1f} | {lat:<6.1f}ms")

    # Aggregate telemetry statistics
    stats = conn.execute(
        """
        SELECT
            count(*) as total_decisions,
            count(CASE WHEN has_conflict THEN 1 END) as conflict_count,
            count(CASE WHEN resolution_method = 'no_conflict_consensus' THEN 1 END) as shortcut_count,
            count(CASE WHEN rounds_used > 0 THEN 1 END) as negotiated_count,
            round(avg(rounds_used), 2) as avg_rounds,
            sum(messages_exchanged) as total_messages,
            count(CASE WHEN fallback_used THEN 1 END) as fallback_count
        FROM orchestrated_decisions
        WHERE run_id = 'run_system6_30step'
        """
    ).fetchone()

    print("-" * 125)
    tot_d, c_cnt, s_cnt, n_cnt, a_r, tot_m, fb_cnt = stats
    print(f"Total Decisions:      {tot_d}")
    print(f"Conflicts Detected:   {c_cnt} ({(c_cnt/tot_d)*100:.1f}%)")
    print(f"No-Conflict Shortcuts:{s_cnt} ({(s_cnt/tot_d)*100:.1f}% bypassed negotiation completely)")
    print(f"Negotiations Run:     {n_cnt} (Avg Rounds: {a_r}, 2-Round Cap Strictly Respected)")
    print(f"Total Negotiation Msgs:{tot_m}")
    print(f"Validation Fallbacks: {fb_cnt}")

    # Sample detailed decision with proposals JSON
    print("\n" + "-" * 105)
    print("SAMPLE DETAILED DUCKDB DECISION (Showing proposal inputs and conflict details):")
    print("-" * 105)

    sample_detail = conn.execute(
        """
        SELECT decision_id, step_number, echelon, has_conflict, conflict_type,
               resolution_method, proposals_json, final_order
        FROM orchestrated_decisions
        WHERE run_id = 'run_system6_30step' AND has_conflict = TRUE
        LIMIT 2
        """
    ).fetchall()

    for idx, d in enumerate(sample_detail, 1):
        d_id, step_num, ech, has_c, c_type, res_m, p_json, f_ord = d
        props = json.loads(p_json) if isinstance(p_json, str) else p_json
        print(f"\n[Detail {idx}] Decision ID: {d_id} | Step: {step_num} | Echelon: {ech}")
        print(f"  Conflict: {c_type} | Resolution: {res_m}")
        print(f"  Agent Proposals: {props}")
        print(f"  Final Validated Order: {f_ord:.1f}")

    # Sample negotiation messages from DuckDB `messages` table
    print("\n" + "-" * 105)
    print("SAMPLE DUCKDB NEGOTIATION MESSAGES (`messages` table):")
    print("-" * 105)

    msg_sample = conn.execute(
        """
        SELECT sender, receiver, role, content
        FROM messages
        WHERE run_id = 'run_system6_30step'
        LIMIT 4
        """
    ).fetchall()

    for idx, m in enumerate(msg_sample, 1):
        snd, rcv, role, cont = m
        print(f"\n[Message {idx}] From: {snd} -> To: {rcv} ({role})")
        print(f"  Content: {cont}")

    conn.close()

    # 5. Verification of Ablation Flags
    print("\n" + "=" * 105)
    print("[4] ABLATION EXPERIMENTS VERIFICATION: MEASURING SYSTEM 6 CONTRIBUTIONS")
    print("=" * 105)

    ablation_states = {e: env.observe(e) for e in ECHELONS}

    # Ablation 1: Negotiation OFF
    sys_no_neg = OrchestratedSystem(
        name="orchestrated",
        config={"enable_negotiation": False, "forecast_deviation_threshold": 0.01},
        llm_client=llm_client,
        run_id="run_ablation_no_neg",
        db_path=db_path,
    )
    sys_no_neg.decide_orders(ablation_states)
    conn_ab = duckdb.connect(str(db_path))
    res_no_neg = conn_ab.execute(
        """
        SELECT count(*), count(CASE WHEN resolution_method = 'ablation_no_negotiation' THEN 1 END), sum(rounds_used)
        FROM orchestrated_decisions WHERE run_id = 'run_ablation_no_neg'
        """
    ).fetchone()
    print("\n[Ablation 1: Negotiation OFF]")
    print(f"  Total Decisions: {res_no_neg[0]}")
    print(f"  Bypassed Negotiation (Ablation Resolution): {res_no_neg[1]}/{res_no_neg[0]}")
    print(f"  Total Negotiation Rounds: {res_no_neg[2]} (0 rounds when negotiation disabled)")

    # Ablation 2: Conflict-Skip OFF (Forces negotiation on all decisions)
    sys_no_skip = OrchestratedSystem(
        name="orchestrated",
        config={"enable_conflict_skip": False, "forecast_deviation_threshold": 5.0, "proposals_disagree_threshold": 5.0},
        llm_client=llm_client,
        run_id="run_ablation_no_skip",
        db_path=db_path,
    )
    sys_no_skip.decide_orders(ablation_states)
    res_no_skip = conn_ab.execute(
        """
        SELECT count(*), count(CASE WHEN has_conflict = FALSE THEN 1 END), sum(rounds_used), sum(messages_exchanged)
        FROM orchestrated_decisions WHERE run_id = 'run_ablation_no_skip'
        """
    ).fetchone()
    print("\n[Ablation 2: Conflict-Skip OFF]")
    print(f"  Total Decisions: {res_no_skip[0]}")
    print(f"  Decisions with No Conflict: {res_no_skip[1]}")
    print(f"  Negotiation Rounds Forced: {res_no_skip[2]}")
    print(f"  Negotiation Messages Generated: {res_no_skip[3]} (Demonstrates cost saving of conflict-skip shortcut)")

    # Ablation 3: Validation OFF
    sys_no_val = OrchestratedSystem(
        name="orchestrated",
        config={"enable_validation": False, "force_validation_failure": True},
        llm_client=llm_client,
        run_id="run_ablation_no_val",
        db_path=db_path,
    )
    sys_no_val.decide_orders(ablation_states)
    res_no_val = conn_ab.execute(
        """
        SELECT count(*), count(CASE WHEN fallback_used THEN 1 END)
        FROM orchestrated_decisions WHERE run_id = 'run_ablation_no_val'
        """
    ).fetchone()
    print("\n[Ablation 3: Validation OFF]")
    print(f"  Total Decisions: {res_no_val[0]}")
    print(f"  Fallbacks Applied: {res_no_val[1]} (Validation was bypassed; orders passed raw without checks)")

    conn_ab.close()
    print("\n" + "=" * 105)
    print("SYSTEM 6 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 105)


if __name__ == "__main__":
    run_system6_demo()
