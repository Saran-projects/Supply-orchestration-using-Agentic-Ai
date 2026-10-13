"""Shock Scenario Experiment Runner for System 3 and System 6 on glm-isoquant.

Requirements:
1. Dry-run first and print projected cost; abort if it exceeds 0.25 USD.
2. Confirm paired baselines and mock_llm runs exist for shock seed 42.
3. Run systems 3 and 6 on glm-isoquant for shock scenario, seed 42, 30 steps, prompt v2, cache on.
4. Enforce hard session cap of 0.35 USD total for this session; stop cleanly if reached.
5. Tag data_source = glm_real on runs, steps, and agent_calls.
6. Print run summary and logged spend reconciliation.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import duckdb

from sca.config import load_models_config
from sca.data.scenarios import generate_scenarios
from sca.db import get_duckdb_connection
from sca.llm.client import LLMClient
from sca.llm.reconcile import print_reconcile_report
from sca.sim.environment import ECHELONS, BeerGameEnv
from sca.systems.orchestrated import OrchestratedSystem
from sca.systems.single_agent import SingleAgentSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DRY_RUN_ABORT_THRESHOLD_USD = 0.25
SESSION_HARD_CAP_USD = 0.35
STEPS_COUNT = 30
SEED = 42
SCENARIO_NAME = "shock"
PROMPT_VERSION = "v2"


def run_dry_run_projection() -> float:
    """Compute and display dry-run cost projection. Abort if exceeding threshold."""
    models_cfg = load_models_config()
    glm_entry = models_cfg.get_model("glm-isoquant")
    if not glm_entry:
        raise ValueError("glm-isoquant model not configured in configs/models.yaml")

    p_in = float(glm_entry.price_per_1k_input)
    p_out = float(glm_entry.price_per_1k_output)

    # Empirical token averages per decision from docs/budget.md
    s3_decisions = STEPS_COUNT * 4  # 120
    s6_decisions = STEPS_COUNT * 4  # 120

    s3_avg_in = 771.0
    s3_avg_out = 1039.7
    s6_avg_in = 1176.2
    s6_avg_out = 1937.2

    s3_cost_per_dec = (s3_avg_in / 1000.0) * p_in + (s3_avg_out / 1000.0) * p_out
    s6_cost_per_dec = (s6_avg_in / 1000.0) * p_in + (s6_avg_out / 1000.0) * p_out

    s3_projected_usd = s3_decisions * s3_cost_per_dec
    s6_projected_usd = s6_decisions * s6_cost_per_dec
    total_projected_usd = s3_projected_usd + s6_projected_usd

    print("\n" + "=" * 90)
    print(" 1. DRY-RUN COST PROJECTION (glm-isoquant, shock scenario, 30 steps, seed 42, prompt v2)")
    print("=" * 90)
    print("Rate Card (models.yaml 2.2x placeholder):")
    print(f"  Input Price:  ${p_in:.6f} / 1k tokens (${p_in * 1000:.3f} / 1M)")
    print(f"  Output Price: ${p_out:.6f} / 1k tokens (${p_out * 1000:.3f} / 1M)")
    print("\nSystem 3 (Single Agent):")
    print(f"  Decisions:          {s3_decisions} (30 steps x 4 echelons)")
    print(f"  Avg In / Out Tok:   {s3_avg_in:.1f} / {s3_avg_out:.1f}")
    print(f"  Est. Cost / Dec:    ${s3_cost_per_dec:.6f} USD")
    print(f"  Subtotal Projected: ${s3_projected_usd:.5f} USD")
    print("\nSystem 6 (Orchestrated):")
    print(f"  Decisions:          {s6_decisions} (30 steps x 4 echelons)")
    print(f"  Avg In / Out Tok:   {s6_avg_in:.1f} / {s6_avg_out:.1f}")
    print(f"  Est. Cost / Dec:    ${s6_cost_per_dec:.6f} USD")
    print(f"  Subtotal Projected: ${s6_projected_usd:.5f} USD")
    print("-" * 90)
    print(f"TOTAL PROJECTED COST: ${total_projected_usd:.5f} USD")
    print(f"ABORT THRESHOLD:      ${DRY_RUN_ABORT_THRESHOLD_USD:.2f} USD")
    print(f"SESSION HARD CAP:     ${SESSION_HARD_CAP_USD:.2f} USD")

    if total_projected_usd > DRY_RUN_ABORT_THRESHOLD_USD:
        print(f"\n[ABORT] Total projected cost (${total_projected_usd:.5f} USD) exceeds ${DRY_RUN_ABORT_THRESHOLD_USD:.2f} USD threshold! Aborting.")
        sys.exit(1)

    print(f"\n[APPROVAL] Projected cost (${total_projected_usd:.5f} USD) <= ${DRY_RUN_ABORT_THRESHOLD_USD:.2f} USD limit. Proceeding to execution.")
    print("=" * 90)
    return total_projected_usd


def ensure_paired_runs(db_path: Path, scenario: Any) -> None:
    """Confirm or execute paired baseline and mock_llm runs for shock seed 42."""
    print("\n" + "=" * 90)
    print(" 2. VERIFYING PAIRED RUNS FOR SHOCK SEED 42")
    print("=" * 90)

    conn = get_duckdb_connection(db_path)
    conn.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS data_source VARCHAR")
    conn.execute("ALTER TABLE agent_calls ADD COLUMN IF NOT EXISTS data_source VARCHAR")
    conn.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS data_source VARCHAR")

    # 1. Classical Heuristic Baselines
    # Copy from sca_metrics.duckdb if present
    src_db = Path("results/sca_metrics.duckdb")
    if src_db.exists():
        con_src = duckdb.connect(str(src_db), read_only=True)
        for b_run in ["baseline_s_S_shock_seed_42", "baseline_eoq_shock_seed_42"]:
            exists = conn.execute("SELECT count(*) FROM runs WHERE run_id = ?", [b_run]).fetchone()[0]
            if not exists:
                r = con_src.execute("SELECT run_id, experiment_name, status, config, metadata FROM runs WHERE run_id = ?", [b_run]).fetchone()
                if r:
                    conn.execute(
                        "INSERT INTO runs (run_id, experiment_name, status, config, metadata, data_source) VALUES (?, ?, ?, ?, ?, 'baseline')",
                        [r[0], r[1], r[2], r[3], r[4]],
                    )
                    steps = con_src.execute(
                        "SELECT step_id, run_id, step_number, inventory_level, backlog_units, demand_units, action_taken, reward, info FROM steps WHERE run_id = ?",
                        [b_run],
                    ).fetchall()
                    for st in steps:
                        conn.execute(
                            "INSERT INTO steps (step_id, run_id, step_number, inventory_level, backlog_units, demand_units, action_taken, reward, info, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'baseline')",
                            list(st) + ["baseline"],
                        )
        con_src.close()
    conn.close()

    # 2. Mock LLM runs for System 3 and System 6 (shock, seed 42, 30 steps)
    mock_runs = [
        ("s3_mock_shock_seed_42", "single_agent", 3),
        ("s6_mock_shock_seed_42", "orchestrated", 6),
    ]

    for run_id, sys_name, sys_num in mock_runs:
        conn = get_duckdb_connection(db_path)
        exists = conn.execute("SELECT count(*) FROM runs WHERE run_id = ?", [run_id]).fetchone()[0]
        if not exists:
            print(f"  Generating paired MockLLM run: {run_id} ({sys_name})...", flush=True)
            conn.execute(
                "INSERT INTO runs (run_id, experiment_name, status, config, data_source) VALUES (?, 'paired_mock_benchmark', 'completed', ?, 'mock_llm')",
                [run_id, json.dumps({"system": sys_name, "model": "mock", "scenario": SCENARIO_NAME, "seed": SEED, "steps": STEPS_COUNT})],
            )
            conn.close()

            mock_client = LLMClient(mock_mode=True, db_path=db_path)
            env_mock = BeerGameEnv(scenario=scenario, seed=SEED)
            env_mock.reset(seed=SEED)

            if sys_num == 3:
                system_instance = SingleAgentSystem(
                    name="single_agent",
                    llm_client=mock_client,
                    db_path=db_path,
                    run_id=run_id,
                    config={"seed": SEED, "model": "mock-model", "prompt_version": PROMPT_VERSION},
                )
            else:
                system_instance = OrchestratedSystem(
                    name="orchestrated",
                    llm_client=mock_client,
                    db_path=db_path,
                    run_id=run_id,
                    config={"seed": SEED, "model": "mock-model", "prompt_version": PROMPT_VERSION, "enable_negotiation": True},
                )

            for _step_i in range(STEPS_COUNT):
                states = {e: env_mock.observe(e) for e in ECHELONS}
                orders = system_instance.decide_orders(states)
                env_mock.step(orders)

            # Tag mock agent_calls
            conn = get_duckdb_connection(db_path)
            conn.execute("UPDATE agent_calls SET data_source = 'mock_llm' WHERE run_id = ?", [run_id])
            conn.close()
        else:
            conn.close()

    # Confirm paired inventory
    conn = get_duckdb_connection(db_path, read_only=True)
    all_paired = [
        "baseline_s_S_shock_seed_42",
        "baseline_eoq_shock_seed_42",
        "s3_mock_shock_seed_42",
        "s6_mock_shock_seed_42",
    ]
    print("\nConfirmed Paired Comparison Runs in Database:")
    for r_id in all_paired:
        r_info = conn.execute("SELECT status, data_source FROM runs WHERE run_id = ?", [r_id]).fetchone()
        status_str = r_info[0] if r_info else "MISSING"
        ds_str = r_info[1] if r_info else "N/A"
        print(f"  [CONFIRMED] {r_id:30s} | status={status_str} | data_source={ds_str}")
    conn.close()
    print("=" * 90)


def run_shock_glm_experiment() -> None:
    """Execute System 3 and System 6 on glm-isoquant for shock scenario."""
    # Step 1: Dry run
    run_dry_run_projection()

    db_path = Path("results/glm_experiment.duckdb")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = generate_scenarios()
    scenario = scenarios[SCENARIO_NAME]

    # Step 2: Ensure paired runs
    ensure_paired_runs(db_path, scenario)

    # Step 3: Run Real GLM-Isoquant Experiments
    print("\n" + "=" * 90)
    print(" 3. EXECUTING REAL GLM-ISOQUANT RUNS (shock scenario, seed 42, 30 steps, prompt v2)")
    print("=" * 90)

    client = LLMClient(mock_mode=False, cache_enabled=True, db_path=db_path)

    # Establish baseline spend in this session
    conn = get_duckdb_connection(db_path, read_only=True)
    initial_all_time_spend = float(conn.execute("SELECT COALESCE(sum(cost_usd), 0.0) FROM agent_calls").fetchone()[0] or 0.0)
    conn.close()

    print(f"Initial DB All-Time Spend: ${initial_all_time_spend:.6f} USD")
    print(f"Session Hard Cap:          ${SESSION_HARD_CAP_USD:.2f} USD")

    session_spend = 0.0
    cap_reached = False


    # -----------------------------------------------------------------------
    # System 3: SingleAgentSystem on shock scenario
    # -----------------------------------------------------------------------
    print("\n[3.1] Running System 3: SingleAgentSystem (prompt v2, cache on)...", flush=True)
    run_id_s3 = "s3_glm_isoquant_shock_seed_42"

    conn = get_duckdb_connection(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (run_id, experiment_name, status, config, metadata, data_source)
        VALUES (?, 'glm_real_benchmark', 'running', ?, ?, 'glm_real')
        """,
        [
            run_id_s3,
            json.dumps({"system": "single_agent", "model": "glm-isoquant", "scenario": SCENARIO_NAME, "seed": SEED, "prompt_version": PROMPT_VERSION}),
            json.dumps({"data_source": "glm_real", "prompt_version": PROMPT_VERSION, "cache": True}),
        ],
    )
    conn.close()

    env_s3 = BeerGameEnv(scenario=scenario, seed=SEED)
    env_s3.reset(seed=SEED)

    sys3 = SingleAgentSystem(
        name="single_agent",
        llm_client=client,
        db_path=db_path,
        run_id=run_id_s3,
        config={"seed": SEED, "model": "glm-isoquant", "use_cache": True, "prompt_version": PROMPT_VERSION},
    )

    s3_start = time.time()
    s3_steps_completed = 0
    import uuid

    for step_idx in range(STEPS_COUNT):
        states = {e: env_s3.observe(e) for e in ECHELONS}
        orders = sys3.decide_orders(states)
        env_s3.step(orders)
        s3_steps_completed += 1

        # Record step in steps table
        conn = get_duckdb_connection(db_path)
        conn.execute(
            """
            INSERT INTO steps (step_id, run_id, step_number, inventory_level, backlog_units, demand_units, action_taken, reward, info, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'glm_real')
            """,
            [
                f"step_{uuid.uuid4().hex[:12]}",
                run_id_s3,
                step_idx + 1,
                int(sum(st.inventory for st in states.values())),
                int(sum(st.backlog for st in states.values())),
                int(states["retailer"].current_demand),
                json.dumps(orders),
                0.0,
                json.dumps({"stage": "simulation"}),
            ],
        )
        conn.close()

        # Check session spend
        conn = get_duckdb_connection(db_path, read_only=True)
        current_all_time = float(conn.execute("SELECT COALESCE(sum(cost_usd), 0.0) FROM agent_calls").fetchone()[0] or 0.0)
        conn.close()

        session_spend = current_all_time - initial_all_time_spend

        if (step_idx + 1) % 5 == 0 or step_idx == 0:
            print(f"  [S3] Step {step_idx + 1:2d}/{STEPS_COUNT}: session spend = ${session_spend:.5f} / ${SESSION_HARD_CAP_USD:.2f}", flush=True)

        if session_spend >= SESSION_HARD_CAP_USD:
            print(f"\n[ALERT] Session spend reached ${session_spend:.5f} USD meeting hard cap ${SESSION_HARD_CAP_USD:.2f}! Stopping cleanly.", flush=True)
            cap_reached = True
            break

    s3_time = time.time() - s3_start
    print(f"System 3 completed {s3_steps_completed}/{STEPS_COUNT} steps in {s3_time:.1f}s. Session spend: ${session_spend:.5f}", flush=True)

    # Tag S3 calls with data_source = glm_real
    conn = get_duckdb_connection(db_path)
    s3_status = "stopped_budget_cap" if cap_reached and s3_steps_completed < STEPS_COUNT else "completed"
    conn.execute("UPDATE runs SET status = ?, ended_at = CURRENT_TIMESTAMP WHERE run_id = ?", [s3_status, run_id_s3])
    conn.execute("UPDATE agent_calls SET data_source = 'glm_real' WHERE run_id = ?", [run_id_s3])
    conn.close()

    # -----------------------------------------------------------------------
    # System 6: OrchestratedSystem on shock scenario
    # -----------------------------------------------------------------------
    s6_steps_completed = 0
    if not cap_reached:
        print("\n[3.2] Running System 6: OrchestratedSystem (prompt v2, cache on)...", flush=True)
        run_id_s6 = "s6_glm_isoquant_shock_seed_42"

        conn = get_duckdb_connection(db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (run_id, experiment_name, status, config, metadata, data_source)
            VALUES (?, 'glm_real_benchmark', 'running', ?, ?, 'glm_real')
            """,
            [
                run_id_s6,
                json.dumps({"system": "orchestrated", "model": "glm-isoquant", "scenario": SCENARIO_NAME, "seed": SEED, "prompt_version": PROMPT_VERSION}),
                json.dumps({"data_source": "glm_real", "prompt_version": PROMPT_VERSION, "cache": True}),
            ],
        )
        conn.close()

        env_s6 = BeerGameEnv(scenario=scenario, seed=SEED)
        env_s6.reset(seed=SEED)

        sys6 = OrchestratedSystem(
            name="orchestrated",
            llm_client=client,
            db_path=db_path,
            run_id=run_id_s6,
            config={
                "seed": SEED,
                "model": "glm-isoquant",
                "use_cache": True,
                "prompt_version": PROMPT_VERSION,
                "enable_negotiation": True,
                "enable_validation": True,
                "enable_conflict_skip": True,
                "forecast_deviation_threshold": 0.25,
                "proposals_disagree_threshold": 0.30,
                "max_rounds": 2,
                "manufacturer_capacity": 300.0,
            },
        )

        s6_start = time.time()
        for step_idx in range(STEPS_COUNT):
            states = {e: env_s6.observe(e) for e in ECHELONS}
            orders = sys6.decide_orders(states)
            env_s6.step(orders)
            s6_steps_completed += 1

            # Record step in steps table
            conn = get_duckdb_connection(db_path)
            conn.execute(
                """
                INSERT INTO steps (step_id, run_id, step_number, inventory_level, backlog_units, demand_units, action_taken, reward, info, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'glm_real')
                """,
                [
                    f"step_{uuid.uuid4().hex[:12]}",
                    run_id_s6,
                    step_idx + 1,
                    int(sum(st.inventory for st in states.values())),
                    int(sum(st.backlog for st in states.values())),
                    int(states["retailer"].current_demand),
                    json.dumps(orders),
                    0.0,
                    json.dumps({"stage": "simulation"}),
                ],
            )
            conn.close()

            # Check session spend
            conn = get_duckdb_connection(db_path, read_only=True)
            current_all_time = float(conn.execute("SELECT COALESCE(sum(cost_usd), 0.0) FROM agent_calls").fetchone()[0] or 0.0)
            conn.close()

            session_spend = current_all_time - initial_all_time_spend

            if (step_idx + 1) % 5 == 0 or step_idx == 0:
                print(f"  [S6] Step {step_idx + 1:2d}/{STEPS_COUNT}: session spend = ${session_spend:.5f} / ${SESSION_HARD_CAP_USD:.2f}", flush=True)

            if session_spend >= SESSION_HARD_CAP_USD:
                print(f"\n[ALERT] Session spend reached ${session_spend:.5f} USD meeting hard cap ${SESSION_HARD_CAP_USD:.2f}! Stopping cleanly.", flush=True)
                cap_reached = True
                break

            time.sleep(0.3)

        s6_time = time.time() - s6_start
        print(f"System 6 completed {s6_steps_completed}/{STEPS_COUNT} steps in {s6_time:.1f}s. Final session spend: ${session_spend:.5f}", flush=True)

        # Tag S6 calls with data_source = glm_real
        conn = get_duckdb_connection(db_path)
        s6_status = "stopped_budget_cap" if cap_reached and s6_steps_completed < STEPS_COUNT else "completed"
        conn.execute("UPDATE runs SET status = ?, ended_at = CURRENT_TIMESTAMP WHERE run_id = ?", [s6_status, run_id_s6])
        conn.execute("UPDATE agent_calls SET data_source = 'glm_real' WHERE run_id = ?", [run_id_s6])
        conn.close()

    # -----------------------------------------------------------------------
    # Step 4: Summary & Reconcile Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" 4. EXPERIMENTAL SESSION SUMMARY (shock scenario, seed 42)")
    print("=" * 90)

    conn = get_duckdb_connection(db_path, read_only=True)

    runs_summary = conn.execute(
        """
        SELECT
            r.run_id,
            COALESCE(r.data_source, 'unspecified') as data_src,
            count(a.call_id) as total_calls,
            count(CASE WHEN a.cache_hit THEN 1 END) as cache_hits,
            COALESCE(sum(a.prompt_tokens), 0) as p_tokens,
            COALESCE(sum(a.completion_tokens), 0) as c_tokens,
            COALESCE(sum(a.cost_usd), 0.0) as run_cost_usd
        FROM runs r
        LEFT JOIN agent_calls a ON r.run_id = a.run_id
        WHERE r.run_id IN (?, ?, ?, ?, ?, ?)
        GROUP BY r.run_id, r.data_source
        ORDER BY r.run_id
        """,
        [
            "baseline_s_S_shock_seed_42",
            "baseline_eoq_shock_seed_42",
            "s3_mock_shock_seed_42",
            "s6_mock_shock_seed_42",
            "s3_glm_isoquant_shock_seed_42",
            "s6_glm_isoquant_shock_seed_42",
        ],
    ).fetchall()
    conn.close()

    print(f"{'Run ID':30s} | {'Data Source':12s} | {'Calls':6s} | {'Hits':4s} | {'In Tokens':10s} | {'Out Tokens':10s} | {'Cost (USD)':10s}")
    print("-" * 95)
    for row in runs_summary:
        r_id, d_src, c_count, hits, p_tok, c_tok, c_cost = row
        print(f"{r_id:30s} | {d_src:12s} | {c_count:6d} | {hits:4d} | {p_tok:10,d} | {c_tok:10,d} | ${c_cost:9.5f}")
    print("=" * 95)
    print(f"Total Session Spend Incurred : ${session_spend:.5f} USD (Hard Cap: ${SESSION_HARD_CAP_USD:.2f} USD)")

    # Call reconcile report to display overall model logged spend
    print("\n" + "=" * 90)
    print(" 5. ALL-TIME SPEND RECONCILIATION")
    print("=" * 90)
    print_reconcile_report([db_path], console_figure=0.21, recompute=False)


if __name__ == "__main__":
    run_shock_glm_experiment()
