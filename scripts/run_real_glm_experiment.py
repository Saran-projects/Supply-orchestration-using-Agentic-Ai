"""Experiment Runner for System 3 and System 6 on glm-isoquant.

Executes:
- System 3 (single_agent) for 30 steps on 'stable' scenario (seed 42)
- System 6 (orchestrated) for 30 steps on 'stable' scenario (seed 42)
Using glm-isoquant via Isoquant endpoint with caching enabled.

Enforces spend cap: stops immediately if cumulative real spend exceeds 0.50 USD.
Measures token consumption, latency, structured output success, and fallbacks.
Writes measurements to results/token_measurements.csv.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from sca.data.scenarios import generate_scenarios
from sca.db import get_duckdb_connection
from sca.llm.client import LLMClient
from sca.sim.environment import ECHELONS, BeerGameEnv
from sca.systems.orchestrated import OrchestratedSystem
from sca.systems.single_agent import SingleAgentSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Budget estimates from user requirements
ESTIMATES = {
    "S3": {"in": 1200, "out": 150},
    "S4": {"in": 2000, "out": 200},
    "S5": {"in": 6000, "out": 600},
    "S6": {"in": 16000, "out": 1600},
}
SPEND_CAP_USD = 0.50


def run_glm_experiment(steps_count: int = 30) -> Dict[str, Any]:
    """Run System 3 and System 6 on glm-isoquant and collect empirical metrics."""
    db_path = Path("results/glm_experiment.duckdb")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    scenarios = generate_scenarios()
    scenario = scenarios["stable"]
    seed = 42

    client = LLMClient(mock_mode=False, cache_enabled=True, db_path=db_path)
    cumulative_cost = 0.0

    print("=" * 100)
    print("REAL LLM EXPERIMENT: GLM-ISOQUANT (GLM-5.3-FLASH)")
    print(f"Scenario: {scenario.name} | Steps: {steps_count} | Seed: {seed} | Cache: ENABLED")
    print(f"Spend Cap: ${SPEND_CAP_USD:.2f} USD")
    print("=" * 100)

    # -----------------------------------------------------------------------
    # 1. System 3: SingleAgentSystem
    # -----------------------------------------------------------------------
    print("\n[1/2] Running System 3: SingleAgentSystem (1 agent per echelon, local visibility)...")
    run_id_s3 = "s3_glm_isoquant_stable_seed_42"
    conn = get_duckdb_connection(db_path)
    conn.execute(
        "INSERT INTO runs (run_id, experiment_name, status, config) VALUES (?, 'glm_benchmark', 'running', ?)",
        [run_id_s3, json.dumps({"system": "single_agent", "model": "glm-isoquant", "scenario": "stable", "seed": seed})],
    )
    conn.close()

    env_s3 = BeerGameEnv(scenario=scenario, seed=seed)
    env_s3.reset(seed=seed)

    sys3 = SingleAgentSystem(
        name="single_agent",
        llm_client=client,
        db_path=db_path,
        run_id=run_id_s3,
        config={"seed": seed, "model": "glm-isoquant", "use_cache": True},
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    s3_start = time.time()
    s3_decisions = 0
    s3_fallbacks = 0
    s3_struct_ok_count = 0

    for step_idx in range(steps_count):
        states = {e: env_s3.observe(e) for e in ECHELONS}
        orders: Dict[str, float] = {}

        def _exec_ech(ech: str) -> tuple[str, float]:
            st = states[ech]
            mo = sys3.get_max_order(ech)
            p = sys3.format_echelon_prompt(echelon=ech, state=st, max_order=mo)
            q, _ = sys3._execute_echelon_decision(
                echelon=ech,
                state=st,
                prompt=p,
                max_order=mo,
            )
            return ech, q

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_exec_ech, ech) for ech in ECHELONS]
            for fut in as_completed(futures):
                ech, qty = fut.result()
                orders[ech] = qty
                s3_decisions += 1

        # Check spend cap
        conn = get_duckdb_connection(db_path, read_only=True)
        cost_row = conn.execute("SELECT sum(cost_usd) FROM agent_calls").fetchone()
        conn.close()
        cumulative_cost = float(cost_row[0] or 0.0)

        if cumulative_cost > SPEND_CAP_USD:
            print(f"\n[ALERT] Cumulative spend reached ${cumulative_cost:.4f} USD exceeding cap ${SPEND_CAP_USD:.2f}! STOPPING.", flush=True)
            break

        env_s3.step(orders)
        if (step_idx + 1) % 5 == 0 or step_idx == 0:
            print(f"  [S3] Step {step_idx + 1:2d}/{steps_count}: cumulative spend = ${cumulative_cost:.5f}", flush=True)

    s3_time = time.time() - s3_start
    print(f"System 3 completed in {s3_time:.1f}s ({s3_decisions} decisions). Cumulative spend: ${cumulative_cost:.5f}", flush=True)

    # -----------------------------------------------------------------------
    # 2. System 6: OrchestratedSystem
    # -----------------------------------------------------------------------
    if cumulative_cost <= SPEND_CAP_USD:
        print("\n[2/2] Running System 6: OrchestratedSystem (Orchestrator + negotiation)...", flush=True)
        run_id_s6 = "s6_glm_isoquant_stable_seed_42"
        conn = get_duckdb_connection(db_path)
        conn.execute(
            "INSERT INTO runs (run_id, experiment_name, status, config) VALUES (?, 'glm_benchmark', 'running', ?)",
            [run_id_s6, json.dumps({"system": "orchestrated", "model": "glm-isoquant", "scenario": "stable", "seed": seed})],
        )
        conn.close()

        env_s6 = BeerGameEnv(scenario=scenario, seed=seed)
        env_s6.reset(seed=seed)

        sys6 = OrchestratedSystem(
            name="orchestrated",
            llm_client=client,
            db_path=db_path,
            run_id=run_id_s6,
            config={
                "seed": seed,
                "model": "glm-isoquant",
                "use_cache": True,
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
        for step_idx in range(steps_count):
            states = {e: env_s6.observe(e) for e in ECHELONS}
            orders = sys6.decide_orders(states)
            env_s6.step(orders)

            # Check spend cap
            conn = get_duckdb_connection(db_path, read_only=True)
            cost_row = conn.execute("SELECT sum(cost_usd) FROM agent_calls").fetchone()
            conn.close()
            cumulative_cost = float(cost_row[0] or 0.0)

            if (step_idx + 1) % 5 == 0 or step_idx == 0:
                print(f"  [S6] Step {step_idx + 1:2d}/{steps_count}: cumulative spend = ${cumulative_cost:.5f}", flush=True)

            if cumulative_cost > SPEND_CAP_USD:
                print(f"\n[ALERT] Cumulative spend reached ${cumulative_cost:.4f} USD exceeding cap ${SPEND_CAP_USD:.2f}! STOPPING.")
                break

            time.sleep(0.5)

        s6_time = time.time() - s6_start
        print(f"System 6 completed in {s6_time:.1f}s. Final cumulative spend: ${cumulative_cost:.5f}")

    # -----------------------------------------------------------------------
    # 3. Analyze Telemetry & Compute Measured Averages
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("EXPERIMENTAL METRICS ANALYSIS (from DuckDB logs)")
    print("=" * 100)

    conn = get_duckdb_connection(db_path, read_only=True)

    # Telemetry for System 3
    s3_stats = conn.execute(
        """
        SELECT
            count(*) AS total_calls,
            sum(prompt_tokens) AS total_in_tok,
            sum(completion_tokens) AS total_out_tok,
            sum(cost_usd) AS total_cost,
            avg(latency_ms) AS avg_lat,
            count(CASE WHEN cache_hit THEN 1 END) AS cache_hits,
            count(CASE WHEN fallback THEN 1 END) AS fallbacks,
            count(CASE WHEN structured_output_ok = TRUE THEN 1 END) AS struct_ok,
            count(CASE WHEN error_type IS NOT NULL THEN 1 END) AS errors
        FROM agent_calls
        WHERE run_id = ?
        """,
        [run_id_s3],
    ).fetchone()

    # Telemetry for System 6
    s6_agent_stats = conn.execute(
        """
        SELECT
            count(*) AS total_calls,
            sum(prompt_tokens) AS total_in_tok,
            sum(completion_tokens) AS total_out_tok,
            sum(cost_usd) AS total_cost,
            avg(latency_ms) AS avg_lat,
            count(CASE WHEN cache_hit THEN 1 END) AS cache_hits,
            count(CASE WHEN fallback THEN 1 END) AS fallbacks,
            count(CASE WHEN structured_output_ok = TRUE THEN 1 END) AS struct_ok,
            count(CASE WHEN error_type IS NOT NULL THEN 1 END) AS errors
        FROM agent_calls
        WHERE run_id = ?
        """,
        [run_id_s6],
    ).fetchone()

    s6_orch_stats = conn.execute(
        """
        SELECT
            count(*) AS total_decisions,
            sum(rounds_used) AS total_rounds,
            sum(messages_exchanged) AS total_msgs,
            count(CASE WHEN has_conflict THEN 1 END) AS conflict_count,
            count(CASE WHEN fallback_used THEN 1 END) AS orch_fallbacks,
            sum(total_tokens) AS total_negotiation_tokens
        FROM orchestrated_decisions
        WHERE run_id = ?
        """,
        [run_id_s6],
    ).fetchone()

    conn.close()

    # Extract S3 values
    s3_calls, s3_in_tok, s3_out_tok, s3_cost, s3_lat, s3_chits, s3_fb, s3_sok, s3_errs = s3_stats
    s3_dec_count = max(s3_calls, 1)
    s3_avg_in = round(float(s3_in_tok or 0) / float(s3_dec_count), 1)
    s3_avg_out = round(float(s3_out_tok or 0) / float(s3_dec_count), 1)
    s3_cost_total = float(s3_cost or 0.0)
    s3_struct_rate = round(float(s3_sok or 0) / float(s3_dec_count) * 100, 1)
    s3_fb_rate = round(float(s3_fb or 0) / float(s3_dec_count) * 100, 1)

    # Extract S6 values
    s6_dec_count = max(s6_orch_stats[0] if s6_orch_stats else 0, 1)
    s6_calls = s6_agent_stats[0] or 0
    s6_in_tok = s6_agent_stats[1] or 0
    s6_out_tok = s6_agent_stats[2] or 0
    s6_cost_total = float(s6_agent_stats[3] or 0.0)
    s6_chits = s6_agent_stats[5] or 0
    s6_fb = s6_orch_stats[4] if s6_orch_stats else 0
    s6_fb_rate = round(float(s6_fb or 0) / float(s6_dec_count) * 100, 1)
    s6_conf_share = round(float(s6_orch_stats[3] or 0) / float(s6_dec_count) * 100, 1) if s6_orch_stats else 0.0
    s6_avg_rounds = round(float(s6_orch_stats[1] or 0) / float(s6_dec_count), 2) if s6_orch_stats else 0.0
    s6_avg_in = round(float(s6_in_tok) / float(s6_dec_count), 1)
    s6_avg_out = round(float(s6_out_tok) / float(s6_dec_count), 1)
    s6_struct_rate = 100.0 if s6_calls == 0 or s6_agent_stats[7] == s6_calls else round(float(s6_agent_stats[7] or 0) / float(max(s6_calls, 1)) * 100, 1)

    # Ratios compared to docs/budget.md estimates
    s3_in_ratio = round(s3_avg_in / ESTIMATES["S3"]["in"], 2)
    s3_out_ratio = round(s3_avg_out / ESTIMATES["S3"]["out"], 2)
    s6_in_ratio = round(s6_avg_in / ESTIMATES["S6"]["in"], 2)
    s6_out_ratio = round(s6_avg_out / ESTIMATES["S6"]["out"], 2)

    # -----------------------------------------------------------------------
    # 4. Write to results/token_measurements.csv
    # -----------------------------------------------------------------------
    csv_path = Path("results/token_measurements.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system",
            "model",
            "steps",
            "decisions",
            "total_calls",
            "cache_hits",
            "measured_avg_input_tokens",
            "measured_avg_output_tokens",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "input_ratio",
            "output_ratio",
            "structured_output_success_rate",
            "fallback_rate",
            "total_cost_usd",
        ])
        writer.writerow([
            "System 3 (Single Agent)",
            "glm-isoquant",
            steps_count,
            s3_dec_count,
            s3_calls,
            s3_chits,
            s3_avg_in,
            s3_avg_out,
            ESTIMATES["S3"]["in"],
            ESTIMATES["S3"]["out"],
            s3_in_ratio,
            s3_out_ratio,
            f"{s3_struct_rate}%",
            f"{s3_fb_rate}%",
            f"${s3_cost_total:.6f}",
        ])
        writer.writerow([
            "System 6 (Orchestrated)",
            "glm-isoquant",
            steps_count,
            s6_dec_count,
            s6_calls,
            s6_chits,
            s6_avg_in,
            s6_avg_out,
            ESTIMATES["S6"]["in"],
            ESTIMATES["S6"]["out"],
            s6_in_ratio,
            s6_out_ratio,
            f"{s6_struct_rate}%",
            f"{s6_fb_rate}%",
            f"${s6_cost_total:.6f}",
        ])

    print(f"Measurements saved to: {csv_path}")

    # -----------------------------------------------------------------------
    # 5. Summary Presentation
    # -----------------------------------------------------------------------
    summary = {
        "s3": {
            "decisions": s3_dec_count,
            "calls": s3_calls,
            "cache_hits": s3_chits,
            "tokens_in": s3_in_tok,
            "tokens_out": s3_out_tok,
            "avg_in": s3_avg_in,
            "avg_out": s3_avg_out,
            "cost_usd": s3_cost_total,
            "in_ratio": s3_in_ratio,
            "out_ratio": s3_out_ratio,
            "struct_ok_rate": s3_struct_rate,
            "fallback_rate": s3_fb_rate,
        },
        "s6": {
            "decisions": s6_dec_count,
            "calls": s6_calls,
            "cache_hits": s6_chits,
            "tokens_in": s6_in_tok,
            "tokens_out": s6_out_tok,
            "avg_in": s6_avg_in,
            "avg_out": s6_avg_out,
            "cost_usd": s6_cost_total,
            "in_ratio": s6_in_ratio,
            "out_ratio": s6_out_ratio,
            "struct_ok_rate": s6_struct_rate,
            "fallback_rate": s6_fb_rate,
            "conflict_share": s6_conf_share,
            "avg_rounds": s6_avg_rounds,
        },
        "total_spend_usd": cumulative_cost,
    }

    return summary


if __name__ == "__main__":
    run_glm_experiment(steps_count=30)
