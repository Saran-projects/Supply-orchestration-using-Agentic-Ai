"""Prompt-Tuning Iteration Runner and Protocol Evaluation.

Executes prompt-tuning iteration runs and confirmation runs strictly under
the empirical study protocol (docs/llm_study_protocol.md).

Models: glm-isoquant only.
Systems: System 3 (single_agent) and System 6 (orchestrated).
Scenario: shock, tuning seed 100 (40 steps, prompt version v3.<N>).
Safety limits: Abort if dry-run projected cost > 0.35 USD or cumulative spend > 1.50 USD.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import duckdb
import numpy as np

import sca.baselines  # noqa: F401
from sca.config import get_config, load_models_config
from sca.data.scenarios import generate_scenarios
from sca.db import DuckDBManager, get_duckdb_connection
from sca.eval.manifest import compute_benchmark_config_hash
from sca.llm.client import LLMClient
from sca.llm.reconcile import find_duckdb_files
from sca.sim.environment import BeerGameEnv
from sca.sim.metrics import SimulationMetrics, compute_simulation_metrics
from sca.systems.base import get_system
from sca.systems.orchestrated import OrchestratedSystem
from sca.systems.single_agent import SingleAgentSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_prompt_tuning")

DRY_RUN_ABORT_THRESHOLD_USD = 0.35
CUMULATIVE_TUNING_SPEND_LIMIT_USD = 1.50
WARMUP_STEPS = 5


def get_all_time_glm_spend(db_path: Path) -> float:
    """Compute total all-time logged spend for glm-isoquant across databases."""
    total_spend = 0.0
    db_files = find_duckdb_files()
    if db_path not in db_files and db_path.exists():
        db_files.append(db_path)

    for db_f in set(db_files):
        if not db_f.is_file():
            continue
        try:
            conn = duckdb.connect(str(db_f), read_only=True)
            tables = [r[0].lower() for r in conn.execute("SHOW TABLES").fetchall()]
            if "agent_calls" in tables:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0.0) FROM agent_calls WHERE LOWER(model_name) LIKE '%glm%'"
                ).fetchone()
                if row and row[0] is not None:
                    total_spend += float(row[0])
            conn.close()
        except Exception:
            pass
    return total_spend


def run_dry_run_projection(steps: int = 40, db_path: Optional[Path] = None) -> float:
    """Dry-run cost projection. Aborts if projected > 0.35 USD or cumulative spend > 1.50 USD."""
    models_cfg = load_models_config()
    glm_entry = models_cfg.get_model("glm-isoquant")
    if not glm_entry:
        raise ValueError("glm-isoquant model not configured in configs/models.yaml")

    p_in = float(glm_entry.price_per_1k_input)
    p_cached = (
        float(glm_entry.price_per_1k_cached_input)
        if getattr(glm_entry, "price_per_1k_cached_input", None) is not None
        else p_in
    )
    p_out = float(glm_entry.price_per_1k_output)

    # Prompt v3.1 empirical token profiles
    # S3 (Single Agent): 160 calls (1 per echelon step), 933.8 in, 0 cached, 795.4 out
    s3_decisions = steps * 4
    s3_avg_in = 933.8
    s3_avg_out = 795.4
    s3_cost_per_dec = (s3_avg_in / 1000.0) * p_in + (s3_avg_out / 1000.0) * p_out
    s3_projected_usd = s3_decisions * s3_cost_per_dec
    s3_out_cost = s3_decisions * (s3_avg_out / 1000.0) * p_out
    s3_out_share = (s3_out_cost / s3_projected_usd * 100.0) if s3_projected_usd > 0 else 0.0

    # S6 (Orchestrated): 282 calls (supervisor + echelons), 665.2 in (59.5% cached), 1691.2 out
    s6_calls = int(round(steps * 4 * (282.0 / 160.0)))  # ~282 calls for 40 steps
    s6_avg_in = 665.2
    s6_avg_cached = 395.8
    s6_avg_uncached = s6_avg_in - s6_avg_cached
    s6_avg_out = 1691.2
    s6_cost_per_call = (
        (s6_avg_uncached / 1000.0) * p_in
        + (s6_avg_cached / 1000.0) * p_cached
        + (s6_avg_out / 1000.0) * p_out
    )
    s6_projected_usd = s6_calls * s6_cost_per_call
    s6_cost_per_dec = s6_projected_usd / (steps * 4)
    s6_out_cost = s6_calls * (s6_avg_out / 1000.0) * p_out
    s6_out_share = (s6_out_cost / s6_projected_usd * 100.0) if s6_projected_usd > 0 else 0.0

    total_projected_usd = s3_projected_usd + s6_projected_usd
    total_out_cost = s3_out_cost + s6_out_cost
    total_out_share = (total_out_cost / total_projected_usd * 100.0) if total_projected_usd > 0 else 0.0

    target_db = db_path or Path("results/glm_experiment.duckdb")
    current_spend = get_all_time_glm_spend(target_db)
    projected_cumulative_spend = current_spend + total_projected_usd

    print("\n" + "=" * 90)
    print(" 1. DRY-RUN COST PROJECTION & SAFETY PRE-CHECKS (PROMPT v3.1 TOKENS)")
    print("=" * 90)
    print("Model: glm-isoquant (reasoning_tokens: disabled)")
    print(f"Rates: In=${p_in:.6f}/1k (${p_in*1000:.3f}/1M) | Cached In=${p_cached:.6f}/1k (${p_cached*1000:.3f}/1M) | Out=${p_out:.6f}/1k (${p_out*1000:.3f}/1M)")
    print(f"Run Horizon: {steps} steps x 4 echelons = {steps * 4} decisions per system")
    print(f"  System 3 Projected Cost: ${s3_projected_usd:.5f} USD (${s3_cost_per_dec:.6f}/dec, Out Share: {s3_out_share:.1f}%)")
    print(f"  System 6 Projected Cost: ${s6_projected_usd:.5f} USD (${s6_cost_per_dec:.6f}/dec, Out Share: {s6_out_share:.1f}%)")
    print(f"  Total Run Projected Cost: ${total_projected_usd:.5f} USD (Combined Out Share: {total_out_share:.1f}%)")
    print(f"  Current Logged Spend:    ${current_spend:.5f} USD")
    print(f"  Est. Cumulative Spend:   ${projected_cumulative_spend:.5f} USD")
    print("-" * 90)
    hard_cap_usd = float(getattr(models_cfg, "hard_cap_usd", 3.60))
    print(f"  Single-Run Abort Ceiling: ${DRY_RUN_ABORT_THRESHOLD_USD:.2f} USD")
    print(f"  Cumulative Spend Ceiling: ${CUMULATIVE_TUNING_SPEND_LIMIT_USD:.2f} USD")
    print(f"  Project Hard Cap Ceiling: ${hard_cap_usd:.2f} USD")

    if total_projected_usd > DRY_RUN_ABORT_THRESHOLD_USD:
        print(f"\n[ABORT] Total projected cost (${total_projected_usd:.5f}) exceeds single-run limit ${DRY_RUN_ABORT_THRESHOLD_USD:.2f} USD!")
        sys.exit(1)

    if projected_cumulative_spend > CUMULATIVE_TUNING_SPEND_LIMIT_USD:
        print(f"\n[ABORT] Cumulative tuning spend (${projected_cumulative_spend:.5f}) would pass ${CUMULATIVE_TUNING_SPEND_LIMIT_USD:.2f} USD limit!")
        sys.exit(1)

    if projected_cumulative_spend > hard_cap_usd:
        print(f"\n[ABORT] Cumulative spend (${projected_cumulative_spend:.5f}) would exceed ${hard_cap_usd:.2f} USD hard cap!")
        sys.exit(1)

    print("\n[SAFETY PASS] Projected costs strictly within authorized budget constraints.")
    print("=" * 90)
    return total_projected_usd


def evaluate_baseline(
    system_id: str,
    scenario_name: str,
    seed: int,
    steps: int = 40,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[SimulationMetrics, Dict[str, Any]]:
    """Evaluate classical baseline on identical scenario and seed."""
    app_cfg = get_config()
    sim_cfg = app_cfg.sim
    scenarios = generate_scenarios()
    sc = scenarios[scenario_name]

    sys_obj = get_system(system_id, config=config)
    sys_obj.calibrate(sc)
    sys_obj.reset(scenario=sc)

    env = BeerGameEnv(config=sim_cfg, scenario=sc, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)

    for _ in range(steps):
        orders = sys_obj.decide_orders(states)
        states = env.step(orders)

    metrics = compute_simulation_metrics(env, warmup_steps=WARMUP_STEPS)
    return metrics, sys_obj.get_parameters()


def run_llm_simulation(
    system_key: str,  # "s3" or "s6"
    scenario_name: str,
    seed: int,
    prompt_version: str,
    steps: int,
    db_path: Path,
    config_hash: str,
) -> Tuple[SimulationMetrics, Dict[str, Any]]:
    """Execute live LLM simulation for System 3 or System 6."""
    app_cfg = get_config()
    sim_cfg = app_cfg.sim
    scenarios = generate_scenarios()
    sc = scenarios[scenario_name]

    client = LLMClient(mock_mode=False, db_path=db_path)
    run_id = f"{system_key}_glm_{scenario_name}_seed_{seed}_{prompt_version.replace('.', '_')}_{uuid.uuid4().hex[:6]}"

    with DuckDBManager(db_path) as mgr:
        mgr.register_run(
            run_id=run_id,
            system="single_agent" if system_key == "s3" else "orchestrated",
            model="glm-isoquant",
            scenario=scenario_name,
            seed=seed,
            prompt_version=prompt_version,
            config_hash=config_hash,
            data_source="glm_real",
            status="running",
        )

    # Initialize Environment
    env = BeerGameEnv(config=sim_cfg, scenario=sc, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)

    if system_key == "s3":
        sys_obj = SingleAgentSystem(
            name="single_agent",
            llm_client=client,
            db_path=db_path,
            run_id=run_id,
            config={
                "seed": seed,
                "model": "glm-isoquant",
                "use_cache": False,
                "prompt_version": prompt_version,
                "tool_mode": "deterministic",
            },
        )
    else:
        sys_obj = OrchestratedSystem(
            name="orchestrated",
            llm_client=client,
            db_path=db_path,
            run_id=run_id,
            config={
                "seed": seed,
                "model": "glm-isoquant",
                "use_cache": False,
                "prompt_version": prompt_version,
                "tool_mode": "deterministic",
                "enable_negotiation": True,
                "enable_validation": True,
                "enable_conflict_skip": True,
                "forecast_deviation_threshold": 0.25,
                "proposals_disagree_threshold": 0.30,
                "max_rounds": 2,
                "manufacturer_capacity": 300.0,
            },
        )

    sys_obj.calibrate(sc)
    sys_obj.reset(scenario=sc)

    print(f"\n[EXEC] Running {sys_obj.name.upper()} ({prompt_version}) on {scenario_name} seed {seed} for {steps} steps...", flush=True)

    start_time = time.time()
    for step_idx in range(steps):
        orders = sys_obj.decide_orders(states)
        states = env.step(orders)

        # Log step into steps table
        try:
            conn = get_duckdb_connection(db_path)
            conn.execute(
                """
                INSERT INTO steps (step_id, run_id, step_number, inventory_level, backlog_units, demand_units, action_taken, reward, info, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'glm_real')
                """,
                [
                    f"{run_id}_s{step_idx}",
                    run_id,
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
        except Exception as e:
            logger.debug("Failed logging step: %s", e)

        if (step_idx + 1) % 10 == 0 or step_idx == 0:
            print(f"  Step {step_idx + 1:2d}/{steps} completed", flush=True)

    elapsed_time = time.time() - start_time
    metrics = compute_simulation_metrics(env, warmup_steps=WARMUP_STEPS)

    # Telemetry query from DuckDB agent_calls
    conn = get_duckdb_connection(db_path, read_only=True)
    telemetry_row = conn.execute(
        """
        SELECT
            COUNT(call_id) as total_calls,
            COALESCE(SUM(prompt_tokens), 0) as p_tok,
            COALESCE(SUM(completion_tokens), 0) as c_tok,
            COALESCE(SUM(cost_usd), 0.0) as cost_usd,
            COUNT(CASE WHEN structured_output_ok = TRUE THEN 1 END) as structured_ok_count,
            COUNT(CASE WHEN fallback = TRUE THEN 1 END) as fallback_count
        FROM agent_calls
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    conn.close()

    total_calls = telemetry_row[0] if telemetry_row else 0
    p_tok = int(telemetry_row[1]) if telemetry_row else 0
    c_tok = int(telemetry_row[2]) if telemetry_row else 0
    run_cost_usd = float(telemetry_row[3]) if telemetry_row else 0.0
    struct_ok = telemetry_row[4] if telemetry_row else 0
    fallbacks = telemetry_row[5] if telemetry_row else 0

    # Finish run in runs_registry
    with DuckDBManager(db_path) as mgr:
        mgr.finish_registered_run(
            run_id=run_id,
            status="completed",
            cost_usd=run_cost_usd,
        )

    decisions_count = steps * 4
    tok_in_per_dec = (p_tok / decisions_count) if decisions_count > 0 else 0.0
    tok_out_per_dec = (c_tok / decisions_count) if decisions_count > 0 else 0.0
    struct_success_rate = (struct_ok / total_calls * 100.0) if total_calls > 0 else 100.0
    fallback_rate = (fallbacks / total_calls * 100.0) if total_calls > 0 else 0.0

    meta = {
        "run_id": run_id,
        "elapsed_sec": elapsed_time,
        "total_calls": total_calls,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "cost_usd": run_cost_usd,
        "tok_in_per_dec": tok_in_per_dec,
        "tok_out_per_dec": tok_out_per_dec,
        "structured_success_rate": struct_success_rate,
        "fallbacks": fallbacks,
        "fallback_rate": fallback_rate,
    }

    return metrics, meta


def print_comparison_table(
    title: str,
    s6_metrics: SimulationMetrics,
    s6_meta: Dict[str, Any],
    s3_metrics: SimulationMetrics,
    s3_meta: Dict[str, Any],
    eoq_metrics: SimulationMetrics,
    s2b_metrics: SimulationMetrics,
    eoq_untuned_bw: float = 7.675,
) -> Dict[str, bool]:
    """Print consolidated comparison report and evaluate protocol success criteria."""
    bw_ceiling = 0.75 * eoq_untuned_bw  # 5.756
    s6_mean_bw = float(np.mean(list(s6_metrics.bullwhip_ratio.values())))
    s6_cost = s6_metrics.total_cost
    eoq_cost = eoq_metrics.total_cost
    cost_diff_pct = ((s6_cost - eoq_cost) / eoq_cost) * 100.0

    ret_fr = s6_metrics.fill_rate["retailer"] * 100.0
    ret_sl = s6_metrics.service_level["retailer"] * 100.0

    # Protocol success criteria evaluation
    criteria_met = {
        "Zero Fallbacks (0.0% rate)": (s6_meta["fallbacks"] == 0 and s3_meta["fallbacks"] == 0),
        "Structured-Output Success >= 99.0%": (s6_meta["structured_success_rate"] >= 99.0 and s3_meta["structured_success_rate"] >= 99.0),
        f"S6 Mean Bullwhip <= {bw_ceiling:.3f} (75% untuned EOQ)": (s6_mean_bw <= bw_ceiling),
        "S6 Cost within +15% of Tuned EOQ": (cost_diff_pct <= 15.0),
        "Retail Fill Rate >= 99.0%": (ret_fr >= 99.0),
        "Retail Step Service Level >= 85.0%": (ret_sl >= 85.0),
    }

    print("\n" + "=" * 115)
    print(f" {title.upper()}")
    print("=" * 115)
    headers = [
        f"{'Metric':<32s}",
        f"{'System 6 (Orch)':<20s}",
        f"{'System 3 (Single)':<20s}",
        f"{'System 2b (Smoothed)':<20s}",
        f"{'EOQ Tuned':<18s}",
    ]
    print(" | ".join(headers))
    print("-" * 115)

    def row(label, s6_v, s3_v, s2b_v, eoq_v):
        print(f" {label:<32s} | {str(s6_v):<20s} | {str(s3_v):<20s} | {str(s2b_v):<20s} | {str(eoq_v):<18s}")

    row("Validity: Fallbacks Count", f"{s6_meta['fallbacks']} ({s6_meta['fallback_rate']:.1f}%)", f"{s3_meta['fallbacks']} ({s3_meta['fallback_rate']:.1f}%)", "0 (0.0%)", "0 (0.0%)")
    row("Structured Output Success", f"{s6_meta['structured_success_rate']:.1f}%", f"{s3_meta['structured_success_rate']:.1f}%", "100.0% (code)", "100.0% (code)")
    row("Tokens In / Dec", f"{s6_meta['tok_in_per_dec']:.1f}", f"{s3_meta['tok_in_per_dec']:.1f}", "n/a", "n/a")
    row("Tokens Out / Dec", f"{s6_meta['tok_out_per_dec']:.1f}", f"{s3_meta['tok_out_per_dec']:.1f}", "n/a", "n/a")
    row("LLM Spend (USD)", f"${s6_meta['cost_usd']:.5f}", f"${s3_meta['cost_usd']:.5f}", "$0.00000", "$0.00000")
    row("Total System Cost ($)", f"${s6_metrics.total_cost:,.2f}", f"${s3_metrics.total_cost:,.2f}", f"${s2b_metrics.total_cost:,.2f}", f"${eoq_metrics.total_cost:,.2f}")
    row("Cost vs EOQ Tuned", f"{cost_diff_pct:+.1f}%", f"{((s3_metrics.total_cost-eoq_cost)/eoq_cost)*100:+.1f}%", f"{((s2b_metrics.total_cost-eoq_cost)/eoq_cost)*100:+.1f}%", "0.0% (baseline)")
    row("Retail Fill Rate", f"{ret_fr:.2f}%", f"{s3_metrics.fill_rate['retailer']*100:.2f}%", f"{s2b_metrics.fill_rate['retailer']*100:.2f}%", f"{eoq_metrics.fill_rate['retailer']*100:.2f}%")
    row("Step Service Level", f"{ret_sl:.1f}%", f"{s3_metrics.service_level['retailer']*100:.1f}%", f"{s2b_metrics.service_level['retailer']*100:.1f}%", f"{eoq_metrics.service_level['retailer']*100:.1f}%")
    row("End-to-End Bullwhip", f"{s6_metrics.end_to_end_bullwhip:.3f}", f"{s3_metrics.end_to_end_bullwhip:.3f}", f"{s2b_metrics.end_to_end_bullwhip:.3f}", f"{eoq_metrics.end_to_end_bullwhip:.3f}")
    row("Retailer Bullwhip", f"{s6_metrics.bullwhip_ratio['retailer']:.3f}", f"{s3_metrics.bullwhip_ratio['retailer']:.3f}", f"{s2b_metrics.bullwhip_ratio['retailer']:.3f}", f"{eoq_metrics.bullwhip_ratio['retailer']:.3f}")
    row("Wholesaler Bullwhip", f"{s6_metrics.bullwhip_ratio['wholesaler']:.3f}", f"{s3_metrics.bullwhip_ratio['wholesaler']:.3f}", f"{s2b_metrics.bullwhip_ratio['wholesaler']:.3f}", f"{eoq_metrics.bullwhip_ratio['wholesaler']:.3f}")
    row("Distributor Bullwhip", f"{s6_metrics.bullwhip_ratio['distributor']:.3f}", f"{s3_metrics.bullwhip_ratio['distributor']:.3f}", f"{s2b_metrics.bullwhip_ratio['distributor']:.3f}", f"{eoq_metrics.bullwhip_ratio['distributor']:.3f}")
    row("Manufacturer Bullwhip", f"{s6_metrics.bullwhip_ratio['manufacturer']:.3f}", f"{s3_metrics.bullwhip_ratio['manufacturer']:.3f}", f"{s2b_metrics.bullwhip_ratio['manufacturer']:.3f}", f"{eoq_metrics.bullwhip_ratio['manufacturer']:.3f}")
    row("Mean Bullwhip Across Tiers", f"{s6_mean_bw:.3f}", f"{np.mean(list(s3_metrics.bullwhip_ratio.values())):.3f}", f"{np.mean(list(s2b_metrics.bullwhip_ratio.values())):.3f}", f"{np.mean(list(eoq_metrics.bullwhip_ratio.values())):.3f}")
    print("=" * 115)

    print("\n--- Protocol Success Criteria Evaluation (docs/llm_study_protocol.md) ---")
    all_met = True
    for criterion, met in criteria_met.items():
        status = "[MET]    " if met else "[NOT MET]"
        print(f"  {status} {criterion}")
        if not met:
            all_met = False
    print(f"\nOverall Pre-Registration Compliance: {'PASSED (All Criteria Met)' if all_met else 'PARTIAL (Some Criteria Missed)'}")
    print("=" * 115)
    return criteria_met


def main():
    parser = argparse.ArgumentParser(description="Run prompt-tuning iteration or confirmation run.")
    parser.add_argument("--iteration", type=int, default=1, help="Prompt tuning iteration number (1..4)")
    parser.add_argument("--prompt-version", type=str, default="v3.1", help="Prompt version tag, e.g. v3.1")
    parser.add_argument("--scenario", type=str, default="shock", help="Evaluation scenario (shock, seasonal, stable)")
    parser.add_argument("--seed", type=int, default=100, help="Tuning seed (100, 101, 102, 103, 104)")
    parser.add_argument("--steps", type=int, default=40, help="Simulation steps (default 40)")
    parser.add_argument("--db-path", type=str, default="results/glm_experiment.duckdb", help="DuckDB target path")
    parser.add_argument("--confirmation", action="store_true", help="Run confirmation on seasonal seed 103 and stable seed 104")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry-run cost projection and budget check only without LLM calls")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    config_hash, _ = compute_benchmark_config_hash()

    if args.dry_run:
        print("\n[DRY RUN ONLY] Executing safety cost projection checks...")
        run_dry_run_projection(steps=args.steps, db_path=db_path)
        print("\nDry-run safety validation complete. No simulation or LLM calls were made.")
        return

    if args.confirmation:
        print("\n" + "#" * 90)
        print(" RUNNING CONFIRMATION EVALUATION ON SEASONAL SEED 103 AND STABLE SEED 104")
        print("#" * 90)

        confirm_runs = [
            ("seasonal", 103),
            ("stable", 104),
        ]

        for sc_name, sc_seed in confirm_runs:
            print(f"\n=== Confirmation Run: Scenario '{sc_name}', Seed {sc_seed}, Prompt {args.prompt_version} ===")
            run_dry_run_projection(steps=args.steps, db_path=db_path)

            app_cfg = get_config()
            sim_cfg = app_cfg.sim
            eoq_cfg = {
                "tuned": True,
                "order_quantity_multiplier": sim_cfg.four_echelon.tuned_eoq_q_mult,
                "reorder_point_factor": sim_cfg.four_echelon.tuned_eoq_rop_factor,
            }
            s2b_cfg = {
                "alpha": sim_cfg.four_echelon.alpha,
                "beta": sim_cfg.four_echelon.beta,
                "z": sim_cfg.four_echelon.safety_factor_z,
                "delta": sim_cfg.four_echelon.rate_limit_delta,
                "forecast_cap_multiple": sim_cfg.four_echelon.forecast_cap_multiple,
            }

            eoq_m, _ = evaluate_baseline("baseline_eoq", sc_name, sc_seed, steps=args.steps, config=eoq_cfg)
            s2b_m, _ = evaluate_baseline("baseline_smoothed_order_up_to", sc_name, sc_seed, steps=args.steps, config=s2b_cfg)

            s3_m, s3_meta = run_llm_simulation("s3", sc_name, sc_seed, args.prompt_version, args.steps, db_path, config_hash)
            s6_m, s6_meta = run_llm_simulation("s6", sc_name, sc_seed, args.prompt_version, args.steps, db_path, config_hash)

            print_comparison_table(
                title=f"Confirmation Run: {sc_name.capitalize()} Scenario (Seed {sc_seed}, Prompt {args.prompt_version})",
                s6_metrics=s6_m,
                s6_meta=s6_meta,
                s3_metrics=s3_m,
                s3_meta=s3_meta,
                eoq_metrics=eoq_m,
                s2b_metrics=s2b_m,
            )

        print("\nConfirmation runs successfully finished.")
        return

    # Regular tuning iteration
    print("\n" + "#" * 90)
    print(f" PROMPT TUNING ITERATION {args.iteration}: {args.scenario.upper()} SEED {args.seed} ({args.prompt_version})")
    print("#" * 90)

    # 1. Dry run
    run_dry_run_projection(steps=args.steps, db_path=db_path)

    # 2. Run Baselines
    app_cfg = get_config()
    sim_cfg = app_cfg.sim
    eoq_cfg = {
        "tuned": True,
        "order_quantity_multiplier": sim_cfg.four_echelon.tuned_eoq_q_mult,
        "reorder_point_factor": sim_cfg.four_echelon.tuned_eoq_rop_factor,
    }
    s2b_cfg = {
        "alpha": sim_cfg.four_echelon.alpha,
        "beta": sim_cfg.four_echelon.beta,
        "z": sim_cfg.four_echelon.safety_factor_z,
        "delta": sim_cfg.four_echelon.rate_limit_delta,
        "forecast_cap_multiple": sim_cfg.four_echelon.forecast_cap_multiple,
    }

    print("\n[BASELINES] Running Paired Tuned Baselines on Shock Seed 100...")
    eoq_metrics, _ = evaluate_baseline("baseline_eoq", args.scenario, args.seed, steps=args.steps, config=eoq_cfg)
    s2b_metrics, _ = evaluate_baseline("baseline_smoothed_order_up_to", args.scenario, args.seed, steps=args.steps, config=s2b_cfg)

    # 3. Live LLM Runs
    s3_metrics, s3_meta = run_llm_simulation("s3", args.scenario, args.seed, args.prompt_version, args.steps, db_path, config_hash)
    s6_metrics, s6_meta = run_llm_simulation("s6", args.scenario, args.seed, args.prompt_version, args.steps, db_path, config_hash)

    # 4. Report & Evaluation
    print_comparison_table(
        title=f"Iteration {args.iteration} Results: {args.scenario.capitalize()} Scenario (Seed {args.seed}, Prompt {args.prompt_version})",
        s6_metrics=s6_metrics,
        s6_meta=s6_meta,
        s3_metrics=s3_metrics,
        s3_meta=s3_meta,
        eoq_metrics=eoq_metrics,
        s2b_metrics=s2b_metrics,
    )


if __name__ == "__main__":
    main()
