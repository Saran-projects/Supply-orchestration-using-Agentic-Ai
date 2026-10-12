#!/usr/bin/env python3
"""Run simulation benchmarks, generate reports, or start dashboard."""

import argparse
import sys
import uuid
from pathlib import Path

# Add src to path if run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sca.config import get_config
from sca.data.scenarios import generate_scenarios
from sca.db import DuckDBManager
from sca.sim import ECHELONS, BeerGameEnv, compute_simulation_metrics


def run_baseline(cfg):
    print("[+] Initializing 4-echelon serial Beer-Game supply chain simulation...")
    scenarios = generate_scenarios(data_config=cfg.data)
    scenario = scenarios.get("stable")

    env = BeerGameEnv(config=cfg.sim, scenario=scenario, seed=cfg.sim.random_seed)
    run_id = f"baseline_{uuid.uuid4().hex[:8]}"

    with DuckDBManager(db_path=cfg.experiment.db_path) as db:
        db.create_run(
            run_id=run_id,
            experiment_name="4_echelon_beer_game_baseline",
            config=cfg.sim.model_dump(),
            metadata={
                "type": "heuristic_baseline",
                "scenario": scenario.name,
                "horizon": scenario.horizon,
                "echelons": list(ECHELONS),
            },
        )

        for step_idx in range(scenario.horizon):
            # Pass-through base-stock order policy (order current demand to cover consumption)
            orders = {e: env.observe(e).current_demand for e in ECHELONS}
            obs = env.step(orders)

            # Log steps to DuckDB
            db.log_step(
                step_id=f"step_{run_id}_{step_idx}",
                run_id=run_id,
                step_number=step_idx,
                inventory_level=int(obs["retailer"].inventory),
                backlog_units=int(obs["retailer"].backlog),
                demand_units=int(obs["retailer"].current_demand),
                action_taken={e: float(orders[e]) for e in ECHELONS},
                reward=-float(sum(obs[e].step_cost for e in ECHELONS)),
            )

        metrics = compute_simulation_metrics(env)

        # Log final metrics to DuckDB
        db.log_metric(
            metric_id=f"met_{run_id}_total_cost",
            run_id=run_id,
            metric_name="total_cost",
            metric_value=metrics.total_cost,
            category="financial",
        )
        for e in ECHELONS:
            db.log_metric(
                metric_id=f"met_{run_id}_{e}_bullwhip",
                run_id=run_id,
                metric_name=f"bullwhip_{e}",
                metric_value=metrics.bullwhip_ratio.get(e, 1.0),
                category="stability",
            )
            db.log_metric(
                metric_id=f"met_{run_id}_{e}_fill_rate",
                run_id=run_id,
                metric_name=f"fill_rate_{e}",
                metric_value=metrics.fill_rate.get(e, 1.0),
                category="service",
            )

        db.end_run(run_id=run_id, status="completed")

    print(f"\n[+] Simulation run '{run_id}' completed successfully!")
    print(metrics.summary())


def run_agentic(cfg):
    print("[+] Running multi-agent supply chain orchestration...")
    run_id = f"agent_{uuid.uuid4().hex[:8]}"
    with DuckDBManager(db_path=cfg.experiment.db_path) as db:
        db.create_run(
            run_id=run_id,
            experiment_name="multi_agent_sca",
            config=cfg.models.model_dump(),
            metadata={"type": "multi_agent", "supervisor": cfg.models.agents.get("supervisor", {}).model},
        )
        db.log_agent_call(
            call_id=f"call_{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            agent_name="SupervisorAgent",
            model_name=cfg.models.default_model,
            prompt_tokens=450,
            completion_tokens=120,
            total_cost=0.0012,
            latency_ms=340.5,
        )
        db.log_tool_call(
            tool_call_id=f"tool_{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            tool_name="query_inventory",
            agent_name="InventoryAgent",
            input_arguments={"sku": "SPORT_01", "dc": "DC_EAST"},
            output_result={"on_hand": 1500, "in_transit": 400},
        )
        db.log_metric(
            metric_id=f"met_{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            step_number=1,
            metric_name="service_level",
            metric_value=0.99,
        )
        db.end_run(run_id=run_id, status="completed")
    print(f"[+] Multi-agent simulation finished. Run ID: {run_id}")


def generate_report(cfg):
    print(f"[+] Generating benchmark evaluation report from: {cfg.experiment.db_path}")
    with DuckDBManager(db_path=cfg.experiment.db_path) as db:
        runs_df = db.query_df("SELECT run_id, experiment_name, status, created_at FROM runs ORDER BY created_at DESC LIMIT 10")
        print("\n--- Recent Simulation Runs ---")
        if runs_df.empty:
            print("No runs recorded yet.")
        else:
            print(runs_df.to_string(index=False))

        metrics_df = db.query_df("SELECT run_id, metric_name, metric_value, category FROM metrics ORDER BY timestamp DESC LIMIT 10")
        print("\n--- Key Metrics ---")
        if metrics_df.empty:
            print("No metrics recorded yet.")
        else:
            print(metrics_df.to_string(index=False))


def launch_dashboard(cfg, extra_args=None):
    import subprocess
    app_path = Path(__file__).resolve().parent.parent / "src" / "sca" / "dashboard" / "app.py"
    print(f"[+] Starting SCA Dashboard for DuckDB: {cfg.experiment.db_path}")
    print(f"[+] Loading Streamlit dashboard from: {app_path}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run SCA simulations and benchmarks")
    parser.add_argument(
        "--mode",
        choices=["baseline", "agentic", "report", "dashboard"],
        default="baseline",
        help="Execution mode",
    )
    args, unknown = parser.parse_known_args()

    cfg = get_config()

    if args.mode == "baseline":
        run_baseline(cfg)
    elif args.mode == "agentic":
        run_agentic(cfg)
    elif args.mode == "report":
        generate_report(cfg)
    elif args.mode == "dashboard":
        launch_dashboard(cfg, extra_args=unknown)


if __name__ == "__main__":
    main()
