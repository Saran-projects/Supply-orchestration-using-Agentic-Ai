"""Evaluation Grid Runner for Supply Chain Agents and Baselines.

Runs (system x scenario x seed x model) grids defined in experiment configuration:
- Writes every step, state transition, and metric to DuckDB
- Is RESUMABLE: queries DuckDB runs table and skips already completed run_ids
- Concurrency: Multiprocessing for non-LLM baselines and bounded concurrency for LLM systems
- Generates and prints statistical benchmark results tables (mean +/- std)

CLI:
    python -m sca.eval.runner --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import duckdb
import numpy as np
import pandas as pd

import sca.baselines  # noqa: F401 (Registers baseline policies)
from sca.config import (
    ExperimentConfig,
    SimConfig,
    SystemSpec,
    load_experiment_config,
    load_models_config,
    load_sim_config,
)
from sca.data.scenarios import Scenario, generate_scenarios
from sca.db import DuckDBManager
from sca.eval.manifest import (
    compute_benchmark_config_hash,
    verify_config_matches_manifest,
)
from sca.sim.environment import ECHELONS, BeerGameEnv
from sca.sim.metrics import SimulationMetrics, compute_simulation_metrics
from sca.systems.base import SYSTEM_REGISTRY, OrderingSystem, get_system, list_systems

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sca.eval.runner")


@dataclass
class RunSpec:
    """Specification of a single grid evaluation run."""

    system_id: str
    system_type: str
    scenario_name: str
    seed: int
    model_name: Optional[str] = None
    run_id: str = ""

    def __post_init__(self):
        if not self.run_id:
            if self.model_name and self.model_name.lower() not in ("none", "null", ""):
                self.run_id = f"{self.system_id}_{self.model_name}_{self.scenario_name}_seed_{self.seed}"
            else:
                self.run_id = f"{self.system_id}_{self.scenario_name}_seed_{self.seed}"


@dataclass
class RunResult:
    """Collected outputs and metrics from a completed simulation run."""

    spec: RunSpec
    status: str  # "completed" | "failed"
    experiment_name: str
    error: Optional[str] = None
    execution_time_sec: float = 0.0
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    steps_data: List[Dict[str, Any]] = field(default_factory=list)
    metrics_data: List[Dict[str, Any]] = field(default_factory=list)
    system_parameters: Dict[str, Any] = field(default_factory=dict)
    config_dict: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_single_simulation(
    spec: RunSpec,
    scenario: Scenario,
    sim_config_dict: Dict[str, Any],
    experiment_name: str,
) -> RunResult:
    """Execute a single simulation run in worker process/thread.

    Returns complete step logs and granular performance metrics for DuckDB persistence.
    """
    start_time = time.time()
    try:
        sim_cfg = SimConfig.model_validate(sim_config_dict)

        # 1. Instantiate and calibrate ordering system
        if spec.system_id.lower() not in SYSTEM_REGISTRY:
            raise KeyError(f"Ordering system '{spec.system_id}' not found in registry: {list_systems()}")

        system: OrderingSystem = get_system(spec.system_id)
        system.calibrate(scenario)
        system.reset(scenario=scenario)

        # 2. Initialize environment with seed
        env = BeerGameEnv(config=sim_cfg, scenario=scenario, seed=spec.seed)
        states = env.reset(seed=spec.seed)

        steps_data: List[Dict[str, Any]] = []
        horizon = len(scenario.demand)

        # 3. Step execution loop
        for t in range(horizon):
            orders = system.decide_orders(states)
            states = env.step(orders)

            sim_date = (
                scenario.dates[t].strftime("%Y-%m-%d")
                if t < len(scenario.dates)
                else None
            )

            total_inv = env.total_inventory
            total_back = env.total_backlog
            dem_unit = env.external_demand[t] if t < len(env.external_demand) else 0.0
            step_reward = -float(sum(env.step_costs.values()))

            info_dict = {
                e: {
                    "inventory": env.inventory[e],
                    "backlog": env.backlog[e],
                    "pipeline": sum(s.quantity for s in env.pipelines[e]),
                    "incoming": env.incoming_shipments[e],
                    "last_order": env.last_order[e],
                    "cost": env.step_costs[e],
                }
                for e in ECHELONS
            }

            step_record = {
                "step_id": f"{spec.run_id}_s{t}",
                "run_id": spec.run_id,
                "step_number": t,
                "sim_date": sim_date,
                "inventory_level": int(round(total_inv)),
                "backlog_units": int(round(total_back)),
                "demand_units": int(round(dem_unit)),
                "action_taken": {e: float(orders.get(e, 0.0)) for e in ECHELONS},
                "reward": step_reward,
                "info": info_dict,
            }
            steps_data.append(step_record)

        # 4. Conservation check and metric calculation
        env.check_unit_conservation()
        sim_metrics: SimulationMetrics = compute_simulation_metrics(env)

        summary = {
            "total_cost": float(sim_metrics.total_cost),
            "retailer_fill_rate": float(sim_metrics.fill_rate["retailer"]),
            "retailer_service_level": float(sim_metrics.service_level["retailer"]),
            "retailer_bullwhip": float(sim_metrics.bullwhip_ratio["retailer"]),
            "wholesaler_bullwhip": float(sim_metrics.bullwhip_ratio["wholesaler"]),
            "distributor_bullwhip": float(sim_metrics.bullwhip_ratio["distributor"]),
            "manufacturer_bullwhip": float(sim_metrics.bullwhip_ratio["manufacturer"]),
            "end_to_end_bullwhip": float(sim_metrics.end_to_end_bullwhip),
            "retailer_stockouts": float(sim_metrics.stockout_steps["retailer"]),
            "mean_inventory": float(np.mean(list(sim_metrics.mean_inventory.values()))),
            "mean_backlog": float(np.mean(list(sim_metrics.mean_backlog.values()))),
        }

        # 5. Granular metrics rows for DuckDB
        metrics_data: List[Dict[str, Any]] = []

        # Run-level aggregated metrics (step_number = None)
        agg_metrics = [
            ("total_cost", sim_metrics.total_cost),
            ("holding_cost_total", sum(sim_metrics.holding_cost_by_echelon.values())),
            ("backlog_cost_total", sum(sim_metrics.backlog_cost_by_echelon.values())),
            ("fill_rate_retailer", sim_metrics.fill_rate["retailer"]),
            ("service_level_retailer", sim_metrics.service_level["retailer"]),
            ("bullwhip_ratio_retailer", sim_metrics.bullwhip_ratio["retailer"]),
            ("bullwhip_ratio_wholesaler", sim_metrics.bullwhip_ratio["wholesaler"]),
            ("bullwhip_ratio_distributor", sim_metrics.bullwhip_ratio["distributor"]),
            ("bullwhip_ratio_manufacturer", sim_metrics.bullwhip_ratio["manufacturer"]),
            ("end_to_end_bullwhip", sim_metrics.end_to_end_bullwhip),
            ("order_variance_retailer", sim_metrics.order_variance_by_echelon.get("retailer", 0.0)),
            ("order_variance_wholesaler", sim_metrics.order_variance_by_echelon.get("wholesaler", 0.0)),
            ("order_variance_distributor", sim_metrics.order_variance_by_echelon.get("distributor", 0.0)),
            ("order_variance_manufacturer", sim_metrics.order_variance_by_echelon.get("manufacturer", 0.0)),
            ("stockout_steps_retailer", float(sim_metrics.stockout_steps["retailer"])),
            ("mean_inventory_retailer", sim_metrics.mean_inventory["retailer"]),
            ("mean_backlog_retailer", sim_metrics.mean_backlog["retailer"]),
        ]
        for m_name, m_val in agg_metrics:
            val = float(m_val) if not np.isinf(m_val) and not np.isnan(m_val) else 9999.0
            metrics_data.append({
                "metric_id": f"{spec.run_id}_{m_name}",
                "run_id": spec.run_id,
                "step_number": None,
                "metric_name": m_name,
                "metric_value": val,
                "category": "performance",
                "metadata": {
                    "system_id": spec.system_id,
                    "scenario": spec.scenario_name,
                    "seed": spec.seed,
                },
            })

        # Per-echelon costs
        for e in ECHELONS:
            metrics_data.append({
                "metric_id": f"{spec.run_id}_cost_{e}",
                "run_id": spec.run_id,
                "step_number": None,
                "metric_name": f"cost_{e}",
                "metric_value": float(sim_metrics.cost_by_echelon[e]),
                "category": "echelon_cost",
                "metadata": {"echelon": e},
            })

        elapsed = time.time() - start_time

        meta = {
            "system_id": spec.system_id,
            "system_type": spec.system_type,
            "scenario": spec.scenario_name,
            "seed": spec.seed,
            "model": spec.model_name,
            "horizon": horizon,
            "elapsed_sec": round(elapsed, 4),
            **summary,
        }

        config_dict = {
            "system": system.get_parameters(),
            "sim_config": sim_config_dict,
            "scenario_metadata": scenario.metadata,
        }

        return RunResult(
            spec=spec,
            status="completed",
            experiment_name=experiment_name,
            execution_time_sec=elapsed,
            summary_metrics=summary,
            steps_data=steps_data,
            metrics_data=metrics_data,
            system_parameters=system.get_parameters(),
            config_dict=config_dict,
            metadata=meta,
        )

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.exception("Failed simulation run '%s': %s", spec.run_id, exc)
        return RunResult(
            spec=spec,
            status="failed",
            experiment_name=experiment_name,
            error=str(exc),
            execution_time_sec=elapsed,
            metadata={
                "system_id": spec.system_id,
                "scenario": spec.scenario_name,
                "seed": spec.seed,
                "error": str(exc),
            },
        )


class EvaluationRunner:
    """Manages grid generation, resumable DuckDB persistence, and concurrent execution."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        sim_config: Optional[SimConfig] = None,
        db_path: Optional[Path | str] = None,
        force: bool = False,
        allow_config_change: bool = False,
        config_change_reason: Optional[str] = None,
        manifest_path: Optional[Path | str] = None,
    ):
        self.exp_cfg = experiment_config
        self.sim_cfg = sim_config or load_sim_config()
        self.db_path = Path(db_path or self.exp_cfg.db_path)
        self.force = force
        self.allow_config_change = allow_config_change
        self.config_change_reason = config_change_reason
        self.manifest_path = Path(manifest_path or "results/benchmark_manifest.json")
        self.current_config_hash, _ = compute_benchmark_config_hash()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Pre-initialize DuckDB schema
        with DuckDBManager(self.db_path) as mgr:
            mgr.verify()

    def get_completed_run_ids(self) -> Set[str]:
        """Fetch all run_ids that are already marked 'completed' in DuckDB."""
        if not self.db_path.exists():
            return set()
        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            tables = [r[0].lower() for r in conn.execute("SHOW TABLES").fetchall()]
            if "runs" not in tables:
                conn.close()
                return set()
            rows = conn.execute("SELECT run_id FROM runs WHERE status = 'completed'").fetchall()
            conn.close()
            return {r[0] for r in rows}
        except Exception as e:
            logger.warning("Could not check completed runs from DuckDB: %s", e)
            return set()

    def build_grid(
        self,
        systems_filter: Optional[List[str]] = None,
        scenarios_filter: Optional[List[str]] = None,
        seeds_override: Optional[List[int]] = None,
    ) -> List[RunSpec]:
        """Build the (system x scenario x seed x model) grid."""
        # 1. Resolve systems
        enabled_specs: List[SystemSpec] = [
            s for s in self.exp_cfg.systems if s.enabled
        ]
        if systems_filter:
            filter_lower = {s.lower() for s in systems_filter}
            enabled_specs = [s for s in enabled_specs if s.id.lower() in filter_lower]

        # Check against registered systems
        available = {s.lower() for s in list_systems()}
        valid_systems: List[SystemSpec] = []
        for s in enabled_specs:
            if s.id.lower() in available:
                valid_systems.append(s)
            else:
                logger.warning(
                    "System '%s' (type=%s) is enabled in experiment config but not registered in SYSTEM_REGISTRY; skipping.",
                    s.id,
                    s.type,
                )

        # 2. Resolve scenarios
        scenarios = list(self.exp_cfg.scenarios)
        if scenarios_filter:
            scenarios = [sc for sc in scenarios if sc in scenarios_filter]

        # 3. Resolve seeds
        if seeds_override is not None:
            seeds = seeds_override
        elif isinstance(self.exp_cfg.seeds, int):
            base_seed = self.exp_cfg.seed
            seeds = [base_seed + i for i in range(self.exp_cfg.seeds)]
        elif isinstance(self.exp_cfg.seeds, list):
            seeds = [int(s) for s in self.exp_cfg.seeds]
        else:
            seeds = [42]

        # 4. Generate grid
        grid: List[RunSpec] = []
        for sys_spec in valid_systems:
            for sc_name in scenarios:
                for seed in seeds:
                    model_name = None
                    if sys_spec.type.lower() == "agentic":
                        model_name = "default"  # Can be expanded for LLM models grid
                    spec = RunSpec(
                        system_id=sys_spec.id,
                        system_type=sys_spec.type,
                        scenario_name=sc_name,
                        seed=seed,
                        model_name=model_name,
                    )
                    grid.append(spec)

        return grid

    def persist_result(self, result: RunResult, db_mgr: DuckDBManager) -> None:
        """Atomically persist run, steps, and metrics into DuckDB."""
        conn = db_mgr.conn

        # Use transaction for fast insertion
        conn.execute("BEGIN TRANSACTION")
        try:
            # 1. runs table
            db_mgr.create_run(
                run_id=result.spec.run_id,
                experiment_name=result.experiment_name,
                config=result.config_dict,
                metadata=result.metadata,
            )

            # 2. steps table (batch insert via appender or executemany for high performance)
            if result.steps_data:
                step_tuples = [
                    (
                        s["step_id"],
                        s["run_id"],
                        s["step_number"],
                        s.get("sim_date"),
                        s.get("inventory_level"),
                        s.get("backlog_units"),
                        s.get("demand_units"),
                        json.dumps(s.get("action_taken")),
                        s.get("reward"),
                        json.dumps(s.get("info")),
                    )
                    for s in result.steps_data
                ]
                conn.executemany(
                    """
                    INSERT INTO steps (step_id, run_id, step_number, sim_date, inventory_level,
                                       backlog_units, demand_units, action_taken, reward, info)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    step_tuples,
                )

            # 3. metrics table
            if result.metrics_data:
                metric_tuples = [
                    (
                        m["metric_id"],
                        m["run_id"],
                        m.get("step_number"),
                        m["metric_name"],
                        float(m["metric_value"]),
                        m.get("category", "performance"),
                        json.dumps(m.get("metadata")),
                    )
                    for m in result.metrics_data
                ]
                conn.executemany(
                    """
                    INSERT INTO metrics (metric_id, run_id, step_number, metric_name,
                                         metric_value, category, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    metric_tuples,
                )

            # 4. update runs_registry table
            # Compute actual per-run LLM token cost from agent_calls if present
            llm_cost_row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM agent_calls WHERE run_id = ?",
                [result.spec.run_id],
            ).fetchone()
            if llm_cost_row and llm_cost_row[0] > 0:
                cost_val = float(round(llm_cost_row[0], 6))
            else:
                cost_val = float(result.summary_metrics.get("cost_usd", 0.0))
            db_mgr.register_run(
                run_id=result.spec.run_id,
                system=result.spec.system_id,
                model=result.spec.model_name,
                scenario=result.spec.scenario_name,
                seed=result.spec.seed,
                prompt_version=result.metadata.get("prompt_version"),
                config_hash=self.current_config_hash,
                data_source="DataCoSupplyChainDataset.csv",
                status=result.status,
                cost_usd=cost_val,
            )
            db_mgr.finish_registered_run(
                run_id=result.spec.run_id,
                status=result.status,
                cost_usd=cost_val,
            )

            # 5. update run status
            db_mgr.end_run(result.spec.run_id, status=result.status)
            conn.execute("COMMIT")

        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error("Failed to persist run '%s' to DuckDB: %s", result.spec.run_id, e)
            raise e

    def run_grid(
        self,
        systems_filter: Optional[List[str]] = None,
        scenarios_filter: Optional[List[str]] = None,
        seeds_override: Optional[List[int]] = None,
        max_workers_override: Optional[int] = None,
    ) -> List[RunResult]:
        """Execute full evaluation grid with resumability and bounded concurrency."""
        scenarios = generate_scenarios()
        all_specs = self.build_grid(
            systems_filter=systems_filter,
            scenarios_filter=scenarios_filter,
            seeds_override=seeds_override,
        )

        completed_ids = self.get_completed_run_ids()
        pending_specs: List[RunSpec] = []
        skipped_count = 0

        for spec in all_specs:
            if spec.run_id in completed_ids and not self.force:
                skipped_count += 1
            else:
                pending_specs.append(spec)

        logger.info(
            "Evaluation Grid: %d total runs (%d already completed, %d pending to run)",
            len(all_specs),
            skipped_count,
            len(pending_specs),
        )

        if not pending_specs:
            logger.info("All %d runs already completed in DuckDB. Resuming complete!", len(all_specs))
            return []

        # Partition pending runs into non-LLM (heuristics) and LLM (agentic)
        non_llm_specs = [s for s in pending_specs if s.system_type.lower() != "agentic"]
        llm_specs = [s for s in pending_specs if s.system_type.lower() == "agentic"]

        concurrency_cfg = self.exp_cfg.concurrency or {}
        max_workers_non_llm = max_workers_override or concurrency_cfg.get("max_workers_non_llm", 8)
        max_workers_llm = concurrency_cfg.get("max_workers_llm", 2)

        results: List[RunResult] = []
        sim_config_dict = self.sim_cfg.model_dump()
        exp_name = self.exp_cfg.experiment_name

        total_pending = len(pending_specs)
        finished_count = 0

        with DuckDBManager(self.db_path) as db_mgr:
            # 1. Execute Non-LLM runs (Multiprocessing / ProcessPool)
            if non_llm_specs:
                logger.info(
                    "Executing %d non-LLM baseline runs with pool size=%d...",
                    len(non_llm_specs),
                    max_workers_non_llm,
                )

                # ProcessPoolExecutor for multiprocessing non-LLM runs
                executor_cls = concurrent.futures.ProcessPoolExecutor
                try:
                    with executor_cls(max_workers=max_workers_non_llm) as executor:
                        future_to_spec = {
                            executor.submit(
                                run_single_simulation,
                                spec,
                                scenarios[spec.scenario_name],
                                sim_config_dict,
                                exp_name,
                            ): spec
                            for spec in non_llm_specs
                        }

                        for future in concurrent.futures.as_completed(future_to_spec):
                            spec = future_to_spec[future]
                            finished_count += 1
                            try:
                                res: RunResult = future.result()
                                self.persist_result(res, db_mgr)
                                results.append(res)
                                logger.info(
                                    "[%d/%d] Completed %s on %s (seed=%d): Total Cost=$%s, Fill Rate=%.1f%%",
                                    finished_count,
                                    total_pending,
                                    spec.system_id,
                                    spec.scenario_name,
                                    spec.seed,
                                    f"{res.summary_metrics.get('total_cost', 0.0):,.0f}",
                                    res.summary_metrics.get("retailer_fill_rate", 0.0) * 100,
                                )
                            except Exception as e:
                                logger.error("Worker failed for run '%s': %s", spec.run_id, e)

                except Exception as pool_err:
                    logger.warning(
                        "ProcessPool execution encountered an issue (%s); falling back to ThreadPoolExecutor",
                        pool_err,
                    )
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_non_llm) as fallback_executor:
                        future_to_spec = {
                            fallback_executor.submit(
                                run_single_simulation,
                                spec,
                                scenarios[spec.scenario_name],
                                sim_config_dict,
                                exp_name,
                            ): spec
                            for spec in non_llm_specs[finished_count:]
                        }
                        for future in concurrent.futures.as_completed(future_to_spec):
                            spec = future_to_spec[future]
                            finished_count += 1
                            res = future.result()
                            self.persist_result(res, db_mgr)
                            results.append(res)

            # 2. Execute LLM runs (Bounded Concurrency via ThreadPoolExecutor)
            if llm_specs:
                models_cfg = load_models_config()
                # If mock_mode is False, this is a real-LLM run!
                if not models_cfg.mock_mode:
                    is_match, curr_hash, man_hash = verify_config_matches_manifest(self.manifest_path)
                    if not is_match:
                        if self.allow_config_change:
                            logger.warning(
                                "ALLOWED CONFIG CHANGE: Current configuration hash '%s' does not match "
                                "frozen benchmark manifest '%s' (manifest: %s). Reason: %s",
                                curr_hash,
                                man_hash,
                                self.manifest_path,
                                self.config_change_reason or "Unspecified user override",
                            )
                        else:
                            raise RuntimeError(
                                f"Refusing to execute real-LLM run: current configuration hash '{curr_hash}' "
                                f"differs from frozen benchmark manifest '{man_hash}' (manifest: {self.manifest_path}). "
                                "Pass --allow-config-change and provide --config-change-reason to override."
                            )

                logger.info(
                    "Executing %d LLM agentic runs with bounded concurrency=%d...",
                    len(llm_specs),
                    max_workers_llm,
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_llm) as llm_executor:
                    future_to_spec = {
                        llm_executor.submit(
                            run_single_simulation,
                            spec,
                            scenarios[spec.scenario_name],
                            sim_config_dict,
                            exp_name,
                        ): spec
                        for spec in llm_specs
                    }

                    for future in concurrent.futures.as_completed(future_to_spec):
                        spec = future_to_spec[future]
                        finished_count += 1
                        try:
                            res = future.result()
                            self.persist_result(res, db_mgr)
                            results.append(res)
                            logger.info(
                                "[%d/%d] Completed LLM %s on %s (seed=%d)",
                                finished_count,
                                total_pending,
                                spec.system_id,
                                spec.scenario_name,
                                spec.seed,
                            )
                        except Exception as e:
                            logger.error("LLM worker failed for run '%s': %s", spec.run_id, e)

        logger.info("Evaluation grid execution completed. Saved results to %s", self.db_path)
        return results


def generate_benchmark_summary(db_path: Path | str) -> pd.DataFrame:
    """Query DuckDB and compute mean +/- std summary for all completed runs."""
    conn = duckdb.connect(str(db_path), read_only=True)
    query = """
    SELECT
        json_extract_string(r.metadata, '$.system_id') AS system,
        json_extract_string(r.metadata, '$.scenario') AS scenario,
        COUNT(DISTINCT r.run_id) AS seeds,
        AVG(CASE WHEN m.metric_name = 'total_cost' THEN m.metric_value END) AS cost_mean,
        STDDEV(CASE WHEN m.metric_name = 'total_cost' THEN m.metric_value END) AS cost_std,
        AVG(CASE WHEN m.metric_name = 'service_level_retailer' THEN m.metric_value END) * 100 AS sl_mean,
        STDDEV(CASE WHEN m.metric_name = 'service_level_retailer' THEN m.metric_value END) * 100 AS sl_std,
        AVG(CASE WHEN m.metric_name = 'fill_rate_retailer' THEN m.metric_value END) * 100 AS fr_mean,
        STDDEV(CASE WHEN m.metric_name = 'fill_rate_retailer' THEN m.metric_value END) * 100 AS fr_std,
        AVG(CASE WHEN m.metric_name = 'bullwhip_ratio_retailer' THEN m.metric_value END) AS ret_bw_mean,
        STDDEV(CASE WHEN m.metric_name = 'bullwhip_ratio_retailer' THEN m.metric_value END) AS ret_bw_std,
        AVG(CASE WHEN m.metric_name = 'bullwhip_ratio_manufacturer' THEN m.metric_value END) AS mfr_bw_mean,
        STDDEV(CASE WHEN m.metric_name = 'bullwhip_ratio_manufacturer' THEN m.metric_value END) AS mfr_bw_std,
        AVG(CASE WHEN m.metric_name = 'stockout_steps_retailer' THEN m.metric_value END) AS so_mean,
        STDDEV(CASE WHEN m.metric_name = 'stockout_steps_retailer' THEN m.metric_value END) AS so_std
    FROM runs r
    JOIN metrics m ON r.run_id = m.run_id
    WHERE r.status = 'completed'
      AND m.step_number IS NULL
      AND json_extract_string(r.metadata, '$.system_id') IS NOT NULL
      AND json_extract_string(r.metadata, '$.system_id') != ''
      AND json_extract_string(r.metadata, '$.system_id') != 'null'
    GROUP BY 1, 2
    ORDER BY 1, 2;
    """
    df = conn.execute(query).df()
    conn.close()
    return df


def format_results_table(df: pd.DataFrame) -> str:
    """Format DataFrame of mean and std into a clean, human-readable markdown table."""
    if df.empty:
        return "No completed runs found in database."

    lines = [
        "| System | Scenario | Seeds | Total Cost ($) | Service Level (%) | Fill Rate (%) | Retailer Bullwhip | Mfr Bullwhip | Retailer Stockout Days |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for _, row in df.iterrows():
        sys_name = row["system"]
        sc_name = row["scenario"]
        seeds = int(row["seeds"])

        cost_m = row["cost_mean"]
        cost_s = row["cost_std"] if pd.notnull(row["cost_std"]) else 0.0
        cost_str = f"${cost_m:,.0f} +/- {cost_s:,.0f}"

        sl_m = row["sl_mean"]
        sl_s = row["sl_std"] if pd.notnull(row["sl_std"]) else 0.0
        sl_str = f"{sl_m:.1f} +/- {sl_s:.1f}%"

        fr_m = row["fr_mean"]
        fr_s = row["fr_std"] if pd.notnull(row["fr_std"]) else 0.0
        fr_str = f"{fr_m:.1f} +/- {fr_s:.1f}%"

        rbw_m = row["ret_bw_mean"]
        rbw_s = row["ret_bw_std"] if pd.notnull(row["ret_bw_std"]) else 0.0
        rbw_str = f"{rbw_m:.2f} +/- {rbw_s:.2f}"

        mbw_m = row["mfr_bw_mean"]
        mbw_s = row["mfr_bw_std"] if pd.notnull(row["mfr_bw_std"]) else 0.0
        mbw_str = f"{mbw_m:.2f} +/- {mbw_s:.2f}"

        so_m = row["so_mean"]
        so_s = row["so_std"] if pd.notnull(row["so_std"]) else 0.0
        so_str = f"{so_m:.1f} +/- {so_s:.1f}"

        lines.append(
            f"| `{sys_name}` | **{sc_name}** | {seeds} | {cost_str} | {sl_str} | {fr_str} | {rbw_str} | {mbw_str} | {so_str} |"
        )

    return "\n".join(lines)


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Run supply chain evaluation grid across systems, scenarios, and random seeds."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment configuration YAML file.",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default=None,
        help="Optional comma-separated list of system IDs to evaluate (e.g. 'baseline_s_S,baseline_eoq').",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Optional comma-separated list of scenarios to evaluate (e.g. 'stable,seasonal,shock').",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Optional override for number of seeds to run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running and overwriting already completed runs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override max workers for non-LLM multiprocessing pool.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip simulation execution and print benchmark summary table from existing DuckDB.",
    )

    parser.add_argument(
        "--allow-config-change",
        action="store_true",
        help="Allow running real LLMs even if the configuration differs from benchmark_manifest.json.",
    )
    parser.add_argument(
        "--config-change-reason",
        type=str,
        default=None,
        help="Reason required when --allow-config-change is passed.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="results/benchmark_manifest.json",
        help="Path to benchmark manifest JSON file.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path.resolve())
        sys.exit(1)

    exp_cfg = load_experiment_config(config_path.parent)
    sim_cfg = load_sim_config(config_path.parent)

    runner = EvaluationRunner(
        experiment_config=exp_cfg,
        sim_config=sim_cfg,
        db_path=exp_cfg.db_path,
        force=args.force,
        allow_config_change=args.allow_config_change,
        config_change_reason=args.config_change_reason,
        manifest_path=args.manifest,
    )

    if not args.summary_only:
        sys_filter = [s.strip() for s in args.systems.split(",")] if args.systems else None
        sc_filter = [s.strip() for s in args.scenarios.split(",")] if args.scenarios else None
        seeds_override = (
            [exp_cfg.seed + i for i in range(args.seeds)]
            if args.seeds is not None
            else None
        )

        runner.run_grid(
            systems_filter=sys_filter,
            scenarios_filter=sc_filter,
            seeds_override=seeds_override,
            max_workers_override=args.max_workers,
        )

    # Print and save final statistical summary table
    df_summary = generate_benchmark_summary(runner.db_path)
    table_str = format_results_table(df_summary)

    print("\n" + "=" * 115)
    print("                      SUPPLY CHAIN EVALUATION BENCHMARK RESULTS (mean +/- std)")
    print("=" * 115)
    print(table_str)
    print("=" * 115 + "\n")

    # Save markdown summary
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "baseline_results.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# Baseline Evaluation Results (mean +/- std)\n\n")
        f.write(table_str + "\n")
    logger.info("Saved summary table to: %s", summary_file)


if __name__ == "__main__":
    main()
