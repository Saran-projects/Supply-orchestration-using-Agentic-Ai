"""Smoke tests for SCA repository skeleton, configs, imports, and DuckDB schema."""

import importlib
from pathlib import Path

import pytest

from sca import (
    AppConfig,
    DataConfig,
    DuckDBManager,
    ExperimentConfig,
    ModelsConfig,
    SimConfig,
    load_data_config,
    load_experiment_config,
    load_models_config,
    load_sim_config,
    verify_schema,
)
from sca.db import REQUIRED_TABLES


class TestSmokeImports:
    """Verify that all core modules and subpackages import cleanly."""

    SUBPACKAGES = [
        "sca.data",
        "sca.sim",
        "sca.baselines",
        "sca.tools",
        "sca.forecast",
        "sca.llm",
        "sca.agents",
        "sca.systems",
        "sca.eval",
        "sca.logging",
        "sca.dashboard",
    ]

    @pytest.mark.parametrize("subpackage", SUBPACKAGES)
    def test_subpackage_import(self, subpackage: str):
        mod = importlib.import_module(subpackage)
        assert mod is not None, f"Failed to import {subpackage}"


class TestSmokeConfigs:
    """Verify loading and Pydantic validation of all YAML configuration files."""

    def test_load_individual_configs(self, project_root: Path):
        configs_dir = project_root / "configs"

        data_cfg = load_data_config(configs_dir)
        assert isinstance(data_cfg, DataConfig)
        assert data_cfg.dataset_filename == "DataCoSupplyChainDataset.csv"
        assert data_cfg.target_column == "Late_delivery_risk"

        sim_cfg = load_sim_config(configs_dir)
        assert isinstance(sim_cfg, SimConfig)
        assert sim_cfg.time_horizon_days > 0
        assert len(sim_cfg.nodes.suppliers) > 0
        assert len(sim_cfg.nodes.distribution_centers) > 0

        exp_cfg = load_experiment_config(configs_dir)
        assert isinstance(exp_cfg, ExperimentConfig)
        assert exp_cfg.experiment_name != ""
        assert len(exp_cfg.systems) > 0

        models_cfg = load_models_config(configs_dir)
        assert isinstance(models_cfg, ModelsConfig)
        assert models_cfg.default_model != ""
        assert "supervisor" in models_cfg.agents

    def test_load_unified_app_config(self, app_config: AppConfig):
        assert isinstance(app_config, AppConfig)
        assert isinstance(app_config.data, DataConfig)
        assert isinstance(app_config.sim, SimConfig)
        assert isinstance(app_config.experiment, ExperimentConfig)
        assert isinstance(app_config.models, ModelsConfig)


class TestSmokeDuckDB:
    """Verify DuckDB connection helper, schema creation, and table operations."""

    def test_schema_creation_and_tables(self, memory_db):
        status = verify_schema(memory_db)
        for tbl in REQUIRED_TABLES:
            assert tbl in status, f"Missing table verification key: {tbl}"
            assert status[tbl] is True, f"Table was not created: {tbl}"

    def test_duckdb_manager_crud_cycle(self, db_manager: DuckDBManager):
        assert db_manager.verify() is True

        run_id = "test_run_001"
        step_id = "test_step_001"
        call_id = "test_call_001"
        tool_id = "test_tool_001"
        msg_id = "test_msg_001"
        metric_id = "test_metric_001"

        # 1. Runs
        db_manager.create_run(
            run_id=run_id,
            experiment_name="smoke_test_experiment",
            config={"seed": 42},
            metadata={"env": "pytest"},
        )
        runs_df = db_manager.query_df("SELECT * FROM runs WHERE run_id = ?", [run_id])
        assert len(runs_df) == 1
        assert runs_df.iloc[0]["status"] == "running"

        # 2. Steps
        db_manager.log_step(
            step_id=step_id,
            run_id=run_id,
            step_number=1,
            sim_date="2026-10-06",
            inventory_level=5000,
            backlog_units=0,
            demand_units=120,
            action_taken={"order_qty": 200},
            reward=10.5,
            info={"note": "smoke step"},
        )
        steps_df = db_manager.query_df("SELECT * FROM steps WHERE step_id = ?", [step_id])
        assert len(steps_df) == 1
        assert steps_df.iloc[0]["inventory_level"] == 5000

        # 3. Agent Calls
        db_manager.log_agent_call(
            call_id=call_id,
            run_id=run_id,
            agent_name="InventoryAgent",
            step_id=step_id,
            model_name="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            total_cost=0.0002,
            latency_ms=150.0,
        )
        calls_df = db_manager.query_df("SELECT * FROM agent_calls WHERE call_id = ?", [call_id])
        assert len(calls_df) == 1
        assert calls_df.iloc[0]["agent_name"] == "InventoryAgent"

        # 4. Tool Calls
        db_manager.log_tool_call(
            tool_call_id=tool_id,
            call_id=call_id,
            run_id=run_id,
            step_id=step_id,
            agent_name="InventoryAgent",
            tool_name="get_stock_level",
            input_arguments={"dc_id": "DC_EAST"},
            output_result={"stock": 5000},
        )
        tools_df = db_manager.query_df("SELECT * FROM tool_calls WHERE tool_call_id = ?", [tool_id])
        assert len(tools_df) == 1
        assert tools_df.iloc[0]["tool_name"] == "get_stock_level"

        # 5. Messages
        db_manager.log_message(
            message_id=msg_id,
            run_id=run_id,
            step_id=step_id,
            sender="InventoryAgent",
            receiver="SupervisorAgent",
            role="agent",
            content="Stock level is sufficient for upcoming demand spike.",
        )
        msgs_df = db_manager.query_df("SELECT * FROM messages WHERE message_id = ?", [msg_id])
        assert len(msgs_df) == 1
        assert msgs_df.iloc[0]["sender"] == "InventoryAgent"

        # 6. Metrics
        db_manager.log_metric(
            metric_id=metric_id,
            run_id=run_id,
            step_number=1,
            metric_name="service_level",
            metric_value=0.975,
            category="operational",
        )
        metrics_df = db_manager.query_df("SELECT * FROM metrics WHERE metric_id = ?", [metric_id])
        assert len(metrics_df) == 1
        assert metrics_df.iloc[0]["metric_value"] == 0.975

        # End Run
        db_manager.end_run(run_id=run_id, status="completed")
        ended_run = db_manager.query_df("SELECT status FROM runs WHERE run_id = ?", [run_id])
        assert ended_run.iloc[0]["status"] == "completed"


class TestSmokeDataset:
    """Verify presence of DataCo CSV in data/raw."""

    def test_dataco_raw_file_exists(self, project_root: Path):
        raw_csv = project_root / "data" / "raw" / "DataCoSupplyChainDataset.csv"
        assert raw_csv.is_file(), f"Expected DataCo dataset at {raw_csv}"
        assert raw_csv.stat().st_size > 0, "Dataset file is empty"
