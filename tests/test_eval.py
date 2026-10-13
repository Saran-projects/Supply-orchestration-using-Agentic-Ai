"""Unit tests for EvaluationRunner, resumability, and DuckDB reporting."""


import pytest

from sca.config import ExperimentConfig, SystemSpec
from sca.db import DuckDBManager
from sca.eval.runner import (
    EvaluationRunner,
    format_results_table,
    generate_benchmark_summary,
)


@pytest.fixture
def temp_db(tmp_path):
    """Temporary DuckDB database path."""
    return tmp_path / "test_metrics.duckdb"


@pytest.fixture
def mini_exp_config(temp_db, tmp_path):
    """Minimal experiment config for fast testing."""
    return ExperimentConfig(
        experiment_name="test_experiment",
        db_path=temp_db,
        output_dir=tmp_path / "results",
        systems=[
            SystemSpec(id="baseline_s_S", type="heuristic", enabled=True),
            SystemSpec(id="baseline_eoq", type="heuristic", enabled=True),
        ],
        scenarios=["stable"],
        seeds=[42, 43],
    )


def test_runner_grid_and_execution(mini_exp_config, temp_db):
    """Verify EvaluationRunner executes grid, persists to DuckDB, and is resumable."""
    runner = EvaluationRunner(experiment_config=mini_exp_config, db_path=temp_db)

    # 1. Build grid: 2 systems x 1 scenario x 2 seeds = 4 runs
    grid = runner.build_grid()
    assert len(grid) == 4

    # 2. First execution pass: 4 runs should execute
    results = runner.run_grid()
    assert len(results) == 4
    for r in results:
        assert r.status == "completed"
        assert len(r.steps_data) > 0
        assert len(r.metrics_data) > 0

    # Verify DuckDB records
    completed = runner.get_completed_run_ids()
    assert len(completed) == 4

    # 3. Resumability check: Second execution should skip all completed runs
    resumed_results = runner.run_grid()
    assert len(resumed_results) == 0

    # 4. Generate summary
    summary_df = generate_benchmark_summary(temp_db)
    assert not summary_df.empty
    assert len(summary_df) == 2  # baseline_s_S and baseline_eoq
    assert "cost_mean" in summary_df.columns
    assert "cost_std" in summary_df.columns

    table = format_results_table(summary_df)
    assert "baseline_s_S" in table
    assert "baseline_eoq" in table

    # 5. Verify runs_registry table is populated
    with DuckDBManager(temp_db) as db_mgr:
        rows = db_mgr.query("SELECT run_id, system, scenario, seed, config_hash, status FROM runs_registry")
        assert len(rows) == 4
        for r in rows:
            assert r[5] == "completed"  # status
            assert r[4] is not None and len(r[4]) > 0  # config_hash


def test_runner_refuses_real_llm_on_config_drift(tmp_path, temp_db, monkeypatch):
    """Verify runner refuses to run real LLMs if manifest hash differs unless allowed."""
    import sca.eval.runner as runner_mod
    from sca.config import ModelsConfig

    # Configure an agentic system in experiment
    exp_cfg = ExperimentConfig(
        experiment_name="test_llm_drift",
        db_path=temp_db,
        output_dir=tmp_path / "results",
        systems=[
            SystemSpec(id="single_agent", type="agentic", enabled=True),
        ],
        scenarios=["stable"],
        seeds=[42],
    )

    # Mock load_models_config to return mock_mode=False (real LLM mode)
    real_models_cfg = ModelsConfig(mock_mode=False)
    monkeypatch.setattr(runner_mod, "load_models_config", lambda *args, **kwargs: real_models_cfg)

    # Point manifest to a fake file with mismatched hash
    manifest_file = tmp_path / "fake_manifest.json"
    manifest_file.write_text('{"config_hash": "mismatched_deadbeef_hash"}', encoding="utf-8")

    runner = EvaluationRunner(
        experiment_config=exp_cfg,
        db_path=temp_db,
        manifest_path=manifest_file,
        allow_config_change=False,
    )

    with pytest.raises(RuntimeError, match="Refusing to execute real-LLM run"):
        runner.run_grid()

    # Now verify that passing allow_config_change=True permits running
    # (Here we mock run_single_simulation so it doesn't make real API calls)
    from sca.eval.runner import RunResult, RunSpec
    dummy_result = RunResult(
        spec=RunSpec(system_id="single_agent", system_type="agentic", scenario_name="stable", seed=42, run_id="test_run_1"),
        status="completed",
        experiment_name="test_llm_drift",
    )
    monkeypatch.setattr(runner_mod, "run_single_simulation", lambda *args, **kwargs: dummy_result)

    runner_allowed = EvaluationRunner(
        experiment_config=exp_cfg,
        db_path=temp_db,
        manifest_path=manifest_file,
        allow_config_change=True,
        config_change_reason="Testing authorized parameter override",
    )
    res = runner_allowed.run_grid()
    assert len(res) == 1
    assert res[0].status == "completed"

