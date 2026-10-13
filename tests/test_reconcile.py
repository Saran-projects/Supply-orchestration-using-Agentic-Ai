"""Unit tests for sca.llm.reconcile module."""

import duckdb
import pytest

from sca.llm.reconcile import (
    print_reconcile_report,
    query_spend_by_model,
    recompute_model_costs,
)


def test_recompute_and_query_spend(tmp_path):
    """Test recomputing cost_usd from tokens and querying spend per model."""
    db_file = tmp_path / "test_telemetry.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE agent_calls (
            call_id VARCHAR PRIMARY KEY,
            model_name VARCHAR,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            cache_hit BOOLEAN,
            cost_usd DOUBLE,
            total_cost DOUBLE
        )
    """)
    # Insert test calls:
    # 1: 1000 prompt, 1000 comp, non-cached (initial cost 0)
    # 2: 2000 prompt, 500 comp, cache hit (initial cost 0)
    con.execute("""
        INSERT INTO agent_calls VALUES
        ('c1', 'glm-isoquant', 1000, 1000, FALSE, 0.0, 0.0),
        ('c2', 'glm-isoquant', 2000, 500, TRUE, 0.0, 0.0),
        ('c3', 'mock-model', 500, 500, FALSE, 0.0, 0.0)
    """)
    con.close()

    # Recompute costs using models.yaml rates
    updated = recompute_model_costs(model_name="glm-isoquant", db_paths=[db_file])
    assert updated == 2

    # Query spend
    stats = query_spend_by_model([db_file])
    assert len(stats) == 2

    glm_stat = next(s for s in stats if s["model"] == "glm-isoquant")
    assert glm_stat["calls"] == 2
    assert glm_stat["cache_hits"] == 1
    assert glm_stat["prompt_tokens"] == 3000
    assert glm_stat["completion_tokens"] == 1500
    # Expected cost for c1: (1000/1000)*0.000027 + (1000/1000)*0.00049 = 0.000517
    # c2 is cache hit -> 0.0
    assert glm_stat["cost_usd"] == pytest.approx(0.000517, abs=1e-6)

    # Test report function runs without error when within threshold
    print_reconcile_report([db_file], console_figure=0.000517, recompute=False, fail_on_threshold=True)

    # Test report function raises ValueError when discrepancy exceeds 5%
    with pytest.raises(ValueError, match="exceeds the allowable 5.0% threshold"):
        print_reconcile_report([db_file], console_figure=0.00100, recompute=False, fail_on_threshold=True)


def test_cached_tokens_recompute(tmp_path):
    """Test that cached input tokens are billed at price_per_1k_cached_input ($0.018/1M)."""
    db_file = tmp_path / "test_cached.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE agent_calls (
            call_id VARCHAR PRIMARY KEY,
            model_name VARCHAR,
            prompt_tokens INTEGER,
            cached_tokens INTEGER,
            completion_tokens INTEGER,
            cache_hit BOOLEAN,
            cost_usd DOUBLE,
            total_cost DOUBLE
        )
    """)
    # 2000 prompt, 1000 cached in, 1000 completion:
    # uncached = 1000 * 0.000027 = 0.000027
    # cached = 1000 * 0.000018 = 0.000018
    # out = 1000 * 0.000490 = 0.000490
    # expected total = 0.000535
    con.execute("""
        INSERT INTO agent_calls VALUES
        ('c_cached', 'glm-isoquant', 2000, 1000, 1000, FALSE, 0.0, 0.0)
    """)
    con.close()

    updated = recompute_model_costs(model_name="glm-isoquant", db_paths=[db_file])
    assert updated == 1

    stats = query_spend_by_model([db_file])
    glm_stat = next(s for s in stats if s["model"] == "glm-isoquant")
    assert glm_stat["cost_usd"] == pytest.approx(0.000535, abs=1e-6)

