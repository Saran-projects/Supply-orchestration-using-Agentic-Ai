"""DuckDB connection helper and schema manager for Supply Chain Agents (SCA).

Manages connection pooling, schema initialization, and structured logging for:
- runs
- steps
- agent_calls
- tool_calls
- messages
- metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb

REQUIRED_TABLES = [
    "runs",
    "steps",
    "agent_calls",
    "tool_calls",
    "messages",
    "metrics",
    "runs_registry",
]

SCHEMA_DDL = """
-- 1. Runs: Tracks overall simulation/benchmark runs
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    experiment_name VARCHAR NOT NULL,
    status VARCHAR DEFAULT 'running',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    config JSON,
    git_commit VARCHAR,
    metadata JSON
);

-- 2. Steps: Simulation time steps and environment state transitions
CREATE TABLE IF NOT EXISTS steps (
    step_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs(run_id),
    step_number INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sim_date VARCHAR,
    inventory_level INTEGER,
    backlog_units INTEGER,
    demand_units INTEGER,
    action_taken JSON,
    reward DOUBLE,
    info JSON
);

-- 3. Agent Calls: LLM invocations, token usage, latency, and costs
CREATE TABLE IF NOT EXISTS agent_calls (
    call_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs(run_id),
    step_id VARCHAR,
    agent_name VARCHAR NOT NULL,
    model_name VARCHAR,
    provider VARCHAR,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cached_tokens INTEGER DEFAULT 0,
    total_cost DOUBLE DEFAULT 0.0,
    cost_usd DOUBLE DEFAULT 0.0,
    latency_ms DOUBLE DEFAULT 0.0,
    retry_count INTEGER DEFAULT 0,
    structured_output_ok BOOLEAN,
    error_type VARCHAR,
    cache_hit BOOLEAN DEFAULT FALSE,
    usage_estimated BOOLEAN DEFAULT FALSE,
    system_name VARCHAR,
    echelon VARCHAR,
    step_number INTEGER,
    prompt_version VARCHAR,
    fallback BOOLEAN DEFAULT FALSE,
    order_quantity DOUBLE,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Tool Calls: Actions executed by agents (orders, forecasts, route adjustments)
CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id VARCHAR PRIMARY KEY,
    call_id VARCHAR,
    run_id VARCHAR NOT NULL REFERENCES runs(run_id),
    step_id VARCHAR,
    agent_name VARCHAR,
    tool_name VARCHAR NOT NULL,
    input_arguments JSON,
    output_result JSON,
    is_error BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    duration_ms DOUBLE DEFAULT 0.0,
    expected_tool VARCHAR,
    is_correct_tool BOOLEAN,
    is_parameter_valid BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Messages: Inter-agent or agent-user communications and debate logs
CREATE TABLE IF NOT EXISTS messages (
    message_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs(run_id),
    step_id VARCHAR,
    sender VARCHAR NOT NULL,
    receiver VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    content TEXT NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Metrics: Granular simulation, business, and operational KPIs
CREATE TABLE IF NOT EXISTS metrics (
    metric_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs(run_id),
    step_number INTEGER,
    metric_name VARCHAR NOT NULL,
    metric_value DOUBLE NOT NULL,
    category VARCHAR DEFAULT 'performance',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- 7. Orchestrated Decisions (System 6): Conflict detection, negotiation rounds, and resolution telemetry
CREATE TABLE IF NOT EXISTS orchestrated_decisions (
    decision_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs(run_id),
    step_number INTEGER NOT NULL,
    echelon VARCHAR NOT NULL,
    has_conflict BOOLEAN NOT NULL,
    conflict_type VARCHAR,
    rounds_used INTEGER NOT NULL DEFAULT 0,
    messages_exchanged INTEGER NOT NULL DEFAULT 0,
    resolution_method VARCHAR NOT NULL,
    final_order DOUBLE NOT NULL,
    total_tokens INTEGER DEFAULT 0,
    latency_ms DOUBLE DEFAULT 0.0,
    fallback_used BOOLEAN DEFAULT FALSE,
    proposals_json JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. Behavioural Metrics (Evaluation): Tool accuracy, groundedness, instruction following, efficiency
CREATE TABLE IF NOT EXISTS behavioural_metrics (
    metric_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs(run_id),
    system_name VARCHAR NOT NULL,
    scenario_name VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    seed INTEGER NOT NULL,
    tool_choice_accuracy DOUBLE,
    parameter_accuracy DOUBLE,
    instruction_following_rate DOUBLE,
    groundedness_score DOUBLE,
    fallback_rate DOUBLE,
    error_rate DOUBLE,
    avg_rounds DOUBLE,
    avg_messages DOUBLE,
    conflict_share DOUBLE,
    avg_tokens DOUBLE,
    avg_cost_usd DOUBLE,
    avg_latency_ms DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 9. Runs Registry: Tracks registered benchmark runs with provenance, config hash, and costs
CREATE TABLE IF NOT EXISTS runs_registry (
    run_id VARCHAR PRIMARY KEY,
    system VARCHAR NOT NULL,
    model VARCHAR,
    scenario VARCHAR NOT NULL,
    seed INTEGER NOT NULL,
    prompt_version VARCHAR,
    config_hash VARCHAR NOT NULL,
    data_source VARCHAR,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP,
    status VARCHAR DEFAULT 'running',
    cost_usd DOUBLE DEFAULT 0.0
);

-- Indexes for fast analytical queries and dashboard aggregation
CREATE INDEX IF NOT EXISTS idx_steps_run_num ON steps(run_id, step_number);
CREATE INDEX IF NOT EXISTS idx_agent_calls_run ON agent_calls(run_id, agent_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id, tool_name);
CREATE INDEX IF NOT EXISTS idx_messages_run ON messages(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run_name ON metrics(run_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_orchestrated_decisions_run ON orchestrated_decisions(run_id, step_number);
CREATE INDEX IF NOT EXISTS idx_behavioural_metrics_run ON behavioural_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_registry_config ON runs_registry(config_hash);
CREATE INDEX IF NOT EXISTS idx_runs_registry_system ON runs_registry(system, scenario, seed);
"""


def _to_json_str(val: Any) -> Optional[str]:
    """Serialize dictionary, list, or object to valid JSON string."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    return str(val)


def get_duckdb_connection(
    db_path: Union[str, Path] = ":memory:",
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Connect to a DuckDB database and ensure schema is initialized if writable."""
    if isinstance(db_path, Path):
        if not read_only:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        path_str = str(db_path)
    else:
        path_str = str(db_path)
        if path_str != ":memory:" and not read_only:
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(database=path_str, read_only=read_only)
    if not read_only:
        init_schema(conn)
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Execute DDL statements to create all 6 core tables and indices, and apply migrations."""
    conn.execute(SCHEMA_DDL)
    # Apply schema migrations for new agent_calls columns if existing table
    columns_to_add = [
        ("provider", "VARCHAR"),
        ("cached_tokens", "INTEGER DEFAULT 0"),
        ("cost_usd", "DOUBLE DEFAULT 0.0"),
        ("retry_count", "INTEGER DEFAULT 0"),
        ("structured_output_ok", "BOOLEAN"),
        ("error_type", "VARCHAR"),
        ("cache_hit", "BOOLEAN DEFAULT FALSE"),
        ("usage_estimated", "BOOLEAN DEFAULT FALSE"),
        ("system_name", "VARCHAR"),
        ("echelon", "VARCHAR"),
        ("step_number", "INTEGER"),
        ("prompt_version", "VARCHAR"),
        ("fallback", "BOOLEAN DEFAULT FALSE"),
        ("order_quantity", "DOUBLE"),
        ("reasoning", "TEXT"),
    ]
    for col_name, col_def in columns_to_add:
        try:
            conn.execute(f"ALTER TABLE agent_calls ADD COLUMN IF NOT EXISTS {col_name} {col_def};")
        except Exception:
            pass

    tool_cols_to_add = [
        ("expected_tool", "VARCHAR"),
        ("is_correct_tool", "BOOLEAN"),
        ("is_parameter_valid", "BOOLEAN"),
    ]
    for col_name, col_def in tool_cols_to_add:
        try:
            conn.execute(f"ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS {col_name} {col_def};")
        except Exception:
            pass


def verify_schema(conn: duckdb.DuckDBPyConnection) -> Dict[str, bool]:
    """Verify presence of all 6 required tables in the database."""
    tables_result = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    existing = {row[0].lower() for row in tables_result}
    return {table: (table in existing) for table in REQUIRED_TABLES}


class DuckDBManager:
    """Helper manager providing transactional logging for supply chain agent runs."""

    def __init__(self, db_path: Union[str, Path] = ":memory:", read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        self.conn = get_duckdb_connection(db_path=self.db_path, read_only=self.read_only)

    def close(self) -> None:
        """Close connection cleanly."""
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "DuckDBManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def verify(self) -> bool:
        """Return True if all 6 schema tables exist."""
        status = verify_schema(self.conn)
        return all(status.values())

    # --- Run Logging ---
    def create_run(
        self,
        run_id: str,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        git_commit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO runs (run_id, experiment_name, status, config, git_commit, metadata)
            VALUES (?, ?, 'running', ?, ?, ?)
            """,
            [
                run_id,
                experiment_name,
                _to_json_str(config),
                git_commit,
                _to_json_str(metadata),
            ],
        )

    def end_run(self, run_id: str, status: str = "completed") -> None:
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, ended_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
            """,
            [status, run_id],
        )

    # --- Runs Registry Logging ---
    def register_run(
        self,
        run_id: str,
        system: str,
        scenario: str,
        seed: int,
        config_hash: str,
        model: Optional[str] = None,
        prompt_version: Optional[str] = None,
        data_source: Optional[str] = None,
        status: str = "running",
        cost_usd: float = 0.0,
        started_at: Optional[Any] = None,
    ) -> None:
        """Register a benchmark run in runs_registry table."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs_registry (
                run_id, system, model, scenario, seed, prompt_version,
                config_hash, data_source, started_at, status, cost_usd
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?, ?)
            """,
            [
                run_id,
                system,
                model,
                scenario,
                seed,
                prompt_version,
                config_hash,
                data_source,
                started_at,
                status,
                cost_usd,
            ],
        )

    def finish_registered_run(
        self,
        run_id: str,
        status: str = "completed",
        cost_usd: Optional[float] = None,
        finished_at: Optional[Any] = None,
    ) -> None:
        """Update finished_at, status, and cost_usd for a registered run."""
        if cost_usd is not None:
            self.conn.execute(
                """
                UPDATE runs_registry
                SET status = ?, finished_at = COALESCE(?, CURRENT_TIMESTAMP), cost_usd = ?
                WHERE run_id = ?
                """,
                [status, finished_at, cost_usd, run_id],
            )
        else:
            self.conn.execute(
                """
                UPDATE runs_registry
                SET status = ?, finished_at = COALESCE(?, CURRENT_TIMESTAMP)
                WHERE run_id = ?
                """,
                [status, finished_at, run_id],
            )

    # --- Step Logging ---
    def log_step(
        self,
        step_id: str,
        run_id: str,
        step_number: int,
        sim_date: Optional[str] = None,
        inventory_level: Optional[int] = None,
        backlog_units: Optional[int] = None,
        demand_units: Optional[int] = None,
        action_taken: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO steps (step_id, run_id, step_number, sim_date, inventory_level,
                               backlog_units, demand_units, action_taken, reward, info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                step_id,
                run_id,
                step_number,
                sim_date,
                inventory_level,
                backlog_units,
                demand_units,
                _to_json_str(action_taken),
                reward,
                _to_json_str(info),
            ],
        )

    # --- Agent Call Logging ---
    def log_agent_call(
        self,
        call_id: str,
        run_id: str,
        agent_name: str,
        step_id: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_cost: float = 0.0,
        latency_ms: float = 0.0,
        provider: Optional[str] = None,
        cached_tokens: int = 0,
        cost_usd: Optional[float] = None,
        retry_count: int = 0,
        structured_output_ok: Optional[bool] = None,
        error_type: Optional[str] = None,
        cache_hit: bool = False,
        usage_estimated: bool = False,
        system_name: Optional[str] = None,
        echelon: Optional[str] = None,
        step_number: Optional[int] = None,
        prompt_version: Optional[str] = None,
        fallback: bool = False,
        order_quantity: Optional[float] = None,
        reasoning: Optional[str] = None,
    ) -> None:
        effective_cost = cost_usd if cost_usd is not None else total_cost
        self.conn.execute(
            """
            INSERT INTO agent_calls (
                call_id, run_id, step_id, agent_name, model_name, provider,
                prompt_tokens, completion_tokens, cached_tokens, total_cost, cost_usd,
                latency_ms, retry_count, structured_output_ok, error_type, cache_hit, usage_estimated,
                system_name, echelon, step_number, prompt_version, fallback, order_quantity, reasoning
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                call_id,
                run_id,
                step_id,
                agent_name,
                model_name,
                provider,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                total_cost,
                effective_cost,
                latency_ms,
                retry_count,
                structured_output_ok,
                error_type,
                cache_hit,
                usage_estimated,
                system_name,
                echelon,
                step_number,
                prompt_version,
                fallback,
                order_quantity,
                reasoning,
            ],
        )

    # --- Tool Call Logging ---
    def log_tool_call(
        self,
        tool_call_id: str,
        run_id: str,
        tool_name: str,
        call_id: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        input_arguments: Optional[Dict[str, Any]] = None,
        output_result: Optional[Dict[str, Any]] = None,
        is_error: bool = False,
        error_message: Optional[str] = None,
        duration_ms: float = 0.0,
        expected_tool: Optional[str] = None,
        is_correct_tool: Optional[bool] = None,
        is_parameter_valid: Optional[bool] = None,
    ) -> None:
        eff_correct = (
            is_correct_tool
            if is_correct_tool is not None
            else ((tool_name == expected_tool) if expected_tool else None)
        )
        eff_param_valid = (
            is_parameter_valid
            if is_parameter_valid is not None
            else (not is_error)
        )
        self.conn.execute(
            """
            INSERT INTO tool_calls (tool_call_id, call_id, run_id, step_id, agent_name,
                                    tool_name, input_arguments, output_result, is_error,
                                    error_message, duration_ms, expected_tool,
                                    is_correct_tool, is_parameter_valid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                tool_call_id,
                call_id,
                run_id,
                step_id,
                agent_name,
                tool_name,
                _to_json_str(input_arguments),
                _to_json_str(output_result),
                is_error,
                error_message,
                duration_ms,
                expected_tool,
                eff_correct,
                eff_param_valid,
            ],
        )

    # --- Message Logging ---
    def log_message(
        self,
        message_id: str,
        run_id: str,
        sender: str,
        receiver: str,
        role: str,
        content: str,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO messages (message_id, run_id, step_id, sender, receiver,
                                  role, content, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                message_id,
                run_id,
                step_id,
                sender,
                receiver,
                role,
                content,
                _to_json_str(metadata),
            ],
        )

    # --- Metric Logging ---
    def log_metric(
        self,
        metric_id: str,
        run_id: str,
        metric_name: str,
        metric_value: float,
        step_number: Optional[int] = None,
        category: str = "performance",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO metrics (metric_id, run_id, step_number, metric_name,
                                 metric_value, category, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                metric_id,
                run_id,
                step_number,
                metric_name,
                metric_value,
                category,
                _to_json_str(metadata),
            ],
        )

    # --- Orchestrated Decision Logging (System 6) ---
    def log_orchestrated_decision(
        self,
        decision_id: str,
        run_id: str,
        step_number: int,
        echelon: str,
        has_conflict: bool,
        conflict_type: Optional[str],
        rounds_used: int,
        messages_exchanged: int,
        resolution_method: str,
        final_order: float,
        total_tokens: int = 0,
        latency_ms: float = 0.0,
        fallback_used: bool = False,
        proposals: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO orchestrated_decisions (
                decision_id, run_id, step_number, echelon, has_conflict, conflict_type,
                rounds_used, messages_exchanged, resolution_method, final_order,
                total_tokens, latency_ms, fallback_used, proposals_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                decision_id,
                run_id,
                step_number,
                echelon,
                has_conflict,
                conflict_type,
                rounds_used,
                messages_exchanged,
                resolution_method,
                final_order,
                total_tokens,
                latency_ms,
                fallback_used,
                _to_json_str(proposals),
            ],
        )

    # --- Query Utilities ---
    def query(self, sql: str, params: Optional[List[Any]] = None) -> List[Any]:
        """Execute a query and fetch all rows."""
        return self.conn.execute(sql, params or []).fetchall()

    def query_df(self, sql: str, params: Optional[List[Any]] = None):
        """Execute a query and return a pandas DataFrame."""
        return self.conn.execute(sql, params or []).df()
