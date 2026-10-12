"""Behavioural Evaluation Metrics for Supply Chain Multi-Agent Systems.

Computes comprehensive behavioural metrics from DuckDB execution logs per
(system, scenario, model, seed):
1. Tool-choice accuracy: actual tool called vs expected_tool label.
2. Parameter accuracy: tool inputs vs true observed state within tolerance.
3. Instruction-following rate: valid, in-range outputs on the first attempt (no retries/fallbacks).
4. Groundedness: programmatic regex extraction of numbers in reasoning text checked against
   true state numbers within tolerance (no LLM judge required).
5. Fallback and error rates: frequency of execution fallbacks and system errors.
6. Negotiation efficiency: average rounds, messages per decision, and conflict share.
7. Resource consumption: tokens, cost in USD (from models.yaml pricing, cache hits cost $0),
   and response latency per decision.

Writes all results to DuckDB (both `metrics` and `behavioural_metrics` tables).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import duckdb

from sca.db import get_duckdb_connection
from sca.llm.client import calculate_cost, load_models_config

logger = logging.getLogger(__name__)

# Regex for extracting integers and floating point numbers (e.g. 100, 100.5, -20.0)
NUMBER_REGEX = re.compile(r"(?<![A-Za-z_])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?![A-Za-z_])")


# ---------------------------------------------------------------------------
# 1. Behavioural Metrics Data Class
# ---------------------------------------------------------------------------
@dataclass
class BehaviouralMetrics:
    """Evaluated behavioural metrics for a single simulation run."""

    system: str
    scenario: str
    model: str
    seed: int
    run_id: str

    # 1. Tool-Use Quality
    tool_choice_accuracy: Optional[float] = None
    parameter_accuracy: Optional[float] = None
    total_tool_calls: int = 0

    # 2. Instruction Adherence & Resilience
    instruction_following_rate: float = 1.0
    fallback_rate: float = 0.0
    error_rate: float = 0.0
    total_decisions: int = 0

    # 3. Groundedness (Programmatic State Alignment)
    groundedness_score: float = 1.0
    total_numbers_checked: int = 0
    grounded_numbers_count: int = 0

    # 4. Multi-Agent Negotiation Efficiency
    avg_rounds_per_decision: float = 0.0
    avg_messages_per_decision: float = 0.0
    conflict_share: float = 0.0

    # 5. Resource Consumption & Latency
    avg_tokens_per_decision: float = 0.0
    avg_cost_usd_per_decision: float = 0.0
    avg_latency_ms_per_decision: float = 0.0
    cache_hit_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# 2. Programmatic Groundedness Evaluator
# ---------------------------------------------------------------------------
def extract_numbers_from_text(text: str) -> List[float]:
    """Extract all numerical values from text using regular expressions."""
    if not text:
        return []
    matches = NUMBER_REGEX.findall(text)
    nums: List[float] = []
    for m in matches:
        try:
            val = float(m)
            nums.append(val)
        except ValueError:
            pass
    return nums


def is_number_grounded(
    num: float,
    candidate_values: Set[float],
    rel_tol: float = 0.05,
    abs_tol: float = 1.0,
) -> bool:
    """Return True if num is within relative or absolute tolerance of any candidate value."""
    for c in candidate_values:
        diff = abs(num - c)
        threshold = max(abs_tol, abs(c) * rel_tol)
        if diff <= threshold:
            return True
    return False


def collect_candidate_state_values(step_info: Dict[str, Any], echelon: str) -> Set[float]:
    """Extract all true ground-truth numeric values from a simulation step state."""
    candidates: Set[float] = set()

    if not isinstance(step_info, dict):
        return candidates

    # 1. Nested echelon data
    ech_info = step_info.get(echelon, {}) if isinstance(step_info.get(echelon), dict) else {}
    for _k, v in ech_info.items():
        if isinstance(v, (int, float)):
            candidates.add(float(v))

    # 2. Top-level state fields (flat dicts or summary metrics)
    for _k, v in step_info.items():
        if isinstance(v, (int, float)):
            candidates.add(float(v))

    # 3. Add inventory position if components are available
    inv = float(ech_info.get("inventory", step_info.get("inventory", 0.0)))
    pipe = float(ech_info.get("pipeline", step_info.get("pipeline", 0.0)))
    back = float(ech_info.get("backlog", step_info.get("backlog", 0.0)))
    candidates.add(inv + pipe - back)

    # 4. Standard echelon capacity limits
    if echelon == "manufacturer":
        candidates.add(300.0)
    else:
        candidates.add(500.0)

    return candidates


# ---------------------------------------------------------------------------
# 3. Modular Metric Calculators
# ---------------------------------------------------------------------------
def compute_tool_choice_accuracy(conn: duckdb.DuckDBPyConnection, run_id: str) -> Tuple[Optional[float], int]:
    """Compute tool-choice accuracy: actual vs expected_tool label."""
    row = conn.execute(
        """
        SELECT
            count(*) AS total_calls,
            count(CASE WHEN is_correct_tool THEN 1 END) AS correct_calls
        FROM tool_calls
        WHERE run_id = ? AND expected_tool IS NOT NULL
        """,
        [run_id],
    ).fetchone()

    if not row or row[0] == 0:
        return None, 0

    total, correct = row
    return round(float(correct) / float(total), 4), total


def compute_parameter_accuracy(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    tolerance: float = 0.05,
) -> Tuple[Optional[float], int]:
    """Compute parameter accuracy: tool inputs vs true observed state within tolerance."""
    rows = conn.execute(
        """
        SELECT
            tc.tool_call_id,
            tc.tool_name,
            tc.input_arguments,
            tc.is_parameter_valid,
            tc.is_error,
            s.info AS step_info
        FROM tool_calls tc
        LEFT JOIN steps s ON tc.run_id = s.run_id AND (tc.step_id = s.step_id OR tc.step_id = CAST(s.step_number AS VARCHAR))
        WHERE tc.run_id = ?
        """,
        [run_id],
    ).fetchall()

    if not rows:
        return None, 0

    valid_count = 0
    total_calls = len(rows)

    for _t_id, tool_name, in_args_json, is_valid_flag, is_error, step_info_json in rows:
        if is_error:
            continue

        in_args = json.loads(in_args_json) if isinstance(in_args_json, str) else (in_args_json or {})
        step_info = json.loads(step_info_json) if isinstance(step_info_json, str) else (step_info_json or {})

        # Direct schema validity flag
        if is_valid_flag is False:
            continue

        # Ground-truth cross-checks against state
        accurate = True
        if tool_name == "ss_policy_order" and step_info:
            arg_inv = float(in_args.get("inventory", -999.0))
            if arg_inv >= 0:
                # Check against flat state
                flat_inv = float(step_info.get("inventory", -999.0))
                if flat_inv >= 0 and abs(arg_inv - flat_inv) > max(1.0, flat_inv * tolerance):
                    accurate = False
                # Check against nested echelon states
                for ech in ("retailer", "wholesaler", "distributor", "manufacturer"):
                    ech_data = step_info.get(ech, {})
                    if isinstance(ech_data, dict):
                        state_inv = float(ech_data.get("inventory", -999.0))
                        if state_inv >= 0 and abs(arg_inv - state_inv) > max(1.0, state_inv * tolerance):
                            accurate = False

        elif tool_name == "eoq_quantity" and step_info:
            arg_dem = float(in_args.get("mean_demand", -999.0))
            if arg_dem >= 0:
                flat_dem = float(step_info.get("demand", step_info.get("current_demand", -999.0)))
                if flat_dem >= 0 and abs(arg_dem - flat_dem) > max(1.0, flat_dem * tolerance):
                    accurate = False

        if accurate:
            valid_count += 1

    return round(float(valid_count) / float(total_calls), 4), total_calls


def compute_instruction_following_rate(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    max_capacity: float = 500.0,
) -> Tuple[float, int]:
    """Instruction-following rate: valid, in-range outputs on the first attempt."""
    # Check agent_calls (Systems 3, 4, 5, 6)
    agent_row = conn.execute(
        """
        SELECT
            count(*) AS total_calls,
            count(CASE
                WHEN retry_count = 0
                 AND (structured_output_ok IS NULL OR structured_output_ok = TRUE)
                 AND fallback = FALSE
                 AND order_quantity >= 0.0
                 AND order_quantity <= ?
                THEN 1 END) AS valid_first_attempt
        FROM agent_calls
        WHERE run_id = ?
        """,
        [max_capacity, run_id],
    ).fetchone()

    # Check orchestrated_decisions (System 6)
    orch_row = conn.execute(
        """
        SELECT
            count(*) AS total_decisions,
            count(CASE
                WHEN fallback_used = FALSE
                 AND final_order >= 0.0
                 AND final_order <= ?
                THEN 1 END) AS valid_decisions
        FROM orchestrated_decisions
        WHERE run_id = ?
        """,
        [max_capacity, run_id],
    ).fetchone()

    total_decisions = 0
    valid_decisions = 0

    if agent_row and agent_row[0] > 0:
        total_decisions += agent_row[0]
        valid_decisions += agent_row[1]
    elif orch_row and orch_row[0] > 0:
        total_decisions += orch_row[0]
        valid_decisions += orch_row[1]

    if total_decisions == 0:
        return 1.0, 0

    return round(float(valid_decisions) / float(total_decisions), 4), total_decisions


def compute_groundedness(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    tolerance: float = 0.05,
) -> Tuple[float, int, int]:
    """Programmatic groundedness score: extract numbers from reasoning with regex and check state."""
    texts_and_states: List[Tuple[str, Set[float]]] = []

    # 1. From agent_calls reasoning
    ac_rows = conn.execute(
        """
        SELECT ac.reasoning, ac.echelon, s.info
        FROM agent_calls ac
        LEFT JOIN steps s ON ac.run_id = s.run_id AND ac.step_number = s.step_number
        WHERE ac.run_id = ? AND ac.reasoning IS NOT NULL AND ac.reasoning != ''
        """,
        [run_id],
    ).fetchall()

    for reasoning, echelon, step_info_json in ac_rows:
        step_info = json.loads(step_info_json) if isinstance(step_info_json, str) else (step_info_json or {})
        candidates = collect_candidate_state_values(step_info, echelon or "retailer")
        texts_and_states.append((reasoning, candidates))

    # 2. From orchestrated_decisions proposals
    od_rows = conn.execute(
        """
        SELECT od.proposals_json, od.echelon, s.info
        FROM orchestrated_decisions od
        LEFT JOIN steps s ON od.run_id = s.run_id AND od.step_number = s.step_number
        WHERE od.run_id = ? AND od.proposals_json IS NOT NULL
        """,
        [run_id],
    ).fetchall()

    for p_json, echelon, step_info_json in od_rows:
        proposals = json.loads(p_json) if isinstance(p_json, str) else (p_json or {})
        step_info = json.loads(step_info_json) if isinstance(step_info_json, str) else (step_info_json or {})
        candidates = collect_candidate_state_values(step_info, echelon or "retailer")
        for _name, val in proposals.items():
            if isinstance(val, (int, float)):
                candidates.add(float(val))

    # 3. From messages content
    msg_rows = conn.execute(
        """
        SELECT content FROM messages WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()
    for (content,) in msg_rows:
        # Check against generic supply chain values
        candidates = {0.0, 100.0, 150.0, 200.0, 300.0, 500.0}
        texts_and_states.append((content, candidates))

    total_numbers = 0
    grounded_numbers = 0

    for text, candidates in texts_and_states:
        nums = extract_numbers_from_text(text)
        for n in nums:
            total_numbers += 1
            if is_number_grounded(n, candidates, rel_tol=tolerance):
                grounded_numbers += 1

    if total_numbers == 0:
        return 1.0, 0, 0

    score = round(float(grounded_numbers) / float(total_numbers), 4)
    return score, grounded_numbers, total_numbers


def compute_fallback_and_error_rate(conn: duckdb.DuckDBPyConnection, run_id: str) -> Tuple[float, float]:
    """Compute fallback rate and execution error rate."""
    # Fallbacks from agent_calls and orchestrated_decisions
    fb_agent = conn.execute(
        "SELECT count(*), count(CASE WHEN fallback THEN 1 END) FROM agent_calls WHERE run_id = ?",
        [run_id],
    ).fetchone()

    fb_orch = conn.execute(
        "SELECT count(*), count(CASE WHEN fallback_used THEN 1 END) FROM orchestrated_decisions WHERE run_id = ?",
        [run_id],
    ).fetchone()

    tot_dec = 0
    tot_fb = 0

    if fb_agent and fb_agent[0] > 0:
        tot_dec += fb_agent[0]
        tot_fb += fb_agent[1]
    elif fb_orch and fb_orch[0] > 0:
        tot_dec += fb_orch[0]
        tot_fb += fb_orch[1]

    fallback_rate = round(float(tot_fb) / float(tot_dec), 4) if tot_dec > 0 else 0.0

    # Errors across agent_calls and tool_calls
    err_agent = conn.execute(
        "SELECT count(*), count(CASE WHEN error_type IS NOT NULL THEN 1 END) FROM agent_calls WHERE run_id = ?",
        [run_id],
    ).fetchone()
    err_tool = conn.execute(
        "SELECT count(*), count(CASE WHEN is_error THEN 1 END) FROM tool_calls WHERE run_id = ?",
        [run_id],
    ).fetchone()

    total_calls = (err_agent[0] if err_agent else 0) + (err_tool[0] if err_tool else 0)
    total_errors = (err_agent[1] if err_agent else 0) + (err_tool[1] if err_tool else 0)

    error_rate = round(float(total_errors) / float(total_calls), 4) if total_calls > 0 else 0.0

    return fallback_rate, error_rate


def compute_negotiation_efficiency(conn: duckdb.DuckDBPyConnection, run_id: str) -> Tuple[float, float, float]:
    """Compute average negotiation rounds, messages exchanged, and conflict share."""
    row = conn.execute(
        """
        SELECT
            count(*) AS total_decisions,
            sum(rounds_used) AS total_rounds,
            sum(messages_exchanged) AS total_messages,
            count(CASE WHEN has_conflict THEN 1 END) AS conflict_count
        FROM orchestrated_decisions
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()

    if not row or row[0] == 0:
        return 0.0, 0.0, 0.0

    tot_d, tot_r, tot_m, conf_cnt = row
    avg_rounds = round(float(tot_r or 0) / float(tot_d), 3)
    avg_msgs = round(float(tot_m or 0) / float(tot_d), 3)
    conflict_share = round(float(conf_cnt or 0) / float(tot_d), 4)

    return avg_rounds, avg_msgs, conflict_share


def compute_resource_consumption(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    models_config_path: Optional[Path] = None,
) -> Tuple[float, float, float, float]:
    """Compute tokens, cost in USD (cache hits cost $0), latency (ms), and cache hit rate."""
    rows = conn.execute(
        """
        SELECT
            model_name,
            prompt_tokens,
            completion_tokens,
            cost_usd,
            latency_ms,
            cache_hit
        FROM agent_calls
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()

    # Also check orchestrated_decisions if agent_calls was not logged directly
    orch_rows = conn.execute(
        """
        SELECT total_tokens, latency_ms FROM orchestrated_decisions WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()

    if not rows and not orch_rows:
        return 0.0, 0.0, 0.0, 0.0

    total_tokens = 0
    total_cost_usd = 0.0
    total_latency_ms = 0.0
    total_calls = len(rows)
    cache_hits = 0

    models_cfg = None
    try:
        models_cfg = load_models_config(models_config_path)
    except Exception:
        pass

    for mod_name, p_tok, c_tok, logged_cost, lat_ms, is_cache_hit in rows:
        p_t = p_tok or 0
        c_t = c_tok or 0
        total_tokens += (p_t + c_t)
        total_latency_ms += (lat_ms or 0.0)

        if is_cache_hit:
            cache_hits += 1
            call_cost = 0.0
        else:
            if logged_cost is not None and logged_cost > 0.0:
                call_cost = float(logged_cost)
            else:
                call_cost = calculate_cost(
                    model=mod_name or "mock-model",
                    prompt_tokens=p_t,
                    completion_tokens=c_t,
                    config=models_cfg,
                )
        total_cost_usd += call_cost

    if not rows and orch_rows:
        total_calls = len(orch_rows)
        for t_tok, lat in orch_rows:
            total_tokens += (t_tok or 0)
            total_latency_ms += (lat or 0.0)
            # Estimate cost for mock negotiation tokens
            total_cost_usd += calculate_cost("mock-model", t_tok or 0, 0, config=models_cfg)

    denom = max(total_calls, 1)
    avg_tokens = round(float(total_tokens) / float(denom), 2)
    avg_cost = round(float(total_cost_usd) / float(denom), 6)
    avg_lat = round(float(total_latency_ms) / float(denom), 2)
    cache_hit_rate = round(float(cache_hits) / float(denom), 4)

    return avg_tokens, avg_cost, avg_lat, cache_hit_rate


# ---------------------------------------------------------------------------
# 4. Master Evaluation Functions
# ---------------------------------------------------------------------------
def extract_run_metadata(conn: duckdb.DuckDBPyConnection, run_id: str) -> Dict[str, Any]:
    """Extract system, scenario, model, seed metadata for a run."""
    meta: Dict[str, Any] = {
        "system": "unknown",
        "scenario": "unknown",
        "model": "mock",
        "seed": 42,
    }

    row = conn.execute("SELECT config, metadata FROM runs WHERE run_id = ?", [run_id]).fetchone()
    if row:
        cfg_j, meta_j = row
        cfg = json.loads(cfg_j) if isinstance(cfg_j, str) else (cfg_j or {})
        m = json.loads(meta_j) if isinstance(meta_j, str) else (meta_j or {})
        for src in (cfg, m):
            if "system" in src:
                meta["system"] = src["system"]
            elif "system_name" in src:
                meta["system"] = src["system_name"]
            if "scenario" in src:
                meta["scenario"] = src["scenario"]
            elif "scenario_name" in src:
                meta["scenario"] = src["scenario_name"]
            if "model" in src:
                meta["model"] = src["model"]
            elif "model_name" in src:
                meta["model"] = src["model_name"]
            if "seed" in src:
                meta["seed"] = int(src["seed"])

    # Fallback parsing from run_id naming pattern: <system>_<model>_<scenario>_seed_<seed>
    parts = run_id.split("_")
    if meta["system"] == "unknown" and len(parts) >= 2:
        meta["system"] = parts[0]
    if "seed" in run_id:
        try:
            seed_idx = parts.index("seed")
            meta["seed"] = int(parts[seed_idx + 1])
        except Exception:
            pass

    return meta


def compute_behavioural_metrics(
    db_path: Union[str, Path],
    run_id: str,
    tolerance: float = 0.05,
    write_to_db: bool = True,
    models_config_path: Optional[Path] = None,
) -> BehaviouralMetrics:
    """Compute all behavioral evaluation metrics for a specific run and optionally persist to DuckDB."""
    conn = get_duckdb_connection(db_path)

    # 1. Extract metadata
    meta = extract_run_metadata(conn, run_id)

    # 2. Compute individual components
    tool_acc, total_tools = compute_tool_choice_accuracy(conn, run_id)
    param_acc, _ = compute_parameter_accuracy(conn, run_id, tolerance=tolerance)
    instr_rate, total_dec = compute_instruction_following_rate(conn, run_id)
    ground_score, ground_cnt, total_nums = compute_groundedness(conn, run_id, tolerance=tolerance)
    fb_rate, err_rate = compute_fallback_and_error_rate(conn, run_id)
    avg_rounds, avg_msgs, conf_share = compute_negotiation_efficiency(conn, run_id)
    avg_tok, avg_cost, avg_lat, cache_rate = compute_resource_consumption(
        conn, run_id, models_config_path=models_config_path
    )

    bm = BehaviouralMetrics(
        system=meta["system"],
        scenario=meta["scenario"],
        model=meta["model"],
        seed=meta["seed"],
        run_id=run_id,
        tool_choice_accuracy=tool_acc,
        parameter_accuracy=param_acc,
        total_tool_calls=total_tools,
        instruction_following_rate=instr_rate,
        fallback_rate=fb_rate,
        error_rate=err_rate,
        total_decisions=total_dec,
        groundedness_score=ground_score,
        total_numbers_checked=total_nums,
        grounded_numbers_count=ground_cnt,
        avg_rounds_per_decision=avg_rounds,
        avg_messages_per_decision=avg_msgs,
        conflict_share=conf_share,
        avg_tokens_per_decision=avg_tok,
        avg_cost_usd_per_decision=avg_cost,
        avg_latency_ms_per_decision=avg_lat,
        cache_hit_rate=cache_rate,
    )

    # 3. Write results to DuckDB tables if requested
    if write_to_db:
        # 3a. Insert into behavioural_metrics table
        conn.execute(
            """
            INSERT OR REPLACE INTO behavioural_metrics (
                metric_id, run_id, system_name, scenario_name, model_name, seed,
                tool_choice_accuracy, parameter_accuracy, instruction_following_rate,
                groundedness_score, fallback_rate, error_rate,
                avg_rounds, avg_messages, conflict_share,
                avg_tokens, avg_cost_usd, avg_latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"bm_{uuid.uuid4().hex[:12]}",
                bm.run_id,
                bm.system,
                bm.scenario,
                bm.model,
                bm.seed,
                bm.tool_choice_accuracy,
                bm.parameter_accuracy,
                bm.instruction_following_rate,
                bm.groundedness_score,
                bm.fallback_rate,
                bm.error_rate,
                bm.avg_rounds_per_decision,
                bm.avg_messages_per_decision,
                bm.conflict_share,
                bm.avg_tokens_per_decision,
                bm.avg_cost_usd_per_decision,
                bm.avg_latency_ms_per_decision,
            ],
        )

        # 3b. Insert granular rows into metrics table
        meta_json = json.dumps({
            "system": bm.system,
            "scenario": bm.scenario,
            "model": bm.model,
            "seed": bm.seed,
        })
        for m_name, m_val in [
            ("tool_choice_accuracy", bm.tool_choice_accuracy),
            ("parameter_accuracy", bm.parameter_accuracy),
            ("instruction_following_rate", bm.instruction_following_rate),
            ("groundedness_score", bm.groundedness_score),
            ("fallback_rate", bm.fallback_rate),
            ("error_rate", bm.error_rate),
            ("avg_rounds_per_decision", bm.avg_rounds_per_decision),
            ("avg_messages_per_decision", bm.avg_messages_per_decision),
            ("conflict_share", bm.conflict_share),
            ("avg_tokens_per_decision", bm.avg_tokens_per_decision),
            ("avg_cost_usd_per_decision", bm.avg_cost_usd_per_decision),
            ("avg_latency_ms_per_decision", bm.avg_latency_ms_per_decision),
        ]:
            if m_val is not None:
                conn.execute(
                    """
                    INSERT INTO metrics (metric_id, run_id, step_number, metric_name,
                                         metric_value, category, metadata)
                    VALUES (?, ?, NULL, ?, ?, 'behaviour', ?)
                    """,
                    [
                        f"met_{m_name}_{uuid.uuid4().hex[:8]}",
                        bm.run_id,
                        f"behaviour_{m_name}",
                        float(m_val),
                        meta_json,
                    ],
                )

    conn.close()
    return bm


def evaluate_all_runs(
    db_path: Union[str, Path],
    tolerance: float = 0.05,
    write_to_db: bool = True,
) -> List[BehaviouralMetrics]:
    """Evaluate behavioural metrics across all runs found in DuckDB."""
    conn = get_duckdb_connection(db_path, read_only=True)
    rows = conn.execute(
        """
        SELECT DISTINCT run_id FROM (
            SELECT run_id FROM runs
            UNION
            SELECT run_id FROM agent_calls
            UNION
            SELECT run_id FROM tool_calls
            UNION
            SELECT run_id FROM orchestrated_decisions
        ) WHERE run_id IS NOT NULL AND run_id != ''
        """
    ).fetchall()
    conn.close()

    results: List[BehaviouralMetrics] = []
    for (r_id,) in rows:
        try:
            m = compute_behavioural_metrics(
                db_path=db_path,
                run_id=r_id,
                tolerance=tolerance,
                write_to_db=write_to_db,
            )
            results.append(m)
        except Exception as e:
            logger.warning("Failed computing behavioural metrics for run '%s': %s", r_id, e)

    return results


# ---------------------------------------------------------------------------
# 5. Summary Table Formatter
# ---------------------------------------------------------------------------
def generate_behaviour_summary_table(metrics_list: List[BehaviouralMetrics]) -> str:
    """Format evaluated behavioural metrics as a markdown table."""
    if not metrics_list:
        return "No behavioural metrics recorded."

    lines: List[str] = [
        "| System | Scenario | Model | Seed | Tool Acc | Param Acc | Instr Follow | Grounded | Fallback | Avg Rounds | Conflict % | Avg Cost ($) | Avg Lat (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for m in metrics_list:
        tool_s = f"{m.tool_choice_accuracy * 100:.1f}%" if m.tool_choice_accuracy is not None else "-"
        param_s = f"{m.parameter_accuracy * 100:.1f}%" if m.parameter_accuracy is not None else "-"
        instr_s = f"{m.instruction_following_rate * 100:.1f}%"
        ground_s = f"{m.groundedness_score * 100:.1f}%"
        fb_s = f"{m.fallback_rate * 100:.1f}%"
        r_s = f"{m.avg_rounds_per_decision:.2f}"
        conf_s = f"{m.conflict_share * 100:.1f}%"
        cost_s = f"${m.avg_cost_usd_per_decision:.5f}"
        lat_s = f"{m.avg_latency_ms_per_decision:.1f}"

        lines.append(
            f"| {m.system:<10} | {m.scenario:<8} | {m.model:<6} | {m.seed:<4} | {tool_s:<8} | {param_s:<9} | "
            f"{instr_s:<12} | {ground_s:<8} | {fb_s:<8} | {r_s:<10} | {conf_s:<10} | {cost_s:<12} | {lat_s:<12} |"
        )

    return "\n".join(lines)


__all__ = [
    "BehaviouralMetrics",
    "extract_numbers_from_text",
    "is_number_grounded",
    "compute_tool_choice_accuracy",
    "compute_parameter_accuracy",
    "compute_instruction_following_rate",
    "compute_groundedness",
    "compute_fallback_and_error_rate",
    "compute_negotiation_efficiency",
    "compute_resource_consumption",
    "compute_behavioural_metrics",
    "evaluate_all_runs",
    "generate_behaviour_summary_table",
]


def main() -> None:
    """CLI entry point for evaluating behavioural metrics from DuckDB logs."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate behavioural metrics from DuckDB execution logs.")
    parser.add_argument("--db", type=str, default="results/sca_metrics.duckdb", help="Path to DuckDB database file.")
    parser.add_argument("--run-id", type=str, default=None, help="Specific run ID to evaluate (default: all runs).")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Numerical tolerance for grounding and parameters.")
    parser.add_argument("--no-write", action="store_true", help="Do not persist results back into DuckDB.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database file '{db_path}' does not exist.")
        return

    write_to_db = not args.no_write
    if args.run_id:
        m = compute_behavioural_metrics(db_path, args.run_id, tolerance=args.tolerance, write_to_db=write_to_db)
        print(generate_behaviour_summary_table([m]))
    else:
        results = evaluate_all_runs(db_path, tolerance=args.tolerance, write_to_db=write_to_db)
        print(generate_behaviour_summary_table(results))


if __name__ == "__main__":
    main()

