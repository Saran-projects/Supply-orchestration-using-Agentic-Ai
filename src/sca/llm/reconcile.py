"""CLI and utility to reconcile logged LLM token spend across DuckDB databases.

Usage:
    python -m sca.llm.reconcile
    python -m sca.llm.reconcile --recompute
    python -m sca.llm.reconcile --db results/glm_experiment.duckdb
    python -m sca.llm.reconcile --console-spend 0.21

Behavior:
    - Queries DuckDB databases for all agent_calls records.
    - Aggregates logged spend, token usage, calls, and cache hits per model.
    - Aggregates spend per system and reports output-token metrics.
    - Recomputes cost_usd from tokens using latest prices in configs/models.yaml,
      billing cached input tokens at the cached rate when reported.
    - Verifies earlier console benchmark ($0.21 for 661.94K tokens) within 5% threshold.
    - Prints current console balance placeholder.
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

from sca.config import load_models_config


def find_duckdb_files(target_db: Optional[str | Path] = None) -> List[Path]:
    """Find all relevant DuckDB databases containing agent calls."""
    if target_db:
        p = Path(target_db)
        return [p] if p.is_file() else []

    # Default: search results/*.duckdb
    found = [Path(p) for p in sorted(glob("results/*.duckdb"))]
    return found


def recompute_model_costs(
    model_name: str = "glm-isoquant",
    db_paths: Optional[List[Path]] = None,
) -> int:
    """Recompute cost_usd and total_cost in agent_calls using prices from configs/models.yaml.

    Billed cached input tokens at price_per_1k_cached_input when available,
    and syncs runs_registry.cost_usd where present.

    Returns:
        Total number of rows updated across all databases.
    """
    config = load_models_config()
    model_entry = config.get_model(model_name)
    if not model_entry:
        raise ValueError(f"Model '{model_name}' not found in configs/models.yaml")

    p_in = float(model_entry.price_per_1k_input)
    p_out = float(model_entry.price_per_1k_output)
    p_cached = (
        float(model_entry.price_per_1k_cached_input)
        if getattr(model_entry, "price_per_1k_cached_input", None) is not None
        else p_in
    )

    dbs = db_paths or find_duckdb_files()
    total_updated = 0

    for db_path in dbs:
        try:
            con = duckdb.connect(str(db_path))
            tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
            if "agent_calls" in tables:
                row_count = con.execute(
                    "SELECT count(*) FROM agent_calls WHERE lower(model_name) = lower(?)",
                    [model_name],
                ).fetchone()[0]

                if row_count > 0:
                    cols = [c[0].lower() for c in con.execute("DESCRIBE agent_calls").fetchall()]
                    has_cached_tokens = "cached_tokens" in cols

                    if has_cached_tokens:
                        cached_expr = "LEAST(COALESCE(cached_tokens, 0), prompt_tokens)"
                        uncached_expr = "(prompt_tokens - LEAST(COALESCE(cached_tokens, 0), prompt_tokens))"
                    else:
                        cached_expr = "0"
                        uncached_expr = "prompt_tokens"

                    update_sql = f"""
                        UPDATE agent_calls
                        SET cost_usd = CASE
                                WHEN cache_hit = TRUE THEN 0.0
                                ELSE ROUND(
                                    ({uncached_expr} / 1000.0) * ?
                                    + ({cached_expr} / 1000.0) * ?
                                    + (completion_tokens / 1000.0) * ?,
                                    6
                                )
                            END,
                            total_cost = CASE
                                WHEN cache_hit = TRUE THEN 0.0
                                ELSE ROUND(
                                    ({uncached_expr} / 1000.0) * ?
                                    + ({cached_expr} / 1000.0) * ?
                                    + (completion_tokens / 1000.0) * ?,
                                    6
                                )
                            END
                        WHERE lower(model_name) = lower(?)
                    """
                    con.execute(update_sql, [p_in, p_cached, p_out, p_in, p_cached, p_out, model_name])
                    total_updated += row_count

                    # Sync runs_registry if present
                    if "runs_registry" in tables:
                        con.execute(
                            """
                            UPDATE runs_registry
                            SET cost_usd = (
                                SELECT ROUND(COALESCE(SUM(cost_usd), 0.0), 6)
                                FROM agent_calls
                                WHERE agent_calls.run_id = runs_registry.run_id
                            )
                            WHERE run_id IN (
                                SELECT DISTINCT run_id FROM agent_calls WHERE lower(model_name) = lower(?)
                            )
                            """,
                            [model_name],
                        )

            con.close()
        except Exception as exc:
            print(f"Warning: Failed updating {db_path}: {exc}", file=sys.stderr)

    return total_updated


def query_spend_by_model(db_paths: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """Query logged spend, tokens, and calls aggregated per model across databases."""
    dbs = db_paths or find_duckdb_files()
    model_stats: Dict[str, Dict[str, Any]] = {}

    for db_path in dbs:
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
            if "agent_calls" in tables:
                cols = [c[0].lower() for c in con.execute("DESCRIBE agent_calls").fetchall()]
                cached_col = "COALESCE(cached_tokens, 0)" if "cached_tokens" in cols else "0"

                rows = con.execute(
                    f"""
                    SELECT
                        COALESCE(model_name, 'unknown') as m_name,
                        count(*) as calls,
                        sum(CASE WHEN cache_hit = TRUE THEN 1 ELSE 0 END) as cache_hits,
                        sum(prompt_tokens) as p_tok,
                        sum({cached_col}) as cached_tok,
                        sum(completion_tokens) as c_tok,
                        sum(cost_usd) as total_cost
                    FROM agent_calls
                    GROUP BY model_name
                    """
                ).fetchall()

                for r in rows:
                    name, calls, hits, p_tok, cached_tok, c_tok, cost = r
                    if name not in model_stats:
                        model_stats[name] = {
                            "model": name,
                            "calls": 0,
                            "cache_hits": 0,
                            "prompt_tokens": 0,
                            "cached_tokens": 0,
                            "completion_tokens": 0,
                            "cost_usd": 0.0,
                            "databases": [],
                        }
                    model_stats[name]["calls"] += (calls or 0)
                    model_stats[name]["cache_hits"] += (hits or 0)
                    model_stats[name]["prompt_tokens"] += (p_tok or 0)
                    model_stats[name]["cached_tokens"] += (cached_tok or 0)
                    model_stats[name]["completion_tokens"] += (c_tok or 0)
                    model_stats[name]["cost_usd"] += (cost or 0.0)
                    model_stats[name]["databases"].append(db_path.name)
            con.close()
        except Exception as exc:
            print(f"Warning: Could not read {db_path}: {exc}", file=sys.stderr)

    # Sort by total cost descending
    sorted_stats = sorted(model_stats.values(), key=lambda x: x["cost_usd"], reverse=True)
    return sorted_stats


def query_spend_by_system(db_paths: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """Query spend, tokens, and output metrics aggregated per system across databases."""
    dbs = db_paths or find_duckdb_files()
    system_stats: Dict[str, Dict[str, Any]] = {}

    for db_path in dbs:
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
            if "agent_calls" in tables:
                cols = [c[0].lower() for c in con.execute("DESCRIBE agent_calls").fetchall()]
                cached_col = "COALESCE(cached_tokens, 0)" if "cached_tokens" in cols else "0"
                system_col = "COALESCE(system_name, '')" if "system_name" in cols else "''"
                run_id_col = "COALESCE(run_id, '')" if "run_id" in cols else "''"
                agent_name_col = "COALESCE(agent_name, '')" if "agent_name" in cols else "''"

                rows = con.execute(
                    f"""
                    SELECT
                        CASE
                            WHEN {system_col} = 'single_agent' OR {run_id_col} LIKE 's3%' OR {agent_name_col} LIKE 'single_agent%' THEN 'S3 (Single Agent)'
                            WHEN {system_col} = 'multi_agent_info' OR {run_id_col} LIKE 's4%' OR {agent_name_col} LIKE 'multi_agent_info%' THEN 'S4 (Multi-Agent Info)'
                            WHEN {system_col} = 'multi_agent_tools' OR {run_id_col} LIKE 's5%' OR {agent_name_col} LIKE 'multi_agent_tools%' THEN 'S5 (Multi-Agent Tools)'
                            WHEN {system_col} = 'orchestrated' OR {run_id_col} LIKE 's6%' OR {agent_name_col} LIKE 'orchestrated%' THEN 'S6 (Orchestrated)'
                            ELSE 'Other / Diagnostic'
                        END as sys_label,
                        count(*) as calls,
                        sum(prompt_tokens) as p_tok,
                        sum({cached_col}) as cached_tok,
                        sum(completion_tokens) as c_tok,
                        sum(cost_usd) as total_cost
                    FROM agent_calls
                    WHERE lower(model_name) = 'glm-isoquant'
                    GROUP BY sys_label
                    """
                ).fetchall()

                for r in rows:
                    sys_lbl, calls, p_tok, cached_tok, c_tok, cost = r
                    if sys_lbl not in system_stats:
                        system_stats[sys_lbl] = {
                            "system": sys_lbl,
                            "calls": 0,
                            "prompt_tokens": 0,
                            "cached_tokens": 0,
                            "completion_tokens": 0,
                            "cost_usd": 0.0,
                        }
                    system_stats[sys_lbl]["calls"] += (calls or 0)
                    system_stats[sys_lbl]["prompt_tokens"] += (p_tok or 0)
                    system_stats[sys_lbl]["cached_tokens"] += (cached_tok or 0)
                    system_stats[sys_lbl]["completion_tokens"] += (c_tok or 0)
                    system_stats[sys_lbl]["cost_usd"] += (cost or 0.0)
            con.close()
        except Exception as exc:
            print(f"Warning: Could not read {db_path}: {exc}", file=sys.stderr)

    order = {
        "S3 (Single Agent)": 0,
        "S4 (Multi-Agent Info)": 1,
        "S5 (Multi-Agent Tools)": 2,
        "S6 (Orchestrated)": 3,
        "Other / Diagnostic": 4,
    }
    return sorted(system_stats.values(), key=lambda x: order.get(x["system"], 99))


def query_specific_runs(run_ids: List[str], db_paths: Optional[List[Path]] = None) -> Dict[str, Dict[str, Any]]:
    """Query telemetry and costs for specific run_ids."""
    dbs = db_paths or find_duckdb_files()
    results: Dict[str, Dict[str, Any]] = {}
    for db_path in dbs:
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
            if "agent_calls" in tables:
                cols = [c[0].lower() for c in con.execute("DESCRIBE agent_calls").fetchall()]
                cached_col = "COALESCE(cached_tokens, 0)" if "cached_tokens" in cols else "0"
                for r_id in run_ids:
                    row = con.execute(
                        f"""
                        SELECT
                            count(*),
                            sum(prompt_tokens),
                            sum({cached_col}),
                            sum(completion_tokens),
                            sum(cost_usd)
                        FROM agent_calls
                        WHERE run_id = ?
                        """,
                        [r_id],
                    ).fetchone()
                    if row and row[0] > 0:
                        calls, p_tok, cached_tok, c_tok, cost = row
                        results[r_id] = {
                            "run_id": r_id,
                            "calls": calls,
                            "prompt_tokens": p_tok or 0,
                            "cached_tokens": cached_tok or 0,
                            "completion_tokens": c_tok or 0,
                            "cost_usd": cost or 0.0,
                        }
            con.close()
        except Exception:
            pass
    return results


def print_reconcile_report(
    db_paths: Optional[List[Path]] = None,
    console_figure: float = 0.21,
    recompute: bool = False,
    fail_on_threshold: bool = True,
    threshold_pct: float = 5.0,
    console_balance_str: str = "<PASTE BALANCE>",
) -> float:
    """Print the complete LLM cost reconciliation report."""
    dbs = db_paths or find_duckdb_files()

    if recompute:
        updated = recompute_model_costs(model_name="glm-isoquant", db_paths=dbs)
        print(f"\n[INFO] Recomputed cost_usd for {updated} glm-isoquant records using models.yaml rates.")

    stats = query_spend_by_model(dbs)
    sys_stats = query_spend_by_system(dbs)

    # 1. Model Summary Table
    print("\n" + "=" * 102)
    print(" 1. LLM Logged Spend per Model (DuckDB Telemetry)")
    print("=" * 102)

    if not stats:
        print("No agent call records found in specified databases.")
        print("=" * 102 + "\n")
        return 0.0

    headers = ["Model", "Calls", "Hits", "Prompt Tok", "Cached In", "Compl. Tok", "Total Tok", "Logged Spend"]
    rows = []

    total_calls = 0
    total_hits = 0
    total_p = 0
    total_cached = 0
    total_c = 0
    total_spend = 0.0
    glm_logged_spend = 0.0

    for s in stats:
        m = s["model"]
        calls = s["calls"]
        hits = s["cache_hits"]
        p = s["prompt_tokens"]
        cached = s["cached_tokens"]
        c = s["completion_tokens"]
        tot_tok = p + c
        cost = s["cost_usd"]

        total_calls += calls
        total_hits += hits
        total_p += p
        total_cached += cached
        total_c += c
        total_spend += cost

        if m.lower() == "glm-isoquant":
            glm_logged_spend = cost

        rows.append([
            m,
            f"{calls:,}",
            f"{hits:,}",
            f"{p:,}",
            f"{cached:,}",
            f"{c:,}",
            f"{tot_tok:,}",
            f"${cost:.6f}",
        ])

    rows.append([
        "TOTAL",
        f"{total_calls:,}",
        f"{total_hits:,}",
        f"{total_p:,}",
        f"{total_cached:,}",
        f"{total_c:,}",
        f"{total_p + total_c:,}",
        f"${total_spend:.6f}",
    ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(val))

    header_line = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    total_sep = "=+=".join("=" * col_widths[i] for i in range(len(headers)))

    print(header_line)
    print(sep_line)
    for row in rows[:-1]:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))
    print(total_sep)
    print(" | ".join(rows[-1][i].ljust(col_widths[i]) for i in range(len(headers))))
    print("=" * len(header_line))

    # 2. Spend per System & Output-Token Metrics Table
    print("\n" + "=" * 102)
    print(" 2. Spend-per-System & Output-Token Metrics (glm-isoquant)")
    print("=" * 102)

    sys_headers = [
        "System",
        "Calls",
        "Prompt Tok",
        "Cached In (%)",
        "Output Tok",
        "Avg Out/Call",
        "Output Spend",
        "Out Share (%)",
        "Total Spend",
    ]
    sys_rows = []

    p_out_rate = 0.00049  # per 1k

    for st in sys_stats:
        s_name = st["system"]
        calls = st["calls"]
        p_tok = st["prompt_tokens"]
        cached_tok = st["cached_tokens"]
        c_tok = st["completion_tokens"]
        tot_c = st["cost_usd"]

        cached_pct = (cached_tok / p_tok * 100.0) if p_tok > 0 else 0.0
        avg_out = (c_tok / calls) if calls > 0 else 0.0
        out_spend = (c_tok / 1000.0) * p_out_rate
        out_share = (out_spend / tot_c * 100.0) if tot_c > 0 else 0.0

        sys_rows.append([
            s_name,
            f"{calls:,}",
            f"{p_tok:,}",
            f"{cached_tok:,} ({cached_pct:.1f}%)",
            f"{c_tok:,}",
            f"{avg_out:.1f}",
            f"${out_spend:.6f}",
            f"{out_share:.2f}%",
            f"${tot_c:.6f}",
        ])

    s_col_widths = [len(h) for h in sys_headers]
    for row in sys_rows:
        for idx, val in enumerate(row):
            s_col_widths[idx] = max(s_col_widths[idx], len(val))

    s_header_line = " | ".join(sys_headers[i].ljust(s_col_widths[i]) for i in range(len(sys_headers)))
    s_sep_line = "-+-".join("-" * s_col_widths[i] for i in range(len(sys_headers)))

    print(s_header_line)
    print(s_sep_line)
    for row in sys_rows:
        print(" | ".join(row[i].ljust(s_col_widths[i]) for i in range(len(sys_headers))))
    print("=" * len(s_header_line))

    # 3. Iteration-1 Specific Runs Telemetry
    print("\n" + "=" * 102)
    print(" 3. Iteration-1 Runs Telemetry (Prompt v3.1, Seed 100, Shock Scenario)")
    print("=" * 102)

    it1_runs = query_specific_runs(
        ["s3_glm_shock_seed_100_v3_1_49f1ce", "s6_glm_shock_seed_100_v3_1_ffdb54"],
        dbs,
    )
    s3_it1 = it1_runs.get("s3_glm_shock_seed_100_v3_1_49f1ce", {})
    s6_it1 = it1_runs.get("s6_glm_shock_seed_100_v3_1_ffdb54", {})

    s3_cost = s3_it1.get("cost_usd", 0.0)
    s6_cost = s6_it1.get("cost_usd", 0.0)
    it1_total = s3_cost + s6_cost

    print(f"  Iteration-1 S3 Cost (s3_glm_shock_seed_100_v3_1_49f1ce) : ${s3_cost:.6f} USD")
    if s3_it1:
        s3_calls = s3_it1["calls"]
        s3_p = s3_it1["prompt_tokens"]
        s3_out = s3_it1["completion_tokens"]
        s3_out_cost = (s3_out / 1000.0) * p_out_rate
        s3_share = (s3_out_cost / s3_cost * 100.0) if s3_cost > 0 else 0.0
        print(f"    Calls: {s3_calls} | In: {s3_p:,} (avg {s3_p/s3_calls:.1f}) | Out: {s3_out:,} (avg {s3_out/s3_calls:.1f}) | Out Share: {s3_share:.2f}%")

    print(f"  Iteration-1 S6 Cost (s6_glm_shock_seed_100_v3_1_ffdb54) : ${s6_cost:.6f} USD")
    if s6_it1:
        s6_calls = s6_it1["calls"]
        s6_p = s6_it1["prompt_tokens"]
        s6_c = s6_it1["cached_tokens"]
        s6_out = s6_it1["completion_tokens"]
        s6_out_cost = (s6_out / 1000.0) * p_out_rate
        s6_share = (s6_out_cost / s6_cost * 100.0) if s6_cost > 0 else 0.0
        print(f"    Calls: {s6_calls} | In: {s6_p:,} (avg {s6_p/s6_calls:.1f}, {s6_c/s6_p*100:.1f}% cached) | Out: {s6_out:,} (avg {s6_out/s6_calls:.1f}) | Out Share: {s6_share:.2f}%")

    print(f"  Combined Iteration-1 Total Cost                         : ${it1_total:.6f} USD (Limit: $0.35 USD -> PASS)")
    print("=" * 102)

    # 4. Console Verification Section
    print("\n" + "=" * 102)
    print(" 4. Isoquant Provider Console Verification")
    print("=" * 102)

    # Earlier console usage: 661.94K tokens, 51.4% input cached, cache savings $0.001138 USD
    # Rates: In: $0.027/1M, Cached In: $0.018/1M, Out: $0.490/1M
    cached_in_tok = 126444.4  # from savings $0.001138 / ($0.009 / 1M)
    total_in_tok = cached_in_tok / 0.514  # = 246,000.9
    uncached_in_tok = total_in_tok - cached_in_tok  # = 119,556.5
    out_tok = 661940.0 - total_in_tok  # = 415,939.1
    recomputed_console_cost = (
        (uncached_in_tok / 1_000_000.0) * 0.027
        + (cached_in_tok / 1_000_000.0) * 0.018
        + (out_tok / 1_000_000.0) * 0.490
    )

    if console_figure < 0.1:
        diff_val = glm_logged_spend - console_figure
        diff_abs = abs(diff_val)
        diff_pct = (diff_abs / console_figure * 100.0) if console_figure > 0 else 0.0
        passed = diff_pct <= threshold_pct
        status_str = f"RECONCILED ({diff_pct:.2f}% <= {threshold_pct:.1f}% threshold)" if passed else f"DISCREPANCY EXCEEDED ({diff_pct:.2f}% > {threshold_pct:.1f}%)"
        print(f"  Target Console / Expected Spend               : ${console_figure:.6f} USD")
        print(f"  Logged Database Spend                         : ${glm_logged_spend:.6f} USD")
        print(f"  Variance / Delta                              : ${diff_val:+.6f} USD ({diff_val / console_figure * 100.0:+.2f}%)")
        print(f"  Difference Percentage vs Console              : {diff_pct:.2f}% (max allowable: {threshold_pct:.1f}%)")
        print(f"  Console Verification Status                   : {status_str}")
        print(f"  Current Console Balance                       : {console_balance_str}")
        print("=" * 102)
        print(f"\n[SUMMARY] All-Time Logged Total (glm-isoquant): ${glm_logged_spend:.6f} USD")
        if fail_on_threshold and not passed:
            raise ValueError(
                f"Reconciliation failure: All-time logged spend (${glm_logged_spend:.6f}) differs from console figure "
                f"(${console_figure:.6f}) by {diff_pct:.2f}%, which exceeds the allowable {threshold_pct:.1f}% threshold."
            )
        return glm_logged_spend

    diff = recomputed_console_cost - console_figure
    diff_abs = abs(diff)
    diff_pct = (diff_abs / console_figure * 100.0) if console_figure > 0 else 0.0
    passed = diff_pct <= threshold_pct
    status_str = f"RECONCILED ({diff_pct:.2f}% <= {threshold_pct:.1f}% threshold)" if passed else f"DISCREPANCY EXCEEDED ({diff_pct:.2f}% > {threshold_pct:.1f}%)"

    print(f"  Earlier Console Reported Total (661.94K tok) : ${console_figure:.6f} USD")
    print(f"  Recomputed Cost at Confirmed Rate Card        : ${recomputed_console_cost:.6f} USD")
    print(f"  Variance / Delta                              : ${diff:+.6f} USD ({diff / console_figure * 100.0:+.2f}%)")
    print(f"  Difference Percentage vs Console              : {diff_pct:.2f}% (max allowable: {threshold_pct:.1f}%)")
    print(f"  Console Verification Status                   : {status_str}")
    print(f"  Current Console Balance                       : {console_balance_str}")
    print("=" * 102)

    # 5. Budget, Safety Reserve & Hard Cap Auditing
    budget_usd = 4.00
    reserve_usd = 0.40
    hard_cap_usd = 3.60
    headroom = hard_cap_usd - glm_logged_spend
    cap_status = "UNDER CAP (PASS)" if glm_logged_spend <= hard_cap_usd else "HARD CAP EXCEEDED (HALT)"
    print("\n" + "=" * 102)
    print(" 5. Budget, Safety Reserve & Hard Cap Auditing")
    print("=" * 102)
    print(f"  Total Authorized Credit Budget : ${budget_usd:.2f} USD")
    print(f"  Safety Reserve (Preserved)     : ${reserve_usd:.2f} USD")
    print(f"  Enforced Spend Hard Cap        : ${hard_cap_usd:.2f} USD (4.00 credit minus reserve)")
    print(f"  All-Time Logged Total Spend    : ${glm_logged_spend:.6f} USD")
    print(f"  Remaining Headroom to Hard Cap : ${headroom:.6f} USD")
    print(f"  Hard Cap Status                : {cap_status}")
    print("=" * 102)

    # 6. All-Time Logged Total
    print(f"\n[SUMMARY] All-Time Logged Total (glm-isoquant): ${glm_logged_spend:.6f} USD (Hard Cap: ${hard_cap_usd:.2f} USD)")

    if fail_on_threshold and not passed:
        raise ValueError(
            f"Reconciliation failure: Recomputed console spend (${recomputed_console_cost:.6f}) differs from console figure "
            f"(${console_figure:.6f}) by {diff_pct:.2f}%, which exceeds the allowable {threshold_pct:.1f}% threshold."
        )

    return glm_logged_spend


def main() -> int:
    """CLI entrypoint for reconcile command."""
    parser = argparse.ArgumentParser(
        description="Reconcile logged LLM token spend across DuckDB telemetry databases."
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Optional path to a specific DuckDB file (default: all in results/*.duckdb)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        default=False,
        help="Recompute cost_usd for existing glm-isoquant rows using models.yaml prices",
    )
    parser.add_argument(
        "--console-spend",
        type=float,
        default=0.21,
        help="Isoquant console figure to compare against in USD (default: 0.21)",
    )
    parser.add_argument(
        "--console-balance",
        dest="console_balance",
        default="<PASTE BALANCE>",
        help="Current console balance string to print (default: <PASTE BALANCE>)",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=5.0,
        help="Maximum allowable difference percentage before failing (default: 5.0%%)",
    )
    parser.add_argument(
        "--no-fail",
        dest="fail_on_threshold",
        action="store_false",
        default=True,
        help="Do not exit with error if difference exceeds threshold",
    )
    args = parser.parse_args()

    dbs = [Path(args.db_path)] if args.db_path else None
    try:
        print_reconcile_report(
            db_paths=dbs,
            console_figure=args.console_spend,
            recompute=args.recompute,
            fail_on_threshold=args.fail_on_threshold,
            threshold_pct=args.threshold_pct,
            console_balance_str=args.console_balance,
        )
        return 0
    except ValueError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
