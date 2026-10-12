"""CLI and utility to reconcile logged LLM token spend across DuckDB databases.

Usage:
    python -m sca.llm.reconcile
    python -m sca.llm.reconcile --recompute
    python -m sca.llm.reconcile --db results/glm_experiment.duckdb
    python -m sca.llm.reconcile --console-spend 0.21

Behavior:
    - Queries DuckDB databases for all agent_calls records.
    - Aggregates logged spend, token usage, calls, and cache hits per model.
    - Optionally recomputes cost_usd from tokens using latest prices in configs/models.yaml.
    - Compares glm-isoquant spend against the Isoquant console figure (default: $0.21 USD).
    - Prints clean summary tables.
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
import sys
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
    
    Returns:
        Total number of rows updated across all databases.
    """
    config = load_models_config()
    model_entry = config.get_model(model_name)
    if not model_entry:
        raise ValueError(f"Model '{model_name}' not found in configs/models.yaml")
    
    p_in = float(model_entry.price_per_1k_input)
    p_out = float(model_entry.price_per_1k_output)
    
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
                    con.execute(
                        """
                        UPDATE agent_calls
                        SET cost_usd = CASE 
                                WHEN cache_hit = TRUE THEN 0.0
                                ELSE ROUND((prompt_tokens / 1000.0) * ? + (completion_tokens / 1000.0) * ?, 6)
                            END,
                            total_cost = CASE 
                                WHEN cache_hit = TRUE THEN 0.0
                                ELSE ROUND((prompt_tokens / 1000.0) * ? + (completion_tokens / 1000.0) * ?, 6)
                            END
                        WHERE lower(model_name) = lower(?)
                        """,
                        [p_in, p_out, p_in, p_out, model_name],
                    )
                    total_updated += row_count
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
                rows = con.execute(
                    """
                    SELECT 
                        COALESCE(model_name, 'unknown') as m_name,
                        count(*) as calls,
                        sum(CASE WHEN cache_hit = TRUE THEN 1 ELSE 0 END) as cache_hits,
                        sum(prompt_tokens) as p_tok,
                        sum(completion_tokens) as c_tok,
                        sum(cost_usd) as total_cost
                    FROM agent_calls
                    GROUP BY model_name
                    """
                ).fetchall()
                
                for r in rows:
                    name, calls, hits, p_tok, c_tok, cost = r
                    if name not in model_stats:
                        model_stats[name] = {
                            "model": name,
                            "calls": 0,
                            "cache_hits": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "cost_usd": 0.0,
                            "databases": [],
                        }
                    model_stats[name]["calls"] += (calls or 0)
                    model_stats[name]["cache_hits"] += (hits or 0)
                    model_stats[name]["prompt_tokens"] += (p_tok or 0)
                    model_stats[name]["completion_tokens"] += (c_tok or 0)
                    model_stats[name]["cost_usd"] += (cost or 0.0)
                    model_stats[name]["databases"].append(db_path.name)
            con.close()
        except Exception as exc:
            print(f"Warning: Could not read {db_path}: {exc}", file=sys.stderr)
            
    # Sort by total cost descending
    sorted_stats = sorted(model_stats.values(), key=lambda x: x["cost_usd"], reverse=True)
    return sorted_stats


def print_reconcile_report(
    db_paths: Optional[List[Path]] = None,
    console_figure: float = 0.21,
    recompute: bool = False,
) -> None:
    """Print the complete LLM cost reconciliation report."""
    dbs = db_paths or find_duckdb_files()
    
    if recompute:
        updated = recompute_model_costs(model_name="glm-isoquant", db_paths=dbs)
        print(f"\n[INFO] Recomputed cost_usd for {updated} glm-isoquant records using models.yaml rates.")
        
    stats = query_spend_by_model(dbs)
    
    print("\n" + "=" * 90)
    print(" LLM Logged Spend per Model (DuckDB Telemetry)")
    print("=" * 90)
    
    if not stats:
        print("No agent call records found in specified databases.")
        print("=" * 90 + "\n")
        return

    headers = ["Model", "Calls", "Cache Hits", "Prompt Tok", "Compl. Tok", "Total Tok", "Logged Spend"]
    rows = []
    
    total_calls = 0
    total_hits = 0
    total_p = 0
    total_c = 0
    total_spend = 0.0
    
    glm_logged_spend = 0.0
    
    for s in stats:
        m = s["model"]
        calls = s["calls"]
        hits = s["cache_hits"]
        p = s["prompt_tokens"]
        c = s["completion_tokens"]
        tot_tok = p + c
        cost = s["cost_usd"]
        
        total_calls += calls
        total_hits += hits
        total_p += p
        total_c += c
        total_spend += cost
        
        if m.lower() == "glm-isoquant":
            glm_logged_spend = cost
            
        rows.append([
            m,
            f"{calls:,}",
            f"{hits:,}",
            f"{p:,}",
            f"{c:,}",
            f"{tot_tok:,}",
            f"${cost:.6f}",
        ])
        
    # Summary totals row
    rows.append([
        "TOTAL",
        f"{total_calls:,}",
        f"{total_hits:,}",
        f"{total_p:,}",
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
    
    # Reconciliation comparison section
    print("\n" + "=" * 90)
    print(" Isoquant Console Reconciliation")
    print("=" * 90)
    
    diff = glm_logged_spend - console_figure
    pct = (glm_logged_spend / console_figure * 100.0) if console_figure > 0 else 0.0
    status_str = "RECONCILED (Within rounding margin)" if abs(diff) < 0.01 else "DISCREPANCY"
    
    print(f"  All-Time Logged Total (glm-isoquant) : ${glm_logged_spend:.6f} USD")
    print(f"  Isoquant Console Billed Figure      : ${console_figure:.6f} USD")
    print(f"  Variance / Delta (Logged - Console) : ${diff:+.6f} USD ({diff / console_figure * 100.0:+.2f}%)")
    print(f"  Coverage Ratio                      : {pct:.2f}%")
    print(f"  Status                              : {status_str}")
    print("=" * 90 + "\n")


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
    args = parser.parse_args()
    
    dbs = [Path(args.db_path)] if args.db_path else None
    print_reconcile_report(
        db_paths=dbs,
        console_figure=args.console_spend,
        recompute=args.recompute,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
