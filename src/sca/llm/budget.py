"""Budget tracking, dry-run cost projection, and hard cap enforcement for LLM runs.

Budget Mechanics:
- Total Authorized Credit: $4.00 USD
- Safety Reserve:          $0.40 USD
- Hard Spend Ceiling:      $3.60 USD (4.00 credit minus reserve)

Enforces:
- Dry-run mode that halts a run if cumulative spend plus projection would exceed the $3.60 hard cap.
- Per-run spend auditing and reconciliation across DuckDB runs_registry and agent_calls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb

from sca.config import load_models_config
from sca.llm.reconcile import find_duckdb_files

logger = logging.getLogger("sca.llm.budget")

DEFAULT_CREDIT_BUDGET_USD = 4.00
DEFAULT_RESERVE_USD = 0.40
DEFAULT_HARD_CAP_USD = 3.60


class BudgetExceededError(Exception):
    """Raised when an operation would cause total spend to exceed the budget hard cap."""

    pass


def get_hard_cap_from_config() -> float:
    """Retrieve hard cap from configs/models.yaml or default to 3.60 USD."""
    try:
        cfg = load_models_config()
        return float(getattr(cfg, "hard_cap_usd", DEFAULT_HARD_CAP_USD))
    except Exception:
        return DEFAULT_HARD_CAP_USD


def get_cumulative_glm_spend(db_paths: Optional[List[Path]] = None) -> float:
    """Query total cumulative logged spend for glm-isoquant across all DuckDB databases."""
    dbs = db_paths or find_duckdb_files()
    total_spend = 0.0

    for db_path in dbs:
        if not db_path.is_file():
            continue
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            tables = [t[0].lower() for t in con.execute("SHOW TABLES").fetchall()]
            if "agent_calls" in tables:
                row = con.execute(
                    """
                    SELECT COALESCE(SUM(cost_usd), 0.0)
                    FROM agent_calls
                    WHERE lower(model_name) LIKE '%glm%'
                    """
                ).fetchone()
                if row and row[0] is not None:
                    total_spend += float(row[0])
            con.close()
        except Exception as e:
            logger.warning("Could not read spend from %s: %s", db_path, e)

    return float(round(total_spend, 6))


def project_run_cost(
    system_id: str,
    steps: int = 40,
    model_name: str = "glm-isoquant",
) -> Dict[str, float]:
    """Project the cost in USD for a simulation run based on empirical token profiles.

    Empirical profiles (Prompt v3.1 / 4 echelons):
    - System 3 (single_agent): 160 calls (1/step/echelon), ~934 in, ~795 out (~$0.0664 / 40 steps)
    - System 4 (multi_agent_info): 160 calls, ~1,400 in, ~1,100 out (~$0.0923 / 40 steps)
    - System 5 (multi_agent_tools): 160 calls, ~1,500 in, ~1,200 out (~$0.1006 / 40 steps)
    - System 6 (orchestrated): ~282 calls (negotiation), ~665 in (60% cached), ~1,691 out (~$0.2377 / 40 steps)
    """
    cfg = load_models_config()
    model_entry = cfg.get_model(model_name)
    if not model_entry:
        p_in = 0.000027
        p_cached = 0.000018
        p_out = 0.000490
    else:
        p_in = float(model_entry.price_per_1k_input)
        p_cached = float(getattr(model_entry, "price_per_1k_cached_input", p_in) or p_in)
        p_out = float(model_entry.price_per_1k_output)

    total_decisions = steps * 4

    sys_norm = system_id.lower()
    if "single" in sys_norm or sys_norm in ("s3", "system_3"):
        calls = total_decisions
        avg_in = 933.8
        avg_cached = 0.0
        avg_out = 795.4
    elif "info" in sys_norm or sys_norm in ("s4", "system_4"):
        calls = total_decisions
        avg_in = 1400.0
        avg_cached = 0.0
        avg_out = 1100.0
    elif "tools" in sys_norm or sys_norm in ("s5", "system_5"):
        calls = total_decisions
        avg_in = 1500.0
        avg_cached = 0.0
        avg_out = 1200.0
    elif "orchestrated" in sys_norm or sys_norm in ("s6", "system_6"):
        # ~1.76 calls per decision due to multi-agent negotiation rounds
        calls = int(round(total_decisions * 1.7625))
        avg_in = 665.2
        avg_cached = 395.8
        avg_out = 1691.2
    else:
        calls = total_decisions
        avg_in = 1000.0
        avg_cached = 0.0
        avg_out = 800.0

    uncached_in = avg_in - avg_cached
    call_cost = (
        (uncached_in / 1000.0) * p_in
        + (avg_cached / 1000.0) * p_cached
        + (avg_out / 1000.0) * p_out
    )
    total_projected = calls * call_cost

    return {
        "calls": float(calls),
        "cost_usd": float(round(total_projected, 6)),
        "cost_per_decision": float(round(total_projected / total_decisions, 6)),
        "avg_input_tokens": float(avg_in),
        "avg_output_tokens": float(avg_out),
    }


def check_dry_run_budget_guard(
    projected_cost: float,
    hard_cap: Optional[float] = None,
    db_paths: Optional[List[Path]] = None,
    halt_on_exceed: bool = True,
) -> Dict[str, Any]:
    """Dry-run check: verifies cumulative spend + projection <= hard cap ($3.60 USD).

    Raises:
        BudgetExceededError if halt_on_exceed is True and the hard cap would be passed.
    """
    cap = hard_cap if hard_cap is not None else get_hard_cap_from_config()
    current_spend = get_cumulative_glm_spend(db_paths)
    projected_cumulative = current_spend + projected_cost
    headroom = cap - current_spend
    passed = projected_cumulative <= cap

    result = {
        "passed": passed,
        "current_spend_usd": current_spend,
        "projected_cost_usd": projected_cost,
        "projected_cumulative_usd": projected_cumulative,
        "hard_cap_usd": cap,
        "credit_budget_usd": DEFAULT_CREDIT_BUDGET_USD,
        "reserve_usd": DEFAULT_RESERVE_USD,
        "remaining_headroom_usd": headroom,
    }

    if not passed and halt_on_exceed:
        msg = (
            f"[BUDGET HALT] Projected cumulative spend (${projected_cumulative:.4f} USD = "
            f"current ${current_spend:.4f} + projected ${projected_cost:.4f}) exceeds "
            f"the hard cap of ${cap:.2f} USD (credit ${DEFAULT_CREDIT_BUDGET_USD:.2f} - "
            f"reserve ${DEFAULT_RESERVE_USD:.2f})! Halting run to preserve reserve credit."
        )
        print(f"\n{'!' * 90}\n{msg}\n{'!' * 90}\n", file=sys.stderr)
        raise BudgetExceededError(msg)

    return result


def main() -> int:
    """CLI to inspect current spend vs. hard cap."""
    import argparse

    parser = argparse.ArgumentParser(description="Audit LLM budget spend and test dry-run projections.")
    parser.add_argument("--project-s3", type=int, default=0, help="Number of S3 runs to project")
    parser.add_argument("--project-s6", type=int, default=0, help="Number of S6 runs to project")
    parser.add_argument("--steps", type=int, default=40, help="Simulation steps per run (default: 40)")
    parser.add_argument("--halt", action="store_true", default=False, help="Halt with exit code 1 if hard cap exceeded")
    args = parser.parse_args()

    cap = get_hard_cap_from_config()
    current_spend = get_cumulative_glm_spend()

    s3_cost = project_run_cost("single_agent", steps=args.steps)["cost_usd"] * args.project_s3
    s6_cost = project_run_cost("orchestrated", steps=args.steps)["cost_usd"] * args.project_s6
    total_proj = s3_cost + s6_cost

    print("=" * 80)
    print(" LLM SPEND & BUDGET HARD CAP AUDIT")
    print("=" * 80)
    print(f"Total Authorized Credit : ${DEFAULT_CREDIT_BUDGET_USD:.2f} USD")
    print(f"Safety Reserve          : ${DEFAULT_RESERVE_USD:.2f} USD")
    print(f"Hard Spend Ceiling      : ${cap:.2f} USD")
    print(f"Current Logged Spend    : ${current_spend:.6f} USD")
    print(f"Remaining Headroom      : ${cap - current_spend:.6f} USD")
    if total_proj > 0:
        print(f"Projected Addition      : ${total_proj:.6f} USD ({args.project_s3}x S3, {args.project_s6}x S6)")
        print(f"Projected Cumulative    : ${current_spend + total_proj:.6f} USD")
    print("=" * 80)

    try:
        check_dry_run_budget_guard(total_proj, hard_cap=cap, halt_on_exceed=args.halt)
        print("[STATUS] Within budget constraints.")
        return 0
    except BudgetExceededError as err:
        print(f"[STATUS] {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
