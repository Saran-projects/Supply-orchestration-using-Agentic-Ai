"""Provider-agnostic LLM client, prompt templates, and structured output parsers."""

from sca.llm.client import (
    MODEL_PRICING,
    BaseLLMAdapter,
    DiskCache,
    LLMClient,
    LLMResponse,
    MockLLM,
    StructuredOutputError,
    calculate_cost,
    log_agent_call_to_duckdb,
)


def __getattr__(name: str):
    if name in {"recompute_model_costs", "query_spend_by_model", "print_reconcile_report"}:
        import sca.llm.reconcile as _rec
        return getattr(_rec, name)
    if name in {"BudgetExceededError", "check_dry_run_budget_guard", "get_cumulative_glm_spend", "project_run_cost"}:
        import sca.llm.budget as _bud
        return getattr(_bud, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LLMClient",
    "LLMResponse",
    "MockLLM",
    "DiskCache",
    "BaseLLMAdapter",
    "StructuredOutputError",
    "calculate_cost",
    "MODEL_PRICING",
    "log_agent_call_to_duckdb",
    "recompute_model_costs",
    "query_spend_by_model",
    "print_reconcile_report",
    "BudgetExceededError",
    "check_dry_run_budget_guard",
    "project_run_cost",
    "get_cumulative_glm_spend",
]


