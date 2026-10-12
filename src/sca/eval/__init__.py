"""Evaluation runner and metrics reporting for Supply Chain Agents."""

__all__ = [
    "EvaluationRunner",
    "RunResult",
    "RunSpec",
    "generate_benchmark_summary",
    "format_results_table",
]


def __getattr__(name: str):
    if name in __all__:
        import sca.eval.runner as runner_module

        return getattr(runner_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
