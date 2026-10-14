"""Run-validity rules for LLM runs.

A run is invalid (and the batch must stop) when:
- more than `max_fallback_rate` of decisions used the fallback order, or
- fewer LLM calls were made than decisions (some decisions never reached the LLM).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

DEFAULT_MAX_FALLBACK_RATE = 0.05


class RunInvalidError(RuntimeError):
    """Raised when a run breaks a validity rule."""


class BatchHaltedError(RuntimeError):
    """Raised when a batch stops early because a run was invalid or hit a fatal API error."""


def check_run_validity(
    stats: Mapping[str, int],
    max_fallback_rate: float = DEFAULT_MAX_FALLBACK_RATE,
) -> Optional[str]:
    """Return the reason a run is invalid, or None if it is valid."""
    decisions = int(stats.get("decisions", 0))
    fallbacks = int(stats.get("fallbacks", 0))
    llm_calls = int(stats.get("llm_calls", 0))

    if decisions <= 0:
        return None
    rate = fallbacks / decisions
    if rate > max_fallback_rate:
        return f"fallback rate {rate:.1%} ({fallbacks}/{decisions}) exceeds {max_fallback_rate:.1%}"
    if llm_calls < decisions:
        return f"LLM calls ({llm_calls}) fewer than decisions ({decisions})"
    return None


def assert_run_valid(system: Any, max_fallback_rate: float = DEFAULT_MAX_FALLBACK_RATE) -> None:
    """Raise RunInvalidError if `system` exposes llm_stats() and breaks a validity rule."""
    if not hasattr(system, "llm_stats"):
        return
    reason = check_run_validity(system.llm_stats(), max_fallback_rate)
    if reason:
        raise RunInvalidError(reason)
