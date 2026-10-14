"""Provider-agnostic LLMClient with adapters for Isoquant (GLM), OpenAI, Grok, Gemini, and Mock.

Features:
- Driven by configs/models.yaml with four model families (openai_compatible, gemini, mock)
- Backoff retries with jitter on 429/5xx/timeouts
- Per-model concurrency limits and requests-per-minute (RPM) limits
- JSON-schema-structured output via Pydantic with single retry and StructuredOutputError
- Disk cache keyed by hash(model name, model_id, prompt, params)
- Tokenizer fallback with usage_estimated = true when token counts are missing
- DuckDB agent_calls logging: model name, provider, latency_ms, input_tokens, output_tokens,
  cached_tokens, cost_usd, retry_count, structured_output_ok, error_type, cache_hit, usage_estimated
- Offline MockLLM test double with error simulation and canned responses
- Optional Opik tracing integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from sca.config import (
    ModelEntryConfig,
    ModelsConfig,
    ProviderConfig,
    get_config,
    load_models_config,
)
from sca.db import get_duckdb_connection

# Load environment variables once at module import
load_dotenv(find_dotenv(usecwd=True))

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when LLM output fails validation against expected Pydantic schema after retries."""

    def __init__(self, message: str = "", prompt_tokens: int = 0, completion_tokens: int = 0, cached_tokens: int = 0):
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached_tokens = cached_tokens


class TruncatedOutputError(RuntimeError):
    """Raised when the provider stops on the token limit or returns empty content.

    Retrying with the same max_tokens cannot succeed, so this is never retried. Token counts are
    kept so the cost of the failed call is still recorded.
    """

    def __init__(
        self,
        message: str,
        finish_reason: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
    ):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached_tokens = cached_tokens


def _usage_counts(usage: Any) -> Tuple[int, int, int]:
    """(prompt, completion, cached) token counts from an OpenAI-style usage object."""
    if not usage:
        return 0, 0, 0
    cached_details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(cached_details, "cached_tokens", 0) if cached_details else 0
    return int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0), int(cached or 0)


def check_completion(res: Any, model: str) -> Tuple[str, int, int, int]:
    """Return (content, prompt, completion, cached) or raise TruncatedOutputError on length/empty output."""
    choice = res.choices[0]
    content = choice.message.content or ""
    p_tok, c_tok, cached_tok = _usage_counts(getattr(res, "usage", None))
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length" or not content.strip():
        kind = "truncated" if finish_reason == "length" else "empty"
        raise TruncatedOutputError(
            f"Model '{model}' returned {kind} output (finish_reason={finish_reason}, "
            f"completion_tokens={c_tok}); raise max_tokens or reduce reasoning.",
            finish_reason=finish_reason,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cached_tokens=cached_tok,
        )
    return content, p_tok, c_tok, cached_tok


class PaidCallsDisabledError(RuntimeError):
    """Raised when a real (billable) network call is attempted without ALLOW_PAID_CALLS=1."""


class FatalAPIError(RuntimeError):
    """Raised on HTTP 401/402/403: the run must stop immediately, with no retry and no fallback."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


PAID_CALLS_ENV_VAR = "ALLOW_PAID_CALLS"
FATAL_STATUS_CODES = (401, 402, 403)


def paid_calls_allowed() -> bool:
    """Real network calls are allowed only when ALLOW_PAID_CALLS=1 is set (default: not set)."""
    return os.environ.get(PAID_CALLS_ENV_VAR, "").strip() == "1"


def get_status_code(exc: Exception) -> Optional[int]:
    """Extract an HTTP status code from provider SDK exceptions, if present."""
    for attr in ("status_code", "code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    response = getattr(exc, "response", None)
    val = getattr(response, "status_code", None)
    return val if isinstance(val, int) else None


def is_fatal_api_error(exc: Exception) -> bool:
    """True for auth/billing failures (HTTP 401, 402, 403) that must never be retried."""
    return isinstance(exc, FatalAPIError) or get_status_code(exc) in FATAL_STATUS_CODES


# ---------------------------------------------------------------------------
# Fallback Pricing & Cost Calculation
# ---------------------------------------------------------------------------
# Cost in USD per 1,000,000 tokens (input_cost_per_m, output_cost_per_m)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    # Google
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
    # Isoquant GLM
    "glm-5.3-flash": (0.027, 0.49, 0.018),
    "glm-isoquant": (0.027, 0.49, 0.018),
    # Local & Mock
    "llama3.1:8b": (0.0, 0.0),
    "local-model": (0.0, 0.0),
    "mock": (0.0, 0.0),
    "mock-model": (0.0, 0.0),
}


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    config: Optional[ModelsConfig] = None,
    cached_tokens: int = 0,
) -> float:
    """Calculate cost in USD based on models.yaml entry prices or fallback pricing table."""
    lookup_key = model.lower().strip()

    # Check if model entry is registered in models.yaml with per-1k prices
    eff_config = config
    if eff_config is None:
        try:
            eff_config = load_models_config()
        except Exception:
            eff_config = None

    if eff_config is not None:
        entry = eff_config.get_model(lookup_key)
        if entry is None:
            # Check model_id matching
            for m in eff_config.models:
                if m.model_id.lower().strip() == lookup_key:
                    entry = m
                    break

        if entry is not None:
            try:
                p_in = float(entry.price_per_1k_input)
                p_out = float(entry.price_per_1k_output)
                p_cached = (
                    float(entry.price_per_1k_cached_input)
                    if getattr(entry, "price_per_1k_cached_input", None) is not None
                    else p_in
                )
                eff_cached = max(0, min(cached_tokens, prompt_tokens))
                eff_uncached = max(0, prompt_tokens - eff_cached)
                in_cost = (eff_uncached / 1000.0) * p_in + (eff_cached / 1000.0) * p_cached
                out_cost = (completion_tokens / 1000.0) * p_out
                return float(round(in_cost + out_cost, 6))
            except (ValueError, TypeError):
                pass

    rate = MODEL_PRICING.get(lookup_key)
    if not rate:
        for k, v in MODEL_PRICING.items():
            if lookup_key.startswith(k) or k in lookup_key:
                rate = v
                break

    if not rate:
        rate = (0.50, 2.00)  # Conservative fallback

    p_in_m = rate[0]
    p_out_m = rate[1]
    p_cached_m = rate[2] if len(rate) > 2 else rate[0]
    eff_cached = max(0, min(cached_tokens, prompt_tokens))
    eff_uncached = max(0, prompt_tokens - eff_cached)
    input_cost = (eff_uncached / 1_000_000.0) * p_in_m + (eff_cached / 1_000_000.0) * p_cached_m
    output_cost = (completion_tokens / 1_000_000.0) * p_out_m
    return float(round(input_cost + output_cost, 6))


def estimate_tokens(text: str) -> int:
    """Estimate token count with tiktoken or fallback length heuristic."""
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def clean_code_fences(content: str) -> str:
    """Strip markdown JSON code fences from model outputs."""
    s = content.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def is_retryable_error(exc: Exception) -> bool:
    """Determine if an exception represents a 429, 5xx, or network timeout."""
    if is_fatal_api_error(exc) or isinstance(exc, (PaidCallsDisabledError, TruncatedOutputError)):
        return False
    status_code = getattr(exc, "status_code", None)
    if status_code in (429, 500, 502, 503, 504):
        return True

    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        if isinstance(exc, (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError)):
            return True
    except ImportError:
        pass

    try:
        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.APIError) and exc.code in (429, 500, 502, 503, 504):
            return True
    except ImportError:
        pass

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    msg = str(exc).lower()
    return any(term in msg for term in ("429", "rate limit", "timed out", "timeout", "500", "502", "503", "504"))


# ---------------------------------------------------------------------------
# Rate Limiting & Concurrency
# ---------------------------------------------------------------------------
class RateLimiter:
    """Sliding-window requests-per-minute rate limiter."""

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.lock = threading.Lock()
        self.timestamps: List[float] = []

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        with self.lock:
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < 60.0]
            if len(self.timestamps) >= self.rpm:
                oldest = self.timestamps[0]
                sleep_sec = max(0.01, 60.0 - (now - oldest))
                time.sleep(sleep_sec)
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < 60.0]
            self.timestamps.append(time.time())


@dataclass
class LLMResponse(Generic[T]):
    """Standardized response container across all LLM providers."""

    content: str
    structured: Optional[T] = None
    model: str = "unknown"
    provider: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    usage_estimated: bool = False
    structured_output_ok: Optional[bool] = None
    retry_count: int = 0
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Disk Caching
# ---------------------------------------------------------------------------
class DiskCache:
    """Local disk-based response cache keyed by SHA-256 hash."""

    def __init__(self, cache_dir: Union[str, Path] = ".cache/llm"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(
        self,
        model_name: str,
        model_id: str,
        prompt: str,
        params: Dict[str, Any],
        system_prompt: Optional[str] = None,
        response_model_name: Optional[str] = None,
    ) -> str:
        """Generate deterministic cache key based on model name, model_id, prompt, params."""
        payload = {
            "model_name": model_name,
            "model_id": model_id,
            "prompt": prompt,
            "system_prompt": system_prompt or "",
            "params": params,
            "response_model": response_model_name or "",
        }
        raw_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if present."""
        path = self.cache_dir / f"{key}.json"
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.debug("Failed to read cache file %s: %s", path, e)
                return None
        return None

    def set(self, key: str, data: Dict[str, Any]) -> None:
        """Write response to cache."""
        path = self.cache_dir / f"{key}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.debug("Failed to write cache file %s: %s", path, e)

    def clear(self) -> int:
        """Clear all cached entries."""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
        return count


# ---------------------------------------------------------------------------
# DuckDB Logging
# ---------------------------------------------------------------------------
def log_agent_call_to_duckdb(
    run_id: str,
    agent_name: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_cost: float,
    latency_ms: float,
    step_id: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
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
) -> str:
    """Log an individual LLM invocation into DuckDB agent_calls table."""
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    effective_run_id = run_id or "standalone_llm_session"
    effective_cost = cost_usd if cost_usd is not None else total_cost

    if db_path is False or db_path == "disabled" or db_path == "":
        return call_id

    target_db = db_path
    if target_db is None:
        try:
            target_db = get_config().experiment.db_path
        except Exception:
            target_db = Path("results/sca_metrics.duckdb")

    try:
        conn = get_duckdb_connection(target_db)
        # Ensure parent run row exists to satisfy foreign key constraint
        conn.execute(
            """
            INSERT OR IGNORE INTO runs (run_id, experiment_name, status, metadata)
            VALUES (?, 'agent_call_session', 'active', ?)
            """,
            [effective_run_id, json.dumps({"source": "llm_client"})],
        )

        conn.execute(
            """
            INSERT INTO agent_calls (
                call_id, run_id, step_id, agent_name, model_name, provider,
                prompt_tokens, completion_tokens, cached_tokens, total_cost, cost_usd,
                latency_ms, retry_count, structured_output_ok, error_type, cache_hit, usage_estimated,
                system_name, echelon, step_number, prompt_version, fallback, order_quantity, reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                call_id,
                effective_run_id,
                step_id,
                agent_name,
                model_name,
                provider,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                round(effective_cost, 6),
                round(effective_cost, 6),
                round(latency_ms, 2),
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
        conn.close()
    except Exception as exc:
        logger.debug("Could not log agent call to DuckDB (%s): %s", target_db, exc)

    return call_id


# ---------------------------------------------------------------------------
# Opik Observability Integration
# ---------------------------------------------------------------------------
def trace_opik_call(
    trace_name: str,
    prompt: str,
    response: str,
    model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_ms: float,
) -> None:
    """Log LLM call to Opik observability platform if available and configured."""
    try:
        if not os.getenv("OPIK_API_KEY") and not os.getenv("OPIK_URL_OVERRIDE"):
            return

        import opik

        client = opik.Opik()
        trace = client.trace(
            name=trace_name,
            input={"prompt": prompt},
            output={"response": response},
        )
        trace.span(
            name="llm_generation",
            type="llm",
            input={"prompt": prompt},
            output={"response": response},
            metadata={
                "model": model,
                "provider": provider,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_cost_usd": cost_usd,
                "latency_ms": latency_ms,
            },
        )
    except Exception as exc:
        logger.debug("Opik trace could not be recorded: %s", exc)


# ---------------------------------------------------------------------------
# Base Provider Adapter
# ---------------------------------------------------------------------------
class BaseLLMAdapter(ABC):
    """Abstract interface for LLM provider adapters."""

    def __init__(self, provider_config: Optional[ProviderConfig] = None):
        self.cfg = provider_config

    @abstractmethod
    def generate_raw(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        response_model: Optional[Type[BaseModel]] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, int, int, int]:
        """Execute call to provider API and return (content, prompt_tokens, completion_tokens, cached_tokens)."""
        pass


# ---------------------------------------------------------------------------
# 1. OpenAI-Compatible Adapter (Isoquant GLM, OpenAI, Grok/xAI)
# ---------------------------------------------------------------------------
class OpenAICompatibleAdapter(BaseLLMAdapter):
    """Adapter for OpenAI-compatible endpoints with configurable base_url and api_key_env."""

    def __init__(self, model_entry: ModelEntryConfig):
        super().__init__()
        self.entry = model_entry
        api_key = os.getenv(model_entry.api_key_env)
        if not api_key:
            raise ValueError(
                f"Required environment variable '{model_entry.api_key_env}' for model '{model_entry.name}' is missing or empty. "
                "Please set it in your .env file."
            )
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=model_entry.base_url,
        )

    def generate_raw(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        response_model: Optional[Type[BaseModel]] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, int, int, int]:
        target_model = model or self.entry.model_id
        temp = temperature if temperature is not None else self.entry.temperature
        timeout = timeout_seconds if timeout_seconds is not None else self.entry.timeout_s
        retries = max_retries if max_retries is not None else 3

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        effective_prompt = prompt
        # If response_model provided and supports_json_schema is False, instruct model directly
        if response_model is not None and not self.entry.supports_json_schema:
            schema_json = json.dumps(response_model.model_json_schema())
            effective_prompt = f"{prompt}\n\nRespond ONLY with a valid JSON object matching this schema:\n{schema_json}"

        messages.append({"role": "user", "content": effective_prompt})

        call_kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "timeout": timeout,
            "temperature": temp,
        }
        if seed is not None:
            call_kwargs["seed"] = seed
        if "max_tokens" in kwargs:
            call_kwargs["max_tokens"] = kwargs["max_tokens"]
        elif self.entry.max_tokens:
            call_kwargs["max_tokens"] = self.entry.max_tokens

        # Optional parameter to limit or disable provider reasoning tokens
        if getattr(self.entry, "reasoning_tokens", None) is not None:
            r_val = self.entry.reasoning_tokens
            if isinstance(r_val, bool) and not r_val or str(r_val).lower() in ("disabled", "none", "0", "false"):
                call_kwargs.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}
            elif isinstance(r_val, int) and r_val > 0:
                call_kwargs.setdefault("extra_body", {})["max_thinking_tokens"] = r_val
            elif isinstance(r_val, str) and r_val.lower() not in ("disabled", "none"):
                call_kwargs.setdefault("extra_body", {})["reasoning_tokens"] = r_val

        if getattr(self.entry, "reasoning_effort", None) is not None:
            call_kwargs["reasoning_effort"] = self.entry.reasoning_effort
        if getattr(self.entry, "extra_body", None):
            call_kwargs.setdefault("extra_body", {}).update(self.entry.extra_body)

        # Backoff retry loop for 429/5xx/timeouts
        for attempt in range(retries + 1):
            try:
                if response_model is not None and self.entry.supports_json_schema:
                    try:
                        res = self.client.beta.chat.completions.parse(
                            response_format=response_model,
                            **call_kwargs,
                        )
                        return check_completion(res, target_model)
                    except TruncatedOutputError:
                        raise
                    except Exception as parse_err:
                        if is_retryable_error(parse_err) or is_fatal_api_error(parse_err):
                            raise parse_err
                        completion = getattr(parse_err, "completion", None)
                        if completion is not None:
                            # Length/content-filter stop: re-sending without response_format hits the same limit
                            return check_completion(completion, target_model)
                        # Fallback to standard chat completions
                        res = self.client.chat.completions.create(**call_kwargs)
                        return check_completion(res, target_model)
                else:
                    res = self.client.chat.completions.create(**call_kwargs)
                    return check_completion(res, target_model)

            except Exception as exc:
                if attempt >= retries or not is_retryable_error(exc):
                    raise exc
                backoff = (2**attempt) * 0.5 + random.uniform(0, 0.2 * (2**attempt))
                logger.warning(
                    "OpenAI-compatible error on attempt %d: %s. Retrying in %.2fs...",
                    attempt + 1,
                    exc.__class__.__name__,
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError("OpenAI-compatible generation failed after retries.")


# ---------------------------------------------------------------------------
# 2. Gemini Adapter (Official Google GenAI SDK)
# ---------------------------------------------------------------------------
class GeminiAdapter(BaseLLMAdapter):
    """Adapter for Google Gemini models via google.genai SDK."""

    def __init__(self, model_entry: ModelEntryConfig):
        super().__init__()
        self.entry = model_entry
        api_key = os.getenv(model_entry.api_key_env)
        if not api_key:
            raise ValueError(
                f"Required environment variable '{model_entry.api_key_env}' for model '{model_entry.name}' is missing or empty. "
                "Please set it in your .env file."
            )
        from google import genai

        self.client = genai.Client(api_key=api_key)

    def generate_raw(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        response_model: Optional[Type[BaseModel]] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, int, int, int]:
        from google.genai import types

        target_model = model or self.entry.model_id
        temp = temperature if temperature is not None else self.entry.temperature
        retries = max_retries if max_retries is not None else 3

        config_kwargs: Dict[str, Any] = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if temp is not None:
            config_kwargs["temperature"] = temp
        if seed is not None:
            config_kwargs["seed"] = seed
        if "max_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs["max_tokens"]
        elif self.entry.max_tokens:
            config_kwargs["max_output_tokens"] = self.entry.max_tokens

        effective_prompt = prompt
        if response_model is not None:
            if self.entry.supports_json_schema:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = response_model
            else:
                schema_json = json.dumps(response_model.model_json_schema())
                effective_prompt += f"\n\nRespond ONLY with a valid JSON object matching this schema:\n{schema_json}"

        cfg = types.GenerateContentConfig(**config_kwargs)

        for attempt in range(retries + 1):
            try:
                res = self.client.models.generate_content(
                    model=target_model,
                    contents=effective_prompt,
                    config=cfg,
                )
                content = res.text or ""
                p_tok = res.usage_metadata.prompt_token_count if res.usage_metadata else 0
                c_tok = res.usage_metadata.candidates_token_count if res.usage_metadata else 0
                cached_tok = getattr(res.usage_metadata, "cached_content_token_count", 0) if res.usage_metadata else 0
                return content, p_tok, c_tok, cached_tok or 0
            except Exception as exc:
                if attempt >= retries or not is_retryable_error(exc):
                    raise exc
                backoff = (2**attempt) * 0.5 + random.uniform(0, 0.2 * (2**attempt))
                logger.warning(
                    "Gemini error on attempt %d: %s. Retrying in %.2fs...",
                    attempt + 1,
                    exc.__class__.__name__,
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError("Google GenAI generation failed after retries.")


# ---------------------------------------------------------------------------
# 3. Legacy/Direct Adapters (OpenAI, Anthropic, Google, Local) for backwards compatibility
# ---------------------------------------------------------------------------
class OpenAIAdapter(BaseLLMAdapter):
    def __init__(self, provider_config: Optional[ProviderConfig] = None):
        super().__init__(provider_config)
        from openai import OpenAI

        api_key = None
        base_url = None
        if self.cfg:
            if self.cfg.api_key_env_var:
                api_key = os.getenv(self.cfg.api_key_env_var)
            base_url = self.cfg.base_url

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "dummy-key-for-init"),
            base_url=base_url,
        )

    def generate_raw(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        response_model: Optional[Type[BaseModel]] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, int, int, int]:
        target_model = model or (self.cfg.default_model if self.cfg else "gpt-4o-mini")
        timeout = timeout_seconds or (self.cfg.timeout_seconds if self.cfg else 45.0)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "timeout": timeout,
        }
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if seed is not None:
            call_kwargs["seed"] = seed

        if response_model is not None:
            res = self.client.beta.chat.completions.parse(
                response_format=response_model,
                **call_kwargs,
            )
        else:
            res = self.client.chat.completions.create(**call_kwargs)

        content = res.choices[0].message.content or ""
        p_tok = res.usage.prompt_tokens if res.usage else len(prompt.split())
        c_tok = res.usage.completion_tokens if res.usage else len(content.split())
        return content, p_tok, c_tok, 0


class AnthropicAdapter(BaseLLMAdapter):
    def __init__(self, provider_config: Optional[ProviderConfig] = None):
        super().__init__(provider_config)
        import anthropic

        api_key = None
        base_url = None
        if self.cfg:
            if self.cfg.api_key_env_var:
                api_key = os.getenv(self.cfg.api_key_env_var)
            base_url = self.cfg.base_url

        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", "dummy-key-for-init"),
            base_url=base_url,
        )

    def generate_raw(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        response_model: Optional[Type[BaseModel]] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, int, int, int]:
        target_model = model or (self.cfg.default_model if self.cfg else "claude-3-5-sonnet-20241022")
        timeout = timeout_seconds or (self.cfg.timeout_seconds if self.cfg else 60.0)
        effective_system = system_prompt or ""
        if response_model is not None:
            schema_json = json.dumps(response_model.model_json_schema())
            effective_system = (
                effective_system
                + f"\nYou MUST respond ONLY with valid JSON conforming strictly to this schema:\n{schema_json}"
            ).strip()

        messages = [{"role": "user", "content": prompt}]
        call_kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "timeout": timeout,
        }
        if effective_system:
            call_kwargs["system"] = effective_system
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        res = self.client.messages.create(**call_kwargs)
        content = "".join([blk.text for blk in res.content if hasattr(blk, "text")])
        p_tok = res.usage.input_tokens
        c_tok = res.usage.output_tokens
        return content, p_tok, c_tok, 0


class GoogleAdapter(BaseLLMAdapter):
    def __init__(self, provider_config: Optional[ProviderConfig] = None):
        super().__init__(provider_config)
        from google import genai

        api_key = None
        if self.cfg and self.cfg.api_key_env_var:
            api_key = os.getenv(self.cfg.api_key_env_var)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "dummy-key")
        self.client = genai.Client(api_key=self.api_key)

    def generate_raw(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        response_model: Optional[Type[BaseModel]] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, int, int, int]:
        from google.genai import types

        target_model = model or (self.cfg.default_model if self.cfg else "gemini-1.5-flash")
        config_kwargs: Dict[str, Any] = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if seed is not None:
            config_kwargs["seed"] = seed
        if response_model is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_model

        cfg = types.GenerateContentConfig(**config_kwargs)
        res = self.client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=cfg,
        )
        content = res.text or ""
        p_tok = res.usage_metadata.prompt_token_count if res.usage_metadata else 0
        c_tok = res.usage_metadata.candidates_token_count if res.usage_metadata else 0
        return content, p_tok, c_tok, 0


class LocalAdapter(BaseLLMAdapter):
    def __init__(self, provider_config: Optional[ProviderConfig] = None):
        super().__init__(provider_config)
        from openai import OpenAI

        base_url = "http://localhost:8000/v1"
        if self.cfg and self.cfg.base_url:
            base_url = self.cfg.base_url
        self.client = OpenAI(api_key="local-token", base_url=base_url)

    def generate_raw(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        response_model: Optional[Type[BaseModel]] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, int, int, int]:
        target_model = model or (self.cfg.default_model if self.cfg else "local-model")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        effective_prompt = prompt
        if response_model is not None:
            schema_json = json.dumps(response_model.model_json_schema())
            effective_prompt += f"\nRespond strictly in JSON conforming to:\n{schema_json}"
        messages.append({"role": "user", "content": effective_prompt})

        call_kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "timeout": timeout_seconds or 60.0,
        }
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if seed is not None:
            call_kwargs["seed"] = seed

        res = self.client.chat.completions.create(**call_kwargs)
        content = res.choices[0].message.content or ""
        p_tok = res.usage.prompt_tokens if res.usage else len(prompt.split())
        c_tok = res.usage.completion_tokens if res.usage else len(content.split())
        return content, p_tok, c_tok, 0


# ---------------------------------------------------------------------------
# 4. MockLLM Adapter (Deterministic Offline Test Double)
# ---------------------------------------------------------------------------
class MockLLM(BaseLLMAdapter):
    """Deterministic mock adapter for unit tests with simulated errors and canned sequences."""

    def __init__(self, model_entry: Optional[ModelEntryConfig] = None):
        super().__init__()
        self.entry = model_entry
        self.call_history: List[Dict[str, Any]] = []
        self.canned_responses: Dict[str, Union[str, BaseModel, Dict[str, Any]]] = {}
        self.sequential_responses: Dict[str, List[Union[str, BaseModel, Dict[str, Any]]]] = {}
        self.error_triggers: Dict[str, Tuple[Exception, int]] = {}  # prompt_substring -> (error, fail_count)
        self.error_counts: Dict[str, int] = {}
        self.default_text_response: str = "Deterministic mock response."
        self.simulate_missing_tokens: bool = False

        # Scripted replenishment policy settings
        self.scripted_policy_enabled: bool = False
        self.scripted_seed: Optional[int] = 42
        self.failure_rate_malformed: float = 0.0
        self.failure_rate_out_of_range: float = 0.0
        self.scripted_base_order: float = 100.0
        self.scripted_noise_std: float = 10.0
        self.scripted_max_order: float = 300.0
        self._scripted_rng: Optional[random.Random] = None

        # Design v5 (ActionDecision) modes: valid | malformed | bad_action | always_follow | detector_copy
        self.v5_mode: Optional[str] = None
        self._v5_rng: random.Random = random.Random(42)

    def configure_scripted_policy(
        self,
        enabled: bool = True,
        seed: Optional[int] = 42,
        failure_rate_malformed: float = 0.0,
        failure_rate_out_of_range: float = 0.0,
        base_order: float = 100.0,
        noise_std: float = 10.0,
        max_order: float = 300.0,
    ) -> None:
        """Configure scripted replenishment policy returning plausible orders and optional failure modes."""
        self.scripted_policy_enabled = enabled
        self.scripted_seed = seed
        self.failure_rate_malformed = failure_rate_malformed
        self.failure_rate_out_of_range = failure_rate_out_of_range
        self.scripted_base_order = base_order
        self.scripted_noise_std = noise_std
        self.scripted_max_order = max_order
        self._scripted_rng = random.Random(seed) if seed is not None else random.Random(42)

    def set_seed(self, seed: int) -> None:
        """Set deterministic RNG seed for scripted policy."""
        self.scripted_seed = seed
        self._scripted_rng = random.Random(seed)
        self._v5_rng = random.Random(seed)

    V5_MODES = ("valid", "malformed", "bad_action", "always_follow", "detector_copy")

    def configure_v5_mode(self, mode: Optional[str], seed: int = 42) -> None:
        """Answer design-v5 ActionDecision requests in one of V5_MODES (None turns it off)."""
        if mode is not None and mode not in self.V5_MODES:
            raise ValueError(f"Unknown v5 mock mode '{mode}'. Must be one of {self.V5_MODES}.")
        self.v5_mode = mode
        self._v5_rng = random.Random(seed)

    def _v5_tool_content(self, mock_context: Optional[Dict[str, Any]]) -> str:
        """Design v5.1 tool request: correct tools and inputs, except in the failure modes."""
        ctx = mock_context or {}
        truth = ctx.get("truth", {})
        if self.v5_mode == "malformed":
            return '{"calls": [{"tool": "regime_detector", "args": '
        if self.v5_mode == "bad_action":
            wrong = {k: (None if k == "demand_history" else 0.0) for k in
                     ("demand_history", "current_demand", "inventory", "pipeline", "backlog", "last_order")}
            return json.dumps({"calls": [{"tool": "eoq_quantity", "args": {**wrong, "demand_history": [1.0]}}]})
        args = {k: truth.get(k) for k in
                ("demand_history", "current_demand", "inventory", "pipeline", "backlog", "last_order")}
        return json.dumps({"calls": [{"tool": t, "args": args} for t in ctx.get("required", [])]})

    def _v5_content(self, mock_context: Optional[Dict[str, Any]]) -> str:
        rng = self._v5_rng
        regimes = ("stable", "trend", "shock")
        actions = ("follow", "boost_small", "boost_large", "trim")
        if self.v5_mode == "malformed":
            return '{"regime": "stable", "action": "follow", "rationale": "unterminated'
        if self.v5_mode == "bad_action":
            return json.dumps({"regime": "stable", "action": "order_more", "rationale": "Invalid action label."})
        if self.v5_mode == "always_follow":
            return json.dumps({"regime": "stable", "action": "follow", "rationale": "Demand looks steady."})
        if self.v5_mode == "detector_copy":
            ctx = mock_context or {}
            phase = ctx.get("detector_phase", "stable")
            action = {"stable": "follow", "trend": "boost_small", "shock": "boost_large", "after_shock": "trim"}[phase]
            return json.dumps({"regime": ctx.get("detector_regime", "stable"), "action": action,
                               "rationale": f"Copied detector phase {phase}."})
        return json.dumps({"regime": rng.choice(regimes), "action": rng.choice(actions),
                           "rationale": "Mock valid decision."})

    def set_response(self, prompt_substring: str, response: Union[str, BaseModel, Dict[str, Any]]) -> None:
        """Register canned response triggered when prompt contains the substring."""
        self.canned_responses[prompt_substring] = response

    def set_sequential_responses(
        self, prompt_substring: str, responses: List[Union[str, BaseModel, Dict[str, Any]]]
    ) -> None:
        """Register a list of responses returned sequentially on consecutive calls matching prompt."""
        self.sequential_responses[prompt_substring] = list(responses)

    def set_error(self, prompt_substring: str, exception: Exception, fail_count: int = 1) -> None:
        """Register an error raised on matching prompts for the specified number of calls before succeeding."""
        self.error_triggers[prompt_substring] = (exception, fail_count)
        self.error_counts[prompt_substring] = 0

    def clear(self) -> None:
        """Reset mock history, responses, and simulated errors."""
        self.call_history.clear()
        self.canned_responses.clear()
        self.sequential_responses.clear()
        self.error_triggers.clear()
        self.error_counts.clear()
        self.simulate_missing_tokens = False
        self.scripted_policy_enabled = False
        self.failure_rate_malformed = 0.0
        self.failure_rate_out_of_range = 0.0
        self._scripted_rng = None
        self.v5_mode = None

    def _synthesize_mock_model(self, model_cls: Type[BaseModel]) -> BaseModel:
        """Synthesize a valid mock instance of any Pydantic model for structured tests."""
        data: Dict[str, Any] = {}
        for f_name, f_info in model_cls.model_fields.items():
            if f_info.default is not None and f_info.default is not PydanticUndefined:
                data[f_name] = f_info.default
            elif f_info.default_factory is not None:
                data[f_name] = f_info.default_factory()
            else:
                ann = f_info.annotation
                if ann in (int, Optional[int]):
                    data[f_name] = 100
                elif ann in (float, Optional[float]):
                    data[f_name] = 100.0
                elif ann in (str, Optional[str]):
                    data[f_name] = f"mock_{f_name}"
                elif ann in (bool, Optional[bool]):
                    data[f_name] = True
                elif getattr(ann, "__origin__", None) is list:
                    data[f_name] = []
                elif getattr(ann, "__origin__", None) is dict:
                    data[f_name] = {}
                else:
                    data[f_name] = None
        return model_cls.model_validate(data)

    def generate_raw(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        response_model: Optional[Type[BaseModel]] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, int, int, int]:
        record = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "response_model": response_model.__name__ if response_model else None,
            "kwargs": kwargs,
        }
        self.call_history.append(record)

        # Check simulated errors
        for trigger, (err, fail_count) in list(self.error_triggers.items()):
            if trigger.lower() in prompt.lower():
                count = self.error_counts[trigger]
                if count < fail_count:
                    self.error_counts[trigger] = count + 1
                    raise err

        # Check sequential responses
        matched_resp = None
        for key, resp_list in self.sequential_responses.items():
            if key.lower() in prompt.lower() and resp_list:
                matched_resp = resp_list.pop(0)
                break

        # Check canned responses
        if matched_resp is None:
            for key, resp in self.canned_responses.items():
                if key.lower() in prompt.lower():
                    matched_resp = resp
                    break

        if matched_resp is not None:
            if isinstance(matched_resp, BaseModel):
                content = matched_resp.model_dump_json()
            elif isinstance(matched_resp, dict):
                content = json.dumps(matched_resp)
            else:
                content = str(matched_resp)
        elif self.v5_mode is not None and response_model is not None and response_model.__name__ == "ActionDecision":
            content = self._v5_content(kwargs.get("mock_context"))
        elif self.v5_mode is not None and response_model is not None and response_model.__name__ == "ToolRequest":
            content = self._v5_tool_content(kwargs.get("mock_context"))
        elif self.scripted_policy_enabled:
            if self._scripted_rng is None:
                eff_seed = seed if seed is not None else (self.scripted_seed if self.scripted_seed is not None else 42)
                self._scripted_rng = random.Random(eff_seed)

            # Failure mode 1: Malformed JSON
            if self.failure_rate_malformed > 0.0 and self._scripted_rng.random() < self.failure_rate_malformed:
                content = "Malformed model response: not valid JSON { quantity: 100 "
            # Failure mode 2: Out of range quantity
            elif self.failure_rate_out_of_range > 0.0 and self._scripted_rng.random() < self.failure_rate_out_of_range:
                bad_qty = -50.0 if self._scripted_rng.random() < 0.5 else 99999.0
                content = json.dumps({
                    "quantity": bad_qty,
                    "reasoning": f"Simulated out-of-range order of {bad_qty}",
                })
            else:
                # Plausible in-range order (forecast-like base order with noise)
                demand_val = None
                for line in prompt.split("\n"):
                    if "Current Demand:" in line:
                        try:
                            demand_val = float(line.split("Current Demand:")[1].strip().split()[0])
                            break
                        except Exception:
                            pass
                base_qty = demand_val if demand_val is not None else self.scripted_base_order
                noise = self._scripted_rng.normalvariate(0.0, self.scripted_noise_std)
                qty = max(0.0, min(self.scripted_max_order, round(base_qty + noise, 1)))
                content = json.dumps({
                    "quantity": qty,
                    "reasoning": f"Plausible replenishment order of {qty:.1f} units",
                })
        elif response_model is not None:
            mock_inst = self._synthesize_mock_model(response_model)
            content = mock_inst.model_dump_json()
        else:
            content = f"{self.default_text_response} (prompt_len={len(prompt)})"

        if self.simulate_missing_tokens:
            return content, 0, 0, 0

        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(content.split()))
        return content, prompt_tokens, completion_tokens, 0


# ---------------------------------------------------------------------------
# Central Provider-Agnostic LLMClient
# ---------------------------------------------------------------------------
class LLMClient:
    """Central provider-agnostic LLM client driven by configs/models.yaml."""

    def __init__(
        self,
        config: Optional[ModelsConfig] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        mock_mode: Optional[bool] = None,
        cache_enabled: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        opik_enabled: Optional[bool] = None,
        db_path: Optional[Union[str, Path]] = None,
        use_cache: Optional[bool] = None,
    ):
        self.config: ModelsConfig = config or load_models_config()
        self.default_provider = provider or self.config.default_provider
        self.default_model = model or self.config.default_model
        self.mock_mode = mock_mode if mock_mode is not None else self.config.mock_mode

        self.cache_enabled = use_cache if use_cache is not None else cache_enabled
        self.cache = DiskCache(cache_dir=cache_dir or self.config.cache_dir)
        self.opik_enabled = opik_enabled if opik_enabled is not None else self.config.opik_enabled
        self.db_path = db_path

        # Cache of initialized provider adapters
        self.mock = MockLLM()
        self._adapters: Dict[str, BaseLLMAdapter] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._concurrency_semaphores: Dict[str, threading.BoundedSemaphore] = {}

    def get_adapter(self, provider_or_model: str) -> BaseLLMAdapter:
        """Resolve and initialize adapter for given provider or model name."""
        p_or_m = provider_or_model.lower().strip()
        if self.mock_mode or p_or_m == "mock":
            return self.mock

        # 1. Match against registered models in models.yaml
        model_entry = self.config.get_model(p_or_m)
        if model_entry is not None:
            if not model_entry.enabled:
                raise ValueError(f"Model '{model_entry.name}' is disabled in configuration.")

            cache_key = f"model:{model_entry.name}"
            if cache_key in self._adapters:
                return self._adapters[cache_key]

            prov = model_entry.provider.lower().strip()
            if prov == "openai_compatible":
                adapter = OpenAICompatibleAdapter(model_entry)
            elif prov in ("gemini", "google"):
                adapter = GeminiAdapter(model_entry)
            elif prov == "mock":
                return self.mock
            else:
                raise ValueError(f"Unsupported provider '{prov}' for model '{model_entry.name}'.")

            self._adapters[cache_key] = adapter
            return adapter

        # 2. Backward compatibility with provider names: "openai", "anthropic", "google", "local"
        if p_or_m in self._adapters:
            return self._adapters[p_or_m]

        prov_cfg = self.config.providers.get(p_or_m)
        if p_or_m == "openai":
            adapter = OpenAIAdapter(prov_cfg)
        elif p_or_m == "anthropic":
            adapter = AnthropicAdapter(prov_cfg)
        elif p_or_m in ("google", "gemini"):
            adapter = GoogleAdapter(prov_cfg)
        elif p_or_m in ("local", "ollama"):
            adapter = LocalAdapter(prov_cfg)
        else:
            raise ValueError(
                f"Unsupported LLM provider or unknown model '{provider_or_model}'. "
                "Ensure model is declared in configs/models.yaml."
            )

        self._adapters[p_or_m] = adapter
        return adapter

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_model: Optional[Type[T]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        use_cache: Optional[bool] = None,
        agent_name: Optional[str] = None,
        run_id: Optional[str] = None,
        step_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse[T]:
        """Generate response with provider fallback, caching, structured parsing, and DuckDB logging."""
        start_time = time.time()
        eff_model_name = model or self.default_model
        eff_agent = agent_name or "supply_chain_agent"
        should_cache = self.cache_enabled if use_cache is None else use_cache

        # Resolve model configuration entry
        model_entry = self.config.get_model(eff_model_name)
        if model_entry is None:
            # Check model_id matching
            for m in self.config.models:
                if m.model_id.lower().strip() == eff_model_name.lower().strip():
                    model_entry = m
                    break

        if model_entry is not None:
            if not model_entry.enabled and not self.mock_mode:
                raise ValueError(f"Model '{model_entry.name}' is disabled in configuration.")
            eff_provider = model_entry.provider
            target_model_id = model_entry.model_id
            eff_temp = temperature if temperature is not None else model_entry.temperature
            eff_max_tokens = kwargs.get("max_tokens", model_entry.max_tokens)
            timeout_s = kwargs.get("timeout_s", model_entry.timeout_s)
            supports_json_schema = model_entry.supports_json_schema
        else:
            eff_provider = (provider or self.default_provider).lower()
            target_model_id = eff_model_name
            eff_temp = temperature
            eff_max_tokens = kwargs.get("max_tokens", 2048)
            timeout_s = kwargs.get("timeout_s", 60.0)
            supports_json_schema = True

        # Test-double context (e.g. detector label for MockLLM detector_copy); never sent to real providers
        mock_context = kwargs.pop("mock_context", None)

        params_sig = {
            "temperature": eff_temp,
            "seed": seed,
            "max_tokens": eff_max_tokens,
            "extra": kwargs,
        }

        system_name = kwargs.get("system_name")
        echelon = kwargs.get("echelon")
        step_number = kwargs.get("step_number")
        prompt_version = kwargs.get("prompt_version")
        fallback = kwargs.get("fallback", False)
        order_quantity = kwargs.get("order_quantity")
        log_to_db = kwargs.get("log_to_db", True)

        # 1. Check Disk Cache: keyed by hash(model name, model_id, prompt, params)
        cache_key = ""
        if should_cache:
            cache_key = self.cache.get_cache_key(
                model_name=eff_model_name,
                model_id=target_model_id,
                prompt=prompt,
                params=params_sig,
                system_prompt=system_prompt,
                response_model_name=response_model.__name__ if response_model else None,
            )
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                content = cached_data["content"]
                structured_obj: Optional[T] = None
                struct_ok = None
                if response_model is not None:
                    try:
                        structured_obj = response_model.model_validate_json(clean_code_fences(content))
                        struct_ok = True
                    except Exception:
                        struct_ok = False

                elapsed_ms = (time.time() - start_time) * 1000.0
                p_tok = cached_data.get("prompt_tokens", 0)
                c_tok = cached_data.get("completion_tokens", 0)
                tot_tok = cached_data.get("total_tokens", p_tok + c_tok)

                # Cache hits log cost_usd = 0 and cache_hit = true
                if log_to_db:
                    log_agent_call_to_duckdb(
                    run_id=run_id or "standalone_llm_session",
                    agent_name=eff_agent,
                    model_name=eff_model_name,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_cost=0.0,
                    latency_ms=elapsed_ms,
                    step_id=step_id,
                    db_path=self.db_path,
                    provider=eff_provider,
                    cached_tokens=cached_data.get("cached_tokens", 0),
                    cost_usd=0.0,
                    retry_count=0,
                    structured_output_ok=struct_ok,
                    error_type=None,
                    cache_hit=True,
                    usage_estimated=cached_data.get("usage_estimated", False),
                    system_name=system_name,
                    echelon=echelon,
                    step_number=step_number,
                    prompt_version=prompt_version,
                    fallback=fallback,
                    order_quantity=order_quantity,
                )

                return LLMResponse(
                    content=content,
                    structured=structured_obj,
                    model=eff_model_name,
                    provider=eff_provider,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=tot_tok,
                    cached_tokens=cached_data.get("cached_tokens", 0),
                    latency_ms=elapsed_ms,
                    cost_usd=0.0,
                    cached=True,
                    usage_estimated=cached_data.get("usage_estimated", False),
                    structured_output_ok=struct_ok,
                    retry_count=0,
                )

        # 2. Enforce Per-Model Rate Limit (RPM) and Concurrency Limit
        if model_entry is not None and not self.mock_mode:
            if eff_model_name not in self._rate_limiters:
                self._rate_limiters[eff_model_name] = RateLimiter(model_entry.requests_per_minute)
            if eff_model_name not in self._concurrency_semaphores:
                self._concurrency_semaphores[eff_model_name] = threading.BoundedSemaphore(
                    model_entry.max_concurrency
                )

            self._rate_limiters[eff_model_name].acquire()
            concurrency_sem = self._concurrency_semaphores[eff_model_name]
        else:
            concurrency_sem = None

        # 3. Execute via Adapter
        adapter_key = eff_model_name if (model_entry is not None and not self.mock_mode) else eff_provider
        adapter = self.get_adapter(adapter_key)
        clean_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "max_tokens",
                "timeout_s",
                "timeout_seconds",
                "system_name",
                "echelon",
                "step_number",
                "prompt_version",
                "fallback",
                "order_quantity",
                "log_to_db",
            )
        }
        eff_max_retries = kwargs.get("max_retries", self.config.max_retries)
        retry_count = 0

        def _call_adapter(call_prompt: str) -> Tuple[str, int, int, int]:
            nonlocal retry_count
            call_resp_model = response_model if supports_json_schema else None
            mock_kwargs = {"mock_context": mock_context} if isinstance(adapter, MockLLM) and mock_context else {}
            if not isinstance(adapter, MockLLM) and not paid_calls_allowed():
                raise PaidCallsDisabledError(
                    f"Refusing real network call via {type(adapter).__name__}: set {PAID_CALLS_ENV_VAR}=1."
                )
            for attempt in range(eff_max_retries + 1):
                try:
                    if concurrency_sem:
                        with concurrency_sem:
                            return adapter.generate_raw(
                                prompt=call_prompt,
                                system_prompt=system_prompt,
                                model=target_model_id,
                                temperature=eff_temp,
                                seed=seed,
                                response_model=call_resp_model,
                                timeout_seconds=timeout_s,
                                max_tokens=eff_max_tokens,
                                **clean_kwargs,
                            )
                    return adapter.generate_raw(
                        prompt=call_prompt,
                        system_prompt=system_prompt,
                        model=target_model_id,
                        temperature=eff_temp,
                        seed=seed,
                        response_model=call_resp_model,
                        timeout_seconds=timeout_s,
                        max_tokens=eff_max_tokens,
                        **clean_kwargs,
                        **mock_kwargs,
                    )
                except Exception as exc:
                    if is_fatal_api_error(exc) and not isinstance(exc, FatalAPIError):
                        raise FatalAPIError(
                            f"Fatal API error (HTTP {get_status_code(exc)}) from '{eff_model_name}': {exc}",
                            status_code=get_status_code(exc),
                        ) from exc
                    if attempt >= eff_max_retries or not is_retryable_error(exc):
                        raise exc
                    retry_count += 1
                    backoff = (2**attempt) * 0.1 + random.uniform(0, 0.05 * (2**attempt))
                    logger.warning(
                        "Retryable error on attempt %d: %s. Retrying in %.2fs...",
                        attempt + 1,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
            raise RuntimeError("Adapter generation failed after retries.")

        content, p_tok, c_tok, cached_tok = _call_adapter(prompt)

        # 4. Tokenizer Fallback: if response has no token usage, estimate tokens
        usage_estimated = False
        if p_tok <= 0:
            p_tok = estimate_tokens(prompt)
            usage_estimated = True
        if c_tok <= 0:
            c_tok = estimate_tokens(content)
            usage_estimated = True

        retry_count = 0
        structured_obj: Optional[T] = None
        structured_ok: Optional[bool] = None
        error_type: Optional[str] = None

        # 5. Structured Output Validation & Single Retry on Failure
        if response_model is not None:
            try:
                cleaned = clean_code_fences(content)
                structured_obj = response_model.model_validate_json(cleaned)
                structured_ok = True
            except Exception as first_val_err:
                # Validation failure: retry once with validation error appended
                retry_count += 1
                logger.info(
                    "Structured validation failed on attempt 1 for %s: %s. Retrying once...",
                    eff_model_name,
                    first_val_err,
                )
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"Previous output was invalid or failed validation:\n{first_val_err}\n"
                    f"Please correct the error and respond strictly in valid JSON conforming to the schema."
                )
                try:
                    retry_content, rp_tok, rc_tok, r_cached_tok = _call_adapter(retry_prompt)
                    p_tok += rp_tok or estimate_tokens(retry_prompt)
                    c_tok += rc_tok or estimate_tokens(retry_content)
                    cached_tok += r_cached_tok
                    content = retry_content
                    cleaned_retry = clean_code_fences(retry_content)
                    structured_obj = response_model.model_validate_json(cleaned_retry)
                    structured_ok = True
                except (FatalAPIError, PaidCallsDisabledError, TruncatedOutputError):
                    raise
                except Exception as second_val_err:
                    structured_ok = False
                    error_type = "StructuredOutputError"
                    elapsed_ms = (time.time() - start_time) * 1000.0
                    cost_usd = calculate_cost(eff_model_name, p_tok, c_tok, self.config, cached_tokens=cached_tok)

                    # Record failure in DuckDB before raising
                    if log_to_db:
                        log_agent_call_to_duckdb(
                            run_id=run_id or "standalone_llm_session",
                            agent_name=eff_agent,
                            model_name=eff_model_name,
                            prompt_tokens=p_tok,
                            completion_tokens=c_tok,
                            total_cost=cost_usd,
                            latency_ms=elapsed_ms,
                            step_id=step_id,
                            db_path=self.db_path,
                            provider=eff_provider,
                            cached_tokens=cached_tok,
                            cost_usd=cost_usd,
                            retry_count=retry_count,
                            structured_output_ok=False,
                            error_type=error_type,
                            cache_hit=False,
                            usage_estimated=usage_estimated,
                            system_name=system_name,
                            echelon=echelon,
                            step_number=step_number,
                            prompt_version=prompt_version,
                            fallback=fallback,
                            order_quantity=order_quantity,
                        )
                    raise StructuredOutputError(
                        f"Structured output validation failed for model '{eff_model_name}' after retry: {second_val_err}",
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        cached_tokens=cached_tok,
                    ) from second_val_err

        elapsed_ms = (time.time() - start_time) * 1000.0
        tot_tok = p_tok + c_tok
        cost_usd = calculate_cost(eff_model_name, p_tok, c_tok, self.config, cached_tokens=cached_tok)

        # 6. Write to Disk Cache
        if should_cache and cache_key:
            self.cache.set(
                cache_key,
                {
                    "content": content,
                    "model": eff_model_name,
                    "provider": eff_provider,
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens": tot_tok,
                    "cached_tokens": cached_tok,
                    "cost_usd": cost_usd,
                    "usage_estimated": usage_estimated,
                },
            )

        # 7. Log to DuckDB agent_calls table
        if log_to_db:
            log_agent_call_to_duckdb(
                run_id=run_id or "standalone_llm_session",
                agent_name=eff_agent,
                model_name=eff_model_name,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_cost=cost_usd,
                latency_ms=elapsed_ms,
                step_id=step_id,
                db_path=self.db_path,
                provider=eff_provider,
                cached_tokens=cached_tok,
                cost_usd=cost_usd,
                retry_count=retry_count,
                structured_output_ok=structured_ok,
                error_type=error_type,
                cache_hit=False,
                usage_estimated=usage_estimated,
                system_name=system_name,
                echelon=echelon,
                step_number=step_number,
                prompt_version=prompt_version,
                fallback=fallback,
                order_quantity=order_quantity,
            )

        # 8. Optional Opik tracing
        if self.opik_enabled:
            trace_opik_call(
                trace_name=f"agent_call_{eff_agent}",
                prompt=prompt,
                response=content,
                model=eff_model_name,
                provider=eff_provider,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                cost_usd=cost_usd,
                latency_ms=elapsed_ms,
            )

        return LLMResponse(
            content=content,
            structured=structured_obj,
            model=eff_model_name,
            provider=eff_provider,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=tot_tok,
            cached_tokens=cached_tok,
            latency_ms=elapsed_ms,
            cost_usd=cost_usd,
            cached=False,
            usage_estimated=usage_estimated,
            structured_output_ok=structured_ok,
            retry_count=retry_count,
            error_type=error_type,
        )
