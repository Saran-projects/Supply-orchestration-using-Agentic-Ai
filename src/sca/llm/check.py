"""CLI check utility for all enabled LLM models in configs/models.yaml.

Usage:
    python -m sca.llm.check

Behavior:
    - Finds all enabled models in configs/models.yaml.
    - Sends one tiny text prompt and one tiny structured-output request per enabled model.
    - Prints an ASCII summary table:
      model, status, latency, tokens in/out, cost, structured output ok (yes/no), exact error text.
    - Exits with return code 1 if any enabled model fails, or 0 if all pass.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

from pydantic import BaseModel, Field

from sca.config import load_models_config
from sca.llm.client import LLMClient, StructuredOutputError


class PingResponse(BaseModel):
    """Test model for structured output validation in health check."""

    status: str = Field(default="ok", description="Status string, e.g. 'ok'")
    code: int = Field(default=200, description="Status code integer")


def run_model_check(target_model: Optional[str] = None) -> int:
    """Execute connectivity and structured output checks on all enabled models or a specific target model."""
    config = load_models_config()
    if target_model:
        enabled_models = [m for m in config.models if m.name.lower() == target_model.lower() or m.model_id.lower() == target_model.lower()]
        if not enabled_models:
            print(f"Model '{target_model}' not found in configs/models.yaml.")
            return 1
    else:
        enabled_models = [m for m in config.models if m.enabled]

    if not enabled_models:
        print("No models are currently enabled in configs/models.yaml.")
        return 0

    client = LLMClient(config=config, cache_enabled=False)
    results: List[dict] = []
    any_failed = False

    for m in enabled_models:
        model_name = m.name
        start_time = time.time()
        status = "PASS"
        total_p_tok = 0
        total_c_tok = 0
        total_cost = 0.0
        struct_ok = "yes"
        error_text = ""

        try:
            # 1. Tiny text prompt
            resp1 = client.generate(
                prompt="Ping. Reply with 'OK'. Keep it brief.",
                model=model_name,
                use_cache=False,
            )
            total_p_tok += resp1.prompt_tokens
            total_c_tok += resp1.completion_tokens
            total_cost += resp1.cost_usd

            # 2. Tiny structured output prompt
            resp2 = client.generate(
                prompt="Return a JSON object with status 'ok' and code 200. Output JSON directly.",
                model=model_name,
                response_model=PingResponse,
                use_cache=False,
            )
            total_p_tok += resp2.prompt_tokens
            total_c_tok += resp2.completion_tokens
            total_cost += resp2.cost_usd

            if resp2.structured is None:
                struct_ok = "no"
                status = "FAIL"
                error_text = "Structured output returned None"
                any_failed = True

        except Exception as exc:
            status = "FAIL"
            struct_ok = "no" if isinstance(exc, StructuredOutputError) else "n/a"
            error_text = str(exc).strip().replace("\n", " ")
            any_failed = True

        total_latency_ms = (time.time() - start_time) * 1000.0
        results.append(
            {
                "model": model_name,
                "status": status,
                "latency": f"{total_latency_ms:.1f}ms",
                "tokens": f"{total_p_tok}/{total_c_tok}",
                "cost": f"${total_cost:.6f}",
                "structured_ok": struct_ok,
                "error": error_text[:60] + "..." if len(error_text) > 60 else (error_text or "-"),
            }
        )

    # Format ASCII table
    headers = [
        "model",
        "status",
        "latency",
        "tokens in/out",
        "cost",
        "structured output ok",
        "exact error text",
    ]
    key_map = [
        "model",
        "status",
        "latency",
        "tokens",
        "cost",
        "structured_ok",
        "error",
    ]

    col_widths = [len(h) for h in headers]
    for row in results:
        for idx, k in enumerate(key_map):
            col_widths[idx] = max(col_widths[idx], len(str(row[k])))

    header_str = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print("\n" + "=" * len(header_str))
    print(" LLM Model Health Check (configs/models.yaml)")
    print("=" * len(header_str))
    print(header_str)
    print(sep_str)
    for row in results:
        print(" | ".join(str(row[key_map[i]]).ljust(col_widths[i]) for i in range(len(headers))))
    print("=" * len(header_str) + "\n")

    return 1 if any_failed else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Health check for LLM models in configs/models.yaml.")
    parser.add_argument("model", nargs="?", default=None, help="Optional model name to check (e.g. glm-isoquant)")
    parser.add_argument("--model", "-m", dest="model_flag", default=None, help="Filter to specific model name")
    args = parser.parse_args()

    chosen_model = args.model_flag or args.model
    sys.exit(run_model_check(target_model=chosen_model))
