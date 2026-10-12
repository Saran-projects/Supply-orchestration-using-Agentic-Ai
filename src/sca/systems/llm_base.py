"""Base infrastructure, shared Pydantic models, and retry/fallback logic for LLM ordering systems."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from sca.config import get_config
from sca.db import get_duckdb_connection
from sca.llm.client import LLMClient, StructuredOutputError, calculate_cost
from sca.sim.environment import ECHELONS, EchelonHistoryRecord, EchelonState
from sca.systems.base import OrderingSystem

logger = logging.getLogger(__name__)


class OrderDecision(BaseModel):
    """Replenishment decision made by an echelon ordering agent."""

    quantity: float = Field(..., description="Replenishment order quantity to place upstream")
    reasoning: str = Field(default="", description="Short reasoning explaining the replenishment decision")


def get_prompts_dir() -> Path:
    """Return absolute path to prompts template directory."""
    return Path(__file__).resolve().parent.parent / "agents" / "prompts"


def load_prompt_template(filename: str) -> str:
    """Load a versioned prompt template from src/sca/agents/prompts/."""
    path = get_prompts_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def format_history_summary(history_window: List[EchelonHistoryRecord], max_records: int = 14) -> str:
    """Format recent history window records compactly."""
    if not history_window:
        return "No prior history (step 0 initial state)."
    records = history_window[-max_records:]
    lines = []
    for h in records:
        lines.append(
            f"Step {h.step}: Demand={h.demand:.1f}, Ordered={h.order_placed:.1f}, "
            f"Inventory={h.inventory:.1f}, Backlog={h.backlog:.1f}, Incoming={h.incoming_shipment:.1f}"
        )
    return "\n".join(lines)


def format_shared_echelon_info(states: Dict[str, EchelonState]) -> str:
    """Format shared supply chain state across all echelons for multi-agent coordination."""
    lines = []
    for ech in ECHELONS:
        st = states.get(ech)
        if st is not None:
            lines.append(
                f"- {ech.capitalize()}: "
                f"Inventory={st.inventory:.1f}, "
                f"Backlog={st.backlog:.1f}, "
                f"Pipeline={st.pipeline:.1f}, "
                f"Current Demand={st.current_demand:.1f}, "
                f"Last Order={st.last_order:.1f}"
            )
    return "\n".join(lines)


_db_write_lock = threading.Lock()


class LLMOrderingSystemBase(OrderingSystem):
    """Base class for LLM-driven supply chain ordering systems.

    Handles:
    - Structured output generation with Pydantic OrderDecision
    - Single retry on malformed JSON or out-of-range output
    - Graceful fallback to state.last_order upon double failure
    - Comprehensive decision telemetry logging to DuckDB agent_calls & metrics
    """

    def __init__(
        self,
        name: str = "llm_ordering_system",
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
        run_id: Optional[str] = None,
        db_path: Optional[Union[str, Path]] = None,
        prompt_version: str = "v1",
        **kwargs: Any,
    ):
        super().__init__(name=name, config=config)
        self.config = config or {}
        self.run_id = run_id or f"run_{self.name}_{uuid.uuid4().hex[:8]}"
        self.db_path = db_path
        self.prompt_version = prompt_version
        self.seed: Optional[int] = self.config.get("seed", 42)
        self.use_cache: bool = self.config.get("use_cache", False)
        self.model_name: Optional[str] = self.config.get("model", None)

        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = LLMClient(mock_mode=True, db_path=self.db_path)

        if self.config.get("db_path") is False or db_path is False:
            self.db_path = None
            self.log_to_db = False
        elif self.db_path is None:
            try:
                self.db_path = get_config().experiment.db_path
            except Exception:
                self.db_path = Path("results/sca_metrics.duckdb")
            self.log_to_db = True
        else:
            self.log_to_db = True

    def reset(self, scenario: Optional[Any] = None) -> None:
        """Reset internal system state at the start of a simulation run."""
        self.calibrated = True
        if hasattr(self.llm_client, "mock") and hasattr(self.llm_client.mock, "set_seed"):
            seed_to_use = self.seed if self.seed is not None else 42
            self.llm_client.mock.set_seed(seed_to_use)

    def get_max_order(self, echelon: str) -> float:
        """Return maximum allowable order quantity for a given echelon."""
        if echelon == "manufacturer":
            try:
                return float(self.config.get("manufacturer_capacity", get_config().sim.four_echelon.manufacturer_capacity))
            except Exception:
                return 300.0
        return float(self.config.get("max_order", 500.0))

    def _execute_echelon_decision(
        self,
        echelon: str,
        state: EchelonState,
        prompt: str,
        max_order: float,
    ) -> Tuple[float, str]:
        """Execute LLM decision for an echelon with 1-retry and fallback policy.

        Returns:
            Tuple of (order_quantity, reasoning)
        """
        start_time = time.time()
        total_p_tok = 0
        total_c_tok = 0
        retry_count = 0
        fallback = False
        final_order = 0.0
        reasoning = ""
        error_type: Optional[str] = None

        # -------------------------------------------------------------
        # 1. Attempt 1: Call LLM
        # -------------------------------------------------------------
        need_retry = False
        first_attempt_err: Optional[str] = None
        first_resp = None

        try:
            first_resp = self.llm_client.generate(
                prompt=prompt,
                response_model=OrderDecision,
                model=self.model_name,
                seed=self.seed,
                use_cache=self.use_cache,
                run_id=self.run_id,
                log_to_db=False,
            )
            total_p_tok += first_resp.prompt_tokens
            total_c_tok += first_resp.completion_tokens
            if first_resp.retry_count > 0:
                retry_count += first_resp.retry_count

            if first_resp.structured is not None:
                qty = float(first_resp.structured.quantity)
                if 0.0 <= qty <= max_order:
                    final_order = qty
                    reasoning = first_resp.structured.reasoning
                else:
                    need_retry = True
                    first_attempt_err = f"Order quantity {qty} out of range [0.0, {max_order}]"
            else:
                need_retry = True
                first_attempt_err = "No structured OrderDecision returned"
        except StructuredOutputError as exc:
            # LLMClient already attempted validation and internal retry failed
            need_retry = False
            fallback = True
            retry_count = 1
            error_type = "StructuredOutputError"
            first_attempt_err = str(exc)
        except Exception as exc:
            # Other error on attempt 1 (e.g. malformed or adapter failure)
            need_retry = True
            first_attempt_err = str(exc)

        # -------------------------------------------------------------
        # 2. Attempt 2: Retry once if needed
        # -------------------------------------------------------------
        if need_retry and not fallback:
            retry_count += 1
            retry_prompt = (
                f"{prompt}\n\n"
                f"CORRECTION REQUIRED: Previous attempt failed validation:\n"
                f"{first_attempt_err}\n"
                f"You must strictly output a valid JSON OrderDecision with quantity between 0.0 and {max_order}."
            )
            try:
                retry_resp = self.llm_client.generate(
                    prompt=retry_prompt,
                    response_model=OrderDecision,
                    model=self.model_name,
                    seed=self.seed,
                    use_cache=self.use_cache,
                    run_id=self.run_id,
                    log_to_db=False,
                )
                total_p_tok += retry_resp.prompt_tokens
                total_c_tok += retry_resp.completion_tokens

                if retry_resp.structured is not None:
                    qty = float(retry_resp.structured.quantity)
                    if 0.0 <= qty <= max_order:
                        final_order = qty
                        reasoning = retry_resp.structured.reasoning
                        fallback = False
                    else:
                        fallback = True
                        error_type = "OutOfRangeError"
                        reasoning = f"Fallback: retry order {qty} still out of range [0.0, {max_order}]"
                else:
                    fallback = True
                    error_type = "StructuredOutputError"
                    reasoning = "Fallback: retry did not yield structured OrderDecision"
            except Exception as exc:
                fallback = True
                error_type = type(exc).__name__
                reasoning = f"Fallback: retry call raised {exc}"

        # -------------------------------------------------------------
        # 3. Fallback to last order if failed
        # -------------------------------------------------------------
        if fallback:
            last_ord = float(state.last_order) if hasattr(state, "last_order") else 0.0
            final_order = max(0.0, min(max_order, last_ord))
            if not reasoning:
                reasoning = f"Fallback to last order {final_order:.1f} after retry failure: {first_attempt_err}"

        latency_ms = (time.time() - start_time) * 1000.0
        is_cache_hit = getattr(first_resp, "cached", False) if first_resp else False
        cost_usd = 0.0 if is_cache_hit else calculate_cost(
            self.model_name or "mock-model",
            total_p_tok,
            total_c_tok,
            getattr(self.llm_client, "config", None),
        )

        # -------------------------------------------------------------
        # 4. Log Decision to DuckDB agent_calls & metrics
        # -------------------------------------------------------------
        self._log_decision_to_duckdb(
            echelon=echelon,
            step_number=state.step,
            final_order=final_order,
            reasoning=reasoning,
            prompt_tokens=total_p_tok,
            completion_tokens=total_c_tok,
            latency_ms=latency_ms,
            retry_count=retry_count,
            fallback=fallback,
            error_type=error_type,
            cost_usd=cost_usd,
            cache_hit=is_cache_hit,
        )

        return final_order, reasoning

    def _log_decision_to_duckdb(
        self,
        echelon: str,
        step_number: int,
        final_order: float,
        reasoning: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        retry_count: int,
        fallback: bool,
        error_type: Optional[str] = None,
        cost_usd: float = 0.0,
        cache_hit: bool = False,
    ) -> None:
        """Write decision row to DuckDB agent_calls and record fallback metric if triggered."""
        if not getattr(self, "log_to_db", True) or not self.db_path:
            return

        call_id = f"dec_{uuid.uuid4().hex[:12]}"
        step_id = f"step_{step_number}"
        agent_name = f"{self.name}_{echelon}"
        model_name = self.model_name or "mock-model"
        provider = getattr(self.llm_client, "default_provider", "mock")

        try:
            with _db_write_lock:
                conn = get_duckdb_connection(self.db_path)
                try:
                    # Ensure parent run exists
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO runs (run_id, experiment_name, status, metadata)
                        VALUES (?, ?, 'active', ?)
                        """,
                        [
                            self.run_id,
                            f"exp_{self.name}",
                            json.dumps({"system": self.name, "seed": self.seed}),
                        ],
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
                            self.run_id,
                            step_id,
                            agent_name,
                            model_name,
                            provider,
                            prompt_tokens,
                            completion_tokens,
                            0,
                            round(cost_usd, 6),
                            round(cost_usd, 6),
                            round(latency_ms, 2),
                            retry_count,
                            not fallback,
                            error_type,
                            cache_hit,
                            False,
                            self.name,
                            echelon,
                            step_number,
                            self.prompt_version,
                            fallback,
                            final_order,
                            reasoning,
                        ],
                    )

                    # Count every fallback in DuckDB metrics table
                    if fallback:
                        metric_id = f"met_{uuid.uuid4().hex[:12]}"
                        conn.execute(
                            """
                            INSERT INTO metrics (
                                metric_id, run_id, step_number, metric_name, metric_value, category, metadata
                            ) VALUES (?, ?, ?, 'fallback_count', 1.0, 'reliability', ?)
                            """,
                            [
                                metric_id,
                                self.run_id,
                                step_number,
                                json.dumps({
                                    "system": self.name,
                                    "echelon": echelon,
                                    "order": final_order,
                                    "reasoning": reasoning,
                                }),
                            ],
                        )
                finally:
                    conn.close()
        except Exception as e:
            logger.debug("Failed to log decision to DuckDB: %s", e)
