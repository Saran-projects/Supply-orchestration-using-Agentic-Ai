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
from sca.tools import safe_order_range

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
        prompt_version: str = "v3",
        tool_mode: str = "deterministic",
        **kwargs: Any,
    ):
        super().__init__(name=name, config=config)
        self.config = config or {}
        self.run_id = run_id or f"run_{self.name}_{uuid.uuid4().hex[:8]}"
        self.db_path = db_path
        self.prompt_version = self.config.get("prompt_version", prompt_version)
        self.tool_mode = self.config.get("tool_mode", tool_mode)
        if self.tool_mode not in ("deterministic", "llm_function_calling"):
            raise ValueError(
                f"Invalid tool_mode '{self.tool_mode}'. Must be 'deterministic' or 'llm_function_calling'."
            )
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

    def compute_deterministic_tools(self, state: EchelonState, echelon: str) -> Dict[str, Any]:
        """Execute deterministic analytical tools in code to provide baselines for LLM decision.

        Implements 'deterministic tools plus LLM decision and negotiation':
        1. forecast_demand: point forecast on observed demand history
        2. eoq_quantity: Wilson EOQ batch sizing
        3. smoothed_order_up_to: heuristic order-up-to target with safety buffer
        4. safe_range: feasible order bounds based on capacity and rate limits
        5. recommendation: primary tool recommendation (smoothed target bounded in safe range)
        """
        import numpy as np

        from sca.tools import (
            EoqQuantityInput,
            ForecastDemandInput,
            SmoothedOrderUpToInput,
            eoq_quantity,
            forecast_demand,
            smoothed_order_up_to,
        )

        # 1. Last 5 orders
        if state.history_window:
            last_5 = [round(float(h.order_placed), 1) for h in state.history_window[-5:]]
        else:
            last_5 = [round(float(state.last_order), 1)]

        # Demand series
        if state.history_window:
            hist_d = [float(h.demand) for h in state.history_window]
        else:
            hist_d = [float(state.current_demand)]

        # 2. Forecast demand
        try:
            fc_out = forecast_demand(
                ForecastDemandInput(recent_history=hist_d, horizon=1, model_type="linear"),
                db_path=False,
            )
            pt_forecast = max(0.0, float(fc_out.point_forecast))
        except Exception:
            pt_forecast = float(state.current_demand)

        # 3. EOQ
        mean_d = max(1.0, float(np.mean(hist_d)))
        h_cost = float(self.config.get("holding_cost_per_unit", 0.50))
        k_cost = float(self.config.get("order_fixed_cost", 100.0))
        try:
            eoq_out = eoq_quantity(
                EoqQuantityInput(
                    mean_demand=mean_d,
                    order_fixed_cost=k_cost,
                    holding_cost_per_unit=h_cost,
                ),
                db_path=False,
            )
            eoq_val = float(eoq_out.eoq)
        except Exception:
            eoq_val = mean_d

        # 4. Smoothed order-up-to target
        alpha = float(self.config.get("alpha", 0.6))
        beta = float(self.config.get("beta", alpha))
        z = float(self.config.get("safety_factor_z", 1.65))
        std_d = float(np.std(hist_d)) if len(hist_d) > 1 else 20.0
        try:
            sm_out = smoothed_order_up_to(
                SmoothedOrderUpToInput(
                    current_demand=float(state.current_demand),
                    inventory=float(state.inventory),
                    pipeline=float(state.pipeline),
                    backlog=float(state.backlog),
                    last_order=float(state.last_order),
                    lead_time_mean=float(state.lead_time),
                    lead_time_std=0.8,
                    demand_std=std_d,
                    safety_stock_factor=z,
                    z=z,
                    alpha=alpha,
                    beta=beta,
                ),
                db_path=False,
            )
            smoothed_target = max(0.0, float(sm_out.order_quantity))
        except Exception:
            smoothed_target = float(state.current_demand)

        # 5. Safe range & recommendation
        cap = self.get_max_order(echelon)
        delta = float(self.config.get("rate_limit_delta", 0.5))
        min_safe, max_safe = safe_order_range(float(state.last_order), float(pt_forecast), delta)
        max_safe = min(cap, max_safe)

        rec_target = min(cap, max(min_safe, smoothed_target))

        return {
            "last_5_orders": last_5,
            "forecast": round(pt_forecast, 1),
            "eoq": round(eoq_val, 1),
            "smoothed_target": round(smoothed_target, 1),
            "recommendation": round(rec_target, 1),
            "safe_range": (round(min_safe, 1), round(max_safe, 1)),
        }

    def _execute_echelon_decision(
        self,
        echelon: str,
        state: EchelonState,
        prompt: str,
        max_order: float,
        recommendation: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Execute LLM decision for an echelon with 1-retry and fallback policy.

        Returns:
            Tuple of (order_quantity, reasoning)
        """
        start_time = time.time()
        total_p_tok = 0
        total_c_tok = 0
        total_cached_tok = 0
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
            total_cached_tok += getattr(first_resp, "cached_tokens", 0)
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
                f"You must strictly output a valid JSON OrderDecision with quantity between 0.0 and {max_order} "
                f"and concise reasoning under 40 words. Respond with JSON only."
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
                total_cached_tok += getattr(retry_resp, "cached_tokens", 0)

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
            cached_tokens=total_cached_tok,
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
            cached_tokens=total_cached_tok,
            latency_ms=latency_ms,
            retry_count=retry_count,
            fallback=fallback,
            error_type=error_type,
            cost_usd=cost_usd,
            cache_hit=is_cache_hit,
            tool_recommendation=recommendation,
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
        tool_recommendation: Optional[float] = None,
        cached_tokens: int = 0,
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
                            json.dumps({"system": self.name, "seed": self.seed, "tool_mode": self.tool_mode}),
                        ],
                    )

                    conn.execute(
                        """
                        INSERT INTO agent_calls (
                            call_id, run_id, step_id, agent_name, model_name, provider,
                            prompt_tokens, completion_tokens, cached_tokens, total_cost, cost_usd,
                            latency_ms, retry_count, structured_output_ok, error_type, cache_hit, usage_estimated,
                            system_name, echelon, step_number, prompt_version, fallback, order_quantity, reasoning, tool_recommendation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            cached_tokens,
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
                            tool_recommendation,
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

                    # Record recommendation metrics if recommendation present
                    if tool_recommendation is not None:
                        tol = max(1.0, 0.05 * tool_recommendation)
                        diff = abs(final_order - tool_recommendation)
                        adopted = 1.0 if diff <= tol else 0.0
                        overridden = 1.0 - adopted

                        conn.execute(
                            """
                            INSERT INTO metrics (
                                metric_id, run_id, step_number, metric_name, metric_value, category, metadata
                            ) VALUES (?, ?, ?, 'recommendation_adopted', ?, 'behaviour', ?)
                            """,
                            [
                                f"met_{uuid.uuid4().hex[:12]}",
                                self.run_id,
                                step_number,
                                adopted,
                                json.dumps({
                                    "system": self.name,
                                    "echelon": echelon,
                                    "tool_recommendation": tool_recommendation,
                                    "order": final_order,
                                    "deviation": round(diff, 2),
                                }),
                            ],
                        )
                        conn.execute(
                            """
                            INSERT INTO metrics (
                                metric_id, run_id, step_number, metric_name, metric_value, category, metadata
                            ) VALUES (?, ?, ?, 'recommendation_deviation', ?, 'behaviour', ?)
                            """,
                            [
                                f"met_{uuid.uuid4().hex[:12]}",
                                self.run_id,
                                step_number,
                                round(diff, 2),
                                json.dumps({
                                    "system": self.name,
                                    "echelon": echelon,
                                    "tool_recommendation": tool_recommendation,
                                    "order": final_order,
                                }),
                            ],
                        )
                        conn.execute(
                            """
                            INSERT INTO metrics (
                                metric_id, run_id, step_number, metric_name, metric_value, category, metadata
                            ) VALUES (?, ?, ?, 'recommendation_override', ?, 'behaviour', ?)
                            """,
                            [
                                f"met_{uuid.uuid4().hex[:12]}",
                                self.run_id,
                                step_number,
                                overridden,
                                json.dumps({
                                    "system": self.name,
                                    "echelon": echelon,
                                    "tool_recommendation": tool_recommendation,
                                    "order": final_order,
                                }),
                            ],
                        )
                finally:
                    conn.close()
        except Exception as e:
            logger.debug("Failed to log decision to DuckDB: %s", e)
