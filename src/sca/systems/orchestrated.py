"""Orchestrated Multi-Agent Ordering System (System 6).

Implements an OrchestratorAgent architecture that:
1. Summarizes local and supply chain echelon state.
2. Dispatches to specialized analytical agents (Demand, Inventory, Procurement from System 5).
3. Collects structured proposals.
4. Detects conflicts via deterministic rules:
   - Requested quantity exceeds supplier capacity.
   - Proposal deviates by more than X% from forecast.
   - Proposals disagree by more than Y%.
5. If NO conflict, skips negotiation LLM calls and applies consensus order (cost-saving shortcut).
6. If conflict exists, executes bounded negotiation with max_rounds = 2 from config,
   logging all exchanged messages.
7. Validates final order with validate_order, applying and counting deterministic fallback on failure.
8. Logs rich per-decision telemetry to DuckDB (orchestrated_decisions, messages, metrics).
9. Supports modular ablation flags in config:
   - enable_negotiation (bool, default True)
   - enable_validation (bool, default True)
   - enable_conflict_skip (bool, default True)
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator

from sca.config import get_config
from sca.db import DuckDBManager
from sca.llm.client import LLMClient
from sca.sim.environment import ECHELONS, EchelonState
from sca.systems.base import OrderingSystem, register_system
from sca.systems.llm_base import format_history_summary, load_prompt_template
from sca.systems.multi_agent_tools import (
    AgentProposal,
    DemandAgent,
    InventoryAgent,
    ProcurementAgent,
)
from sca.tools import (
    SmoothedOrderUpToInput,
    ValidateOrderInput,
    safe_order_range,
    smoothed_order_up_to,
    validate_order,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Pydantic Models for Negotiation & Decision Telemetry
# ---------------------------------------------------------------------------
class NegotiationResponse(BaseModel):
    """Structured response from an agent or mediator during negotiation."""

    agent_name: str = Field(default="mediator", description="Agent or mediator identifier")
    regime: str = Field(default="stable", description="Detected demand regime: 'stable', 'trend', or 'shock'")
    revised_quantity: Optional[float] = Field(default=None, description="Revised proposed order quantity")
    order: Optional[float] = Field(default=None, description="Order quantity alias")
    rationale: str = Field(default="", description="Domain justification for revised concession")
    reasoning: str = Field(default="", description="Reasoning alias")
    concession_made: bool = Field(default=False, description="Whether agent made a concession")

    @model_validator(mode="after")
    def unify_fields(self) -> NegotiationResponse:
        if self.revised_quantity is None and self.order is not None:
            self.revised_quantity = float(self.order)
        elif self.order is None and self.revised_quantity is not None:
            self.order = float(self.revised_quantity)
        elif self.revised_quantity is None and self.order is None:
            raise ValueError("Either 'revised_quantity' or 'order' must be provided.")
        if not self.rationale and self.reasoning:
            self.rationale = self.reasoning
        elif not self.reasoning and self.rationale:
            self.reasoning = self.rationale
        return self


class ConflictDetectionResult(BaseModel):
    """Result of deterministic conflict detection rules."""

    has_conflict: bool = Field(default=False)
    conflict_type: str = Field(default="none", description="Comma-separated trigger labels")
    details: Dict[str, Any] = Field(default_factory=dict)


class OrchestratedDecisionRecord(BaseModel):
    """Telemetry record captured per ordering decision."""

    decision_id: str
    run_id: str
    step_number: int
    echelon: str
    has_conflict: bool
    conflict_type: str = "none"
    rounds_used: int = 0
    messages_exchanged: int = 0
    resolution_method: str
    final_order: float
    total_tokens: int = 0
    latency_ms: float = 0.0
    fallback_used: bool = False
    proposals: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. OrchestratorAgent
# ---------------------------------------------------------------------------
class OrchestratorAgent:
    """Central orchestrator mediating specialized agents for a supply chain echelon."""

    def __init__(
        self,
        echelon: str,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.echelon = echelon
        self.config = config or {}
        self.llm_client = llm_client

        # Thresholds from config
        self.forecast_deviation_threshold: float = float(self.config.get("forecast_deviation_threshold", 0.25))
        self.proposals_disagree_threshold: float = float(self.config.get("proposals_disagree_threshold", 0.30))
        self.max_rounds: int = min(2, int(self.config.get("max_rounds", 2)))
        self.enable_smoothed_target_conflict: bool = bool(
            self.config.get("enable_smoothed_target_conflict", False)
        )
        self.smoothed_target_threshold: float = float(
            self.config.get("smoothed_target_threshold", 0.25)
        )

        # Specialized agents (reused from System 5)
        self.demand_agent = DemandAgent(echelon=echelon, config=self.config)
        self.inventory_agent = InventoryAgent(echelon=echelon, config=self.config)
        self.procurement_agent = ProcurementAgent(
            echelon=echelon,
            capacity_limit=self.get_capacity(),
            config=self.config,
        )

        p_ver = self.config.get("prompt_version", "v3")
        p_norm = p_ver.replace(".", "_")
        tpl_cand = f"orchestrated_{p_norm}.txt"
        prompts_dir = Path(__file__).resolve().parent.parent / "agents" / "prompts"
        if (prompts_dir / tpl_cand).exists():
            tpl_name = tpl_cand
        elif p_ver.startswith("v3") or p_ver.startswith("orchestrated_v3"):
            tpl_name = "orchestrated_v3.txt"
        elif p_ver in ("v2", "orchestrated_v2"):
            tpl_name = "orchestrated_v2.txt"
        else:
            tpl_name = "orchestrated_v1.txt"
        try:
            self.prompt_template = load_prompt_template(tpl_name)
        except Exception:
            self.prompt_template = (
                "Orchestrator mediating proposals: {proposals_summary}. "
                "Conflict: {conflict_details}. Round: {round_num} of {max_rounds}."
            )

    def get_capacity(self) -> float:
        """Resolve capacity limit for this echelon."""
        if self.echelon == "manufacturer":
            try:
                return float(self.config.get("manufacturer_capacity", get_config().sim.four_echelon.manufacturer_capacity))
            except Exception:
                return 300.0
        return float(self.config.get("max_order", 500.0))

    def summarize_state(self, state: EchelonState, capacity_limit: float) -> str:
        """Construct structured summary of local echelon state."""
        hist_summary = format_history_summary(state.history_window, max_records=7)
        return (
            f"Echelon: {self.echelon.capitalize()}, Step: {state.step}\n"
            f"Capacity Limit: {capacity_limit:.1f}\n"
            f"Current Inventory: {state.inventory:.1f}, Backlog: {state.backlog:.1f}, Pipeline: {state.pipeline:.1f}\n"
            f"Current Demand: {state.current_demand:.1f}, Lead Time: {state.lead_time}\n"
            f"Recent History:\n{hist_summary}"
        )

    def dispatch_proposals(
        self,
        state: EchelonState,
        run_id: str,
        db_path: Path,
        step_id: str,
    ) -> Dict[str, AgentProposal]:
        """Dispatch analysis to Demand, Inventory, and Procurement agents."""
        p_demand = self.demand_agent.propose(
            state=state, run_id=run_id, db_path=db_path, step_id=step_id
        )
        p_inventory = self.inventory_agent.propose(
            state=state, run_id=run_id, db_path=db_path, step_id=step_id
        )
        p_procurement = self.procurement_agent.propose(
            state=state, run_id=run_id, db_path=db_path, step_id=step_id
        )

        return {
            "demand": p_demand,
            "inventory": p_inventory,
            "procurement": p_procurement,
        }

    def detect_conflicts(
        self,
        proposals: Dict[str, AgentProposal],
        capacity_limit: float,
        state: Optional[EchelonState] = None,
    ) -> ConflictDetectionResult:
        """Deterministic rule-based conflict detection.

        Rules:
        1. Capacity Exceeded: Any proposed order exceeds supplier capacity ceiling.
        2. Forecast Deviation: A proposal deviates > X% from forecast demand.
        3. Proposals Disagree: Proposals differ by > Y% across agents.
        4. Deviation from Smoothed Target: A proposal deviates > Z% from smoothed target (behind config flag).
        """
        triggers: List[str] = []
        details: Dict[str, Any] = {}

        quantities = {name: p.quantity for name, p in proposals.items()}
        q_vals = list(quantities.values())
        q_forecast = quantities.get("demand", 0.0)

        # Rule 1: Capacity Exceeded
        exceeded = [name for name, q in quantities.items() if q > capacity_limit]
        if exceeded:
            triggers.append("capacity_exceeded")
            details["capacity_exceeded"] = {
                "exceeded_agents": exceeded,
                "capacity_limit": capacity_limit,
                "values": {name: quantities[name] for name in exceeded},
            }

        # Rule 2: Forecast Deviation (> X% deviation from demand forecast)
        if q_forecast > 1.0:
            dev_agents = {}
            for name, q in quantities.items():
                if name == "demand":
                    continue
                dev = abs(q - q_forecast) / q_forecast
                if dev > self.forecast_deviation_threshold:
                    dev_agents[name] = round(dev, 3)
            if dev_agents:
                triggers.append("forecast_deviation")
                details["forecast_deviation"] = {
                    "threshold": self.forecast_deviation_threshold,
                    "deviations": dev_agents,
                }

        # Rule 3: Proposals Disagree (> Y% spread across all proposals)
        max_q = max(q_vals) if q_vals else 0.0
        min_q = min(q_vals) if q_vals else 0.0
        denom = max(max_q, 1.0)
        spread = (max_q - min_q) / denom
        if spread > self.proposals_disagree_threshold:
            triggers.append("proposals_disagree")
            details["proposals_disagree"] = {
                "spread": round(spread, 3),
                "threshold": self.proposals_disagree_threshold,
                "range": [round(min_q, 1), round(max_q, 1)],
            }

        # Rule 4: Deviation from Smoothed Target (behind config flag)
        if self.enable_smoothed_target_conflict and state is not None:
            try:
                hist = (
                    [float(h.demand) for h in state.history_window]
                    if state.history_window
                    else [float(state.current_demand)]
                )
                mean_d = float(np.mean(hist))
                std_d = float(np.std(hist)) if len(hist) > 1 else 20.0
                alpha = float(self.config.get("alpha", 0.2))

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
                        safety_stock_factor=1.65,
                        previous_smoothed_demand=mean_d,
                        alpha=alpha,
                    )
                )
                smoothed_target = float(sm_out.order_quantity)
                denom_target = max(smoothed_target, 1.0)
                sm_dev_agents = {}
                for name, q in quantities.items():
                    dev_sm = abs(q - smoothed_target) / denom_target
                    if dev_sm > self.smoothed_target_threshold:
                        sm_dev_agents[name] = round(dev_sm, 3)

                if sm_dev_agents:
                    triggers.append("deviation_from_smoothed_target")
                    details["deviation_from_smoothed_target"] = {
                        "smoothed_target": round(smoothed_target, 2),
                        "threshold": self.smoothed_target_threshold,
                        "deviations": sm_dev_agents,
                    }
            except Exception as exc:
                logger.warning("Error evaluating smoothed target conflict for %s: %s", self.echelon, exc)

        has_conflict = len(triggers) > 0
        conflict_type = ",".join(triggers) if triggers else "none"

        return ConflictDetectionResult(
            has_conflict=has_conflict,
            conflict_type=conflict_type,
            details=details,
        )

    def run_negotiation(
        self,
        state: EchelonState,
        proposals: Dict[str, AgentProposal],
        conflict: ConflictDetectionResult,
        capacity_limit: float,
        run_id: str,
        db_path: Path,
        step_id: str,
    ) -> Tuple[float, int, int, str, int]:
        """Execute bounded multi-agent negotiation (strictly max_rounds <= 2).

        Returns:
            (resolved_order, rounds_used, messages_exchanged, resolution_method, tokens_used)
        """
        rounds_used = 0
        messages_exchanged = 0
        tokens_used = 0
        curr_proposals = {k: v.quantity for k, v in proposals.items()}

        proposals_summary = "\n".join(
            f"- {name.capitalize()}: {p.quantity:.1f} (Rationale: {p.rationale})"
            for name, p in proposals.items()
        )
        conflict_desc = f"Type: {conflict.conflict_type}, Details: {conflict.details}"

        db = None
        if db_path:
            try:
                db = DuckDBManager(db_path)
                db.conn.execute(
                    "INSERT OR IGNORE INTO runs (run_id, experiment_name, status) VALUES (?, 'system6', 'active')",
                    [run_id],
                )
            except Exception as e:
                logger.warning("Could not open DuckDB for negotiation logging: %s", e)
                db = None

        try:
            for round_idx in range(1, self.max_rounds + 1):
                rounds_used = round_idx

                # Log Orchestrator inquiry
                orch_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
                orch_content = (
                    f"Round {round_idx}/{self.max_rounds}: Resolving conflict [{conflict.conflict_type}]. "
                    f"Current proposals: {curr_proposals}. Capacity ceiling: {capacity_limit:.1f}."
                )
                if db:
                    db.log_message(
                        message_id=orch_msg_id,
                        run_id=run_id,
                        sender=f"orchestrator_{self.echelon}",
                        receiver=f"agents_{self.echelon}",
                        role="system",
                        content=orch_content,
                        step_id=step_id,
                        metadata={"round": round_idx, "echelon": self.echelon},
                    )
                messages_exchanged += 1

                last_5 = (
                    [round(float(h.order_placed), 1) for h in state.history_window[-5:]]
                    if state.history_window
                    else [round(float(state.last_order), 1)]
                )
                fc_val = proposals["demand"].quantity if "demand" in proposals else float(state.current_demand)
                delta = float(self.config.get("rate_limit_delta", 0.5))
                min_safe, max_safe = safe_order_range(float(state.last_order), float(fc_val), delta)
                max_safe = min(capacity_limit, max_safe)
                safe_range_str = f"[{min_safe:.1f}, {max_safe:.1f}]"

                # Format prompt for negotiation concession
                prompt = self.prompt_template.format(
                    echelon=self.echelon.capitalize(),
                    step=state.step,
                    capacity_limit=f"{capacity_limit:.1f}",
                    inventory=f"{state.inventory:.1f}",
                    backlog=f"{state.backlog:.1f}",
                    pipeline=f"{state.pipeline:.1f}",
                    last_5_orders=str(last_5),
                    forecast=f"{fc_val:.1f}",
                    safe_range=safe_range_str,
                    proposals_summary=proposals_summary,
                    conflict_details=conflict_desc,
                    round_num=round_idx,
                    max_rounds=self.max_rounds,
                )

                # LLM concession call with 1-retry fallback
                revised_qty = min(capacity_limit, float(np.median(list(curr_proposals.values()))))
                concession_rationale = f"Adjusted order toward capacity-bounded median ({revised_qty:.1f})"

                if self.llm_client is not None:
                    first_err = None
                    p_ver_str = str(self.config.get("prompt_version", "v3"))
                    try:
                        resp = self.llm_client.generate(
                            prompt=prompt,
                            response_model=NegotiationResponse,
                            model=self.config.get("model", None),
                            use_cache=self.config.get("use_cache", True),
                            run_id=run_id,
                            system_name="orchestrated",
                            echelon=self.echelon,
                            step_number=state.step,
                            prompt_version=p_ver_str,
                        )
                        tokens_used += (resp.prompt_tokens + resp.completion_tokens)
                        if isinstance(resp.structured, NegotiationResponse):
                            qty = resp.structured.revised_quantity
                            if 0.0 <= qty <= capacity_limit:
                                revised_qty = qty
                                concession_rationale = resp.structured.rationale
                            else:
                                first_err = f"Quantity {qty} out of range [0.0, {capacity_limit}]"
                        else:
                            first_err = "No structured NegotiationResponse"
                    except Exception as e:
                        first_err = str(e)

                    # Retry once if first attempt failed
                    if first_err is not None:
                        retry_prompt = (
                            f"{prompt}\n\n"
                            f"CORRECTION REQUIRED: Previous attempt failed validation:\n{first_err}\n"
                            f"Output strictly a valid JSON NegotiationResponse with revised_quantity "
                            f"between 0.0 and {capacity_limit} and rationale under 40 words. Respond with JSON only."
                        )
                        try:
                            retry_resp = self.llm_client.generate(
                                prompt=retry_prompt,
                                response_model=NegotiationResponse,
                                model=self.config.get("model", None),
                                use_cache=self.config.get("use_cache", True),
                                run_id=run_id,
                                system_name="orchestrated",
                                echelon=self.echelon,
                                step_number=state.step,
                                prompt_version=p_ver_str,
                            )
                            tokens_used += (retry_resp.prompt_tokens + retry_resp.completion_tokens)
                            if isinstance(retry_resp.structured, NegotiationResponse):
                                qty = retry_resp.structured.revised_quantity
                                if 0.0 <= qty <= capacity_limit:
                                    revised_qty = qty
                                    concession_rationale = retry_resp.structured.rationale
                                else:
                                    revised_qty = min(capacity_limit, max(0.0, float(state.last_order)))
                                    concession_rationale = f"Fallback to last order {revised_qty:.1f}"
                            else:
                                revised_qty = min(capacity_limit, max(0.0, float(state.last_order)))
                                concession_rationale = f"Fallback to last order {revised_qty:.1f}"
                        except Exception:
                            revised_qty = min(capacity_limit, max(0.0, float(state.last_order)))
                            concession_rationale = f"Fallback to last order {revised_qty:.1f}"

                # Log Agent concession message
                agent_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
                if db:
                    db.log_message(
                        message_id=agent_msg_id,
                        run_id=run_id,
                        sender=f"mediator_{self.echelon}",
                        receiver=f"orchestrator_{self.echelon}",
                        role="assistant",
                        content=f"Concession proposed: {revised_qty:.1f}. Reason: {concession_rationale}",
                        step_id=step_id,
                        metadata={"round": round_idx, "revised_qty": revised_qty},
                    )
                messages_exchanged += 1

                # Update current proposal set
                curr_proposals["compromise"] = revised_qty

                # Convergence check: if revised order respects capacity and narrows disagreement
                q_vals = list(curr_proposals.values())
                spread = (max(q_vals) - min(q_vals)) / max(max(q_vals), 1.0)
                if round_idx == 1 and spread <= self.proposals_disagree_threshold and revised_qty <= capacity_limit:
                    return revised_qty, rounds_used, messages_exchanged, f"negotiation_resolved_round_{round_idx}", tokens_used

            # End of bounded negotiation: synthesize final compromise
            final_compromise = min(capacity_limit, float(np.median(list(curr_proposals.values()))))
            return final_compromise, rounds_used, messages_exchanged, "negotiation_compromise", tokens_used
        finally:
            if db:
                try:
                    db.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 3. System 6: OrchestratedSystem
# ---------------------------------------------------------------------------
@register_system(["orchestrated", "system_6"])
class OrchestratedSystem(OrderingSystem):
    """System 6: Orchestrated multi-agent replenishment architecture.

    Features:
    - State summarization.
    - Delegation to Demand, Inventory, and Procurement agents.
    - Deterministic conflict detection (capacity limit, forecast deviation, proposal spread).
    - No-conflict shortcut: bypasses all negotiation LLM calls to save cost & latency.
    - Bounded negotiation (capped strictly at max_rounds=2).
    - Validation with validate_order and deterministic fallback counting.
    - Modular ablation flags: enable_negotiation, enable_validation, enable_conflict_skip.
    - Transactional DuckDB telemetry logged per decision.
    """

    def __init__(
        self,
        name: str = "orchestrated",
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
        run_id: Optional[str] = None,
        db_path: Optional[Union[str, Path]] = None,
        tool_mode: str = "deterministic",
        **kwargs: Any,
    ):
        super().__init__(name=name, config=config)
        self.config = config or {}
        self.run_id = run_id or f"run_system6_{uuid.uuid4().hex[:8]}"
        self.prompt_version = self.config.get("prompt_version", "v3")
        self.tool_mode = self.config.get("tool_mode", tool_mode)
        if self.tool_mode not in ("deterministic", "llm_function_calling"):
            raise ValueError(f"Invalid tool_mode '{self.tool_mode}'. Must be 'deterministic' or 'llm_function_calling'.")
        if self.config.get("db_path") is False or db_path is False:
            self.db_path = False
        elif db_path is not None:
            self.db_path = Path(db_path)
        else:
            try:
                self.db_path = get_config().experiment.db_path
            except Exception:
                self.db_path = Path("results/sca_metrics.duckdb")

        self.llm_client = llm_client or LLMClient(mock_mode=True, db_path=self.db_path)

        # Ablation Flags (Config-driven)
        self.enable_negotiation: bool = self.config.get("enable_negotiation", True)
        self.enable_validation: bool = self.config.get("enable_validation", True)
        self.enable_conflict_skip: bool = self.config.get("enable_conflict_skip", True)

        # Scripted Failure Flags for Testing
        self.force_validation_failure: bool = self.config.get("force_validation_failure", False)

        # Initialize OrchestratorAgent per echelon
        self.orchestrators: Dict[str, OrchestratorAgent] = {
            ech: OrchestratorAgent(echelon=ech, config=self.config, llm_client=self.llm_client)
            for ech in ECHELONS
        }

        self.decision_history: List[OrchestratedDecisionRecord] = []
        self.total_fallbacks: int = 0

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Orchestrate ordering decisions across all echelons."""
        orders: Dict[str, float] = {}

        for echelon in ECHELONS:
            if echelon not in states:
                continue

            start_t = time.time()
            state = states[echelon]
            step_id = f"step_{state.step}"
            orch = self.orchestrators[echelon]
            capacity_limit = orch.get_capacity()

            # 1. State Summarization
            _ = orch.summarize_state(state, capacity_limit)

            # 2. Dispatch to Specialized Agents (System 5 agents)
            proposals = orch.dispatch_proposals(
                state=state,
                run_id=self.run_id,
                db_path=self.db_path,
                step_id=step_id,
            )
            prop_qtys = {k: v.quantity for k, v in proposals.items()}

            # 3. Detect Conflicts via Deterministic Rules
            conflict = orch.detect_conflicts(proposals, capacity_limit, state=state)

            # 4. Resolve Conflict vs Apply Shortcut
            rounds_used = 0
            messages_exchanged = 0
            tokens_used = 0

            # Determine whether to skip negotiation
            skip_negotiation = (not conflict.has_conflict) if self.enable_conflict_skip else False

            if skip_negotiation:
                # No-conflict shortcut: Zero LLM negotiation calls
                raw_order = float(np.median(list(prop_qtys.values())))
                resolution_method = "no_conflict_consensus"
            elif not self.enable_negotiation:
                # Ablation: Negotiation turned off
                raw_order = min(capacity_limit, float(min(list(prop_qtys.values()))))
                resolution_method = "ablation_no_negotiation"
            else:
                # Conflict exists (or conflict-skip disabled): Run bounded negotiation
                raw_order, rounds_used, messages_exchanged, resolution_method, tokens_used = orch.run_negotiation(
                    state=state,
                    proposals=proposals,
                    conflict=conflict,
                    capacity_limit=capacity_limit,
                    run_id=self.run_id,
                    db_path=self.db_path,
                    step_id=step_id,
                )

            # 5. Validation with validate_order & Fallback
            fallback_used = False
            final_order = raw_order

            if self.enable_validation:
                ip = state.inventory + state.pipeline - state.backlog

                # Optional testing hook to trigger validation failure
                proposed_val = 999999.0 if self.force_validation_failure else raw_order

                try:
                    val_out = validate_order(
                        ValidateOrderInput(
                            proposed_order=proposed_val,
                            inventory_position=ip,
                            min_order=0.0,
                            max_order=capacity_limit,
                            capacity_limit=capacity_limit if echelon == "manufacturer" else None,
                        ),
                        db_path=self.db_path,
                        run_id=self.run_id,
                        step_id=step_id,
                        agent_name=f"orchestrator_{echelon}",
                        expected_tool="validate_order",
                        is_correct_tool=True,
                        is_parameter_valid=True,
                    )

                    if val_out.is_valid and not self.force_validation_failure:
                        final_order = float(val_out.validated_order)
                    else:
                        # Validation rejected: Deterministic fallback to state.last_order
                        final_order = min(capacity_limit, max(0.0, float(state.last_order)))
                        fallback_used = True
                        self.total_fallbacks += 1
                except Exception as e:
                    logger.warning("[%s] Order validation error: %s. Applying fallback.", echelon, e)
                    final_order = min(capacity_limit, max(0.0, float(state.last_order)))
                    fallback_used = True
                    self.total_fallbacks += 1
            else:
                # Ablation: Validation turned off (order applied unvalidated)
                final_order = raw_order

            orders[echelon] = final_order
            elapsed_ms = (time.time() - start_t) * 1000.0

            # 6. Telemetry Logging to DuckDB
            decision_id = f"dec_{uuid.uuid4().hex[:8]}"
            record = OrchestratedDecisionRecord(
                decision_id=decision_id,
                run_id=self.run_id,
                step_number=state.step,
                echelon=echelon,
                has_conflict=conflict.has_conflict,
                conflict_type=conflict.conflict_type,
                rounds_used=rounds_used,
                messages_exchanged=messages_exchanged,
                resolution_method=resolution_method,
                final_order=final_order,
                total_tokens=tokens_used,
                latency_ms=elapsed_ms,
                fallback_used=fallback_used,
                proposals=prop_qtys,
            )
            self.decision_history.append(record)

            self._log_decision_to_duckdb(record)

        return orders

    def _log_decision_to_duckdb(self, record: OrchestratedDecisionRecord) -> None:
        """Write decision metrics and summary to DuckDB tables."""
        try:
            with DuckDBManager(self.db_path) as db:
                db.log_orchestrated_decision(
                    decision_id=record.decision_id,
                    run_id=record.run_id,
                    step_number=record.step_number,
                    echelon=record.echelon,
                    has_conflict=record.has_conflict,
                    conflict_type=record.conflict_type,
                    rounds_used=record.rounds_used,
                    messages_exchanged=record.messages_exchanged,
                    resolution_method=record.resolution_method,
                    final_order=record.final_order,
                    total_tokens=record.total_tokens,
                    latency_ms=record.latency_ms,
                    fallback_used=record.fallback_used,
                    proposals=record.proposals,
                )

                # Also record granular KPIs in metrics table
                step_num = record.step_number
                db.log_metric(
                    metric_id=f"met_{uuid.uuid4().hex[:8]}",
                    run_id=record.run_id,
                    step_number=step_num,
                    metric_name="conflict_detected",
                    metric_value=1.0 if record.has_conflict else 0.0,
                    category="orchestration",
                )
                db.log_metric(
                    metric_id=f"met_{uuid.uuid4().hex[:8]}",
                    run_id=record.run_id,
                    step_number=step_num,
                    metric_name="negotiation_rounds",
                    metric_value=float(record.rounds_used),
                    category="orchestration",
                )
                if record.fallback_used:
                    db.log_metric(
                        metric_id=f"met_{uuid.uuid4().hex[:8]}",
                        run_id=record.run_id,
                        step_number=step_num,
                        metric_name="fallback_count",
                        metric_value=1.0,
                        category="resilience",
                    )
        except Exception as e:
            logger.debug("Failed logging orchestrated decision to DuckDB: %s", e)

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Aggregate summary metrics from decision history."""
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "conflict_rate": 0.0,
                "shortcut_rate": 0.0,
                "avg_rounds": 0.0,
                "total_messages": 0,
                "total_fallbacks": 0,
            }

        total = len(self.decision_history)
        conflicts = sum(1 for d in self.decision_history if d.has_conflict)
        shortcuts = sum(1 for d in self.decision_history if d.resolution_method == "no_conflict_consensus")
        avg_rounds = float(np.mean([d.rounds_used for d in self.decision_history]))
        tot_msgs = sum(d.messages_exchanged for d in self.decision_history)

        return {
            "total_decisions": total,
            "conflict_rate": round(conflicts / total, 3),
            "shortcut_rate": round(shortcuts / total, 3),
            "avg_rounds": round(avg_rounds, 3),
            "total_messages": tot_msgs,
            "total_fallbacks": self.total_fallbacks,
        }


__all__ = [
    "NegotiationResponse",
    "ConflictDetectionResult",
    "OrchestratedDecisionRecord",
    "OrchestratorAgent",
    "OrchestratedSystem",
]
