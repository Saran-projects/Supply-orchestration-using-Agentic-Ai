"""Multi-Agent Information Sharing Ordering System (System 4).

Architecture:
- Shared visibility: Each echelon agent receives both its local EchelonState
  and a synchronized real-time view of inventory, backlog, pipeline, demand,
  and last orders from all other echelons in the supply chain network.
- Decision output: Validated Pydantic OrderDecision(quantity, reasoning). No tools.
- Resilience: 1 retry on malformed JSON or out-of-range orders, falling back to state.last_order.
- Telemetry: Full per-decision logging to DuckDB agent_calls and metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sca.sim.environment import ECHELONS, EchelonState
from sca.systems.base import register_system
from sca.systems.llm_base import (
    LLMOrderingSystemBase,
    OrderDecision,
    format_history_summary,
    format_shared_echelon_info,
    load_prompt_template,
)

logger = logging.getLogger(__name__)


@register_system(["multi_agent_info", "system_4"])
class MultiAgentInfoSystem(LLMOrderingSystemBase):
    """System 4: Multi-agent coordination with full information sharing.

    Per-echelon agents receive shared supply chain visibility (demand, inventory,
    backlog, pipeline, and last orders across all echelons) to synchronize orders
    and dampen the bullwhip effect without external tools.
    """

    def __init__(
        self,
        name: str = "multi_agent_info",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, config=config, prompt_version="multi_agent_info_v1", **kwargs)
        self.prompt_template = load_prompt_template("multi_agent_info_v1.txt")

    def format_echelon_prompt(
        self,
        echelon: str,
        state: EchelonState,
        max_order: float,
        shared_info: str,
    ) -> str:
        """Format multi-agent prompt with fixed instructions first, local state and shared info last."""
        history_str = format_history_summary(state.history_window)
        return self.prompt_template.format(
            echelon=echelon.capitalize(),
            max_order=f"{max_order:.1f}",
            step=state.step,
            current_demand=f"{state.current_demand:.1f}",
            inventory=f"{state.inventory:.1f}",
            backlog=f"{state.backlog:.1f}",
            pipeline=f"{state.pipeline:.1f}",
            incoming_shipments=f"{state.incoming_shipments:.1f}",
            lead_time=state.lead_time,
            history_summary=history_str,
            shared_info=shared_info,
        )

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Decide orders for all echelons with shared supply chain visibility."""
        orders: Dict[str, float] = {}
        shared_info = format_shared_echelon_info(states)

        for echelon in ECHELONS:
            if echelon not in states:
                continue
            state = states[echelon]
            max_order = self.get_max_order(echelon)
            prompt = self.format_echelon_prompt(
                echelon=echelon,
                state=state,
                max_order=max_order,
                shared_info=shared_info,
            )

            qty, _ = self._execute_echelon_decision(
                echelon=echelon,
                state=state,
                prompt=prompt,
                max_order=max_order,
            )
            orders[echelon] = qty

        return orders


__all__ = ["MultiAgentInfoSystem", "OrderDecision"]
