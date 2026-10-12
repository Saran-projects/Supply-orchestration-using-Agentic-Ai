"""Single-Agent LLM Ordering System (System 3).

Architecture:
- Decentralized coordination: One independent LLM agent per echelon.
- Information set: Each agent observes strictly its local EchelonState and local history window.
- Decision output: Validated Pydantic OrderDecision(quantity, reasoning).
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
    load_prompt_template,
)

logger = logging.getLogger(__name__)


@register_system(["single_agent", "system_3"])
class SingleAgentSystem(LLMOrderingSystemBase):
    """System 3: One independent LLM agent per supply chain echelon.

    Each agent operates with local visibility only (its own inventory, backlog, pipeline,
    incoming demand, and history window). No cross-echelon shared information or tools.
    """

    def __init__(
        self,
        name: str = "single_agent",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        cfg = config or {}
        p_ver = cfg.get("prompt_version") or kwargs.get("prompt_version", "v1")
        if p_ver in ("v2", "single_agent_v2"):
            tpl_file = "single_agent_v2.txt"
            ver_tag = "single_agent_v2"
        else:
            tpl_file = "single_agent_v1.txt"
            ver_tag = "single_agent_v1"
        super().__init__(name=name, config=config, prompt_version=ver_tag, **kwargs)
        self.prompt_template = load_prompt_template(tpl_file)

    def format_echelon_prompt(self, echelon: str, state: EchelonState, max_order: float) -> str:
        """Format single agent prompt with fixed instructions first and variable state last."""
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
        )

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Decide orders for all echelons given local states."""
        orders: Dict[str, float] = {}

        for echelon in ECHELONS:
            if echelon not in states:
                continue
            state = states[echelon]
            max_order = self.get_max_order(echelon)
            prompt = self.format_echelon_prompt(echelon=echelon, state=state, max_order=max_order)

            qty, _ = self._execute_echelon_decision(
                echelon=echelon,
                state=state,
                prompt=prompt,
                max_order=max_order,
            )
            orders[echelon] = qty

        return orders


__all__ = ["SingleAgentSystem", "OrderDecision"]
