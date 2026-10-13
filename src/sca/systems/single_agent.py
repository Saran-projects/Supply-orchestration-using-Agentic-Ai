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
from pathlib import Path
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
        tool_mode: str = "deterministic",
        **kwargs: Any,
    ):
        cfg = config or {}
        p_kw = kwargs.pop("prompt_version", None)
        p_ver = cfg.get("prompt_version") or p_kw or "single_agent_v3"
        p_norm = p_ver.replace(".", "_")
        tpl_cand = f"single_agent_{p_norm}.txt"
        prompts_dir = Path(__file__).resolve().parent.parent / "agents" / "prompts"
        if (prompts_dir / tpl_cand).exists():
            tpl_file = tpl_cand
            ver_tag = p_ver
        elif p_ver == "v3" or p_ver == "single_agent_v3":
            tpl_file = "single_agent_v3.txt"
            ver_tag = "single_agent_v3"
        elif p_ver.startswith("v3") or p_ver.startswith("single_agent_v3"):
            tpl_file = "single_agent_v3.txt"
            ver_tag = p_ver
        elif p_ver in ("v2", "single_agent_v2"):
            tpl_file = "single_agent_v2.txt"
            ver_tag = "single_agent_v2"
        else:
            tpl_file = "single_agent_v1.txt"
            ver_tag = "single_agent_v1"
        super().__init__(name=name, config=config, prompt_version=ver_tag, tool_mode=tool_mode, **kwargs)
        self.prompt_template = load_prompt_template(tpl_file)

    def format_echelon_prompt(
        self, echelon: str, state: EchelonState, max_order: float
    ) -> tuple[str, Optional[float]]:
        """Format single agent prompt with fixed instructions first and variable state last."""
        history_str = format_history_summary(state.history_window)
        recommendation: Optional[float] = None

        if self.prompt_version.startswith("v3") or self.prompt_version.startswith("single_agent_v3"):
            tool_data = self.compute_deterministic_tools(state, echelon)
            recommendation = tool_data["recommendation"]
            prompt = self.prompt_template.format(
                echelon=echelon.capitalize(),
                max_order=f"{max_order:.1f}",
                step=state.step,
                current_demand=f"{state.current_demand:.1f}",
                inventory=f"{state.inventory:.1f}",
                backlog=f"{state.backlog:.1f}",
                pipeline=f"{state.pipeline:.1f}",
                incoming_shipments=f"{state.incoming_shipments:.1f}",
                lead_time=state.lead_time,
                last_5_orders=str(tool_data["last_5_orders"]),
                forecast=f"{tool_data['forecast']:.1f}",
                eoq=f"{tool_data['eoq']:.1f}",
                recommendation=f"{tool_data['recommendation']:.1f}",
                safe_range=f"[{tool_data['safe_range'][0]:.1f}, {tool_data['safe_range'][1]:.1f}]",
                history_summary=history_str,
            )
        else:
            prompt = self.prompt_template.format(
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

        return prompt, recommendation

    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Decide orders for all echelons given local states."""
        orders: Dict[str, float] = {}

        for echelon in ECHELONS:
            if echelon not in states:
                continue
            state = states[echelon]
            max_order = self.get_max_order(echelon)
            prompt, rec = self.format_echelon_prompt(echelon=echelon, state=state, max_order=max_order)

            qty, _ = self._execute_echelon_decision(
                echelon=echelon,
                state=state,
                prompt=prompt,
                max_order=max_order,
                recommendation=rec,
            )
            orders[echelon] = qty

        return orders


__all__ = ["SingleAgentSystem", "OrderDecision"]
