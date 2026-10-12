"""Multi-agent coordination architectures and ordering systems."""

import sca.baselines  # noqa: F401 (Ensures heuristic and smoothed baselines are registered)
from sca.systems.base import (
    SYSTEM_REGISTRY,
    OrderingSystem,
    get_system,
    list_systems,
    register_system,
)
from sca.systems.llm_base import OrderDecision
from sca.systems.multi_agent_info import MultiAgentInfoSystem
from sca.systems.multi_agent_tools import AgentProposal, MultiAgentToolsSystem
from sca.systems.orchestrated import OrchestratedSystem, OrchestratorAgent
from sca.systems.single_agent import SingleAgentSystem

__all__ = [
    "OrderingSystem",
    "register_system",
    "get_system",
    "list_systems",
    "SYSTEM_REGISTRY",
    "OrderDecision",
    "AgentProposal",
    "SingleAgentSystem",
    "MultiAgentInfoSystem",
    "MultiAgentToolsSystem",
    "OrchestratorAgent",
    "OrchestratedSystem",
]
