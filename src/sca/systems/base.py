"""Base interfaces and registry for supply chain ordering systems and policies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from sca.data.scenarios import Scenario
from sca.sim.environment import EchelonState

logger = logging.getLogger(__name__)

SYSTEM_REGISTRY: Dict[str, Type["OrderingSystem"]] = {}


def register_system(names: Union[str, List[str]]):
    """Decorator to register an OrderingSystem implementation by name(s)."""
    if isinstance(names, str):
        names = [names]

    def decorator(cls: Type[OrderingSystem]):
        for name in names:
            SYSTEM_REGISTRY[name] = cls
            SYSTEM_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_system(name: str, **kwargs) -> OrderingSystem:
    """Instantiate a registered ordering system by name."""
    lookup = name.lower()
    if lookup not in SYSTEM_REGISTRY:
        raise KeyError(
            f"Ordering system '{name}' not found in registry. "
            f"Available systems: {list(set(SYSTEM_REGISTRY.keys()))}"
        )
    return SYSTEM_REGISTRY[lookup](name=name, **kwargs)


def list_systems() -> List[str]:
    """List unique registered ordering systems."""
    return sorted(list(set(SYSTEM_REGISTRY.keys())))


class OrderingSystem(ABC):
    """Abstract base class for all supply chain ordering systems (heuristics, baselines, multi-agent)."""

    def __init__(self, name: str = "OrderingSystem", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.calibrated = False
        self.parameters: Dict[str, Any] = {}

    @abstractmethod
    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        """Decide orders to place upstream for all echelons given current states.

        Args:
            states: Dict mapping echelon name to its current EchelonState.

        Returns:
            Dict mapping each echelon to non-negative order quantity:
            {'retailer': o_R, 'wholesaler': o_W, 'distributor': o_D, 'manufacturer': o_M}
        """
        pass

    def calibrate(self, scenario: Scenario, config: Optional[Dict[str, Any]] = None) -> None:
        """Derive policy parameters from scenario training history and config.

        Strictly uses scenario.train_demand to ensure no lookahead data leakage.
        """
        self.calibrated = True

    def reset(self, scenario: Optional[Scenario] = None) -> None:
        """Reset internal system state at the start of a simulation run."""
        return None

    def get_parameters(self) -> Dict[str, Any]:
        """Return a copy of fitted/configured parameters for logging and inspection."""
        return dict(self.parameters)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', calibrated={self.calibrated})"
