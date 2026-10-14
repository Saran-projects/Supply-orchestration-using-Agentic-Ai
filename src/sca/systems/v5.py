"""Agent design v5 (docs/agent_design_v5.md): one action space for Systems 3-6.

The LLM answers with JSON only: {regime, action, rationale}. Code turns the action into an order:

    order = clip(reference x multiplier[action], band_low, band_high)

- reference: S3/S4 use the basic reference (trailing mean + inventory-position gap / H);
  S5/S6 use the smoothed order-up-to tool target (frozen System 2c tool parameters).
- band: shared ``safe_order_bounds`` (the same function validate_order enforces). S5/S6 widen it
  in shock, and only the deterministic detector's phase can trigger that. The LLM's regime label
  is logged and scored, never acted on.
- Systems differ only in information (S4: shared echelon view), tools and band (S5) and
  orchestration (S6: deterministic specialist proposals shown to an LLM orchestrator).

Choosers: "llm" (default; MockLLM unless a real client is passed) or the scripted policies
"follow", "detector_map" and "oracle" (sees next-step customer demand; needs ``attach_env``).
"""

from __future__ import annotations

import json
import re
import time
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

from sca.config import AgentV5Config, find_project_root, get_config, load_agent_v5_config
from sca.db import get_duckdb_connection
from sca.llm.client import (
    FatalAPIError,
    LLMClient,
    PaidCallsDisabledError,
    StructuredOutputError,
    TruncatedOutputError,
    calculate_cost,
)
from sca.sim.environment import ECHELONS, EchelonState
from sca.systems.base import OrderingSystem, register_system
from sca.systems.llm_base import format_shared_echelon_info, load_prompt_template
from sca.tools import (
    EoqQuantityInput,
    ForecastDemandInput,
    RegimeDetectorInput,
    SmoothedOrderUpToInput,
    eoq_quantity,
    forecast_demand,
    regime_detector,
    safe_order_bounds,
    safe_order_range,
    smoothed_order_up_to,
)

logger = logging.getLogger(__name__)

ACTIONS: Tuple[str, ...] = ("follow", "boost_small", "boost_large", "trim")
REGIMES: Tuple[str, ...] = ("stable", "trend", "shock")
# Detector-mapped scripted policy (and MockLLM detector_copy mode)
DETECTOR_ACTION_MAP: Dict[str, str] = {
    "stable": "follow",
    "trend": "boost_small",
    "shock": "boost_large",
    "after_shock": "trim",
}
CHOOSERS = ("llm", "follow", "detector_map", "oracle")
MAX_RATIONALE_WORDS = 20
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def groundedness(rationale: str, prompt: str) -> Tuple[int, int]:
    """(numbers quoted in the rationale, how many match a number shown in the prompt within +-0.5 or 1%)."""
    shown = [float(x) for x in _NUMBER.findall(prompt)]
    quoted = [float(x) for x in _NUMBER.findall(rationale or "")]
    ok = sum(any(abs(q - v) <= max(0.5, 0.01 * abs(v)) for v in shown) for q in quoted)
    return len(quoted), ok


class ActionDecision(BaseModel):
    """The only output the LLM produces in design v5."""

    regime: Literal["stable", "trend", "shock"] = Field(description="Demand regime: stable, trend or shock")
    action: Literal["follow", "boost_small", "boost_large", "trim"] = Field(
        description="Adjustment applied to the reference order"
    )
    rationale: str = Field(description=f"At most {MAX_RATIONALE_WORDS} words")

    @field_validator("rationale")
    @classmethod
    def _short_rationale(cls, v: str) -> str:
        if len(v.split()) > MAX_RATIONALE_WORDS:
            raise ValueError(f"rationale must be at most {MAX_RATIONALE_WORDS} words")
        return v


TOOL_NAMES = ("forecast_demand", "regime_detector", "smoothed_order_up_to", "eoq_quantity", "safe_order_range")
TOOL_INPUTS = {
    "forecast_demand": ("demand_history",),
    "regime_detector": ("demand_history",),
    "smoothed_order_up_to": ("current_demand", "inventory", "pipeline", "backlog", "last_order"),
    "eoq_quantity": ("demand_history",),
    "safe_order_range": ("last_order", "current_demand"),
}


class ToolArgs(BaseModel):
    """Inputs a requested tool may take (design v5.1). Unused inputs are null."""

    demand_history: Optional[List[float]] = None
    current_demand: Optional[float] = None
    inventory: Optional[float] = None
    pipeline: Optional[float] = None
    backlog: Optional[float] = None
    last_order: Optional[float] = None


class ToolCall(BaseModel):
    tool: Literal["forecast_demand", "regime_detector", "smoothed_order_up_to", "eoq_quantity", "safe_order_range"]
    args: ToolArgs


class ToolRequest(BaseModel):
    """The model's tool request before a decision (design v5.1)."""

    calls: List[ToolCall] = Field(max_length=5)


def _close(given: Optional[float], truth: float, tol_abs: float, tol_rel: float) -> bool:
    return given is not None and abs(float(given) - float(truth)) <= max(tol_abs, tol_rel * abs(float(truth)))


def score_tool_request(calls: List[ToolCall], truth: Dict[str, Any], required: List[str], allowed: List[str],
                       tol_abs: float, tol_rel: float) -> Dict[str, Any]:
    """Tool-choice correctness and per-call parameter validity against the observed state."""
    names = [c.tool for c in calls]
    relevant = set(required) | set(allowed)
    choice_ok = set(required) <= set(names) and all(n in relevant for n in names)
    valid = total = 0
    for c in calls:
        if c.tool not in relevant:
            continue
        total += 1
        ok = True
        for field in TOOL_INPUTS[c.tool]:
            given = getattr(c.args, field)
            if field == "demand_history":
                hist = truth["demand_history"]
                ok &= (given is not None and len(given) >= min(3, len(hist))
                       and all(_close(g, t, tol_abs, tol_rel) for g, t in zip(reversed(given), reversed(hist))))
            else:
                ok &= _close(given, truth[field], tol_abs, tol_rel)
        valid += int(ok)
    return {"tool_choice_correct": choice_ok, "params_valid": valid, "params_total": total, "tools": names}


def compute_order(reference: float, multiplier: float, low: float, high: float) -> Tuple[float, float]:
    """(order before clipping, order after clipping into [low, high])."""
    pre = max(0.0, float(reference) * float(multiplier))
    return pre, float(min(max(pre, low), high))


def basic_reference(received: List[float], inventory_position: float, window: int, horizon: float,
                    mean_lead_time: float) -> Dict[str, float]:
    """Basic reference for S3/S4: max(0, mean_W + (S_basic - IP) / H), S_basic = mean_W x L."""
    recent = received[-window:] if received else [0.0]
    mean_w = float(np.mean(recent))
    s_basic = mean_w * float(mean_lead_time)
    ref = max(0.0, mean_w + (s_basic - float(inventory_position)) / float(horizon))
    return {"reference": ref, "trailing_mean": mean_w, "s_basic": s_basic}


def nearest_action(proposal: float, reference: float, multipliers: Dict[str, float]) -> str:
    """Action whose multiplier is closest to proposal / reference (follow if reference is ~0)."""
    if reference <= 1e-9:
        return "follow"
    ratio = proposal / reference
    return min(ACTIONS, key=lambda a: (abs(multipliers[a] - ratio), ACTIONS.index(a)))


def load_tool_params(cfg: AgentV5Config) -> Dict[str, Any]:
    """Frozen System 2c tool parameters used for the S5/S6 reference."""
    path = Path(cfg.tool_params_path)
    if not path.is_absolute():
        path = find_project_root() / path
    try:
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f)["tuned_policies"]["system_2c_adaptive_smoothed_out"]["parameters"])
    except (OSError, KeyError, ValueError):
        logger.warning("Frozen 2c tool parameters not found at %s; using tool defaults", path)
        return {}


class RegimeTracker:
    """Deterministic regime detector per echelon with shock countdown and after-shock phase.

    raw label: regime_detector output (stable | trend | shock), used to score the LLM's label.
    phase: stable | trend | shock | after_shock, used for band widening and the detector-mapped policy.
    """

    def __init__(self, cfg: AgentV5Config):
        self.cfg = cfg.detector
        self.reset()

    def reset(self) -> None:
        self.history: Dict[str, List[float]] = {e: [] for e in ECHELONS}
        self.countdown: Dict[str, int] = {e: 0 for e in ECHELONS}
        self.after: Dict[str, int] = {e: 0 for e in ECHELONS}

    def update(self, echelon: str, demand: float) -> Dict[str, Any]:
        self.history[echelon].append(float(demand))
        out = regime_detector(
            RegimeDetectorInput(
                demand_history=self.history[echelon],
                short_window=self.cfg.short_window,
                long_window=self.cfg.long_window,
                shock_threshold=self.cfg.shock_threshold,
                trend_threshold=self.cfg.trend_threshold,
            ),
            db_path=False,
        )
        if out.regime == "shock":
            self.countdown[echelon] = self.cfg.shock_duration
            self.after[echelon] = 0
        if self.countdown[echelon] > 0:
            phase = "shock"
            self.countdown[echelon] -= 1
            if self.countdown[echelon] == 0:
                self.after[echelon] = self.cfg.after_shock_steps
        elif self.after[echelon] > 0:
            phase = "after_shock"
            self.after[echelon] -= 1
        else:
            phase = out.regime
        return {"raw": out.regime, "phase": phase, "ratio": float(out.ratio)}


class V5System(OrderingSystem):
    """Shared engine for design v5. Subclasses only switch information, tools/band and orchestration."""

    system_label = "v5"
    template_file = "single_agent_v5_0.txt"
    shares_info = False      # S4+: prompt includes every echelon's state
    uses_tools = False       # S5+: smoothed order-up-to tool reference and detector output in the prompt
    widens_band = False      # S5+: band widened in the detector's shock phase
    orchestrated = False     # S6: deterministic specialist proposals + LLM orchestrator

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
        run_id: Optional[str] = None,
        db_path: Optional[Union[str, Path, bool]] = None,
        v5_config: Optional[AgentV5Config] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name or self.system_label, config=config)
        self.config = config or {}
        self.cfg: AgentV5Config = v5_config or load_agent_v5_config()
        if "multipliers" in self.config:
            self.cfg = self.cfg.model_copy(update={"multipliers": dict(self.config["multipliers"])})
        self.multipliers: Dict[str, float] = dict(self.cfg.multipliers)
        self.prompt_version: str = self.cfg.prompt_version
        self.chooser: str = self.config.get("chooser", "llm")
        if self.chooser not in CHOOSERS:
            raise ValueError(f"Invalid chooser '{self.chooser}'. Must be one of {CHOOSERS}.")
        self.run_id = run_id or self.config.get("run_id") or f"run_{self.system_label}_{uuid.uuid4().hex[:8]}"
        self.seed: int = int(self.config.get("seed", 42))
        self.model_name: Optional[str] = self.config.get("model")
        self.use_cache: bool = bool(self.config.get("use_cache", False))

        if self.config.get("db_path") is False or db_path is False:
            self.db_path: Union[Path, bool] = False
        elif db_path is not None:
            self.db_path = Path(db_path)
        else:
            try:
                self.db_path = get_config().experiment.db_path
            except Exception:
                self.db_path = Path("results/sca_metrics.duckdb")
        if llm_client is None:
            llm_client = LLMClient(mock_mode=True, db_path=self.db_path or None)
            llm_client.mock.configure_v5_mode(self.config.get("mock_v5_mode", "valid"), seed=self.seed)
        self.llm_client = llm_client
        self.tool_mode: str = self.config.get("tool_mode", self.cfg.tool_mode)
        if self.tool_mode not in ("deterministic", "llm_tools"):
            raise ValueError(f"Invalid tool_mode '{self.tool_mode}'. Must be 'deterministic' or 'llm_tools'.")
        # v5.1 applies to the systems that have tools (S5, S6); S3/S4 have no tools and stay on v5.0
        self.llm_tools = self.tool_mode == "llm_tools" and self.uses_tools and self.chooser == "llm"
        template = self.template_file
        if self.llm_tools:
            template = template.replace("_v5_0", "_v5_1")
            self.prompt_version = self.cfg.llm_tools.prompt_version
            self.tool_request_template = load_prompt_template("tool_request_v5_1.txt")
        self.prompt_template = load_prompt_template(template)

        self.tool_params = load_tool_params(self.cfg) if self.uses_tools else {}
        if self.uses_tools and "tool_params_override" in self.config:
            # Offline design studies only (scripts/v5_s6_design_study.py); not used by registered runs
            self.tool_params = {**self.tool_params, **self.config["tool_params_override"]}
        self.tracker = RegimeTracker(self.cfg)
        self.mean_lead_time: float = float(get_config().sim.four_echelon.default_lead_time)
        self.lead_time_std: float = 0.8
        self.demand_std: float = 20.0
        self.first_demand: float = 100.0
        self.env: Any = None
        self.parameters = {
            "design": "v5",
            "prompt_version": self.prompt_version,
            "multipliers": dict(self.multipliers),
            "reference": "smoothed_order_up_to" if self.uses_tools else "basic",
            "chooser": self.chooser,
            "tool_mode": self.tool_mode if self.uses_tools else "none",
        }
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def calibrate(self, scenario: Any, config: Optional[Dict[str, Any]] = None) -> None:
        """Lead-time and demand statistics from the scenario's training data only."""
        sampler = getattr(scenario, "lead_time_sampler", None)
        if sampler is not None:
            self.mean_lead_time = float(sampler.mean)
            self.lead_time_std = float(sampler.std)
        train = getattr(scenario, "train_demand", [])
        if len(train) > 0:
            self.demand_std = float(np.std(train))
        self.parameters.update({"mean_lead_time": round(self.mean_lead_time, 3)})
        self.calibrated = True

    def reset(self, scenario: Optional[Any] = None) -> None:
        """Per-run state reset (tracker, histories, smoothing, counters, logs, mock RNG)."""
        if scenario is not None and len(getattr(scenario, "demand", [])) > 0:
            self.first_demand = float(scenario.demand[0])
        self.tracker.reset()
        self.received: Dict[str, List[float]] = {e: [] for e in ECHELONS}
        self.smoothed_demand: Dict[str, float] = {e: self.first_demand for e in ECHELONS}
        self.decision_log: List[Dict[str, Any]] = []
        self._pending_rows: List[Dict[str, Any]] = []
        self.decision_count = 0
        self.fallback_count = 0
        self.llm_call_count = 0
        mock = getattr(self.llm_client, "mock", None)
        if mock is not None and hasattr(mock, "set_seed"):
            mock.set_seed(self.seed)

    def attach_env(self, env: Any) -> None:
        """Give the oracle chooser access to the environment's future customer demand."""
        self.env = env

    def llm_stats(self) -> Dict[str, int]:
        """Counts for the run-validity rules (only LLM-chosen decisions are LLM decisions)."""
        if self.chooser != "llm":
            return {"decisions": 0, "fallbacks": 0, "llm_calls": 0}
        return {"decisions": self.decision_count, "fallbacks": self.fallback_count, "llm_calls": self.llm_call_count}

    # ------------------------------------------------------------------ deterministic pieces
    def get_capacity(self, echelon: str) -> float:
        lim = self.cfg.limits
        return float(lim.manufacturer_capacity if echelon == "manufacturer" else lim.max_order)

    def _reference(self, echelon: str, state: EchelonState, observed: float, phase: str) -> Dict[str, float]:
        ip = state.inventory + state.pipeline - state.backlog
        if not self.uses_tools:
            out = basic_reference(self.received[echelon], ip, self.cfg.reference.window,
                                  self.cfg.reference.horizon, self.mean_lead_time)
            out.update({"inventory_position": ip, "demand_estimate": out["trailing_mean"]})
            return out
        p = self.tool_params
        beta = float(p.get("beta", p.get("alpha", 0.8)))
        if phase == "shock":
            beta = float(p.get("beta_shock", beta))
        tool = smoothed_order_up_to(
            SmoothedOrderUpToInput(
                current_demand=float(observed),
                inventory=float(state.inventory),
                pipeline=float(state.pipeline),
                backlog=float(state.backlog),
                last_order=float(state.last_order),
                lead_time_mean=self.mean_lead_time,
                lead_time_std=self.lead_time_std,
                demand_std=self.demand_std,
                safety_stock_factor=float(p.get("z", 1.65)),
                z=float(p.get("z", 1.65)),
                include_lead_time_variance=bool(p.get("include_lead_time_variance", True)),
                previous_smoothed_demand=float(self.smoothed_demand[echelon]),
                alpha=float(p.get("alpha", 0.8)),
                beta=beta,
            ),
            db_path=False,
        )
        self.smoothed_demand[echelon] = float(tool.smoothed_demand)
        return {
            "reference": float(tool.order_quantity),
            "demand_estimate": float(tool.smoothed_demand),
            "order_up_to_target": float(tool.order_up_to_target),
            "inventory_position": ip,
        }

    def _band(self, echelon: str, state: EchelonState, demand_estimate: float, phase: str) -> Dict[str, float]:
        widened = self.widens_band and phase == "shock"
        delta = self.cfg.band.delta_shock if widened else self.cfg.band.delta
        cap = self.get_capacity(echelon)
        low, high = safe_order_bounds(
            last_order=float(state.last_order),
            forecast_demand=float(demand_estimate),
            rate_limit_delta=delta,
            forecast_cap_multiple=self.cfg.band.forecast_cap_multiple,
            capacity_limit=cap if echelon == "manufacturer" else None,
            max_order=cap,
        )
        return {"low": low, "high": high, "delta": delta, "widened": widened}

    def _specialists(self, ctx: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """S6: deterministic specialist proposals, each mapped to the nearest action."""
        ref = ctx["reference"]
        props = {
            "demand": ctx["demand_estimate"],
            "inventory": ref,
            "procurement": float(min(max(ref, ctx["band_low"]), ctx["band_high"])),
        }
        return {k: {"quantity": q, "action": nearest_action(q, ref, self.multipliers)} for k, q in props.items()}

    def _oracle_action(self, ctx: Dict[str, Any]) -> str:
        if self.env is None:
            raise RuntimeError("oracle chooser needs attach_env(env)")
        demand = np.asarray(self.env.external_demand, dtype=float)
        d_next = float(demand[min(ctx["step"] + 1, len(demand) - 1)])
        ideal = max(0.0, d_next + (d_next * self.mean_lead_time - ctx["inventory_position"]) / self.cfg.reference.horizon)
        return min(ACTIONS, key=lambda a: (abs(ctx["candidates"][a] - ideal), ACTIONS.index(a)))

    # ------------------------------------------------------------------ prompt
    def _format_prompt(self, echelon: str, state: EchelonState, ctx: Dict[str, Any],
                       states: Dict[str, EchelonState]) -> str:
        cands = "\n".join(
            f"- {a} (x{self.multipliers[a]:.2f}): order {ctx['candidates'][a]:.1f}" for a in ACTIONS
        )
        recent = [round(x, 1) for x in self.received[echelon][-self.cfg.reference.window:]]
        fields: Dict[str, Any] = {
            "echelon": echelon.capitalize(),
            "step": state.step,
            "inventory": f"{state.inventory:.1f}",
            "backlog": f"{state.backlog:.1f}",
            "pipeline": f"{state.pipeline:.1f}",
            "inventory_position": f"{ctx['inventory_position']:.1f}",
            "current_demand": f"{state.current_demand:.1f}",
            "recent_demand": str(recent),
            "last_order": f"{state.last_order:.1f}",
            "reference": f"{ctx['reference']:.1f}",
            "band_low": f"{ctx['band_low']:.1f}",
            "band_high": f"{ctx['band_high']:.1f}",
            "candidates": cands,
            "max_words": MAX_RATIONALE_WORDS,
        }
        if not self.uses_tools:
            fields.update({
                "trailing_mean": f"{ctx['trailing_mean']:.1f}",
                "s_basic": f"{ctx['s_basic']:.1f}",
                "window": self.cfg.reference.window,
                "horizon": f"{self.cfg.reference.horizon:g}",
                "mean_lead_time": f"{self.mean_lead_time:.2f}",
            })
        else:
            fields.update({
                "smoothed_demand": f"{ctx['demand_estimate']:.1f}",
                "order_up_to_target": f"{ctx['order_up_to_target']:.1f}",
                "detector_regime": ctx["detector_phase"],
                "detector_ratio": f"{ctx['detector_ratio']:.2f}",
                "band_widened": "yes" if ctx["band_widened"] else "no",
            })
        fields["tool_results"] = ctx.get("tool_results", "")
        if self.shares_info:
            fields["shared_info"] = format_shared_echelon_info(states)
        if self.orchestrated:
            fields["proposals"] = "\n".join(
                f"- {k.capitalize()} specialist: {v['quantity']:.1f} (nearest action: {v['action']})"
                for k, v in ctx["specialists"].items()
            )
            acts = {v["action"] for v in ctx["specialists"].values()}
            fields["conflict"] = "none (specialists agree)" if len(acts) == 1 else f"specialists disagree: {sorted(acts)}"
        return self.prompt_template.format(**fields)

    # ------------------------------------------------------------------ choice
    def _choose_llm(self, prompt: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"action": None, "llm_regime": None, "rationale": "", "error_type": None,
                               "error_message": None, "retry_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache_hit": False}
        try:
            resp = self.llm_client.generate(
                prompt=prompt,
                response_model=ActionDecision,
                model=self.model_name,
                seed=self.seed,
                use_cache=self.use_cache,
                run_id=self.run_id,
                log_to_db=False,
                mock_context={"detector_regime": ctx["detector_regime"], "detector_phase": ctx["detector_phase"]},
            )
            self.llm_call_count += 1
            out.update(prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                       retry_count=int(getattr(resp, "retry_count", 0) or 0),
                       cached_tokens=getattr(resp, "cached_tokens", 0), cache_hit=bool(getattr(resp, "cached", False)))
            if isinstance(resp.structured, ActionDecision):
                out.update(action=resp.structured.action, llm_regime=resp.structured.regime,
                           rationale=resp.structured.rationale)
            else:
                out["error_type"] = "StructuredOutputError"
                out["error_message"] = f"no ActionDecision parsed from: {resp.content[:300]!r}"
        except (FatalAPIError, PaidCallsDisabledError):
            raise
        except (StructuredOutputError, TruncatedOutputError) as exc:
            self.llm_call_count += 1
            out.update(error_type=type(exc).__name__, error_message=str(exc)[:500], retry_count=1,
                       prompt_tokens=exc.prompt_tokens,
                       completion_tokens=exc.completion_tokens, cached_tokens=exc.cached_tokens)
        except Exception as exc:
            out["error_type"] = type(exc).__name__
            out["error_message"] = str(exc)[:500]
        return out

    # ------------------------------------------------------------------ design v5.1: tools and negotiation
    def _truth(self, echelon: str, state: EchelonState) -> Dict[str, Any]:
        return {
            "demand_history": [round(x, 1) for x in self.received[echelon][-self.cfg.reference.window:]],
            "current_demand": float(state.current_demand), "inventory": float(state.inventory),
            "pipeline": float(state.pipeline), "backlog": float(state.backlog), "last_order": float(state.last_order),
        }

    def _run_tool(self, call: ToolCall, prev_smooth: float) -> str:
        a = call.args
        missing = [f for f in TOOL_INPUTS[call.tool] if getattr(a, f) in (None, [])]
        if missing:
            return f"{call.tool}: error, missing {', '.join(missing)}"
        p = self.tool_params
        try:
            if call.tool == "forecast_demand":
                out = forecast_demand(ForecastDemandInput(recent_history=list(a.demand_history), horizon=1,
                                                          model_type="linear"), db_path=False)
                return f"forecast_demand: next-step forecast {max(0.0, out.point_forecast):.1f}"
            if call.tool == "regime_detector":
                d = self.cfg.detector
                out = regime_detector(RegimeDetectorInput(demand_history=list(a.demand_history),
                                                          short_window=d.short_window, long_window=d.long_window,
                                                          shock_threshold=d.shock_threshold,
                                                          trend_threshold=d.trend_threshold), db_path=False)
                return f"regime_detector: {out.regime} (short/long ratio {out.ratio:.2f})"
            if call.tool == "smoothed_order_up_to":
                out = smoothed_order_up_to(SmoothedOrderUpToInput(
                    current_demand=float(a.current_demand), inventory=float(a.inventory), pipeline=float(a.pipeline),
                    backlog=float(a.backlog), last_order=float(a.last_order), lead_time_mean=self.mean_lead_time,
                    lead_time_std=self.lead_time_std, demand_std=self.demand_std,
                    safety_stock_factor=float(p.get("z", 1.65)), z=float(p.get("z", 1.65)),
                    include_lead_time_variance=bool(p.get("include_lead_time_variance", True)),
                    previous_smoothed_demand=float(prev_smooth), alpha=float(p.get("alpha", 0.8)),
                    beta=float(p.get("beta", p.get("alpha", 0.8)))), db_path=False)
                return (f"smoothed_order_up_to: order {out.order_quantity:.1f} (order-up-to level "
                        f"{out.order_up_to_target:.1f}, smoothed demand {out.smoothed_demand:.1f})")
            if call.tool == "eoq_quantity":
                out = eoq_quantity(EoqQuantityInput(mean_demand=max(1.0, float(np.mean(a.demand_history))),
                                                    order_fixed_cost=100.0, holding_cost_per_unit=0.5), db_path=False)
                return f"eoq_quantity: batch size {out.eoq:.1f}"
            low, high = safe_order_range(float(a.last_order), float(a.current_demand), self.cfg.band.delta)
            return f"safe_order_range: [{low:.1f}, {high:.1f}]"
        except Exception as exc:  # invalid inputs (e.g. negative values) are the model's error, shown to it
            return f"{call.tool}: error, {type(exc).__name__}"

    def _tool_request(self, echelon: str, state: EchelonState, prev_smooth: float) -> Dict[str, Any]:
        truth = self._truth(echelon, state)
        prompt = self.tool_request_template.format(
            echelon=echelon.capitalize(), step=state.step, inventory=f"{state.inventory:.1f}",
            backlog=f"{state.backlog:.1f}", pipeline=f"{state.pipeline:.1f}", current_demand=f"{state.current_demand:.1f}",
            recent_demand=str(truth["demand_history"]), last_order=f"{state.last_order:.1f}",
            max_calls=self.cfg.llm_tools.max_calls)
        out: Dict[str, Any] = {"text": "(no tool results: the tool request failed)", "tool_choice_correct": False,
                               "params_valid": 0, "params_total": 0, "tools": [], "prompt_tokens": 0,
                               "completion_tokens": 0, "cached_tokens": 0, "error": None}
        try:
            resp = self.llm_client.generate(
                prompt=prompt, response_model=ToolRequest, model=self.model_name, seed=self.seed,
                use_cache=self.use_cache, run_id=self.run_id, log_to_db=False,
                mock_context={"truth": truth, "required": list(self.cfg.llm_tools.required)})
            self.llm_call_count += 1
            out.update(prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                       cached_tokens=getattr(resp, "cached_tokens", 0))
            req = resp.structured
        except (FatalAPIError, PaidCallsDisabledError):
            raise
        except (StructuredOutputError, TruncatedOutputError) as exc:
            self.llm_call_count += 1
            out.update(prompt_tokens=exc.prompt_tokens, completion_tokens=exc.completion_tokens,
                       cached_tokens=exc.cached_tokens, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return out
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            return out
        if not isinstance(req, ToolRequest):
            out["error"] = "no ToolRequest parsed"
            return out
        lt = self.cfg.llm_tools
        out.update(score_tool_request(req.calls, truth, lt.required, lt.allowed, lt.param_tolerance_abs,
                                      lt.param_tolerance_rel))
        results = [self._run_tool(c, prev_smooth) for c in req.calls[: lt.max_calls]]
        out["text"] = "\n".join(f"- {r}" for r in results) if results else "(you requested no tools)"
        return out

    def _objections(self, ctx: Dict[str, Any], action: str, order: float) -> List[str]:
        thr = self.cfg.negotiation.objection_threshold
        msgs = []
        for name, sp in ctx["specialists"].items():
            if sp["action"] != action and abs(order - sp["quantity"]) > thr * max(sp["quantity"], 1.0):
                msgs.append(f"- {name.capitalize()} specialist objects: prefers {sp['action']} "
                            f"(its proposal {sp['quantity']:.1f}; your order {order:.1f}).")
        return msgs

    # ------------------------------------------------------------------ main loop
    def decide_orders(self, states: Dict[str, EchelonState]) -> Dict[str, float]:
        orders: Dict[str, float] = {}
        customer = float(states["retailer"].current_demand) if "retailer" in states else None
        share_target = self.uses_tools and bool(self.tool_params.get("info_sharing", False))

        for echelon in ECHELONS:
            if echelon not in states:
                continue
            state = states[echelon]
            self.received[echelon].append(float(state.current_demand))
            observed = customer if (share_target and customer is not None) else float(state.current_demand)
            prev_smooth = self.smoothed_demand[echelon]
            det = self.tracker.update(echelon, observed)
            ref = self._reference(echelon, state, observed, det["phase"])
            band = self._band(echelon, state, ref["demand_estimate"], det["phase"])
            candidates = {a: compute_order(ref["reference"], self.multipliers[a], band["low"], band["high"])[1]
                          for a in ACTIONS}
            ctx: Dict[str, Any] = {
                **ref,
                "step": int(state.step),
                "detector_regime": det["raw"],
                "detector_phase": det["phase"],
                "detector_ratio": det["ratio"],
                "band_low": band["low"],
                "band_high": band["high"],
                "band_widened": band["widened"],
                "candidates": candidates,
            }
            if self.orchestrated:
                ctx["specialists"] = self._specialists(ctx)

            choice: Dict[str, Any] = {"action": None, "llm_regime": None, "rationale": "", "error_type": None,
                                      "error_message": None, "retry_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
                                      "cache_hit": False}
            t0 = time.perf_counter()
            prompt = ""
            tools: Dict[str, Any] = {}
            rounds = messages = 0
            if self.chooser == "llm":
                if self.llm_tools:
                    tools = self._tool_request(echelon, state, prev_smooth)
                    ctx["tool_results"] = tools["text"]
                prompt = self._format_prompt(echelon, state, ctx, states)
                choice = self._choose_llm(prompt, ctx)
                rounds, messages = 1, 1
                if self.llm_tools and self.orchestrated:
                    messages += len(ctx["specialists"])
                    if choice["action"] is not None and self.cfg.negotiation.max_rounds > 1:
                        order1 = compute_order(ref["reference"], self.multipliers[choice["action"]], band["low"],
                                               band["high"])[1]
                        objections = self._objections(ctx, choice["action"], order1)
                        if objections:
                            round2 = (f"{prompt}\n\nNEGOTIATION ROUND 2 of 2:\nYour previous choice: {choice['action']} "
                                      f"(order {order1:.1f}).\n" + "\n".join(objections) +
                                      "\nReconsider and answer with the same JSON object.")
                            second = self._choose_llm(round2, ctx)
                            rounds += 1
                            messages += 1 + len(ctx["specialists"])
                            for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                                choice[k] += second[k]
                            if second["action"] is not None:
                                for k in ("action", "llm_regime", "rationale"):
                                    choice[k] = second[k]
                                prompt = round2
                for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                    choice[k] += tools.get(k, 0)
            elif self.chooser == "follow":
                choice["action"] = "follow"
            elif self.chooser == "detector_map":
                choice["action"] = DETECTOR_ACTION_MAP[det["phase"]]
            else:
                choice["action"] = self._oracle_action(ctx)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            n_quoted, n_grounded = groundedness(choice["rationale"], prompt) if prompt else (0, 0)
            fallback = choice["action"] is None
            if fallback:
                # Protocol section 8: previous-period order, kept inside the safe range
                multiplier = None
                pre = float(state.last_order)
                post = float(min(max(pre, band["low"]), band["high"]))
                self.fallback_count += 1
            else:
                multiplier = self.multipliers[choice["action"]]
                pre, post = compute_order(ref["reference"], multiplier, band["low"], band["high"])
            self.decision_count += 1
            orders[echelon] = post

            cost = 0.0
            if self.chooser == "llm" and not choice["cache_hit"]:
                cost = calculate_cost(self.model_name or "mock-model", choice["prompt_tokens"],
                                      choice["completion_tokens"], getattr(self.llm_client, "config", None),
                                      cached_tokens=choice["cached_tokens"])
            row = {
                "run_id": self.run_id,
                "system": self.system_label,
                "prompt_version": self.prompt_version,
                "chooser": self.chooser,
                "step_number": int(state.step),
                "echelon": echelon,
                "action": choice["action"] or "fallback",
                "llm_regime": choice["llm_regime"],
                "detector_regime": det["raw"],
                "detector_phase": det["phase"],
                "reference": round(ref["reference"], 4),
                "multiplier": multiplier,
                "order_pre_clip": round(pre, 4),
                "order_post_clip": round(post, 4),
                "clipped": abs(pre - post) > 1e-9,
                "band_low": round(band["low"], 4),
                "band_high": round(band["high"], 4),
                "band_widened": band["widened"],
                "fallback": fallback,
                "error_type": choice["error_type"],
                "error_message": choice["error_message"],
                "llm_retry_count": int(choice["retry_count"]),
                "first_pass_valid": (not fallback) and int(choice["retry_count"]) == 0,
                "rationale": choice["rationale"],
                "rationale_numbers": n_quoted,
                "rationale_numbers_grounded": n_grounded,
                "latency_ms": round(latency_ms, 2),
                "tool_mode": "llm_tools" if self.llm_tools else ("deterministic" if self.uses_tools else "none"),
                "tools_requested": ",".join(tools.get("tools", [])) if self.llm_tools else None,
                "tool_choice_correct": tools.get("tool_choice_correct") if self.llm_tools else None,
                "params_valid": tools.get("params_valid") if self.llm_tools else None,
                "params_total": tools.get("params_total") if self.llm_tools else None,
                "tool_error": tools.get("error") if self.llm_tools else None,
                "negotiation_rounds": rounds if (self.llm_tools and self.orchestrated) else None,
                "negotiation_messages": messages if (self.llm_tools and self.orchestrated) else None,
                "prompt_tokens": int(choice["prompt_tokens"]),
                "completion_tokens": int(choice["completion_tokens"]),
                "cost_usd": float(cost),
            }
            self.decision_log.append(row)
            self._pending_rows.append(row)

        self._flush_log()
        return orders

    def _flush_log(self) -> None:
        """Write this step's decision rows to DuckDB table v5_decisions (skipped when db_path is False)."""
        rows, self._pending_rows = self._pending_rows, []
        if self.db_path is False or not rows:
            return
        try:
            conn = get_duckdb_connection(self.db_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v5_decisions (
                    run_id VARCHAR, system VARCHAR, prompt_version VARCHAR, chooser VARCHAR,
                    step_number INTEGER, echelon VARCHAR, action VARCHAR, llm_regime VARCHAR,
                    detector_regime VARCHAR, detector_phase VARCHAR, reference DOUBLE, multiplier DOUBLE,
                    order_pre_clip DOUBLE, order_post_clip DOUBLE, clipped BOOLEAN, band_low DOUBLE,
                    band_high DOUBLE, band_widened BOOLEAN, fallback BOOLEAN, error_type VARCHAR,
                    error_message VARCHAR, llm_retry_count INTEGER, first_pass_valid BOOLEAN, rationale VARCHAR,
                    rationale_numbers INTEGER, rationale_numbers_grounded INTEGER, latency_ms DOUBLE,
                    tool_mode VARCHAR, tools_requested VARCHAR, tool_choice_correct BOOLEAN, params_valid INTEGER,
                    params_total INTEGER, tool_error VARCHAR, negotiation_rounds INTEGER, negotiation_messages INTEGER,
                    prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cols = list(rows[0].keys())
            conn.executemany(
                f"INSERT INTO v5_decisions ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [[r[c] for c in cols] for r in rows],
            )
            conn.close()
        except Exception as exc:
            logger.warning("Failed to log v5 decisions: %s", exc)


def summarize_decisions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Action mix, clip rate, fallback rate and LLM regime accuracy (vs the detector's raw label)."""
    n = len(rows)
    if n == 0:
        return {"decisions": 0}
    mix = {a: sum(r["action"] == a for r in rows) / n for a in (*ACTIONS, "fallback")}
    scored = [r for r in rows if r["llm_regime"] is not None]
    acc = (sum(r["llm_regime"] == r["detector_regime"] for r in scored) / len(scored)) if scored else None
    return {
        "decisions": n,
        "action_mix": mix,
        "clip_rate": sum(r["clipped"] for r in rows) / n,
        "fallback_rate": sum(r["fallback"] for r in rows) / n,
        "regime_accuracy": acc,
    }


@register_system(["single_agent_v5", "system_3_v5"])
class SingleAgentV5(V5System):
    """System 3 (v5): local information, basic reference, fixed band."""

    system_label = "single_agent_v5"
    template_file = "single_agent_v5_0.txt"


@register_system(["multi_agent_info_v5", "system_4_v5"])
class MultiAgentInfoV5(V5System):
    """System 4 (v5): System 3 plus every echelon's state in the prompt."""

    system_label = "multi_agent_info_v5"
    template_file = "multi_agent_info_v5_0.txt"
    shares_info = True


@register_system(["multi_agent_tools_v5", "system_5_v5"])
class MultiAgentToolsV5(V5System):
    """System 5 (v5): System 4 plus deterministic tools (smoothed OUT reference, detector) and shock band."""

    system_label = "multi_agent_tools_v5"
    template_file = "multi_agent_tools_v5_0.txt"
    shares_info = True
    uses_tools = True
    widens_band = True


@register_system(["orchestrated_v5", "system_6_v5"])
class OrchestratedV5(V5System):
    """System 6 (v5): System 5 plus deterministic specialist proposals arbitrated by an LLM orchestrator."""

    system_label = "orchestrated_v5"
    template_file = "orchestrated_v5_0.txt"
    shares_info = True
    uses_tools = True
    widens_band = True
    orchestrated = True


V5_SYSTEM_IDS = ["single_agent_v5", "multi_agent_info_v5", "multi_agent_tools_v5", "orchestrated_v5"]

__all__ = [
    "ACTIONS",
    "ActionDecision",
    "ToolRequest",
    "score_tool_request",
    "DETECTOR_ACTION_MAP",
    "MultiAgentInfoV5",
    "MultiAgentToolsV5",
    "OrchestratedV5",
    "RegimeTracker",
    "SingleAgentV5",
    "V5System",
    "V5_SYSTEM_IDS",
    "basic_reference",
    "compute_order",
    "nearest_action",
    "summarize_decisions",
]
