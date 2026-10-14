"""Design v5 paid run: shock scenario, one seed, 40 steps, System 3 then System 6 on glm-isoquant.

Safety:
- ``--dry-run`` (default) makes no network call: it builds the real v5.0 prompts with MockLLM and
  projects the cost. A real run needs ``--execute`` and ALLOW_PAID_CALLS=1 (LLMClient guard).
- Hard stop when cumulative spend (including ``--prior-spend``) would exceed ``--cap`` USD: every call
  is checked against its worst case (prompt + max_tokens) before it is sent.
- HTTP 401/402/403 raise FatalAPIError: the run stops at once, with no retry and no fallback order.
- System 6 runs only if System 3 had LLM calls >= decisions and 0 fallbacks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

from sca.config import find_project_root, get_config, load_models_config  # noqa: E402
from sca.data.scenarios import generate_scenarios  # noqa: E402
from sca.llm.client import LLMClient, PaidCallsDisabledError, calculate_cost, estimate_tokens  # noqa: E402
from sca.sim.environment import ECHELONS, BeerGameEnv  # noqa: E402
from sca.sim.metrics import compute_simulation_metrics  # noqa: E402
from sca.systems.base import get_system  # noqa: E402
from sca.systems.v5 import ACTIONS, summarize_decisions  # noqa: E402

ROOT = find_project_root()
MODEL = "glm-isoquant"
SYSTEMS = [("System 3", "single_agent_v5"), ("System 6", "orchestrated_v5")]


class BudgetExceeded(PaidCallsDisabledError):
    """Raised before a call whose worst case would push spend past the cap (propagates like the paid guard)."""


class GuardedClient(LLMClient):
    """LLMClient that tracks real spend and refuses calls whose worst case would exceed the cap."""

    def __init__(self, cap: float, spent: float, **kw: Any):
        super().__init__(**kw)
        self.cap, self.spent, self.calls = cap, spent, 0
        entry = self.config.get_model(MODEL)
        self.max_tokens = int(entry.max_tokens)

    def generate(self, prompt: str, **kw: Any):
        # Worst case: this prompt plus a full max_tokens completion, twice (one schema retry inside the client)
        worst = 2 * calculate_cost(MODEL, estimate_tokens(prompt), self.max_tokens, self.config)
        if self.spent + worst > self.cap:
            raise BudgetExceeded(f"Hard stop: spent ${self.spent:.4f} + worst case ${worst:.4f} > cap ${self.cap:.2f}")
        self.calls += 1
        try:
            resp = super().generate(prompt, **kw)
        except Exception as exc:
            self.spent += calculate_cost(MODEL, getattr(exc, "prompt_tokens", 0), getattr(exc, "completion_tokens", 0),
                                         self.config, cached_tokens=getattr(exc, "cached_tokens", 0))
            raise
        if not resp.cached:
            self.spent += float(resp.cost_usd)
        return resp


def simulate(system, scenario, seed):
    env = BeerGameEnv(config=get_config().sim, scenario=scenario, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)
    if hasattr(system, "attach_env"):
        system.attach_env(env)
    for _ in range(40):
        states = env.step(system.decide_orders(states))
    return compute_simulation_metrics(env, warmup_steps=5)


def run_system(sid, scenario, seed, client=None, mock_mode="always_follow", chooser="llm", params=None):
    cfg = {"db_path": False, "seed": seed, "chooser": chooser, "model": MODEL if client else None,
           "mock_v5_mode": mock_mode, "run_id": f"{sid}_shock_seed{seed}_v5_0_{uuid.uuid4().hex[:6]}",
           **(params or {})}
    system = get_system(sid, config=cfg, llm_client=client) if client else get_system(sid, config=cfg)
    system.calibrate(scenario, config=cfg if params else None)
    system.reset(scenario)
    return system, simulate(system, scenario, seed)


def metrics_row(label, m, system=None):
    row = {"policy": label, "cost": m.total_cost, "fill_rate": 100 * m.fill_rate["retailer"],
           "e2e_bw": m.end_to_end_bullwhip, **{f"bw_{e[:4]}": m.bullwhip_ratio[e] for e in ECHELONS}}
    if system is not None and hasattr(system, "decision_log"):
        s = summarize_decisions(system.decision_log)
        rows = system.decision_log
        row.update(follow_rate=s["action_mix"]["follow"], clip_rate=s["clip_rate"],
                   regime_acc=s["regime_accuracy"], fallback_rate=s["fallback_rate"])
        row["first_pass_valid"] = float(np.mean([r.get("first_pass_valid", False) for r in rows]))
    return row


def dry_run(scenario, seed, prior, cap, threshold):
    models = load_models_config()
    entry = models.get_model(MODEL)
    total = 0.0
    print(f"DRY RUN (no network). model={entry.model_id} max_tokens={entry.max_tokens}")
    for label, sid in SYSTEMS:
        system, _ = run_system(sid, scenario, seed)
        prompts = [c["prompt"] for c in system.llm_client.mock.call_history]
        p_tok = [estimate_tokens(p) for p in prompts]
        n = len(prompts)
        expected = sum(calculate_cost(MODEL, t, min(entry.max_tokens, 800), models) for t in p_tok)
        worst = sum(calculate_cost(MODEL, t, entry.max_tokens, models) for t in p_tok)
        total += worst
        print(f"  {label}: {n} calls, prompt tokens mean {np.mean(p_tok):.0f} (max {max(p_tok)}), "
              f"expected ${expected:.4f}, worst case (full max_tokens) ${worst:.4f}, with one schema retry each ${2 * worst:.4f}")
    check = 2 * calculate_cost(MODEL, 40, entry.max_tokens, models)
    print(f"  model check (2 tiny calls) worst ${check:.4f}")
    projected = total + check
    print(f"Projected (worst case, no retries): ${projected:.4f} | threshold ${threshold:.2f} | "
          f"prior spend ${prior:.4f} + projection = ${prior + projected:.4f} vs cap ${cap:.2f}")
    ok = projected <= threshold and prior + projected <= cap
    print("DRY RUN RESULT:", "OK to execute" if ok else "ABORT (projection over limit)")
    return ok


def execute(scenario, seed, prior, cap, db_path):
    client = GuardedClient(cap=cap, spent=prior, mock_mode=False, cache_enabled=False, db_path=db_path)
    results: Dict[str, Any] = {}
    for label, sid in SYSTEMS:
        start = client.spent
        system, m = run_system(sid, scenario, seed, client=client)
        stats = system.llm_stats()
        rows = system.decision_log
        results[label] = (system, m, client.spent - start)
        tok_in = np.mean([r["prompt_tokens"] for r in rows])
        tok_out = np.mean([r["completion_tokens"] for r in rows])
        print(f"{label}: {stats} | tokens/decision in {tok_in:.0f} out {tok_out:.0f} | USD {client.spent - start:.4f} "
              f"| cumulative {client.spent:.4f}")
        if label == "System 3" and (stats["llm_calls"] < stats["decisions"] or stats["fallbacks"] > 0):
            errs = [r for r in rows if r["fallback"]]
            print("System 3 failed the gate; System 6 skipped. First error, exactly:")
            if errs:
                print(f"  step {errs[0]['step_number']} {errs[0]['echelon']}: {errs[0]['error_type']}: {errs[0]['error_message']}")
            break
    return results, client.spent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--execute", action="store_true", help="make real calls (requires ALLOW_PAID_CALLS=1)")
    ap.add_argument("--cap", type=float, default=0.45, help="hard stop on cumulative USD")
    ap.add_argument("--threshold", type=float, default=0.30, help="abort if projected cost is above this")
    ap.add_argument("--prior-spend", type=float, default=0.0, help="USD already spent in this task")
    ap.add_argument("--db", default="results/v5_paid.duckdb")
    args = ap.parse_args()

    scenario = generate_scenarios()["shock"]
    if not dry_run(scenario, args.seed, args.prior_spend, args.cap, args.threshold):
        return 2
    if not args.execute:
        return 0

    results, spent = execute(scenario, args.seed, args.prior_spend, args.cap, ROOT / args.db)
    frozen = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))["tuned_policies"]
    rows: List[Dict[str, Any]] = []
    for label, (system, m, usd) in results.items():
        r = metrics_row(f"{label} (glm, v5.0)", m, system)
        r["usd"] = usd
        rows.append(r)
        mix = pd.DataFrame(system.decision_log).groupby("echelon")["action"].value_counts(normalize=True).unstack(fill_value=0)
        print(f"\n{label} action mix per echelon:\n{mix.reindex(columns=[*ACTIONS, 'fallback'], fill_value=0).round(2).to_string()}")
    for sid, fam in (("single_agent_v5", "basic"), ("multi_agent_tools_v5", "tool")):
        for chooser in ("follow", "detector_map", "oracle"):
            system, m = run_system(sid, scenario, args.seed, chooser=chooser)
            rows.append(metrics_row(f"scripted {chooser} [{fam}]", m, system))
    for label, sid, key in (("System 2c tuned", "baseline_2c", "system_2c_adaptive_smoothed_out"),
                            ("EOQ tuned", "baseline_eoq", "system_2_eoq")):
        params = {**frozen[key]["parameters"], "db_path": False}
        system = get_system(sid, config=params)
        system.calibrate(scenario, config=params)
        system.reset(scenario)
        rows.append(metrics_row(label, simulate(system, scenario, args.seed)))
    df = pd.DataFrame(rows).set_index("policy")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + df.round(3).to_string())

    eoq = df.loc["EOQ tuned"]
    print("\nProtocol success criteria (section 3), shock seed", args.seed)
    for label, (system, m, _) in results.items():
        r = df.loc[f"{label} (glm, v5.0)"]
        crit = {
            "1 zero fallbacks": system.llm_stats()["fallbacks"] == 0,
            "2 first-pass validity >= 99%": r["first_pass_valid"] >= 0.99,
            "3 e2e bullwhip <= tuned EOQ": r["e2e_bw"] <= eoq["e2e_bw"],
            "4 cost <= 1.15 x tuned EOQ": r["cost"] <= 1.15 * eoq["cost"],
            "5 fill rate >= 99%": r["fill_rate"] >= 99.0,
        }
        print(f"  {label}: " + "; ".join(f"{k}: {'MET' if v else 'not met'}" for k, v in crit.items()))
    print(f"\nTotal spend this task: ${spent:.4f} (cap ${args.cap:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
