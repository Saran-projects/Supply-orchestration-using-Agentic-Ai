"""Final evaluation, Systems 1-6 (docs/v5_final_eval_plan.md): 3 scenarios x test seeds 0-7.

- Systems 1-2 (and references 2c, scripted always-follow) are deterministic: no network.
- Systems 3-6 use glm-isoquant through one shared, thread-safe GuardedClient (one RPM limiter for the batch).
  A real run needs --execute and ALLOW_PAID_CALLS=1; without --execute only the dry-run projection is printed.
- Hard stop on cumulative USD (--cap, including --prior-spend); abort on HTTP 401/402/403.
- An invalid run (fallbacks > 5% or LLM calls < decisions) halts the batch.
- Resumable: finished runs are appended to results/v5_final_runs.csv and skipped on restart.
  Decision rows go to results/v5_final.duckdb (table v5_decisions), written under a lock.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

from sca.config import find_project_root, get_config, load_models_config  # noqa: E402
from sca.data.scenarios import generate_scenarios  # noqa: E402
from sca.eval.validity import check_run_validity  # noqa: E402
from sca.llm.client import (  # noqa: E402
    FatalAPIError,
    LLMClient,
    PaidCallsDisabledError,
    RateLimiter,
    calculate_cost,
    estimate_tokens,
)
from sca.sim.environment import ECHELONS, BeerGameEnv  # noqa: E402
from sca.sim.metrics import compute_simulation_metrics  # noqa: E402
from sca.systems.base import get_system  # noqa: E402
from sca.systems.v5 import ACTIONS, summarize_decisions  # noqa: E402

ROOT = find_project_root()
MODEL = "glm-isoquant"
SCENARIOS = ["stable", "seasonal", "shock"]
SEEDS = list(range(8))
HORIZON, WARMUP = 40, 5
RUNS_CSV = ROOT / "results/v5_final_runs.csv"
DB_PATH = ROOT / "results/v5_final.duckdb"
FROZEN = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))["tuned_policies"]
DETERMINISTIC = {
    "S1 (s,S) tuned": ("baseline_s_S", FROZEN["system_1_s_S"]["parameters"]),
    "S2 EOQ tuned": ("baseline_eoq", FROZEN["system_2_eoq"]["parameters"]),
    "ref 2c tuned": ("baseline_2c", FROZEN["system_2c_adaptive_smoothed_out"]["parameters"]),
}
SCRIPTED = {"ref follow [basic]": "single_agent_v5", "ref follow [tool]": "multi_agent_tools_v5"}
LLM_SYSTEMS = {
    "S3 single agent": "single_agent_v5",
    "S4 multi-agent info": "multi_agent_info_v5",
    "S5 multi-agent tools": "multi_agent_tools_v5",
    "S6 orchestrated": "orchestrated_v5",
}
SC = generate_scenarios()
_csv_lock = threading.Lock()
_db_lock = threading.Lock()


class BudgetExceeded(PaidCallsDisabledError):
    """Raised before a call whose worst case would push spend past the cap."""


class GuardedClient(LLMClient):
    """Shared client: tracks spend under a lock and refuses calls whose worst case would exceed the cap."""

    def __init__(self, cap: float, spent: float, **kw: Any):
        super().__init__(**kw)
        self.cap, self.spent = cap, spent
        self.lock = threading.Lock()
        entry = self.config.get_model(MODEL)
        self.max_tokens = int(entry.max_tokens)
        # Create the limiter/semaphore up front so concurrent first calls share them
        self._rate_limiters[entry.name] = RateLimiter(entry.requests_per_minute)
        self._concurrency_semaphores[entry.name] = threading.BoundedSemaphore(entry.max_concurrency)

    def generate(self, prompt: str, **kw: Any):
        worst = 2 * calculate_cost(MODEL, estimate_tokens(prompt), self.max_tokens, self.config)
        with self.lock:
            if self.spent + worst > self.cap:
                raise BudgetExceeded(f"Hard stop: spent ${self.spent:.4f} + worst case ${worst:.4f} > cap ${self.cap:.2f}")
            self.spent += worst  # reserve, settled below
        try:
            resp = super().generate(prompt, **kw)
        except Exception as exc:
            actual = calculate_cost(MODEL, getattr(exc, "prompt_tokens", 0), getattr(exc, "completion_tokens", 0),
                                    self.config, cached_tokens=getattr(exc, "cached_tokens", 0))
            with self.lock:
                self.spent += actual - worst
            raise
        with self.lock:
            self.spent += (0.0 if resp.cached else float(resp.cost_usd)) - worst
        return resp


def simulate(system, scenario: str, seed: int):
    sc = SC[scenario]
    env = BeerGameEnv(config=get_config().sim, scenario=sc, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)
    if hasattr(system, "attach_env"):
        system.attach_env(env)
    t_decide = 0.0
    for _ in range(HORIZON):
        t0 = time.perf_counter()
        orders = system.decide_orders(states)
        t_decide += time.perf_counter() - t0
        states = env.step(orders)
    m = compute_simulation_metrics(env, warmup_steps=WARMUP)
    return m, 1000.0 * t_decide / (HORIZON * len(ECHELONS))


def metrics_row(label: str, scenario: str, seed: int, m, latency_ms: float, system=None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "system": label, "scenario": scenario, "seed": seed,
        "total_cost": m.total_cost,
        "holding_cost": float(sum(m.holding_cost_by_echelon.values())),
        "backlog_cost": float(sum(m.backlog_cost_by_echelon.values())),
        "fill_rate": 100.0 * m.fill_rate["retailer"],
        "e2e_bullwhip": m.end_to_end_bullwhip,
        **{f"bw_{e}": m.bullwhip_ratio[e] for e in ECHELONS},
        "stockout_steps_retailer": m.stockout_steps["retailer"],
        "stockout_steps_total": int(sum(m.stockout_steps.values())),
        "latency_ms": latency_ms,
    }
    if system is not None and getattr(system, "chooser", None) == "llm":
        rows = system.decision_log
        s = summarize_decisions(rows)
        stats = system.llm_stats()
        quoted = sum(r["rationale_numbers"] for r in rows)
        row.update({
            "decisions": stats["decisions"], "fallbacks": stats["fallbacks"], "llm_calls": stats["llm_calls"],
            "fallback_rate": s["fallback_rate"],
            "error_rate": float(np.mean([r["error_type"] is not None for r in rows])),
            "instruction_following": float(np.mean([r["first_pass_valid"] for r in rows])),
            "groundedness": (sum(r["rationale_numbers_grounded"] for r in rows) / quoted) if quoted else np.nan,
            "rationales_with_numbers": float(np.mean([r["rationale_numbers"] > 0 for r in rows])),
            "tokens_in": float(np.mean([r["prompt_tokens"] for r in rows])),
            "tokens_out": float(np.mean([r["completion_tokens"] for r in rows])),
            "usd": float(sum(r["cost_usd"] for r in rows)),
            "clip_rate": s["clip_rate"], "regime_accuracy": s["regime_accuracy"],
            **{f"mix_{a}": s["action_mix"][a] for a in (*ACTIONS, "fallback")},
            "llm_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
            "invalid_reason": check_run_validity(stats),
        })
    return row


def append_row(row: Dict[str, Any]) -> None:
    with _csv_lock:
        df = pd.DataFrame([row])
        df.to_csv(RUNS_CSV, mode="a", header=not RUNS_CSV.exists(), index=False)


def persist_decisions(rows: List[Dict[str, Any]], scenario: str, seed: int) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["scenario"], df["seed"] = scenario, seed
    with _db_lock:
        con = duckdb.connect(str(DB_PATH))
        con.register("df", df)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if "v5_decisions" in tables:
            con.execute("INSERT INTO v5_decisions BY NAME SELECT * FROM df")
        else:
            con.execute("CREATE TABLE v5_decisions AS SELECT * FROM df")
        con.close()


def run_llm(label: str, sid: str, scenario: str, seed: int, client: GuardedClient) -> Dict[str, Any]:
    run_id = f"final_{sid}_{scenario}_seed{seed}"
    system = get_system(sid, config={"db_path": False, "seed": seed, "model": MODEL, "run_id": run_id},
                        llm_client=client)
    system.calibrate(SC[scenario])
    system.reset(SC[scenario])
    m, latency = simulate(system, scenario, seed)
    persist_decisions(system.decision_log, scenario, seed)
    row = metrics_row(label, scenario, seed, m, latency, system)
    append_row(row)
    return row


def done_runs() -> set:
    if not RUNS_CSV.exists():
        return set()
    df = pd.read_csv(RUNS_CSV)
    return set(zip(df.system, df.scenario, df.seed))


def dry_run(prior: float, cap: float) -> float:
    models = load_models_config()
    entry = models.get_model(MODEL)
    per_run = {}
    for label, sid in LLM_SYSTEMS.items():
        system = get_system(sid, config={"db_path": False, "mock_v5_mode": "always_follow"})
        system.calibrate(SC["shock"])
        system.reset(SC["shock"])
        simulate(system, "shock", 0)
        p = [estimate_tokens(c["prompt"]) for c in system.llm_client.mock.call_history]
        per_run[label] = (sum(calculate_cost(MODEL, t, 85, models) for t in p),
                          sum(calculate_cost(MODEL, t, entry.max_tokens, models) for t in p))
    n_runs = len(SCENARIOS) * len(SEEDS)
    expected = n_runs * sum(v[0] for v in per_run.values())
    worst = n_runs * sum(v[1] for v in per_run.values())
    for label, (e, w) in per_run.items():
        print(f"  {label}: per run expected ${e:.4f} (85 output tokens/call, as measured on seed 102), worst ${w:.4f}")
    print(f"Projected for {n_runs} runs x 4 systems: expected ${expected:.3f}, worst case (no retries) ${worst:.3f}")
    print(f"Prior spend ${prior:.3f}; cap ${cap:.2f}; the guard stops before any call that could cross the cap.")
    calls = n_runs * 4 * HORIZON * len(ECHELONS)
    rpm = entry.requests_per_minute
    print(f"{calls} LLM calls at {rpm} RPM: at least {calls / rpm / 60:.1f} h")
    return expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--cap", type=float, default=1.20)
    ap.add_argument("--prior-spend", type=float, default=0.140)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    expected = dry_run(args.prior_spend, args.cap)
    if args.prior_spend + expected > args.cap:
        print("ABORT: expected spend exceeds cap")
        return 2

    done = done_runs()
    # Deterministic systems and references (no network)
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for label, (sid, params) in DETERMINISTIC.items():
                if (label, scenario, seed) in done:
                    continue
                cfg = {**params, "db_path": False}
                system = get_system(sid, config=cfg)
                system.calibrate(SC[scenario], config=cfg)
                system.reset(SC[scenario])
                m, lat = simulate(system, scenario, seed)
                append_row(metrics_row(label, scenario, seed, m, lat))
            for label, sid in SCRIPTED.items():
                if (label, scenario, seed) in done:
                    continue
                system = get_system(sid, config={"db_path": False, "chooser": "follow"})
                system.calibrate(SC[scenario])
                system.reset(SC[scenario])
                m, lat = simulate(system, scenario, seed)
                append_row(metrics_row(label, scenario, seed, m, lat))
    print("Deterministic systems done.")
    if not args.execute:
        return 0

    client = GuardedClient(cap=args.cap, spent=args.prior_spend, mock_mode=False, cache_enabled=False,
                           db_path=None)
    jobs = [(label, sid, sc, seed) for seed in SEEDS for sc in SCENARIOS for label, sid in LLM_SYSTEMS.items()
            if (label, sc, seed) not in done]
    print(f"LLM runs to do: {len(jobs)}", flush=True)
    halt: Optional[str] = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_llm, *job, client): job for job in jobs}
        for fut in concurrent.futures.as_completed(futures):
            label, sid, sc, seed = futures[fut]
            try:
                row = fut.result()
            except (FatalAPIError, PaidCallsDisabledError) as exc:
                halt = f"{label} {sc} seed {seed}: {type(exc).__name__}: {exc}"
            except Exception as exc:
                halt = f"{label} {sc} seed {seed}: unexpected {type(exc).__name__}: {exc}"
            else:
                print(f"[{time.strftime('%H:%M:%S')}] {label:22s} {sc:8s} seed {seed}: cost {row['total_cost']:.0f} "
                      f"FR {row['fill_rate']:.1f} BW {row['e2e_bullwhip']:.2f} fallbacks {row['fallbacks']} "
                      f"| spent ${client.spent:.4f}", flush=True)
                if row["invalid_reason"]:
                    halt = f"{label} {sc} seed {seed} INVALID: {row['invalid_reason']}"
            if halt:
                print(f"HALT: {halt}", flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                break
    print(f"Total spend: ${client.spent:.4f} (cap ${args.cap:.2f})")
    return 1 if halt else 0


if __name__ == "__main__":
    sys.exit(main())
