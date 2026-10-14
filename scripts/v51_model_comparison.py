"""Design v5.1 model comparison (protocol Amendment A2): Systems 3-6 x models x 3 scenarios, tuning seed 102.

Modes:
  --probe     one real decision-sized call per model (System 6 v5.1 prompt) to measure output tokens; tiny cost.
  (default)   dry run: projected cost per model from prompt sizes (MockLLM) and measured/assumed output tokens.
  --execute   paid runs (needs ALLOW_PAID_CALLS=1). Shared hard stop on cumulative USD; abort on 401/402/403;
              an invalid run (fallbacks > 5% or LLM calls < decisions) halts everything.
Resumable: finished runs are appended to results/v51_model_runs.csv; decisions go to results/v51_models.duckdb.
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
from sca.systems.v5 import ACTIONS, ActionDecision, summarize_decisions  # noqa: E402

ROOT = find_project_root()
MODELS = ["glm-isoquant", "deepseek-v4.1-flash", "gemini-3.8-flash"]
SYSTEMS = {"S3 single agent": "single_agent_v5", "S4 multi-agent info": "multi_agent_info_v5",
           "S5 multi-agent tools": "multi_agent_tools_v5", "S6 orchestrated": "orchestrated_v5"}
SCENARIOS = ["stable", "seasonal", "shock"]
SEED = 102
RUNS_CSV = ROOT / "results/v51_model_runs.csv"
DB_PATH = ROOT / "results/v51_models.duckdb"
PROBE_JSON = ROOT / "results/v51_probe.json"
SC = generate_scenarios()
FROZEN = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))["tuned_policies"]
_lock = threading.Lock()
TEXT_COLUMNS = ["run_id", "system", "prompt_version", "chooser", "echelon", "action", "llm_regime", "detector_regime",
                "detector_phase", "error_type", "error_message", "rationale", "tool_mode", "tools_requested",
                "tool_error", "model", "scenario"]


class Budget:
    """Spend shared by every model's client."""

    def __init__(self, cap: float, spent: float):
        self.cap, self.spent, self.lock = cap, spent, threading.Lock()


class BudgetExceeded(PaidCallsDisabledError):
    pass


class GuardedClient(LLMClient):
    """One model's client: reserves each call's worst case against the shared budget before sending it."""

    def __init__(self, model: str, budget: Budget, **kw: Any):
        super().__init__(**kw)
        self.model, self.budget = model, budget
        entry = self.config.get_model(model)
        self.max_tokens = int(entry.max_tokens)
        self._rate_limiters[entry.name] = RateLimiter(entry.requests_per_minute)
        self._concurrency_semaphores[entry.name] = threading.BoundedSemaphore(entry.max_concurrency)

    def generate(self, prompt: str, **kw: Any):
        worst = 2 * calculate_cost(self.model, estimate_tokens(prompt), self.max_tokens, self.config)
        b = self.budget
        with b.lock:
            if b.spent + worst > b.cap:
                raise BudgetExceeded(f"Hard stop: spent ${b.spent:.4f} + worst ${worst:.4f} > cap ${b.cap:.2f}")
            b.spent += worst
        try:
            resp = super().generate(prompt, **kw)
        except Exception as exc:
            actual = calculate_cost(self.model, getattr(exc, "prompt_tokens", 0), getattr(exc, "completion_tokens", 0),
                                    self.config, cached_tokens=getattr(exc, "cached_tokens", 0))
            with b.lock:
                b.spent += actual - worst
            raise
        with b.lock:
            b.spent += (0.0 if resp.cached else float(resp.cost_usd)) - worst
        return resp


def simulate(system, scenario: str):
    env = BeerGameEnv(config=get_config().sim, scenario=SC[scenario], seed=SEED, steady_state_init=True)
    states = env.reset(seed=SEED)
    if hasattr(system, "attach_env"):
        system.attach_env(env)
    t = 0.0
    for _ in range(40):
        t0 = time.perf_counter()
        orders = system.decide_orders(states)
        t += time.perf_counter() - t0
        states = env.step(orders)
    return compute_simulation_metrics(env, warmup_steps=5), 1000 * t / 160


def metrics_row(model, label, scenario, m, lat, system=None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model": model, "system": label, "scenario": scenario, "seed": SEED, "total_cost": m.total_cost,
        "fill_rate": 100 * m.fill_rate["retailer"], "e2e_bullwhip": m.end_to_end_bullwhip,
        **{f"bw_{e}": m.bullwhip_ratio[e] for e in ECHELONS},
        "stockout_steps": int(m.stockout_steps["retailer"]), "latency_ms": lat,
    }
    if system is not None and getattr(system, "chooser", None) == "llm":
        d = pd.DataFrame(system.decision_log)
        s = summarize_decisions(system.decision_log)
        stats = system.llm_stats()
        quoted = d.rationale_numbers.sum()
        tools = d[d.tool_choice_correct.notna()]
        neg = d[d.negotiation_rounds.notna()]
        row.update({
            "prompt_version": system.prompt_version, "decisions": stats["decisions"], "fallbacks": stats["fallbacks"],
            "llm_calls": stats["llm_calls"], "fallback_rate": s["fallback_rate"],
            "error_rate": float(d.error_type.notna().mean()),
            "instruction_following": float(d.first_pass_valid.mean()),
            "groundedness": float(d.rationale_numbers_grounded.sum() / quoted) if quoted else np.nan,
            "rationales_with_numbers": float((d.rationale_numbers > 0).mean()),
            "tool_choice_accuracy": float(tools.tool_choice_correct.astype(float).mean()) if len(tools) else np.nan,
            "parameter_accuracy": (float(tools.params_valid.sum() / tools.params_total.sum())
                                   if len(tools) and tools.params_total.sum() else np.nan),
            "tool_errors": int(tools.tool_error.notna().sum()) if len(tools) else 0,
            "negotiation_rounds": float(neg.negotiation_rounds.mean()) if len(neg) else np.nan,
            "negotiation_messages": float(neg.negotiation_messages.mean()) if len(neg) else np.nan,
            "tokens_in": float(d.prompt_tokens.mean()), "tokens_out": float(d.completion_tokens.mean()),
            "usd": float(d.cost_usd.sum()), "llm_latency_ms": float(d.latency_ms.mean()),
            "clip_rate": s["clip_rate"], "regime_accuracy": s["regime_accuracy"],
            **{f"mix_{a}": s["action_mix"][a] for a in (*ACTIONS, "fallback")},
            "invalid_reason": check_run_validity(stats),
        })
    return row


def append(row):
    """Add one run, keeping a single header that covers every column (deterministic rows have fewer)."""
    with _lock:
        new = pd.DataFrame([row])
        df = pd.concat([pd.read_csv(RUNS_CSV), new], ignore_index=True) if RUNS_CSV.exists() else new
        df.to_csv(RUNS_CSV, index=False)


def persist(system, model, scenario):
    if not system.decision_log:
        return
    df = pd.DataFrame(system.decision_log)
    df["model"], df["scenario"], df["seed"] = model, scenario, SEED
    # Text columns can be all-empty in some systems (e.g. no tools in S3); fix their type so DuckDB never infers numbers
    for c in TEXT_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("string")
    with _lock:
        con = duckdb.connect(str(DB_PATH))
        con.register("df", df)
        if "v5_decisions" in [r[0] for r in con.execute("SHOW TABLES").fetchall()]:
            con.execute("INSERT INTO v5_decisions BY NAME SELECT * FROM df")
        else:
            con.execute("CREATE TABLE v5_decisions AS SELECT * FROM df")
        con.close()


def make_system(sid, model, client, scenario):
    cfg = {"db_path": False, "seed": SEED, "model": model, "tool_mode": "llm_tools",
           "run_id": f"v51_{model}_{sid}_{scenario}_seed{SEED}"}
    system = get_system(sid, config=cfg, llm_client=client) if client else \
        get_system(sid, config={**cfg, "mock_v5_mode": "detector_copy", "model": None})
    system.calibrate(SC[scenario])
    system.reset(SC[scenario])
    return system


def prompt_sizes() -> Dict[str, Dict[str, float]]:
    """Calls per run and mean prompt tokens per call, per system (MockLLM, shock)."""
    out = {}
    for label, sid in SYSTEMS.items():
        s = make_system(sid, None, None, "shock")
        simulate(s, "shock")
        p = [estimate_tokens(c["prompt"]) for c in s.llm_client.mock.call_history]
        out[label] = {"calls": len(p), "prompt_tokens": float(np.mean(p))}
    return out


def probe(budget: Budget) -> Dict[str, Any]:
    s = make_system("orchestrated_v5", None, None, "shock")
    env = BeerGameEnv(config=get_config().sim, scenario=SC["shock"], seed=SEED, steady_state_init=True)
    s.decide_orders(env.reset(seed=SEED))
    prompt = [c["prompt"] for c in s.llm_client.mock.call_history if c["response_model"] == "ActionDecision"][0]
    res = {}
    for model in MODELS:
        client = GuardedClient(model, budget, mock_mode=False, cache_enabled=False, db_path=None)
        t0 = time.time()
        try:
            r = client.generate(prompt, model=model, response_model=ActionDecision, seed=SEED, log_to_db=False,
                                use_cache=False)
            res[model] = {"ok": r.structured is not None, "prompt_tokens": r.prompt_tokens,
                          "completion_tokens": r.completion_tokens, "usd": r.cost_usd, "seconds": round(time.time() - t0, 1),
                          "answer": r.structured.model_dump() if r.structured else r.content[:200]}
        except (FatalAPIError, BudgetExceeded):
            raise
        except Exception as exc:
            res[model] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                          "completion_tokens": getattr(exc, "completion_tokens", None)}
        print(model, res[model], flush=True)
    PROBE_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def dry_run(prior: float, cap: float) -> Dict[str, float]:
    models = load_models_config()
    sizes = prompt_sizes()
    measured = json.loads(PROBE_JSON.read_text(encoding="utf-8")) if PROBE_JSON.exists() else {}
    total_exp = total_worst = 0.0
    per_model = {}
    print("Calls per run and prompt tokens per call:", {k: (v["calls"], round(v["prompt_tokens"])) for k, v in sizes.items()})
    for model in MODELS:
        e = models.get_model(model)
        out_tok = (measured.get(model) or {}).get("completion_tokens") or 150
        exp = worst = 0.0
        for v in sizes.values():
            exp += len(SCENARIOS) * v["calls"] * calculate_cost(model, v["prompt_tokens"], out_tok, models)
            worst += len(SCENARIOS) * v["calls"] * calculate_cost(model, v["prompt_tokens"], e.max_tokens, models)
        per_model[model] = exp
        total_exp += exp
        total_worst += worst
        src = "measured" if (measured.get(model) or {}).get("completion_tokens") else "assumed"
        print(f"  {model:20s} output tokens/call {out_tok} ({src}): expected ${exp:.3f}, worst ${worst:.2f}")
    print(f"Total expected ${total_exp:.3f} (worst ${total_worst:.2f}); prior spend ${prior:.4f}; cap ${cap:.2f}")
    return per_model


def run_job(model, label, sid, scenario, client):
    system = make_system(sid, model, client, scenario)
    m, lat = simulate(system, scenario)
    row = metrics_row(model, label, scenario, m, lat, system)
    append(row)  # the run's results are saved first, so a decision-log error cannot lose a paid run
    try:
        persist(system, model, scenario)
    except Exception as exc:
        print(f"WARNING: decision log not saved for {model} {label} {scenario}: {type(exc).__name__}: {exc}", flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--prior-spend", type=float, default=0.0)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--workers", type=int, default=0, help="parallel runs (default: 2 per model)")
    args = ap.parse_args()
    budget = Budget(args.cap, args.prior_spend)

    if args.probe:
        probe(budget)
        print(f"Probe spend ${budget.spent - args.prior_spend:.5f}")
        return 0
    per_model = dry_run(args.prior_spend, args.cap)
    chosen = [m for m in args.models if m in MODELS]
    exp = sum(per_model[m] for m in chosen)
    if args.prior_spend + exp > args.cap:
        print(f"ABORT: expected spend for {chosen} (${exp:.3f}) + prior ${args.prior_spend:.3f} exceeds cap ${args.cap:.2f}")
        return 2

    done = set()
    if RUNS_CSV.exists():
        d = pd.read_csv(RUNS_CSV)
        done = set(zip(d.model, d.system, d.scenario))
    for scenario in SCENARIOS:  # deterministic references (free)
        for label, sid, params in (("S1 (s,S) tuned", "baseline_s_S", FROZEN["system_1_s_S"]["parameters"]),
                                   ("S2 EOQ tuned", "baseline_eoq", FROZEN["system_2_eoq"]["parameters"]),
                                   ("ref 2c tuned", "baseline_2c", FROZEN["system_2c_adaptive_smoothed_out"]["parameters"])):
            if ("deterministic", label, scenario) in done:
                continue
            cfg = {**params, "db_path": False}
            s = get_system(sid, config=cfg)
            s.calibrate(SC[scenario], config=cfg)
            s.reset(SC[scenario])
            m, lat = simulate(s, scenario)
            append(metrics_row("deterministic", label, scenario, m, lat))
    if not args.execute:
        return 0

    clients = {m: GuardedClient(m, budget, mock_mode=False, cache_enabled=False, db_path=None) for m in chosen}
    jobs = [(m, label, sid, sc) for sc in SCENARIOS for label, sid in SYSTEMS.items() for m in chosen
            if (m, label, sc) not in done]
    print(f"Runs to do: {len(jobs)}", flush=True)
    halt: Optional[str] = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers or 2 * len(chosen)) as ex:
        futs = {ex.submit(run_job, *job, clients[job[0]]): job for job in jobs}
        for fut in concurrent.futures.as_completed(futs):
            model, label, _sid, sc = futs[fut]
            try:
                row = fut.result()
            except (FatalAPIError, PaidCallsDisabledError) as exc:
                halt = f"{model} {label} {sc}: {type(exc).__name__}: {exc}"
            except Exception as exc:
                halt = f"{model} {label} {sc}: unexpected {type(exc).__name__}: {exc}"
            else:
                print(f"[{time.strftime('%H:%M:%S')}] {model:20s} {label:22s} {sc:8s} cost {row['total_cost']:.0f} "
                      f"FR {row['fill_rate']:.1f} BW {row['e2e_bullwhip']:.2f} fb {row['fallbacks']} "
                      f"usd {row['usd']:.4f} | spent ${budget.spent:.4f}", flush=True)
                if row["invalid_reason"]:
                    halt = f"{model} {label} {sc} INVALID: {row['invalid_reason']}"
            if halt:
                print(f"HALT: {halt}", flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                break
    print(f"Total spend ${budget.spent:.4f} (cap ${args.cap:.2f})")
    return 1 if halt else 0


if __name__ == "__main__":
    sys.exit(main())
