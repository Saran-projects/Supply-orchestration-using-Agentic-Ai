"""Dashboard datasets from data that already exists (no LLM calls, no network).

- Seed-102 shock comparison: S1, S2 and references computed here (deterministic); S3/S6 real glm v5 results parsed
  from results/v5_paid_run_seed102_attempt3_effort_low.log (the only valid real-LLM v5 run). S4/S5: never run.
- Agent behaviour, v5 (same log) and earlier designs v3.1-v4.1 (results/glm_experiment.duckdb).
- Traditional baselines on test seeds 0-7 (results/v5_final_runs.csv, deterministic part).
Writes results/dashboard/existing/*.json.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time

import duckdb
import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

from sca.config import find_project_root, get_config  # noqa: E402
from sca.data.scenarios import generate_scenarios  # noqa: E402
from sca.sim.environment import ECHELONS, BeerGameEnv  # noqa: E402
from sca.sim.metrics import compute_simulation_metrics  # noqa: E402
from sca.systems.base import get_system  # noqa: E402

ROOT = find_project_root()
OUT = ROOT / "results/dashboard/existing"
LOG = ROOT / "results/v5_paid_run_seed102_attempt3_effort_low.log"
FROZEN = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))["tuned_policies"]


def simulate(system, sc, seed):
    env = BeerGameEnv(config=get_config().sim, scenario=sc, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)
    t = 0.0
    for _ in range(40):
        t0 = time.perf_counter()
        orders = system.decide_orders(states)
        t += time.perf_counter() - t0
        states = env.step(orders)
    return compute_simulation_metrics(env, warmup_steps=5), 1000 * t / 160


def det_row(name, kind, sid, params, sc, seed):
    cfg = {**params, "db_path": False}
    s = get_system(sid, config=cfg)
    s.calibrate(sc, config=cfg)
    s.reset(sc)
    m, lat = simulate(s, sc, seed)
    return {
        "system": name, "kind": kind, "status": "Ran (deterministic)",
        "total_cost": round(m.total_cost), "fill_rate": round(m.fill_rate["retailer"], 4),
        "e2e_bullwhip": round(m.end_to_end_bullwhip, 2),
        **{f"bw_{e}": round(m.bullwhip_ratio[e], 2) for e in ECHELONS},
        "stockout_steps": int(m.stockout_steps["retailer"]), "latency_ms": round(lat, 2),
    }


def parse_log():
    """Rows of the final table in the attempt-3 log, keyed by policy label."""
    lines = LOG.read_text(encoding="utf-8").splitlines()
    head = next(i for i, l in enumerate(lines) if l.strip().startswith("cost") and "fill_rate" in l)
    cols = lines[head].split()
    rows = {}
    for l in lines[head + 2:]:
        if not l.strip():
            break
        vals = l.split()[-len(cols):]
        label = l[: len(l) - len(" ".join(vals))].strip()
        label = re.sub(r"\s+", " ", l.split("  ")[0]).strip()
        rows[label] = dict(zip(cols, vals))
    sysline = {k: re.search(r"tokens/decision in (\d+) out (\d+) \| USD ([\d.]+)", l)
               for k, l in (("System 3", next(x for x in lines if x.startswith("System 3:"))),
                            ("System 6", next(x for x in lines if x.startswith("System 6:"))))}
    mix = {}
    for name in ("System 3", "System 6"):
        i = lines.index(f"{name} action mix per echelon:")
        for l in lines[i + 3:i + 7]:
            p = l.split()
            mix.setdefault(name, []).append({"echelon": p[0], "follow": float(p[1]), "boost_small": float(p[2]),
                                             "boost_large": float(p[3]), "trim": float(p[4]), "fallback": float(p[5])})
    return rows, sysline, mix


def f(x):
    return None if x in (None, "NaN", "nan") else float(x)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sc = generate_scenarios()["shock"]
    seed = 102
    rows, sysline, mix = parse_log()

    compare = [
        det_row("S1 (s,S) policy", "Traditional", "baseline_s_S", FROZEN["system_1_s_S"]["parameters"], sc, seed),
        det_row("S2 EOQ", "Traditional", "baseline_eoq", FROZEN["system_2_eoq"]["parameters"], sc, seed),
    ]
    for name, key in (("S3 Single LLM agent", "System 3 (glm, v5.0)"), ("S6 Orchestrated (proposed)", "System 6 (glm, v5.0)")):
        r = rows[key]
        compare.append({
            "system": name, "kind": "LLM", "status": "Ran (glm-5.3-flash, v5.0)",
            "total_cost": round(f(r["cost"])), "fill_rate": round(f(r["fill_rate"]) / 100, 4),
            "e2e_bullwhip": round(f(r["e2e_bw"]), 2),
            "bw_retailer": f(r["bw_reta"]), "bw_wholesaler": f(r["bw_whol"]),
            "bw_distributor": f(r["bw_dist"]), "bw_manufacturer": f(r["bw_manu"]),
            "stockout_steps": None, "latency_ms": None,
        })
    for name in ("S4 Multi-agent + info sharing", "S5 Multi-agent + tools"):
        compare.append({"system": name, "kind": "LLM", "status": "Not run with a real LLM",
                        "total_cost": None, "fill_rate": None, "e2e_bullwhip": None,
                        **{f"bw_{e}": None for e in ECHELONS}, "stockout_steps": None, "latency_ms": None})
    compare.append(det_row("Ref: 2c adaptive heuristic", "Reference", "baseline_2c",
                           FROZEN["system_2c_adaptive_smoothed_out"]["parameters"], sc, seed))
    s = get_system("multi_agent_tools_v5", config={"db_path": False, "chooser": "follow"})
    s.calibrate(sc)
    s.reset(sc)
    m, lat = simulate(s, sc, seed)
    compare.append({"system": "Ref: always-follow (S6 floor)", "kind": "Reference", "status": "Ran (scripted)",
                    "total_cost": round(m.total_cost), "fill_rate": round(m.fill_rate["retailer"], 4),
                    "e2e_bullwhip": round(m.end_to_end_bullwhip, 2),
                    **{f"bw_{e}": round(m.bullwhip_ratio[e], 2) for e in ECHELONS},
                    "stockout_steps": int(m.stockout_steps["retailer"]), "latency_ms": round(lat, 2)})
    order = ["S1 (s,S) policy", "S2 EOQ", "S3 Single LLM agent", "S4 Multi-agent + info sharing",
             "S5 Multi-agent + tools", "S6 Orchestrated (proposed)", "Ref: 2c adaptive heuristic",
             "Ref: always-follow (S6 floor)"]
    compare.sort(key=lambda r: order.index(r["system"]))

    bw = [{"system": r["system"], "echelon": e, "bullwhip": r[f"bw_{e}"]}
          for r in compare if r["e2e_bullwhip"] is not None for e in ECHELONS]

    behaviour = []
    for name, key, sl in (("S3 Single LLM agent", "System 3 (glm, v5.0)", "System 3"),
                          ("S6 Orchestrated (proposed)", "System 6 (glm, v5.0)", "System 6")):
        r = rows[key]
        tin, tout, usd = sysline[sl].groups()
        behaviour.append({
            "system": name, "decisions": 160, "llm_calls": 160,
            "instruction_following": f(r["first_pass_valid"]), "fallback_rate": f(r["fallback_rate"]),
            "error_rate": f(r["fallback_rate"]), "regime_accuracy": f(r["regime_acc"]),
            "follow_rate": f(r["follow_rate"]), "clip_rate": f(r["clip_rate"]),
            "tokens_in": int(tin), "tokens_out": int(tout), "usd_per_run": float(usd),
            "groundedness": None, "latency_ms": None,
            "tool_choice_accuracy": "n/a: tools are deterministic",
            "parameter_accuracy": "n/a: tools are deterministic",
            "negotiation": "n/a: 1 orchestrator call, 0 rounds" if sl == "System 6" else "n/a: no negotiation",
        })
    mix_rows = [{"system": "S3 Single LLM agent" if k == "System 3" else "S6 Orchestrated (proposed)", **m}
                for k, v in mix.items() for m in v]

    con = duckdb.connect(str(ROOT / "results/glm_experiment.duckdb"), read_only=True)
    calls = con.execute("""
        select run_id, prompt_version, count(*) calls, avg(latency_ms) latency_ms, avg(prompt_tokens) tokens_in,
               avg(completion_tokens) tokens_out, sum(cost_usd) usd, avg(cast(structured_output_ok as int)) valid_rate,
               sum(cast(fallback as int)) fallbacks
        from agent_calls where run_id like 's%_glm_shock_seed_10%' group by 1, 2""").df()
    neg = con.execute("""
        select run_id, avg(rounds_used) rounds, avg(messages_exchanged) messages, count(*) decisions
        from orchestrated_decisions where run_id like 's6_glm_shock_seed_10%' group by 1""").df()
    con.close()
    earlier = calls.merge(neg, on="run_id", how="left")
    earlier["system"] = np.where(earlier.run_id.str.startswith("s3"), "S3 Single LLM agent", "S6 Orchestrated")
    earlier["seed"] = earlier.run_id.str.extract(r"seed_(\d+)")[0].astype(int)
    earlier["note"] = np.where(earlier.prompt_version == "v4.1", "Invalid: every call truncated", "Valid")
    earlier = earlier.sort_values(["system", "prompt_version"])
    earlier_rows = [{k: (None if (isinstance(v, float) and np.isnan(v)) else (round(v, 4) if isinstance(v, float) else v))
                     for k, v in rec.items()} for rec in earlier.drop(columns=["run_id"]).to_dict("records")]
    for rec in earlier_rows:
        rec["key"] = f"{rec['system']}|{rec['prompt_version']}"

    test = pd.read_csv(ROOT / "results/v5_final_runs.csv")
    names = {"S1 (s,S) tuned": "S1 (s,S) policy", "S2 EOQ tuned": "S2 EOQ", "ref 2c tuned": "Ref: 2c adaptive heuristic",
             "ref follow [tool]": "Ref: always-follow (S6 floor)"}
    test = test[test.system.isin(names)].drop_duplicates(["system", "scenario", "seed"])
    tagg = test.groupby(["system", "scenario"]).agg(total_cost=("total_cost", "mean"), fill_rate=("fill_rate", "mean"),
                                                   e2e_bullwhip=("e2e_bullwhip", "mean"),
                                                   stockout_steps=("stockout_steps_retailer", "mean"),
                                                   runs=("seed", "count")).reset_index()
    tagg["system"] = tagg.system.map(names)
    tagg["fill_rate"] = (tagg.fill_rate / 100).round(4)
    tagg = tagg.round({"total_cost": 0, "e2e_bullwhip": 2, "stockout_steps": 1})
    tagg["key"] = tagg.system + "|" + tagg.scenario

    for name, data in (("compare_seed102", compare), ("bullwhip_echelon_seed102", bw), ("behaviour_v5", behaviour),
                       ("action_mix_v5", mix_rows), ("behaviour_earlier", earlier_rows),
                       ("baselines_test_seeds", tagg.to_dict("records"))):
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(json.dumps(compare, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
