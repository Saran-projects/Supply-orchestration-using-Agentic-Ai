"""Design v5: scripted policies (no LLM) and multiplier tuning on tuning seeds 100-104.

Selection follows docs/tuning_rule_v2_1.md:
  feasible  = mean retailer fill rate >= 99% and mean end-to-end bullwhip <= 0.75 x untuned-EOQ ceiling
  objective = argmin mean total cost over feasible candidates
  fallback  = if none feasible: lowest bullwhip among FR >= 99%; if none reach 99%: highest FR (caveat)
  interior  = a chosen value on a grid edge is extended outward until interior (or a hard limit).
Multipliers are shared by Systems 3-6, so each candidate is scored on the detector-mapped policy
pooled over both reference families (basic: S3/S4, tool: S5/S6) x 3 scenarios x 5 seeds = 30 runs.

Usage:
  python scripts/v5_scripted_tuning.py            # report + tune, no config change
  python scripts/v5_scripted_tuning.py --freeze   # also write the chosen multipliers to configs/agent_v5.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

logging.disable(logging.WARNING)

from sca.config import find_project_root, get_config, load_agent_v5_config  # noqa: E402
from sca.data.scenarios import generate_scenarios  # noqa: E402
from sca.sim.environment import BeerGameEnv  # noqa: E402
from sca.sim.metrics import compute_simulation_metrics  # noqa: E402
from sca.systems.base import get_system  # noqa: E402

TUNING_SEEDS = [100, 101, 102, 103, 104]
SCENARIOS = ["stable", "seasonal", "shock"]
FAMILIES = {"basic": "single_agent_v5", "tool": "multi_agent_tools_v5"}
HORIZON, WARMUP = 40, 5
FR_MIN = 99.0
STEP = {"boost_small": 0.05, "boost_large": 0.1, "trim": 0.05}
# trim <= 0.95 keeps all four actions distinct (user decision 2026-10-14, docs/agent_design_v5.md)
HARD = {"boost_small": (1.0, 2.0), "boost_large": (1.0, 3.0), "trim": (0.3, 0.95)}
START_GRID = {
    "boost_small": [1.05, 1.10, 1.15, 1.20, 1.25],
    "boost_large": [1.2, 1.3, 1.4, 1.5, 1.6],
    "trim": [0.75, 0.80, 0.85, 0.90, 0.95],
}
ROOT = find_project_root()
SCENARIO_CACHE = generate_scenarios()


def run_one(system_id: str, chooser: str, scenario: str, seed: int, multipliers: Dict[str, float]) -> Dict[str, float]:
    sc = SCENARIO_CACHE[scenario]
    system = get_system(system_id, config={"db_path": False, "chooser": chooser, "multipliers": multipliers})
    system.calibrate(sc)
    system.reset(sc)
    env = BeerGameEnv(config=get_config().sim, scenario=sc, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)
    system.attach_env(env)
    for _ in range(HORIZON):
        states = env.step(system.decide_orders(states))
    m = compute_simulation_metrics(env, warmup_steps=WARMUP)
    return {
        "cost": float(m.total_cost),
        "fill_rate": 100.0 * float(m.fill_rate["retailer"]),
        "e2e_bw": float(m.end_to_end_bullwhip),
        "clip_rate": float(np.mean([r["clipped"] for r in system.decision_log])),
    }


def run_baseline(system_id: str, params: Dict, scenario: str, seed: int) -> Dict[str, float]:
    sc = SCENARIO_CACHE[scenario]
    system = get_system(system_id, config={**params, "db_path": False})
    system.calibrate(sc, config={**params, "db_path": False})
    system.reset(sc)
    env = BeerGameEnv(config=get_config().sim, scenario=sc, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)
    for _ in range(HORIZON):
        states = env.step(system.decide_orders(states))
    m = compute_simulation_metrics(env, warmup_steps=WARMUP)
    return {"cost": float(m.total_cost), "fill_rate": 100.0 * float(m.fill_rate["retailer"]),
            "e2e_bw": float(m.end_to_end_bullwhip), "clip_rate": np.nan}


def evaluate(mult: Dict[str, float]) -> Dict[str, float]:
    rows = [run_one(sid, "detector_map", sc, s, mult)
            for sid in FAMILIES.values() for sc in SCENARIOS for s in TUNING_SEEDS]
    df = pd.DataFrame(rows)
    return {"cost": df.cost.mean(), "fill_rate": df.fill_rate.mean(), "e2e_bw": df.e2e_bw.mean()}


def select(results: Dict[Tuple[float, float, float], Dict[str, float]], ceiling: float):
    items = list(results.items())
    feasible = [kv for kv in items if kv[1]["fill_rate"] >= FR_MIN and kv[1]["e2e_bw"] <= ceiling]
    if feasible:
        return min(feasible, key=lambda kv: kv[1]["cost"]), "feasible optimum (min cost)"
    fr_ok = [kv for kv in items if kv[1]["fill_rate"] >= FR_MIN]
    if fr_ok:
        return min(fr_ok, key=lambda kv: kv[1]["e2e_bw"]), "no feasible candidate: lowest bullwhip with FR >= 99%"
    return (max(items, key=lambda kv: (kv[1]["fill_rate"], -kv[1]["cost"])),
            "CAVEAT: no candidate reaches FR >= 99%: highest fill rate, then lowest cost")


def tune(ceiling: float):
    grid = {k: list(v) for k, v in START_GRID.items()}
    results: Dict[Tuple[float, float, float], Dict[str, float]] = {}
    dims = ["boost_small", "boost_large", "trim"]
    for it in range(12):
        for combo in itertools.product(*(grid[d] for d in dims)):
            if combo not in results:
                results[combo] = evaluate({"follow": 1.0, **dict(zip(dims, combo))})
        (best, metrics), note = select(results, ceiling)
        expanded = False
        for i, d in enumerate(dims):
            lo, hi = HARD[d]
            if best[i] == min(grid[d]) and round(best[i] - STEP[d], 4) >= lo:
                grid[d].insert(0, round(best[i] - STEP[d], 4))
                expanded = True
            elif best[i] == max(grid[d]) and round(best[i] + STEP[d], 4) <= hi:
                grid[d].append(round(best[i] + STEP[d], 4))
                expanded = True
        print(f"  iteration {it}: {len(results)} candidates, best={dict(zip(dims, best))} "
              f"cost={metrics['cost']:.0f} FR={metrics['fill_rate']:.2f} BW={metrics['e2e_bw']:.2f} [{note}]"
              f"{' -> grid expanded' if expanded else ''}")
        if not expanded:
            break
    edges = {d: (best[i] in (min(grid[d]), max(grid[d]))) for i, d in enumerate(dims)}
    return dict(zip(dims, best)), metrics, note, results, grid, edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="write chosen multipliers to configs/agent_v5.yaml")
    args = ap.parse_args()

    frozen = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))
    ceiling = float(frozen["bullwhip_reference"]["target_75_percent_ceiling"])
    start = load_agent_v5_config().multipliers

    print(f"Scripted policies, tuning seeds {TUNING_SEEDS}, starting multipliers {start}")
    rows = []
    for fam, sid in FAMILIES.items():
        for chooser in ("follow", "detector_map", "oracle"):
            for sc in SCENARIOS:
                for s in TUNING_SEEDS:
                    rows.append({"policy": f"{chooser}[{fam}]", "scenario": sc, "seed": s,
                                 **run_one(sid, chooser, sc, s, start)})
    for label, sid, params in (("2c tuned", "baseline_2c", frozen["tuned_policies"]["system_2c_adaptive_smoothed_out"]["parameters"]),
                               ("EOQ tuned", "baseline_eoq", frozen["tuned_policies"]["system_2_eoq"]["parameters"])):
        for sc in SCENARIOS:
            for s in TUNING_SEEDS:
                rows.append({"policy": label, "scenario": sc, "seed": s, **run_baseline(sid, params, sc, s)})
    df = pd.DataFrame(rows)
    out_dir = ROOT / "results"
    df.to_csv(out_dir / "v5_scripted_policies_seeds100_104.csv", index=False)
    summ = df.groupby("policy", sort=False).agg(cost=("cost", "mean"), fill_rate=("fill_rate", "mean"),
                                                e2e_bw=("e2e_bw", "mean"), clip_rate=("clip_rate", "mean"))
    print(summ.round(2).to_string())
    print(df.pivot_table(index="policy", columns="scenario", values=["cost", "fill_rate", "e2e_bw"], sort=False)
          .round(1).to_string())

    print(f"\nTuning (detector-mapped, pooled basic+tool, 30 runs/candidate); BW ceiling {ceiling:.3f}")
    best, metrics, note, results, grid, edges = tune(ceiling)
    tdf = pd.DataFrame([{"boost_small": k[0], "boost_large": k[1], "trim": k[2], **v} for k, v in results.items()])
    tdf.to_csv(out_dir / "v5_multiplier_tuning.csv", index=False)
    print(f"Chosen: {best} | cost={metrics['cost']:.0f} FR={metrics['fill_rate']:.2f} BW={metrics['e2e_bw']:.2f} | {note}")
    print(f"Final grid: {grid} | on edge: {edges}")

    if args.freeze:
        cfg_path = ROOT / "configs/agent_v5.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        new_mult = {"follow": 1.0, **{k: float(v) for k, v in best.items()}}
        block_old = "multipliers:\n" + "".join(f"  {k}: {v}\n" for k, v in data["multipliers"].items())
        block_new = "multipliers:\n" + "".join(f"  {k}: {v}\n" for k, v in new_mult.items())
        assert block_old in text, "unexpected multipliers block layout in configs/agent_v5.yaml"
        log = (f"results/v5_multiplier_tuning.csv ({date.today().isoformat()}, seeds 100-104, "
               f"{note}; cost={metrics['cost']:.0f}, FR={metrics['fill_rate']:.2f}, BW={metrics['e2e_bw']:.2f})")
        text = text.replace(block_old, block_new)
        text = text.replace("multipliers_frozen: false", "multipliers_frozen: true")
        text = text.replace("multipliers_tuning_log: null", f'multipliers_tuning_log: "{log}"')
        cfg_path.write_text(text, encoding="utf-8")
        print(f"Froze multipliers {new_mult} in configs/agent_v5.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
