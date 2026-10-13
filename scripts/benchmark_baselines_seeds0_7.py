"""Benchmark evaluation of System 1 (s, S), System 2 (EOQ), and System 2b (Smoothed OUT) on seeds 0 to 7.

Evaluates across all 3 scenarios: stable, seasonal, shock.
Applies steady-state initialization and 5-step warm-up identically to every system.
Computes and reports mean +/- std for:
- Total Cost ($)
- Holding Cost ($) and Backlog Cost ($)
- Bullwhip ratio by echelon (Retailer, Wholesaler, Distributor, Manufacturer)
- Service Level (%) and Fill Rate (%)
- Retailer Stockout Steps
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import sca.baselines  # noqa: F401
from sca.config import get_config
from sca.data.scenarios import LeadTimeSampler, generate_scenarios
from sca.sim.environment import BeerGameEnv
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import get_system

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark_baselines")

EVAL_SEEDS = list(range(8))  # Seeds 0..7
SCENARIOS_LIST = ["stable", "seasonal", "shock"]
SYSTEMS_TO_TEST = [
    ("system_1", "System 1: (s, S)", "baseline_s_S"),
    ("system_2", "System 2: EOQ", "baseline_eoq"),
    ("system_2b", "System 2b: Smoothed OUT", "baseline_smoothed_order_up_to"),
]
WARMUP_STEPS = 5


def run_benchmark():
    cfg = get_config()
    sim_cfg = cfg.sim
    all_scenarios = generate_scenarios()

    records: List[Dict[str, Any]] = []

    print("=" * 100)
    print("BENCHMARK EVALUATION ON SEEDS 0 TO 7 (8 SEEDS x 3 SCENARIOS x 3 SYSTEMS = 72 RUNS)")
    print(f"Warm-up: {WARMUP_STEPS} steps | Steady-state initialization: True")
    print("=" * 100)

    for sys_key, sys_label, sys_id in SYSTEMS_TO_TEST:
        for sc_name in SCENARIOS_LIST:
            base_sc = all_scenarios[sc_name]
            for seed in EVAL_SEEDS:
                sc = base_sc.clone() if hasattr(base_sc, "clone") else base_sc
                sc.lead_time_sampler = LeadTimeSampler(seed=seed)

                system = get_system(sys_id)
                system.calibrate(sc)
                system.reset(scenario=sc)

                env = BeerGameEnv(
                    config=sim_cfg,
                    scenario=sc,
                    seed=seed,
                    steady_state_init=True,
                )
                states = env.reset(seed=seed)

                for _ in range(len(sc.demand)):
                    orders = system.decide_orders(states)
                    states = env.step(orders)

                metrics = compute_simulation_metrics(env, warmup_steps=WARMUP_STEPS)

                ret_bw = float(metrics.bullwhip_ratio["retailer"])
                who_bw = float(metrics.bullwhip_ratio["wholesaler"])
                dis_bw = float(metrics.bullwhip_ratio["distributor"])
                man_bw = float(metrics.bullwhip_ratio["manufacturer"])

                records.append({
                    "system_key": sys_key,
                    "system_label": sys_label,
                    "system_id": sys_id,
                    "scenario": sc_name,
                    "seed": seed,
                    "total_cost": float(metrics.total_cost),
                    "holding_cost": float(sum(metrics.holding_cost_by_echelon.values())),
                    "backlog_cost": float(sum(metrics.backlog_cost_by_echelon.values())),
                    "retailer_cost": float(metrics.cost_by_echelon["retailer"]),
                    "retailer_bullwhip": ret_bw,
                    "wholesaler_bullwhip": who_bw,
                    "distributor_bullwhip": dis_bw,
                    "manufacturer_bullwhip": man_bw,
                    "retailer_service_level": float(metrics.service_level["retailer"]),
                    "retailer_fill_rate": float(metrics.fill_rate["retailer"]),
                    "retailer_stockouts": int(metrics.stockout_steps["retailer"]),
                })

    df = pd.DataFrame(records)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "benchmark_seeds0_7_raw.csv", index=False)

    # 1. Scenario-level Summary
    print("\n### 1. PERFORMANCE SUMMARY BY SCENARIO (SEEDS 0-7, MEAN ± STD)\n")
    sc_summary = df.groupby(["scenario", "system_label"]).agg(
        cost_mean=("total_cost", "mean"),
        cost_std=("total_cost", "std"),
        holding_mean=("holding_cost", "mean"),
        backlog_mean=("backlog_cost", "mean"),
        ret_bw_mean=("retailer_bullwhip", "mean"),
        ret_bw_std=("retailer_bullwhip", "std"),
        man_bw_mean=("manufacturer_bullwhip", "mean"),
        man_bw_std=("manufacturer_bullwhip", "std"),
        sl_mean=("retailer_service_level", "mean"),
        sl_std=("retailer_service_level", "std"),
        fr_mean=("retailer_fill_rate", "mean"),
        fr_std=("retailer_fill_rate", "std"),
        so_mean=("retailer_stockouts", "mean"),
    ).reset_index()

    print("| Scenario | System | Total Cost ($) | Ret Bullwhip | Mfr Bullwhip | Service Level (%) | Fill Rate (%) | Stockout Days |")
    print("|:---|:---|---:|---:|---:|---:|---:|---:|")
    for _, r in sc_summary.iterrows():
        c_str = f"${r['cost_mean']:,.2f} ± {r['cost_std']:,.1f}"
        rbw_str = f"{r['ret_bw_mean']:.2f} ± {r['ret_bw_std']:.2f}"
        mbw_str = f"{r['man_bw_mean']:.2f} ± {r['man_bw_std']:.2f}"
        sl_str = f"{r['sl_mean']*100:.1f} ± {r['sl_std']*100:.1f}"
        fr_str = f"{r['fr_mean']*100:.1f} ± {r['fr_std']*100:.1f}"
        so_str = f"{r['so_mean']:.1f}"
        print(f"| {r['scenario'].capitalize()} | **{r['system_label']}** | {c_str} | {rbw_str} | {mbw_str} | {sl_str}% | {fr_str}% | {so_str} |")

    # 2. Overall Aggregate Summary across all scenarios & seeds (n=24 per system)
    print("\n### 2. OVERALL BENCHMARK ACROSS ALL SCENARIOS (n=24 RUNS PER SYSTEM, MEAN ± STD)\n")
    ov_summary = df.groupby("system_label").agg(
        cost_mean=("total_cost", "mean"),
        cost_std=("total_cost", "std"),
        holding_mean=("holding_cost", "mean"),
        backlog_mean=("backlog_cost", "mean"),
        ret_bw_mean=("retailer_bullwhip", "mean"),
        ret_bw_std=("retailer_bullwhip", "std"),
        who_bw_mean=("wholesaler_bullwhip", "mean"),
        who_bw_std=("wholesaler_bullwhip", "std"),
        dis_bw_mean=("distributor_bullwhip", "mean"),
        dis_bw_std=("distributor_bullwhip", "std"),
        man_bw_mean=("manufacturer_bullwhip", "mean"),
        man_bw_std=("manufacturer_bullwhip", "std"),
        sl_mean=("retailer_service_level", "mean"),
        sl_std=("retailer_service_level", "std"),
        fr_mean=("retailer_fill_rate", "mean"),
        fr_std=("retailer_fill_rate", "std"),
        so_mean=("retailer_stockouts", "mean"),
    ).loc[[s[1] for s in SYSTEMS_TO_TEST]].reset_index()

    print("| System | Total Cost ($) | Retailer BW | Wholesaler BW | Distributor BW | Mfr BW | Service Level (%) | Fill Rate (%) | Stockout Days |")
    print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in ov_summary.iterrows():
        c_str = f"${r['cost_mean']:,.2f} ± {r['cost_std']:,.1f}"
        rbw = f"{r['ret_bw_mean']:.2f} ± {r['ret_bw_std']:.2f}"
        wbw = f"{r['who_bw_mean']:.2f} ± {r['who_bw_std']:.2f}"
        dbw = f"{r['dis_bw_mean']:.2f} ± {r['dis_bw_std']:.2f}"
        mbw = f"{r['man_bw_mean']:.2f} ± {r['man_bw_std']:.2f}"
        sl_str = f"{r['sl_mean']*100:.1f}% ± {r['sl_std']*100:.1f}%"
        fr_str = f"{r['fr_mean']*100:.1f}% ± {r['fr_std']*100:.1f}%"
        so_str = f"{r['so_mean']:.1f}"
        print(f"| **{r['system_label']}** | {c_str} | {rbw} | {wbw} | {dbw} | {mbw} | {sl_str} | {fr_str} | {so_str} |")

    ov_summary.to_csv(out_dir / "benchmark_seeds0_7_summary.csv", index=False)
    print(f"\nBenchmark summary written to: {out_dir / 'benchmark_seeds0_7_summary.csv'}")

    return df, sc_summary, ov_summary


if __name__ == "__main__":
    run_benchmark()
