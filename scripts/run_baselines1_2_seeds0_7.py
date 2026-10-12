"""Run System 1 ((s, S)) and System 2 (EOQ) on seeds 0 to 7 at common 40-step horizon.

Prints per-scenario tables (mean +/- std) with explicit metric labeling:
- Service Level: share of steps without stockout (%)
- Fill Rate: units shipped / demanded (%)
- Total Cost, Holding Cost, Backlog Cost
- Bullwhip ratios per echelon and mean bullwhip across echelons
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import sca.baselines  # noqa: F401
from sca.config import get_config
from sca.data.scenarios import LeadTimeSampler, generate_scenarios
from sca.sim.environment import BeerGameEnv
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import get_system

logging.basicConfig(level=logging.WARNING)

SEEDS = list(range(8))  # seeds 0 to 7
SCENARIOS = ["stable", "seasonal", "shock"]
SYSTEMS = [
    ("system_1", "System 1: (s, S)", "baseline_s_S"),
    ("system_2", "System 2: EOQ", "baseline_eoq"),
]
WARMUP_STEPS = 5
COMMON_HORIZON = 40


def run():
    cfg = get_config()
    sim_cfg = cfg.sim
    all_scenarios = generate_scenarios()

    records: List[Dict[str, Any]] = []

    for sys_key, sys_label, sys_id in SYSTEMS:
        for sc_name in SCENARIOS:
            base_sc = all_scenarios[sc_name]
            for seed in SEEDS:
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

                steps_to_run = min(COMMON_HORIZON, len(sc.demand))
                for _ in range(steps_to_run):
                    orders = system.decide_orders(states)
                    states = env.step(orders)

                metrics = compute_simulation_metrics(env, warmup_steps=WARMUP_STEPS)

                ret_bw = float(metrics.bullwhip_ratio["retailer"])
                who_bw = float(metrics.bullwhip_ratio["wholesaler"])
                dis_bw = float(metrics.bullwhip_ratio["distributor"])
                man_bw = float(metrics.bullwhip_ratio["manufacturer"])
                mean_bw = (ret_bw + who_bw + dis_bw + man_bw) / 4.0

                records.append({
                    "system_key": sys_key,
                    "system_label": sys_label,
                    "scenario": sc_name,
                    "seed": seed,
                    "total_cost": float(metrics.total_cost),
                    "holding_cost": float(sum(metrics.holding_cost_by_echelon.values())),
                    "backlog_cost": float(sum(metrics.backlog_cost_by_echelon.values())),
                    "retailer_bullwhip": ret_bw,
                    "wholesaler_bullwhip": who_bw,
                    "distributor_bullwhip": dis_bw,
                    "manufacturer_bullwhip": man_bw,
                    "mean_bullwhip": mean_bw,
                    "service_level": float(metrics.service_level["retailer"]) * 100.0,
                    "fill_rate": float(metrics.fill_rate["retailer"]) * 100.0,
                    "stockout_steps": int(metrics.stockout_steps["retailer"]),
                    "eval_steps": metrics.total_steps,
                })

    df = pd.DataFrame(records)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "baselines1_2_seeds0_7_raw.csv", index=False)

    print("=" * 110)
    print(f"BASELINES 1 & 2 EVALUATION ON SEEDS 0 TO 7 (COMMON HORIZON: {COMMON_HORIZON} STEPS, WARM-UP: {WARMUP_STEPS} STEPS)")
    print(f"Evaluated steps per run: {COMMON_HORIZON - WARMUP_STEPS} steps | Steady-state initialization: True")
    print("=" * 110)

    for sc_name in SCENARIOS:
        sub = df[df["scenario"] == sc_name]
        print(f"\n### Scenario: {sc_name.upper()} (Seeds 0-7, n=8 runs per system)")
        print("-" * 110)

        summary_rows = []
        for sys_key, sys_label, _ in SYSTEMS:
            s_df = sub[sub["system_key"] == sys_key]
            cost_m, cost_s = s_df["total_cost"].mean(), s_df["total_cost"].std()
            hold_m, hold_s = s_df["holding_cost"].mean(), s_df["holding_cost"].std()
            back_m, back_s = s_df["backlog_cost"].mean(), s_df["backlog_cost"].std()
            ret_m, ret_s = s_df["retailer_bullwhip"].mean(), s_df["retailer_bullwhip"].std()
            who_m, who_s = s_df["wholesaler_bullwhip"].mean(), s_df["wholesaler_bullwhip"].std()
            dis_m, dis_s = s_df["distributor_bullwhip"].mean(), s_df["distributor_bullwhip"].std()
            man_m, man_s = s_df["manufacturer_bullwhip"].mean(), s_df["manufacturer_bullwhip"].std()
            mbw_m, mbw_s = s_df["mean_bullwhip"].mean(), s_df["mean_bullwhip"].std()
            sl_m, sl_s = s_df["service_level"].mean(), s_df["service_level"].std()
            fr_m, fr_s = s_df["fill_rate"].mean(), s_df["fill_rate"].std()
            so_m, so_s = s_df["stockout_steps"].mean(), s_df["stockout_steps"].std()

            summary_rows.append({
                "System": sys_label,
                "Total Cost ($)": f"${cost_m:,.2f} ± {cost_s:,.1f}",
                "Holding Cost ($)": f"${hold_m:,.2f} ± {hold_s:,.1f}",
                "Backlog Cost ($)": f"${back_m:,.2f} ± {back_s:,.1f}",
                "Retailer BW": f"{ret_m:.2f} ± {ret_s:.2f}",
                "Wholesaler BW": f"{who_m:.2f} ± {who_s:.2f}",
                "Distributor BW": f"{dis_m:.2f} ± {dis_s:.2f}",
                "Manufacturer BW": f"{man_m:.2f} ± {man_s:.2f}",
                "Mean BW across Echelons": f"{mbw_m:.2f} ± {mbw_s:.2f}",
                "Service Level (% steps without stockout)": f"{sl_m:.1f}% ± {sl_s:.1f}%",
                "Fill Rate (% units)": f"{fr_m:.2f}% ± {fr_s:.2f}%",
                "Stockout Steps": f"{so_m:.1f} ± {so_s:.1f}",
            })

        summary_df = pd.DataFrame(summary_rows)
        # Markdown table
        print(summary_df.to_markdown(index=False))

    return df


if __name__ == "__main__":
    run()
