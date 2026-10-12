"""Benchmark v2.1 Full Acceptance Test & Sensitivity Runner.

Evaluates Systems 0, 1, 2, 2b (untuned), and Oracle on tuning seeds 100..104:
- Primary test at initial inventory S = 687.0 (z = 2.0 implied base stock)
- Sensitivity test at initial inventory S = 400.0
- Verifies Acceptance Gate: System 0 must lose on constrained cost in Seasonal & Shock.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sca.config import get_config
from sca.data.scenarios import LeadTimeSampler, generate_scenarios
from sca.sim.environment import BeerGameEnv
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import get_system
import sca.baselines

logging.basicConfig(level=logging.WARNING)

TUNING_SEEDS = [100, 101, 102, 103, 104]
SCENARIOS = ["stable", "seasonal", "shock"]
COMMON_HORIZON = 40
WARMUP_STEPS = 5

SYSTEMS_UNDER_TEST = [
    ("system_0", "System 0: Constant", "baseline_constant", {}),
    ("system_1_untuned", "System 1: (s, S) Untuned", "baseline_s_S", {}),
    ("system_2_untuned", "System 2: EOQ Untuned", "baseline_eoq", {}),
    ("system_2b_untuned", "System 2b: Smoothed OUT Untuned", "baseline_smoothed_order_up_to", {
        "alpha": 0.8,
        "beta": 0.8,
        "z": 2.0,
        "delta": 0.5,
        "forecast_cap_multiple": 2.0,
        "include_lead_time_variance": True,
        "db_path": False,
    }),
    ("oracle", "Oracle: Clairvoyant", "baseline_oracle", {"lead_time_offset": 4}),
]


def run_evaluation(init_inv: float, backlog_cost: float = 5.0) -> pd.DataFrame:
    cfg = get_config()
    sim_cfg = cfg.sim.model_copy(deep=True)
    sim_cfg.four_echelon.initial_inventory = float(init_inv)
    sim_cfg.four_echelon.backlog_cost_per_unit = float(backlog_cost)
    sim_cfg.four_echelon.holding_cost_per_unit = 0.50
    sim_cfg.four_echelon.steady_state_init = True
    sim_cfg.four_echelon.independent_demand_paths = True

    all_scenarios = generate_scenarios()
    records: List[Dict[str, Any]] = []

    for sys_key, sys_label, sys_id, custom_cfg in SYSTEMS_UNDER_TEST:
        system = get_system(sys_id, config=custom_cfg)

        for sc_name in SCENARIOS:
            sc = all_scenarios[sc_name]
            system.calibrate(sc, config=custom_cfg)

            for seed in TUNING_SEEDS:
                # Rigorous per-run state isolation
                system.reset(sc)
                env = BeerGameEnv(
                    config=sim_cfg,
                    scenario=sc,
                    seed=seed,
                    steady_state_init=True,
                )
                states = env.reset(seed=seed)

                for _ in range(COMMON_HORIZON):
                    orders = system.decide_orders(states)
                    states = env.step(orders)

                metrics = compute_simulation_metrics(env, warmup_steps=WARMUP_STEPS)

                ret_bw = float(metrics.bullwhip_ratio.get("retailer", np.nan))
                who_bw = float(metrics.bullwhip_ratio.get("wholesaler", np.nan))
                dis_bw = float(metrics.bullwhip_ratio.get("distributor", np.nan))
                man_bw = float(metrics.bullwhip_ratio.get("manufacturer", np.nan))
                valid_bws = [b for b in (ret_bw, who_bw, dis_bw, man_bw) if not np.isnan(b)]
                mean_bw = float(np.mean(valid_bws)) if valid_bws else np.nan

                records.append({
                    "init_inv": init_inv,
                    "system_key": sys_key,
                    "system_label": sys_label,
                    "scenario": sc_name,
                    "seed": seed,
                    "total_cost": float(metrics.total_cost),
                    "fill_rate": float(metrics.fill_rate.get("retailer", 0.0)) * 100.0,
                    "service_level": float(metrics.service_level.get("retailer", 0.0)) * 100.0,
                    "mean_bullwhip": mean_bw,
                    "end_to_end_bullwhip": float(metrics.end_to_end_bullwhip),
                })

    df = pd.DataFrame(records)
    return df


def compile_summary_table(df: pd.DataFrame, eoq_bw_threshold: float = None) -> pd.DataFrame:
    summary_rows = []
    
    # Calculate EOQ untuned mean bullwhip if not provided
    if eoq_bw_threshold is None:
        eoq_runs = df[df["system_key"] == "system_2_untuned"]
        eoq_bw_mean = eoq_runs["mean_bullwhip"].mean()
        eoq_bw_threshold = 0.75 * eoq_bw_mean

    for sc_name in SCENARIOS:
        sc_df = df[df["scenario"] == sc_name]

        for sys_key, sys_label, _, _ in SYSTEMS_UNDER_TEST:
            sub = sc_df[sc_df["system_key"] == sys_key]
            mean_cost = sub["total_cost"].mean()
            mean_fr = sub["fill_rate"].mean()
            mean_sl = sub["service_level"].mean()
            mean_bw = sub["mean_bullwhip"].mean()
            e2e_bw = sub["end_to_end_bullwhip"].mean()

            # Pre-registered constraints: FR >= 99%, SL >= 85%
            fr_ok = bool(mean_fr >= 99.0)
            sl_ok = bool(mean_sl >= 85.0)
            feasible = fr_ok and sl_ok

            if sys_key == "oracle":
                status = "REFERENCE"
                constrained_cost = mean_cost
            elif feasible:
                status = "PASS"
                constrained_cost = mean_cost
            else:
                fail_reasons = []
                if not fr_ok:
                    fail_reasons.append(f"FR={mean_fr:.1f}%<99%")
                if not sl_ok:
                    fail_reasons.append(f"SL={mean_sl:.1f}%<85%")
                status = f"DISQUALIFIED ({', '.join(fail_reasons)})"
                constrained_cost = np.nan

            summary_rows.append({
                "Scenario": sc_name.capitalize(),
                "System": sys_label,
                "Total Cost ($)": f"${mean_cost:,.2f}",
                "Fill Rate (%)": f"{mean_fr:.2f}%",
                "Service Level (%)": f"{mean_sl:.1f}%",
                "Mean Bullwhip": f"{mean_bw:.3f}" if not np.isnan(mean_bw) else "n/a",
                "End-to-End BW": f"{e2e_bw:.3f}" if not np.isnan(e2e_bw) else "n/a",
                "Constrained Cost": f"${constrained_cost:,.2f}" if not np.isnan(constrained_cost) else "NaN",
                "Status": status,
                "raw_cost": mean_cost,
                "raw_constrained_cost": constrained_cost,
                "feasible": feasible,
            })

    return pd.DataFrame(summary_rows)


def main():
    print("=" * 110)
    print("BENCHMARK v2.1 FULL ACCEPTANCE TEST ON TUNING SEEDS 100 TO 104")
    print("Cost: h=$0.50, b=$5.00 (b/h = 10.0x) | Horizon: 40 steps (warmup: 5 steps)")
    print("=" * 110)

    # 1. Primary evaluation at I_0 = 687.0 (Implied Base-Stock S with z=2.0)
    print("\n[1] PRIMARY TEST: INITIAL INVENTORY I_0 = 687.0 UNITS (BASE-STOCK S)")
    df_687 = run_evaluation(init_inv=687.0, backlog_cost=5.0)
    summary_687 = compile_summary_table(df_687)
    display_cols = ["Scenario", "System", "Total Cost ($)", "Fill Rate (%)", "Service Level (%)", "Mean Bullwhip", "End-to-End BW", "Constrained Cost", "Status"]
    print(summary_687[display_cols].to_markdown(index=False))

    # 2. Sensitivity evaluation at I_0 = 400.0
    print("\n" + "=" * 110)
    print("[2] SENSITIVITY TEST: INITIAL INVENTORY I_0 = 400.0 UNITS")
    print("=" * 110)
    df_400 = run_evaluation(init_inv=400.0, backlog_cost=5.0)
    summary_400 = compile_summary_table(df_400)
    print(summary_400[display_cols].to_markdown(index=False))

    # 3. Acceptance Gate Evaluation
    print("\n" + "=" * 110)
    print("ACCEPTANCE GATE VERIFICATION (SYSTEM 0 CONSTRAINED COST CHECK)")
    print("=" * 110)

    for sc_name in ["seasonal", "shock"]:
        sc_cap = sc_name.capitalize()
        s0_row = summary_687[(summary_687["Scenario"] == sc_cap) & (summary_687["System"] == "System 0: Constant")].iloc[0]
        eoq_row = summary_687[(summary_687["Scenario"] == sc_cap) & (summary_687["System"] == "System 2: EOQ Untuned")].iloc[0]
        ss_row = summary_687[(summary_687["Scenario"] == sc_cap) & (summary_687["System"] == "System 1: (s, S) Untuned")].iloc[0]

        s0_cc = s0_row["raw_constrained_cost"]
        eoq_cc = eoq_row["raw_constrained_cost"]
        ss_cc = ss_row["raw_constrained_cost"]

        print(f"\nScenario: {sc_cap}")
        print(f"  System 0: Constrained Cost = {s0_row['Constrained Cost']} | Status = {s0_row['Status']}")
        print(f"  EOQ:      Constrained Cost = {eoq_row['Constrained Cost']} | Status = {eoq_row['Status']}")
        print(f"  (s, S):   Constrained Cost = {ss_row['Constrained Cost']} | Status = {ss_row['Status']}")

        if np.isnan(s0_cc):
            print(f"  --> System 0 is DISQUALIFIED (Constrained Cost = NaN). Baseline policies win!")
        elif not np.isnan(eoq_cc) and eoq_cc < s0_cc:
            print(f"  --> System 0 finite cost (${s0_cc:,.2f}) LOSES to EOQ (${eoq_cc:,.2f}). Baseline policies win!")
        elif not np.isnan(ss_cc) and ss_cc < s0_cc:
            print(f"  --> System 0 finite cost (${s0_cc:,.2f}) LOSES to (s, S) (${ss_cc:,.2f}). Baseline policies win!")
        else:
            print(f"  --> WARNING: System 0 achieved lowest constrained cost (${s0_cc:,.2f})!")


if __name__ == "__main__":
    main()
