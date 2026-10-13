"""Benchmark v2.1 Acceptance Test & Reference Evaluation Runner.

Evaluates:
Reference Rows:
- System 0: Constant (Training Mean)
- System 2: EOQ Untuned
- System 1: (s, S) Untuned
- Base-Stock (z = 2.0)
- Oracle: Clairvoyant 4-day lookahead (labelled Reference Only)

Tuned Baseline Rows:
- System 2: EOQ Tuned
- System 1: (s, S) Tuned
- System 2b: Smoothed OUT Tuned

Constraints:
- Unit fill rate >= 99.0%
- End-to-end bullwhip <= 75% of untuned EOQ on tuning seeds
- Step service level is reported, not constrained

Primary cost: h = $0.50, b = $2.50 (b/h = 5.0)
Sensitivities: b/h = 2.0 (b = $1.00) and b/h = 10.0 (b = $5.00)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sca.config import get_config
from sca.data.scenarios import generate_scenarios
from sca.sim.environment import BeerGameEnv
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import get_system

logging.basicConfig(level=logging.WARNING)

TUNING_SEEDS = [100, 101, 102, 103, 104]
SCENARIOS = ["stable", "seasonal", "shock"]
COMMON_HORIZON = 40
WARMUP_STEPS = 5
BASE_STOCK_INIT = 687.0

# Systems to evaluate: (key, label, system_id, config_dict, is_reference)
BENCHMARK_SYSTEMS = [
    # Reference rows
    ("system_0", "System 0: Constant", "baseline_constant", {}, True),
    ("system_2_untuned", "EOQ Untuned", "baseline_eoq", {}, True),
    ("system_1_untuned", "(s, S) Untuned", "baseline_s_S", {}, True),
    ("base_stock_z2", "Base-Stock (z=2.0)", "baseline_base_stock", {"z": 2.0}, True),
    ("oracle", "Oracle (Clairvoyant 4-day)", "baseline_oracle", {"lead_time_offset": 4}, True),
    # Tuned baselines
    ("system_2_tuned", "EOQ Tuned", "baseline_eoq", {
        "order_quantity_multiplier": 0.75,
        "reorder_point_factor": 1.2,
    }, False),
    ("system_1_tuned", "(s, S) Tuned", "baseline_s_S", {
        "s_factor": 1.2,
        "S_factor": 0.25,
    }, False),
    ("system_2b_tuned", "System 2b Tuned (Smoothed OUT)", "baseline_smoothed_order_up_to", {
        "alpha": 0.6,
        "beta": 0.6,
        "z": 2.5,
        "delta": 0.5,
        "forecast_cap_multiple": 1.5,
        "include_lead_time_variance": True,
        "db_path": False,
    }, False),
]


def run_benchmark_eval(
    init_inv: float = BASE_STOCK_INIT,
    backlog_cost: float = 2.50,
    holding_cost: float = 0.50,
    persistent_shock: bool = False,
) -> pd.DataFrame:
    """Run all systems on tuning seeds across scenarios."""
    cfg = get_config()
    sim_cfg = cfg.sim.model_copy(deep=True)
    sim_cfg.four_echelon.initial_inventory = float(init_inv)
    sim_cfg.four_echelon.backlog_cost_per_unit = float(backlog_cost)
    sim_cfg.four_echelon.holding_cost_per_unit = float(holding_cost)
    sim_cfg.four_echelon.backlog_cost_multiplier = float(backlog_cost / holding_cost)
    sim_cfg.four_echelon.steady_state_init = True
    sim_cfg.four_echelon.independent_demand_paths = True

    data_cfg = cfg.data.model_copy(deep=True)
    if persistent_shock:
        # Contingency: make shock shift persistent from step 15 to horizon end
        data_cfg.shock_level_shift_duration = 25

    all_scenarios = generate_scenarios(data_config=data_cfg)
    records: List[Dict[str, Any]] = []

    for sys_key, sys_label, sys_id, custom_cfg, is_ref in BENCHMARK_SYSTEMS:
        system = get_system(sys_id, config=custom_cfg)

        for sc_name in SCENARIOS:
            sc = all_scenarios[sc_name]
            system.calibrate(sc, config=custom_cfg)

            for seed in TUNING_SEEDS:
                # Per-run state isolation
                system.reset(sc)
                env = BeerGameEnv(
                    config=sim_cfg,
                    scenario=sc,
                    seed=seed,
                    steady_state_init=True,
                )
                states = env.reset(seed=seed)

                # For oracle, inject seed's realized demand path
                if sys_id == "baseline_oracle" and hasattr(system, "set_demand"):
                    system.set_demand(env.external_demand)

                for _ in range(COMMON_HORIZON):
                    orders = system.decide_orders(states)
                    states = env.step(orders)

                metrics = compute_simulation_metrics(env, warmup_steps=WARMUP_STEPS)

                ret_bw = float(metrics.bullwhip_ratio.get("retailer", np.nan))
                who_bw = float(metrics.bullwhip_ratio.get("wholesaler", np.nan))
                dis_bw = float(metrics.bullwhip_ratio.get("distributor", np.nan))
                man_bw = float(metrics.bullwhip_ratio.get("manufacturer", np.nan))

                records.append({
                    "system_key": sys_key,
                    "system_label": sys_label,
                    "is_reference": is_ref,
                    "scenario": sc_name,
                    "seed": seed,
                    "total_cost": float(metrics.total_cost),
                    "fill_rate": float(metrics.fill_rate.get("retailer", 0.0)) * 100.0,
                    "service_level": float(metrics.service_level.get("retailer", 0.0)) * 100.0,
                    "bw_ret": ret_bw,
                    "bw_who": who_bw,
                    "bw_dis": dis_bw,
                    "bw_man": man_bw,
                    "end_to_end_bullwhip": float(metrics.end_to_end_bullwhip),
                })

    return pd.DataFrame(records)


def format_bw(val: float) -> str:
    if np.isnan(val):
        return "n/a"
    if np.isinf(val):
        return "inf"
    return f"{val:.2f}"


def build_acceptance_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute summary statistics, apply 75% bullwhip & 99% fill rate constraints."""
    # First calculate Untuned EOQ end-to-end bullwhip thresholds per scenario
    eoq_thresholds: Dict[str, float] = {}
    for sc_name in SCENARIOS:
        eoq_sub = df[(df["scenario"] == sc_name) & (df["system_key"] == "system_2_untuned")]
        eoq_e2e_mean = float(eoq_sub["end_to_end_bullwhip"].mean())
        eoq_thresholds[sc_name] = 0.75 * eoq_e2e_mean

    summary_rows = []

    for sc_name in SCENARIOS:
        sc_df = df[df["scenario"] == sc_name]
        threshold = eoq_thresholds[sc_name]

        for sys_key, sys_label, _, _, is_ref in BENCHMARK_SYSTEMS:
            sub = sc_df[sc_df["system_key"] == sys_key]
            mean_cost = float(sub["total_cost"].mean())
            mean_fr = float(sub["fill_rate"].mean())
            mean_sl = float(sub["service_level"].mean())
            mean_e2e_bw = float(sub["end_to_end_bullwhip"].mean())
            bw_r = float(sub["bw_ret"].mean())
            bw_w = float(sub["bw_who"].mean())
            bw_d = float(sub["bw_dis"].mean())
            bw_m = float(sub["bw_man"].mean())

            per_echelon_str = f"R:{format_bw(bw_r)} / W:{format_bw(bw_w)} / D:{format_bw(bw_d)} / M:{format_bw(bw_m)}"

            # Constraint checks: Fill Rate >= 99.0% and E2E Bullwhip <= 75% of EOQ untuned
            fr_ok = mean_fr >= 99.0
            # Note: For EOQ untuned itself, e2e_bw is 1.0x EOQ untuned > 0.75x threshold by definition
            bw_ok = (mean_e2e_bw <= threshold) if not np.isnan(mean_e2e_bw) else False

            if sys_key == "oracle":
                status = "REFERENCE ONLY"
                constrained_cost_val = mean_cost
                constrained_cost_str = f"${mean_cost:,.2f}*"
            else:
                fail_reasons = []
                if not fr_ok:
                    fail_reasons.append(f"FR={mean_fr:.1f}%<99%")
                if not bw_ok:
                    fail_reasons.append(f"BW={mean_e2e_bw:.2f}>thresh({threshold:.2f})")

                if len(fail_reasons) == 0:
                    status = "PASS"
                    constrained_cost_val = mean_cost
                    constrained_cost_str = f"${mean_cost:,.2f}"
                else:
                    status = f"DISQUALIFIED ({', '.join(fail_reasons)})"
                    constrained_cost_val = np.nan
                    constrained_cost_str = "NaN"

            summary_rows.append({
                "Scenario": sc_name.capitalize(),
                "System": sys_label,
                "Type": "Reference" if is_ref else "Tuned",
                "Cost": f"${mean_cost:,.2f}",
                "Fill (%)": f"{mean_fr:.2f}%",
                "Step SL (%)": f"{mean_sl:.1f}%",
                "End-to-End BW": f"{mean_e2e_bw:.2f}" if not np.isnan(mean_e2e_bw) else "n/a",
                "Per-Echelon BW (R/W/D/M)": per_echelon_str,
                "Constrained Cost": constrained_cost_str,
                "Status": status,
                "raw_cost": mean_cost,
                "raw_constrained_cost": constrained_cost_val,
                "raw_fr": mean_fr,
                "raw_sl": mean_sl,
                "raw_e2e_bw": mean_e2e_bw,
            })

    return pd.DataFrame(summary_rows), eoq_thresholds


def main():
    print("=" * 120)
    print("BENCHMARK v2.1 ACCEPTANCE TEST ON TUNING SEEDS 100-104")
    print(f"Initial Inventory: S = {BASE_STOCK_INIT} (z = 2.0) | b/h = 5.0 (h = $0.50, b = $2.50)")
    print("Constraints: Fill Rate >= 99.0%, End-to-End Bullwhip <= 75% of Untuned EOQ | Step SL Reported")
    print("=" * 120)

    # 1. Primary evaluation b/h = 5.0
    df_primary = run_benchmark_eval(backlog_cost=2.50, holding_cost=0.50)
    table_primary, thresholds = build_acceptance_table(df_primary)

    cols_to_print = [
        "Scenario",
        "System",
        "Cost",
        "Fill (%)",
        "Step SL (%)",
        "End-to-End BW",
        "Per-Echelon BW (R/W/D/M)",
        "Constrained Cost",
        "Status",
    ]

    print("\n### Primary Benchmark Results (b/h = 5.0, h=$0.50, b=$2.50):\n")
    print(table_primary[cols_to_print].to_markdown(index=False))

    print("\nEOQ Untuned 75% E2E Bullwhip Dampening Ceilings:")
    for sc, thresh in thresholds.items():
        print(f"  - {sc.capitalize()}: Threshold = {thresh:.2f}")

    # Check Acceptance Gate: System 0 must lose to best tuned baseline on constrained cost in Seasonal and Shock
    print("\n" + "=" * 120)
    print("ACCEPTANCE GATE VERIFICATION (SYSTEM 0 CONSTRAINED COST CHECK)")
    print("=" * 120)

    contingency_needed = False
    for sc_name in ["seasonal", "shock"]:
        sc_cap = sc_name.capitalize()
        sc_rows = table_primary[table_primary["Scenario"] == sc_cap]

        s0_row = sc_rows[sc_rows["System"] == "System 0: Constant"].iloc[0]
        tuned_rows = sc_rows[sc_rows["Type"] == "Tuned"]

        # Find best tuned baseline (lowest constrained cost among PASS)
        valid_tuned = tuned_rows[~tuned_rows["raw_constrained_cost"].isna()]

        s0_cost = s0_row["raw_constrained_cost"]
        print(f"\nScenario [{sc_cap}]:")
        print(f"  System 0: Constrained Cost = {s0_row['Constrained Cost']}, Status = {s0_row['Status']}")

        if len(valid_tuned) > 0:
            best_tuned = valid_tuned.sort_values("raw_constrained_cost").iloc[0]
            print(f"  Best Tuned Baseline: {best_tuned['System']} -> Constrained Cost = {best_tuned['Constrained Cost']}, Status = {best_tuned['Status']}")

            if np.isnan(s0_cost):
                print(f"  --> System 0 is DISQUALIFIED! {best_tuned['System']} wins on constrained cost. (GATE PASSED)")
            elif s0_cost > best_tuned["raw_constrained_cost"]:
                print(f"  --> System 0 ({s0_row['Constrained Cost']}) LOSES to {best_tuned['System']} ({best_tuned['Constrained Cost']}). (GATE PASSED)")
            else:
                print(f"  --> CONTINGENCY TRIGGER: System 0 achieved lower constrained cost ({s0_row['Constrained Cost']} vs {best_tuned['Constrained Cost']})!")
                contingency_needed = True
        else:
            print("  --> No tuned baseline passed constraints! Contingency may be needed.")
            contingency_needed = True

    if contingency_needed:
        print("\n" + "!" * 120)
        print("APPLYING CONTINGENCY: PERSISTENT LEVEL SHIFT IN SHOCK SCENARIO (DURATION = 25 STEPS)")
        print("!" * 120)
        df_contingency = run_benchmark_eval(backlog_cost=2.50, holding_cost=0.50, persistent_shock=True)
        table_contingency, thresholds_c = build_acceptance_table(df_contingency)
        print("\n### Contingency Benchmark Results:\n")
        print(table_contingency[cols_to_print].to_markdown(index=False))

    # Sensitivity tests: b/h = 2.0 and b/h = 10.0
    print("\n" + "=" * 120)
    print("SENSITIVITY EVALUATION: b/h = 2.0 (h=$0.50, b=$1.00)")
    print("=" * 120)
    df_sens_low = run_benchmark_eval(backlog_cost=1.00, holding_cost=0.50)
    table_sens_low, _ = build_acceptance_table(df_sens_low)
    print(table_sens_low[cols_to_print].to_markdown(index=False))

    print("\n" + "=" * 120)
    print("SENSITIVITY EVALUATION: b/h = 10.0 (h=$0.50, b=$5.00)")
    print("=" * 120)
    df_sens_high = run_benchmark_eval(backlog_cost=5.00, holding_cost=0.50)
    table_sens_high, _ = build_acceptance_table(df_sens_high)
    print(table_sens_high[cols_to_print].to_markdown(index=False))


if __name__ == "__main__":
    main()
