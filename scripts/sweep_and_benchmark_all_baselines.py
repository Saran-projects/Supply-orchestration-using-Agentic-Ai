"""Comprehensive Baseline Tuning, Evaluation, and Reporting Script.

Executes:
1. Demonstrates how seeds affect each scenario (demand variances per seed, lead times, late delivery delays).
2. Benchmarks untuned EOQ on tuning seeds 100..104 to compute baseline mean bullwhip and 75% threshold.
3. Sweeps EOQ (order_quantity_multiplier, reorder_point_factor) on tuning seeds 100..104.
4. Sweeps (s, S) (s_factor, S_factor) on tuning seeds 100..104.
5. Selects lowest-cost configurations satisfying Mean BW <= 75% of untuned EOQ (or Pareto compromise).
6. Freezes chosen parameters in configs/sim.yaml.
7. Evaluates all 6 systems on test seeds 0..7:
   - System 0: Constant
   - System 1: (s, S) Untuned
   - System 1-tuned: (s, S) Tuned
   - System 2: EOQ Untuned
   - System 2-tuned: EOQ Tuned
   - System 2b: Smoothed OUT (Tuned)
8. Plots Total Cost vs Mean Bullwhip per scenario, marking reference baselines, sweep points, and Pareto frontiers.
9. Exports final CSV and prints per-scenario performance tables.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import sca.baselines  # noqa: F401
from sca.config import get_config
from sca.data.scenarios import LeadTimeSampler, generate_scenarios
from sca.sim.environment import BeerGameEnv
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import get_system

logging.basicConfig(level=logging.WARNING)

TUNING_SEEDS = [100, 101, 102, 103, 104]
TEST_SEEDS = list(range(8))  # 0 to 7
SCENARIOS = ["stable", "seasonal", "shock"]
WARMUP_STEPS = 5
COMMON_HORIZON = 40


def simulate_run(system_id: str, scenario_obj, seed: int, cfg, sys_config: Dict[str, Any] = None) -> Dict[str, Any]:
    sc = scenario_obj.clone() if hasattr(scenario_obj, "clone") else scenario_obj
    sc.lead_time_sampler = LeadTimeSampler(seed=seed)

    system = get_system(system_id, config=sys_config or {})
    system.calibrate(sc, config=sys_config or {})
    system.reset(scenario=sc)

    env = BeerGameEnv(
        config=cfg.sim,
        scenario=sc,
        seed=seed,
        steady_state_init=True,
    )
    states = env.reset(seed=seed)

    steps_to_run = min(COMMON_HORIZON, len(env.external_demand))
    for _ in range(steps_to_run):
        orders = system.decide_orders(states)
        states = env.step(orders)

    metrics = compute_simulation_metrics(env, warmup_steps=WARMUP_STEPS)

    ret_bw = float(metrics.bullwhip_ratio["retailer"])
    who_bw = float(metrics.bullwhip_ratio["wholesaler"])
    dis_bw = float(metrics.bullwhip_ratio["distributor"])
    man_bw = float(metrics.bullwhip_ratio["manufacturer"])
    mean_bw = (ret_bw + who_bw + dis_bw + man_bw) / 4.0

    ret_ov = float(metrics.order_variance_by_echelon.get("retailer", 0.0))
    who_ov = float(metrics.order_variance_by_echelon.get("wholesaler", 0.0))
    dis_ov = float(metrics.order_variance_by_echelon.get("distributor", 0.0))
    man_ov = float(metrics.order_variance_by_echelon.get("manufacturer", 0.0))

    ret_dv = float(metrics.demand_variance_by_echelon.get("retailer", 0.0))

    return {
        "total_cost": float(metrics.total_cost),
        "holding_cost": float(sum(metrics.holding_cost_by_echelon.values())),
        "backlog_cost": float(sum(metrics.backlog_cost_by_echelon.values())),
        "retailer_bullwhip": ret_bw,
        "wholesaler_bullwhip": who_bw,
        "distributor_bullwhip": dis_bw,
        "manufacturer_bullwhip": man_bw,
        "mean_bullwhip": mean_bw,
        "end_to_end_bullwhip": float(metrics.end_to_end_bullwhip),
        "order_var_retailer": ret_ov,
        "order_var_wholesaler": who_ov,
        "order_var_distributor": dis_ov,
        "order_var_manufacturer": man_ov,
        "demand_var_retailer": ret_dv,
        "service_level": float(metrics.service_level["retailer"]) * 100.0,
        "fill_rate": float(metrics.fill_rate["retailer"]) * 100.0,
        "stockout_steps": int(metrics.stockout_steps["retailer"]),
    }


def main():
    cfg = get_config()
    all_scenarios = generate_scenarios()

    print("=" * 110)
    print("STEP 1: HOW SEEDS AFFECT EACH SCENARIO (DEMAND VARIANCE & RANDOMIZATION MECHANISMS)")
    print("=" * 110)
    print("Under the circular block bootstrap (block size = 7 days):")
    print("- Customer Demand Series: Each seed generates an independent, non-overlapping circular block trajectory.")
    print("- Transit Lead Times: Sampled from empirical DataCo distribution using seed-specific RNG.")
    print("- Late Deliveries: Late delivery delays (p = 0.5483) drawn independently per shipment using seed-specific RNG.")
    print("\nDemand Variance per Seed (evaluated steps 5..39):")

    demand_var_table = []
    for sc_name in SCENARIOS:
        sc = all_scenarios[sc_name]
        row = {"Scenario": sc_name.capitalize()}
        vars_list = []
        for s in TEST_SEEDS:
            env = BeerGameEnv(config=cfg.sim, scenario=sc, seed=s)
            dem_slice = env.external_demand[WARMUP_STEPS:COMMON_HORIZON]
            v = float(np.var(dem_slice, ddof=1))
            row[f"Seed {s}"] = f"{v:.1f}"
            vars_list.append(v)
        row["Mean ± Std"] = f"{np.mean(vars_list):.1f} ± {np.std(vars_list):.1f}"
        demand_var_table.append(row)
    df_dvar = pd.DataFrame(demand_var_table)
    print(df_dvar.to_markdown(index=False))

    # STEP 2: Benchmark untuned EOQ on tuning seeds 100..104
    print("\n" + "=" * 110)
    print("STEP 2: TUNING SEED BENCHMARK (UNTUNED EOQ ON SEEDS 100..104)")
    print("=" * 110)

    eoq_tune_records = []
    for sc_name in SCENARIOS:
        sc = all_scenarios[sc_name]
        for s in TUNING_SEEDS:
            res = simulate_run("baseline_eoq", sc, s, cfg)
            res["scenario"] = sc_name
            res["seed"] = s
            eoq_tune_records.append(res)
    df_eoq_tune = pd.DataFrame(eoq_tune_records)

    eoq_untuned_cost = df_eoq_tune["total_cost"].mean()
    eoq_untuned_mean_bw = df_eoq_tune["mean_bullwhip"].mean()
    bw_threshold = 0.75 * eoq_untuned_mean_bw

    print(f"Untuned EOQ Mean Cost on Tuning Seeds:      ${eoq_untuned_cost:,.2f}")
    print(f"Untuned EOQ Mean Bullwhip on Tuning Seeds: {eoq_untuned_mean_bw:.3f}")
    print(f"TARGET 75% BULLWHIP THRESHOLD:             <= {bw_threshold:.3f}")

    # STEP 3: Sweep EOQ
    print("\n" + "=" * 110)
    print("STEP 3: SWEEPING SYSTEM 2 (EOQ) ON TUNING SEEDS 100..104")
    print("=" * 110)
    q_mults = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
    rop_factors = [0.80, 0.90, 1.00, 1.10, 1.20]

    eoq_sweep_results = []
    for q_m, rop_f in itertools.product(q_mults, rop_factors):
        sub_costs, sub_bws, sub_sls = [], [], []
        for sc_name in SCENARIOS:
            sc = all_scenarios[sc_name]
            for s in TUNING_SEEDS:
                r = simulate_run("baseline_eoq", sc, s, cfg, {
                    "order_quantity_multiplier": q_m,
                    "reorder_point_factor": rop_f,
                })
                sub_costs.append(r["total_cost"])
                sub_bws.append(r["mean_bullwhip"])
                sub_sls.append(r["service_level"])
        c_m, bw_m = float(np.mean(sub_costs)), float(np.mean(sub_bws))
        eoq_sweep_results.append({
            "order_quantity_multiplier": q_m,
            "reorder_point_factor": rop_f,
            "mean_cost": c_m,
            "mean_bullwhip": bw_m,
            "service_level": float(np.mean(sub_sls)),
            "qualifies": bool(bw_m <= bw_threshold),
        })
    df_eoq_sw = pd.DataFrame(eoq_sweep_results)
    eoq_qualifying = df_eoq_sw[df_eoq_sw["qualifies"]]

    if len(eoq_qualifying) > 0:
        best_eoq = eoq_qualifying.sort_values("mean_cost").iloc[0]
        print(f"EOQ Qualifying configurations: {len(eoq_qualifying)} / {len(df_eoq_sw)}")
    else:
        print(f"No EOQ configuration met BW <= {bw_threshold:.3f}. Selecting best Pareto compromise:")
        # Sort by mean bullwhip and cost
        best_eoq = df_eoq_sw.sort_values(["mean_bullwhip", "mean_cost"]).iloc[0]

    print(f"Chosen Tuned EOQ: Q_mult={best_eoq['order_quantity_multiplier']}, ROP_factor={best_eoq['reorder_point_factor']} | Cost=${best_eoq['mean_cost']:,.2f}, Mean BW={best_eoq['mean_bullwhip']:.3f}")

    # STEP 4: Sweep (s, S)
    print("\n" + "=" * 110)
    print("STEP 4: SWEEPING SYSTEM 1 (s, S) ON TUNING SEEDS 100..104")
    print("=" * 110)
    s_factors = [0.80, 0.90, 1.00, 1.10, 1.20]
    S_factors = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

    ss_sweep_results = []
    for s_f, S_f in itertools.product(s_factors, S_factors):
        sub_costs, sub_bws, sub_sls = [], [], []
        for sc_name in SCENARIOS:
            sc = all_scenarios[sc_name]
            for s in TUNING_SEEDS:
                r = simulate_run("baseline_s_S", sc, s, cfg, {
                    "s_factor": s_f,
                    "S_factor": S_f,
                })
                sub_costs.append(r["total_cost"])
                sub_bws.append(r["mean_bullwhip"])
                sub_sls.append(r["service_level"])
        c_m, bw_m = float(np.mean(sub_costs)), float(np.mean(sub_bws))
        ss_sweep_results.append({
            "s_factor": s_f,
            "S_factor": S_f,
            "mean_cost": c_m,
            "mean_bullwhip": bw_m,
            "service_level": float(np.mean(sub_sls)),
            "qualifies": bool(bw_m <= bw_threshold),
        })
    df_ss_sw = pd.DataFrame(ss_sweep_results)
    ss_qualifying = df_ss_sw[df_ss_sw["qualifies"]]

    if len(ss_qualifying) > 0:
        best_ss = ss_qualifying.sort_values("mean_cost").iloc[0]
        print(f"(s, S) Qualifying configurations: {len(ss_qualifying)} / {len(df_ss_sw)}")
    else:
        print(f"No (s, S) configuration met BW <= {bw_threshold:.3f}. Selecting best Pareto compromise:")
        best_ss = df_ss_sw.sort_values(["mean_bullwhip", "mean_cost"]).iloc[0]

    print(f"Chosen Tuned (s, S): s_factor={best_ss['s_factor']}, S_factor={best_ss['S_factor']} | Cost=${best_ss['mean_cost']:,.2f}, Mean BW={best_ss['mean_bullwhip']:.3f}")

    # STEP 5: Freeze Chosen Parameters in configs/sim.yaml
    sim_yaml_path = Path("configs/sim.yaml")
    with open(sim_yaml_path, "r", encoding="utf-8") as f:
        sim_data = yaml.safe_load(f)

    sim_data["four_echelon"]["tuned_eoq_q_mult"] = float(best_eoq["order_quantity_multiplier"])
    sim_data["four_echelon"]["tuned_eoq_rop_factor"] = float(best_eoq["reorder_point_factor"])
    sim_data["four_echelon"]["tuned_ss_s_factor"] = float(best_ss["s_factor"])
    sim_data["four_echelon"]["tuned_ss_S_factor"] = float(best_ss["S_factor"])

    with open(sim_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sim_data, f, sort_keys=False)

    print("\nFrozen tuned baseline parameters into configs/sim.yaml.")

    # STEP 6: Benchmark ALL systems on Test Seeds 0..7
    print("\n" + "=" * 110)
    print("STEP 6: FINAL BENCHMARK EVALUATION ON SEEDS 0 TO 7 (8 SEEDS x 3 SCENARIOS = 24 RUNS PER SYSTEM)")
    print("=" * 110)

    systems_to_eval = [
        ("system_0", "System 0: Constant", "baseline_constant", {}),
        ("system_1", "System 1: (s, S) [Untuned]", "baseline_s_S", {}),
        ("system_1_tuned", "System 1-tuned: (s, S) [Tuned]", "baseline_s_S", {
            "s_factor": float(best_ss["s_factor"]),
            "S_factor": float(best_ss["S_factor"]),
        }),
        ("system_2", "System 2: EOQ [Untuned]", "baseline_eoq", {}),
        ("system_2_tuned", "System 2-tuned: EOQ [Tuned]", "baseline_eoq", {
            "order_quantity_multiplier": float(best_eoq["order_quantity_multiplier"]),
            "reorder_point_factor": float(best_eoq["reorder_point_factor"]),
        }),
        ("system_2b", "System 2b: Smoothed OUT", "baseline_smoothed_order_up_to", {
            "alpha": 0.8,
            "beta": 0.2,
            "z": 2.0,
            "delta": 0.05,
            "forecast_cap_multiple": 1.5,
            "include_lead_time_variance": True,
            "db_path": False,
        }),
    ]

    all_test_records = []
    for sys_key, sys_label, sys_id, sys_cfg in systems_to_eval:
        for sc_name in SCENARIOS:
            sc = all_scenarios[sc_name]
            for s in TEST_SEEDS:
                res = simulate_run(sys_id, sc, s, cfg, sys_cfg)
                res["system_key"] = sys_key
                res["system"] = sys_label
                res["scenario"] = sc_name
                res["seed"] = s
                all_test_records.append(res)

    df_final = pd.DataFrame(all_test_records)
    df_final.to_csv("results/comprehensive_baselines_seeds0_7.csv", index=False)

    # Print per-scenario summary tables
    for sc_name in SCENARIOS:
        sub = df_final[df_final["scenario"] == sc_name]
        print(f"\n### {sc_name.upper()} SCENARIO BENCHMARK (SEEDS 0-7, n=8 runs per system)")
        print("-" * 140)

        rows = []
        for _, sys_label, _, _ in systems_to_eval:
            s_df = sub[sub["system"] == sys_label]
            c_m, c_s = s_df["total_cost"].mean(), s_df["total_cost"].std()
            h_m, _h_s = s_df["holding_cost"].mean(), s_df["holding_cost"].std()
            b_m, _b_s = s_df["backlog_cost"].mean(), s_df["backlog_cost"].std()
            ret_bw_m, ret_bw_s = s_df["retailer_bullwhip"].mean(), s_df["retailer_bullwhip"].std()
            who_bw_m, _who_bw_s = s_df["wholesaler_bullwhip"].mean(), s_df["wholesaler_bullwhip"].std()
            dis_bw_m, _dis_bw_s = s_df["distributor_bullwhip"].mean(), s_df["distributor_bullwhip"].std()
            man_bw_m, _man_bw_s = s_df["manufacturer_bullwhip"].mean(), s_df["manufacturer_bullwhip"].std()
            e2e_m, e2e_s = s_df["end_to_end_bullwhip"].mean(), s_df["end_to_end_bullwhip"].std()
            mbw_m, mbw_s = s_df["mean_bullwhip"].mean(), s_df["mean_bullwhip"].std()
            sl_m, sl_s = s_df["service_level"].mean(), s_df["service_level"].std()
            fr_m, _fr_s = s_df["fill_rate"].mean(), s_df["fill_rate"].std()
            so_m, so_s = s_df["stockout_steps"].mean(), s_df["stockout_steps"].std()

            rows.append({
                "System": sys_label,
                "Total Cost ($)": f"${c_m:,.2f} ± {c_s:,.1f}",
                "Holding ($)": f"${h_m:,.1f}",
                "Backlog ($)": f"${b_m:,.1f}",
                "Ret BW": f"{ret_bw_m:.2f} ± {ret_bw_s:.2f}",
                "Who BW": f"{who_bw_m:.2f}",
                "Dis BW": f"{dis_bw_m:.2f}",
                "Mfr BW": f"{man_bw_m:.2f}",
                "End-to-End BW": f"{e2e_m:.2f} ± {e2e_s:.2f}",
                "Mean BW": f"{mbw_m:.2f} ± {mbw_s:.2f}",
                "Service Level (%)": f"{sl_m:.1f}% ± {sl_s:.1f}%",
                "Fill Rate (%)": f"{fr_m:.2f}%",
                "Stockouts": f"{so_m:.1f} ± {so_s:.1f}",
            })
        print(pd.DataFrame(rows).to_markdown(index=False))

    # Print absolute order variances table per scenario
    print("\n" + "=" * 110)
    print("ABSOLUTE ORDER VARIANCE PER ECHELON (MEAN ± STD ACROSS SEEDS 0-7)")
    print("=" * 110)
    for sc_name in SCENARIOS:
        sub = df_final[df_final["scenario"] == sc_name]
        print(f"\n#### Scenario: {sc_name.upper()} - Absolute Order Variance")
        vrows = []
        for _, sys_label, _, _ in systems_to_eval:
            s_df = sub[sub["system"] == sys_label]
            vrows.append({
                "System": sys_label,
                "Customer Dem Var": f"{s_df['demand_var_retailer'].mean():,.1f}",
                "Retailer Ord Var": f"{s_df['order_var_retailer'].mean():,.1f} ± {s_df['order_var_retailer'].std():,.1f}",
                "Wholesaler Ord Var": f"{s_df['order_var_wholesaler'].mean():,.1f} ± {s_df['order_var_wholesaler'].std():,.1f}",
                "Distributor Ord Var": f"{s_df['order_var_distributor'].mean():,.1f} ± {s_df['order_var_distributor'].std():,.1f}",
                "Manufacturer Ord Var": f"{s_df['order_var_manufacturer'].mean():,.1f} ± {s_df['order_var_manufacturer'].std():,.1f}",
            })
        print(pd.DataFrame(vrows).to_markdown(index=False))

    # STEP 7: Plot Cost vs Mean Bullwhip per Scenario
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    palette = {
        "System 0: Constant": ("#7f7f7f", "X", 100),
        "System 1: (s, S) [Untuned]": ("#ff7f0e", "^", 120),
        "System 1-tuned: (s, S) [Tuned]": ("#d62728", "^", 150),
        "System 2: EOQ [Untuned]": ("#1f77b4", "s", 120),
        "System 2-tuned: EOQ [Tuned]": ("#17becf", "s", 150),
        "System 2b: Smoothed OUT": ("#2ca02c", "*", 220),
    }

    for ax, sc_name in zip(axes, SCENARIOS, strict=False):
        sub = df_final[df_final["scenario"] == sc_name]
        for _, sys_label, _, _ in systems_to_eval:
            s_df = sub[sub["system"] == sys_label]
            c_val = s_df["total_cost"].mean()
            bw_val = s_df["mean_bullwhip"].mean()
            col, marker, sz = palette[sys_label]
            ax.scatter([bw_val], [c_val], color=col, marker=marker, s=sz, edgecolor="black", linewidth=1.2, label=sys_label, zorder=5)

        ax.set_title(f"{sc_name.capitalize()} Scenario", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Bullwhip across Echelons", fontsize=11)
        ax.set_ylabel("Total Chain Cost ($)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        if sc_name == "stable":
            ax.legend(fontsize=9, loc="upper right")

    plt.suptitle("Total Cost vs Mean Bullwhip across Baselines (Seeds 0-7 with Independent Demand Paths)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_file = Path("results/baselines_cost_vs_bullwhip_scenarios.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFinal plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
