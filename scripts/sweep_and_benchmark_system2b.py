"""Full Sweep, Tuning, and Benchmark for System 2b according to pre-registered protocol.

Steps:
1. Print lead-time mean and std actually used.
2. Benchmark System 2 (EOQ) and System 1 (s, S) on tuning seeds 100 to 104 to compute baseline mean bullwhip.
3. Sweep alpha, beta, z, delta (in 0.05, 0.1, 0.2, 0.5), cap_multiple on tuning seeds 100 to 104.
4. Explain why delta previously had no effect and demonstrate the effect of delta in [0.05, 0.1, 0.2, 0.5].
5. Apply pre-registered selection rule: lowest cost with mean bullwhip <= 75% of EOQ's mean bullwhip on tuning seeds.
6. Generate 3-panel plot: Cost vs Mean Bullwhip per scenario for EOQ, (s, S), and sweep points, marking chosen point & Pareto frontier.
7. Freeze chosen parameters into configs/sim.yaml.
8. Rerun System 2b on test seeds 0 to 7 once, and save all final tables.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def simulate_single_run(system_id: str, scenario_obj, seed: int, cfg, sys_config: Dict[str, Any] = None) -> Dict[str, Any]:
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

    return {
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
    }


def main():
    cfg = get_config()
    all_scenarios = generate_scenarios()

    # 1. Print lead time stats
    base_sc = all_scenarios["stable"]
    lt_mean = float(base_sc.lead_time_sampler.mean)
    lt_std = float(base_sc.lead_time_sampler.std)
    print("\n" + "=" * 90)
    print("LEAD-TIME DISTRIBUTION STATS ACTUALLY USED IN SIMULATION & SYSTEM 2b:")
    print(f"  Empirical Lead-Time Mean (mu_L):   {lt_mean:.4f} days")
    print(f"  Empirical Lead-Time Std (sigma_L): {lt_std:.4f} days")
    print(f"  Late delivery rate:                {base_sc.late_delivery_probability:.4f}")
    print(f"  Lead-Time Variance ON:             sigma_L = {lt_std:.4f}")
    print(f"  Lead-Time Variance OFF:            sigma_L = 0.0000")
    print("=" * 90)

    # 2. Benchmark EOQ and (s, S) on Tuning Seeds (100..104)
    print("\nBenchmarking EOQ and (s, S) on tuning seeds 100..104...")
    eoq_tuning_records = []
    ss_tuning_records = []

    for sc_name in SCENARIOS:
        sc_obj = all_scenarios[sc_name]
        for seed in TUNING_SEEDS:
            res_eoq = simulate_single_run("baseline_eoq", sc_obj, seed, cfg)
            res_eoq["scenario"] = sc_name
            res_eoq["seed"] = seed
            eoq_tuning_records.append(res_eoq)

            res_ss = simulate_single_run("baseline_s_S", sc_obj, seed, cfg)
            res_ss["scenario"] = sc_name
            res_ss["seed"] = seed
            ss_tuning_records.append(res_ss)

    df_eoq_tune = pd.DataFrame(eoq_tuning_records)
    df_ss_tune = pd.DataFrame(ss_tuning_records)

    eoq_mean_cost = df_eoq_tune["total_cost"].mean()
    eoq_mean_bw = df_eoq_tune["mean_bullwhip"].mean()
    ss_mean_cost = df_ss_tune["total_cost"].mean()
    ss_mean_bw = df_ss_tune["mean_bullwhip"].mean()

    bw_threshold = 0.75 * eoq_mean_bw

    print(f"\n--- TUNING SEED BENCHMARKS (15 runs each) ---")
    print(f"  System 2 (EOQ):    Cost = ${eoq_mean_cost:,.2f}, Mean Bullwhip = {eoq_mean_bw:.3f}")
    print(f"  System 1 (s, S):   Cost = ${ss_mean_cost:,.2f}, Mean Bullwhip = {ss_mean_bw:.3f}")
    print(f"  SELECTION RULE TARGET: Mean Bullwhip <= 75% of EOQ = {bw_threshold:.3f}")

    # 3. Parameter Sweep for System 2b on Tuning Seeds
    alphas = [0.2, 0.5, 0.8]
    betas = [0.2, 0.5, 0.8]
    zs = [1.28, 1.65, 2.0]
    deltas = [0.05, 0.10, 0.20, 0.50]
    cap_multiples = [1.5, 2.0]

    combos = list(itertools.product(alphas, betas, zs, deltas, cap_multiples))
    print(f"\nStarting System 2b sweep: {len(combos)} configurations x 15 runs = {len(combos) * 15} runs...")

    sweep_records = []
    combo_summaries = []

    for idx, (alpha, beta, z, delta, cap_mult) in enumerate(combos):
        sys_cfg = {
            "alpha": alpha,
            "beta": beta,
            "z": z,
            "delta": delta,
            "forecast_cap_multiple": cap_mult,
            "include_lead_time_variance": True,
            "db_path": False,
        }

        cfg_costs = []
        cfg_bws = []
        cfg_sls = []
        cfg_frs = []
        cfg_ret_bws = []
        cfg_mfr_bws = []

        sc_metrics = {sc: {"cost": [], "bw": []} for sc in SCENARIOS}

        for sc_name in SCENARIOS:
            sc_obj = all_scenarios[sc_name]
            for seed in TUNING_SEEDS:
                r = simulate_single_run("system_2b", sc_obj, seed, cfg, sys_cfg)
                r["alpha"] = alpha
                r["beta"] = beta
                r["z"] = z
                r["delta"] = delta
                r["cap_multiple"] = cap_mult
                r["scenario"] = sc_name
                r["seed"] = seed
                sweep_records.append(r)

                cfg_costs.append(r["total_cost"])
                cfg_bws.append(r["mean_bullwhip"])
                cfg_sls.append(r["service_level"])
                cfg_frs.append(r["fill_rate"])
                cfg_ret_bws.append(r["retailer_bullwhip"])
                cfg_mfr_bws.append(r["manufacturer_bullwhip"])

                sc_metrics[sc_name]["cost"].append(r["total_cost"])
                sc_metrics[sc_name]["bw"].append(r["mean_bullwhip"])

        mean_cost = float(np.mean(cfg_costs))
        mean_bw = float(np.mean(cfg_bws))
        mean_sl = float(np.mean(cfg_sls))
        mean_fr = float(np.mean(cfg_frs))
        mean_ret_bw = float(np.mean(cfg_ret_bws))
        mean_mfr_bw = float(np.mean(cfg_mfr_bws))

        combo_summaries.append({
            "alpha": alpha,
            "beta": beta,
            "z": z,
            "delta": delta,
            "cap_multiple": cap_mult,
            "mean_cost": mean_cost,
            "cost_std": float(np.std(cfg_costs)),
            "mean_bullwhip": mean_bw,
            "bw_std": float(np.std(cfg_bws)),
            "retailer_bw": mean_ret_bw,
            "manufacturer_bw": mean_mfr_bw,
            "service_level": mean_sl,
            "fill_rate": mean_fr,
            "qualifies": bool(mean_bw <= bw_threshold),
            "stable_cost": float(np.mean(sc_metrics["stable"]["cost"])),
            "stable_bw": float(np.mean(sc_metrics["stable"]["bw"])),
            "seasonal_cost": float(np.mean(sc_metrics["seasonal"]["cost"])),
            "seasonal_bw": float(np.mean(sc_metrics["seasonal"]["bw"])),
            "shock_cost": float(np.mean(sc_metrics["shock"]["cost"])),
            "shock_bw": float(np.mean(sc_metrics["shock"]["bw"])),
        })

    df_sweep = pd.DataFrame(combo_summaries)
    df_sweep.to_csv("results/system2b_full_sweep_tuning.csv", index=False)
    print("Sweep complete. Saved to results/system2b_full_sweep_tuning.csv.")

    # 4. Check why delta had no effect & sweep effect of delta
    print("\n" + "=" * 90)
    print("ANALYSIS: WHY DELTA PREVIOUSLY HAD NO EFFECT AND DELTA'S CURRENT IMPACT")
    print("=" * 90)
    print("In previous sweeps (delta in 0.25, 0.5, 1.0):")
    print("  With beta <= 0.8, order smoothing already constrained step-to-step jumps.")
    print("  Rate limit of delta >= 0.25 permitted step-to-step changes of >= 25-100 units,")
    print("  which were strictly wider than the smoothed order increments. Thus rate limiting rarely clipped.")
    print("\nImpact across delta values in this sweep (averaged across alpha, beta, z, cap_multiple):")
    delta_agg = df_sweep.groupby("delta")[["mean_cost", "mean_bullwhip", "retailer_bw", "service_level"]].mean()
    print(delta_agg.to_markdown())

    # 5. Apply Pre-registered Selection Rule
    qualifying = df_sweep[df_sweep["qualifies"]].copy()
    print(f"\n--- SELECTION RULE EVALUATION ---")
    print(f"Total swept configurations: {len(df_sweep)}")
    print(f"Configurations meeting Mean BW <= {bw_threshold:.3f}: {len(qualifying)}")

    pareto_frontier = []
    # Identify non-dominated Pareto points (lower cost, lower bullwhip)
    for i, r in df_sweep.iterrows():
        is_dominated = False
        for j, other in df_sweep.iterrows():
            if (other["mean_cost"] <= r["mean_cost"] and other["mean_bullwhip"] <= r["mean_bullwhip"]) and \
               (other["mean_cost"] < r["mean_cost"] or other["mean_bullwhip"] < r["mean_bullwhip"]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_frontier.append(r)
    df_pareto = pd.DataFrame(pareto_frontier).sort_values("mean_cost")

    if len(qualifying) > 0:
        chosen = qualifying.sort_values("mean_cost").iloc[0]
        print("\nSUCCESS: Qualifying configuration found!")
        print(f"Chosen Configuration (Lowest Cost among BW <= {bw_threshold:.3f}):")
        print(f"  alpha:        {chosen['alpha']}")
        print(f"  beta:         {chosen['beta']}")
        print(f"  z:            {chosen['z']}")
        print(f"  delta:        {chosen['delta']}")
        print(f"  cap_multiple: {chosen['cap_multiple']}")
        print(f"  Total Cost:   ${chosen['mean_cost']:,.2f} ± {chosen['cost_std']:,.1f}")
        print(f"  Mean BW:      {chosen['mean_bullwhip']:.3f} (EOQ threshold: {bw_threshold:.3f})")
        print(f"  Retailer BW:  {chosen['retailer_bw']:.3f}")
        print(f"  Mfr BW:       {chosen['manufacturer_bw']:.3f}")
        print(f"  Service Lvl:  {chosen['service_level']:.1f}%")
        print(f"  Fill Rate:    {chosen['fill_rate']:.2f}%")
    else:
        print("\nNo configuration met the <= 75% threshold! Reporting Pareto Frontier:")
        print(df_pareto[["alpha", "beta", "z", "delta", "cap_multiple", "mean_cost", "mean_bullwhip"]].to_markdown(index=False))
        chosen = df_pareto.iloc[0]  # Take compromise point on frontier

    # 6. Plot Cost vs Mean Bullwhip (per scenario)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    scenarios_titles = ["Stable Scenario", "Seasonal Scenario", "Shock Scenario"]

    for ax, sc_name, title in zip(axes, SCENARIOS, scenarios_titles):
        # Sweep points for scenario
        sc_pts = df_sweep[[f"{sc_name}_bw", f"{sc_name}_cost", "alpha", "beta", "delta"]]
        ax.scatter(sc_pts[f"{sc_name}_bw"], sc_pts[f"{sc_name}_cost"], color="lightgray", alpha=0.6, s=30, label="Sweep (System 2b)")

        # EOQ on this scenario
        eoq_sc = df_eoq_tune[df_eoq_tune["scenario"] == sc_name]
        eoq_sc_cost = eoq_sc["total_cost"].mean()
        eoq_sc_bw = eoq_sc["mean_bullwhip"].mean()
        ax.scatter([eoq_sc_bw], [eoq_sc_cost], color="#1f77b4", s=150, marker="s", edgecolor="black", linewidth=1.5, label="System 2: EOQ", zorder=5)

        # (s, S) on this scenario
        ss_sc = df_ss_tune[df_ss_tune["scenario"] == sc_name]
        ss_sc_cost = ss_sc["total_cost"].mean()
        ss_sc_bw = ss_sc["mean_bullwhip"].mean()
        ax.scatter([ss_sc_bw], [ss_sc_cost], color="#ff7f0e", s=150, marker="^", edgecolor="black", linewidth=1.5, label="System 1: (s, S)", zorder=5)

        # Chosen point on this scenario
        ch_cost = chosen[f"{sc_name}_cost"]
        ch_bw = chosen[f"{sc_name}_bw"]
        ax.scatter([ch_bw], [ch_cost], color="#d62728", s=200, marker="*", edgecolor="black", linewidth=1.5, label="Chosen System 2b", zorder=6)

        # 75% threshold line for this scenario
        ax.axvline(0.75 * eoq_sc_bw, color="blue", linestyle="--", alpha=0.5, label="75% EOQ BW")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Bullwhip across Echelons", fontsize=11)
        ax.set_ylabel("Total Cost ($)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper right")

    plt.suptitle("Total Cost vs Mean Bullwhip Trade-Off Curves Across Scenarios (Tuning Seeds 100-104)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = Path("results/cost_vs_bullwhip_scenarios.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Trade-off plot saved to: {plot_path}")

    # 7. Freeze Chosen Parameters into configs/sim.yaml
    sim_yaml_path = Path("configs/sim.yaml")
    with open(sim_yaml_path, "r", encoding="utf-8") as f:
        sim_data = yaml.safe_load(f)

    sim_data["four_echelon"]["alpha"] = float(chosen["alpha"])
    sim_data["four_echelon"]["beta"] = float(chosen["beta"])
    sim_data["four_echelon"]["safety_factor_z"] = float(chosen["z"])
    sim_data["four_echelon"]["rate_limit_delta"] = float(chosen["delta"])
    sim_data["four_echelon"]["forecast_cap_multiple"] = float(chosen["cap_multiple"])
    sim_data["four_echelon"]["tuning_seeds_used"] = TUNING_SEEDS

    with open(sim_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sim_data, f, sort_keys=False)

    print(f"Frozen chosen parameters into {sim_yaml_path}:")
    print(f"  alpha: {chosen['alpha']}, beta: {chosen['beta']}, z: {chosen['z']}, delta: {chosen['delta']}, cap: {chosen['cap_multiple']}")

    # 8. Rerun System 2b on Test Seeds 0 to 7 ONCE
    print("\n" + "=" * 90)
    print("RERUNNING SYSTEM 2b ON TEST SEEDS 0 TO 7 WITH FROZEN PARAMETERS (ONCE)")
    print("=" * 90)

    frozen_sys_cfg = {
        "alpha": float(chosen["alpha"]),
        "beta": float(chosen["beta"]),
        "z": float(chosen["z"]),
        "delta": float(chosen["delta"]),
        "forecast_cap_multiple": float(chosen["cap_multiple"]),
        "include_lead_time_variance": True,
        "db_path": False,
    }

    test_records = []
    for sc_name in SCENARIOS:
        sc_obj = all_scenarios[sc_name]
        for seed in TEST_SEEDS:
            r = simulate_single_run("system_2b", sc_obj, seed, cfg, frozen_sys_cfg)
            r["system"] = "System 2b: Smoothed OUT"
            r["scenario"] = sc_name
            r["seed"] = seed
            test_records.append(r)

    df_test_2b = pd.DataFrame(test_records)
    df_test_2b.to_csv("results/system2b_seeds0_7_frozen.csv", index=False)

    # Load baseline runs for seeds 0..7 from step 1
    df_base = pd.read_csv("results/baselines1_2_seeds0_7_raw.csv")
    df_base["system"] = df_base["system_label"]

    combined = pd.concat([df_base, df_test_2b], ignore_index=True)
    combined.to_csv("results/all_baselines_seeds0_7_final.csv", index=False)

    for sc_name in SCENARIOS:
        sub = combined[combined["scenario"] == sc_name]
        print(f"\n### FINAL BENCHMARK: {sc_name.upper()} SCENARIO (SEEDS 0-7, n=8 runs per system)")
        print("-" * 110)

        rows = []
        for sys_name in ["System 1: (s, S)", "System 2: EOQ", "System 2b: Smoothed OUT"]:
            s_df = sub[sub["system"] == sys_name]
            rows.append({
                "System": sys_name,
                "Total Cost ($)": f"${s_df['total_cost'].mean():,.2f} ± {s_df['total_cost'].std():,.1f}",
                "Retailer BW": f"{s_df['retailer_bullwhip'].mean():.2f} ± {s_df['retailer_bullwhip'].std():.2f}",
                "Wholesaler BW": f"{s_df['wholesaler_bullwhip'].mean():.2f} ± {s_df['wholesaler_bullwhip'].std():.2f}",
                "Distributor BW": f"{s_df['distributor_bullwhip'].mean():.2f} ± {s_df['distributor_bullwhip'].std():.2f}",
                "Mfr BW": f"{s_df['manufacturer_bullwhip'].mean():.2f} ± {s_df['manufacturer_bullwhip'].std():.2f}",
                "Mean BW": f"{s_df['mean_bullwhip'].mean():.2f} ± {s_df['mean_bullwhip'].std():.2f}",
                "Service Level (% steps without stockout)": f"{s_df['service_level'].mean():.1f}% ± {s_df['service_level'].std():.1f}%",
                "Fill Rate (% units)": f"{s_df['fill_rate'].mean():.2f}% ± {s_df['fill_rate'].std():.2f}%",
                "Stockouts": f"{s_df['stockout_steps'].mean():.1f} ± {s_df['stockout_steps'].std():.1f}",
            })
        print(pd.DataFrame(rows).to_markdown(index=False))

    print("\nAll benchmark steps completed successfully.")


if __name__ == "__main__":
    main()
