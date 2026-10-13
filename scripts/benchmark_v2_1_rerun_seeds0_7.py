"""Benchmark v2.1 Final Test Evaluation on Seeds 0 to 7.

Reruns seeds 0 to 7 (n = 8) across scenarios (stable, seasonal, shock):
- Evaluates Reference baselines (System 0, EOQ Untuned, (s,S) Untuned, System 2b Untuned, Base-Stock, Oracle)
- Evaluates Tuned baselines from results/baselines_frozen.json (EOQ Tuned, (s,S) Tuned, System 2b Tuned, System 2c Tuned)
- Prints per-scenario tables (mean +/- std, n=8):
  - Cost with holding and backlog split
  - End-to-end and per-echelon bullwhip
  - Absolute order variances
  - Fill rate, step SL, stockouts
- Plots:
  1. Cost vs end-to-end bullwhip with Pareto frontier (results/cost_vs_bullwhip_pareto.png)
  2. Demand vs orders per echelon in shock scenario (results/shock_demand_vs_orders_echelons.png)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sca.config import get_config
from sca.data.scenarios import generate_scenarios
from sca.sim.environment import BeerGameEnv
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import get_system

logging.disable(logging.INFO)
logging.basicConfig(level=logging.WARNING)

TEST_SEEDS = list(range(8))
SCENARIOS = ["stable", "seasonal", "shock"]
COMMON_HORIZON = 40
WARMUP_STEPS = 5
INIT_INV = 687.0
HOLDING_COST = 0.50
BACKLOG_COST = 2.50


def load_frozen_configs() -> Dict[str, Any]:
    frozen_path = Path("results/baselines_frozen.json")
    if not frozen_path.exists():
        raise FileNotFoundError(f"Frozen baselines file not found: {frozen_path}")
    with open(frozen_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval():
    frozen_data = load_frozen_configs()
    tuned_policies = frozen_data["tuned_policies"]

    eoq_tuned_cfg = tuned_policies["system_2_eoq"]["parameters"]
    ss_tuned_cfg = tuned_policies["system_1_s_S"]["parameters"]
    s2b_tuned_cfg = tuned_policies["system_2b_smoothed_out"]["parameters"]
    s2c_tuned_cfg = tuned_policies["system_2c_adaptive_smoothed_out"]["parameters"]

    systems = [
        ("System 0: Constant", "baseline_constant", {}, True),
        ("EOQ Untuned", "baseline_eoq", {}, True),
        ("EOQ Tuned", "baseline_eoq", eoq_tuned_cfg, False),
        ("(s, S) Untuned", "baseline_s_S", {}, True),
        ("(s, S) Tuned", "baseline_s_S", ss_tuned_cfg, False),
        (
            "System 2b Untuned",
            "baseline_smoothed_order_up_to",
            {
                "alpha": 0.8,
                "beta": 0.8,
                "z": 2.0,
                "delta": 0.5,
                "forecast_cap_multiple": 1.5,
                "info_sharing": False,
                "include_lead_time_variance": True,
                "db_path": False,
            },
            True,
        ),
        ("System 2b Tuned", "baseline_smoothed_order_up_to", s2b_tuned_cfg, False),
        ("System 2c Tuned", "baseline_2c", s2c_tuned_cfg, False),
        ("Base-Stock (z=2.0)", "baseline_base_stock", {"z": 2.0}, True),
        ("Oracle (Reference)", "baseline_oracle", {"lead_time_offset": 4}, True),
    ]

    cfg = get_config()
    sim_cfg = cfg.sim.model_copy(deep=True)
    sim_cfg.four_echelon.initial_inventory = INIT_INV
    sim_cfg.four_echelon.holding_cost_per_unit = HOLDING_COST
    sim_cfg.four_echelon.backlog_cost_per_unit = BACKLOG_COST
    sim_cfg.four_echelon.steady_state_init = True
    sim_cfg.four_echelon.independent_demand_paths = True

    all_scenarios = generate_scenarios()

    records: List[Dict[str, Any]] = []
    # Collect echelon trajectories for shock scenario (seed 0)
    shock_trajectories: Dict[str, Dict[str, List[float]]] = {}

    print("=" * 110)
    print(f"RUNNING BENCHMARK v2.1 ON TEST SEEDS 0 TO 7 (n = {len(TEST_SEEDS)})")
    print(f"Environment: I_0 = {INIT_INV}, h = ${HOLDING_COST:.2f}, b = ${BACKLOG_COST:.2f} (b/h = 5.0x)")
    print("=" * 110)

    for sys_label, sys_id, custom_cfg, is_ref in systems:
        system = get_system(sys_id, config=custom_cfg)

        for sc_name in SCENARIOS:
            sc = all_scenarios[sc_name]
            system.calibrate(sc, config=custom_cfg)

            for seed in TEST_SEEDS:
                system.reset(sc)
                env = BeerGameEnv(
                    config=sim_cfg,
                    scenario=sc,
                    seed=seed,
                    steady_state_init=True,
                )
                states = env.reset(seed=seed)

                if sys_id == "baseline_oracle" and hasattr(system, "set_demand"):
                    system.set_demand(env.external_demand)

                step_orders: Dict[str, List[float]] = {
                    "retailer": [],
                    "wholesaler": [],
                    "distributor": [],
                    "manufacturer": [],
                }
                cust_demands: List[float] = []

                for _ in range(COMMON_HORIZON):
                    orders = system.decide_orders(states)
                    for e in step_orders:
                        step_orders[e].append(float(orders.get(e, 0.0)))
                    cust_demands.append(float(states["retailer"].current_demand))
                    states = env.step(orders)

                metrics = compute_simulation_metrics(env, warmup_steps=WARMUP_STEPS)

                tot_holding = sum(metrics.holding_cost_by_echelon.values())
                tot_backlog = sum(metrics.backlog_cost_by_echelon.values())

                ret_bw = float(metrics.bullwhip_ratio.get("retailer", np.nan))
                who_bw = float(metrics.bullwhip_ratio.get("wholesaler", np.nan))
                dis_bw = float(metrics.bullwhip_ratio.get("distributor", np.nan))
                man_bw = float(metrics.bullwhip_ratio.get("manufacturer", np.nan))
                e2e_bw = float(metrics.end_to_end_bullwhip) if metrics.end_to_end_bullwhip is not None else np.nan

                var_ret = float(metrics.order_variance_by_echelon.get("retailer", np.nan))
                var_who = float(metrics.order_variance_by_echelon.get("wholesaler", np.nan))
                var_dis = float(metrics.order_variance_by_echelon.get("distributor", np.nan))
                var_man = float(metrics.order_variance_by_echelon.get("manufacturer", np.nan))

                ret_fr = float(metrics.fill_rate.get("retailer", 0.0)) * 100.0
                ret_sl = float(metrics.service_level.get("retailer", 0.0)) * 100.0
                ret_stockout = int(metrics.stockout_steps.get("retailer", 0))

                records.append({
                    "system": sys_label,
                    "system_id": sys_id,
                    "is_reference": is_ref,
                    "scenario": sc_name,
                    "seed": seed,
                    "total_cost": float(metrics.total_cost),
                    "holding_cost": tot_holding,
                    "backlog_cost": tot_backlog,
                    "end_to_end_bw": e2e_bw,
                    "bw_retailer": ret_bw,
                    "bw_wholesaler": who_bw,
                    "bw_distributor": dis_bw,
                    "bw_manufacturer": man_bw,
                    "var_order_retailer": var_ret,
                    "var_order_wholesaler": var_who,
                    "var_order_distributor": var_dis,
                    "var_order_manufacturer": var_man,
                    "fill_rate": ret_fr,
                    "service_level": ret_sl,
                    "stockout_steps": ret_stockout,
                })

                # Save shock scenario representative trajectory (seed 0)
                if sc_name == "shock" and seed == 0:
                    shock_trajectories[sys_label] = {
                        "demand": cust_demands,
                        "retailer_orders": step_orders["retailer"],
                        "wholesaler_orders": step_orders["wholesaler"],
                        "distributor_orders": step_orders["distributor"],
                        "manufacturer_orders": step_orders["manufacturer"],
                    }

    df_raw = pd.DataFrame(records)
    Path("results").mkdir(parents=True, exist_ok=True)
    df_raw.to_csv("results/benchmark_v2_1_test_seeds0_7_raw.csv", index=False)
    print(f"Saved raw test results ({len(df_raw)} runs) to results/benchmark_v2_1_test_seeds0_7_raw.csv")

    # Aggregate by (scenario, system)
    summary_list = []
    for (sc_name, sys_label), grp in df_raw.groupby(["scenario", "system"]):
        is_ref = grp["is_reference"].iloc[0]
        n_obs = len(grp)
        summary_list.append({
            "scenario": sc_name,
            "system": sys_label,
            "is_reference": is_ref,
            "n": n_obs,
            "mean_total_cost": grp["total_cost"].mean(),
            "std_total_cost": grp["total_cost"].std(),
            "mean_holding_cost": grp["holding_cost"].mean(),
            "std_holding_cost": grp["holding_cost"].std(),
            "mean_backlog_cost": grp["backlog_cost"].mean(),
            "std_backlog_cost": grp["backlog_cost"].std(),
            "mean_e2e_bw": grp["end_to_end_bw"].mean(),
            "std_e2e_bw": grp["end_to_end_bw"].std(),
            "mean_bw_retailer": grp["bw_retailer"].mean(),
            "std_bw_retailer": grp["bw_retailer"].std(),
            "mean_bw_wholesaler": grp["bw_wholesaler"].mean(),
            "std_bw_wholesaler": grp["bw_wholesaler"].std(),
            "mean_bw_distributor": grp["bw_distributor"].mean(),
            "std_bw_distributor": grp["bw_distributor"].std(),
            "mean_bw_manufacturer": grp["bw_manufacturer"].mean(),
            "std_bw_manufacturer": grp["bw_manufacturer"].std(),
            "mean_var_retailer": grp["var_order_retailer"].mean(),
            "std_var_retailer": grp["var_order_retailer"].std(),
            "mean_var_wholesaler": grp["var_order_wholesaler"].mean(),
            "std_var_wholesaler": grp["var_order_wholesaler"].std(),
            "mean_var_distributor": grp["var_order_distributor"].mean(),
            "std_var_distributor": grp["var_order_distributor"].std(),
            "mean_var_manufacturer": grp["var_order_manufacturer"].mean(),
            "std_var_manufacturer": grp["var_order_manufacturer"].std(),
            "mean_fill_rate": grp["fill_rate"].mean(),
            "std_fill_rate": grp["fill_rate"].std(),
            "mean_service_level": grp["service_level"].mean(),
            "std_service_level": grp["service_level"].std(),
            "mean_stockouts": grp["stockout_steps"].mean(),
            "std_stockouts": grp["stockout_steps"].std(),
        })

    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv("results/benchmark_v2_1_test_seeds0_7_summary.csv", index=False)
    print(f"Saved summary table to results/benchmark_v2_1_test_seeds0_7_summary.csv\n")

    # Print comprehensive per-scenario tables
    for sc_name in SCENARIOS:
        sc_cap = sc_name.upper()
        print("\n" + "=" * 120)
        print(f"SCENARIO: {sc_cap} (Test Seeds 0 to 7, n = 8)")
        print("=" * 120)

        sub = df_summary[df_summary["scenario"] == sc_name].copy()
        # Sort by total cost
        sub = sub.sort_values("mean_total_cost")

        # Table 1: Cost breakdown and service metrics
        tbl_cost_service = []
        for _, r in sub.iterrows():
            tbl_cost_service.append({
                "System": r["system"],
                "Total Cost ($)": f"{r['mean_total_cost']:,.1f} ± {r['std_total_cost']:,.1f}",
                "Holding Cost ($)": f"{r['mean_holding_cost']:,.1f} ± {r['std_holding_cost']:,.1f}",
                "Backlog Cost ($)": f"{r['mean_backlog_cost']:,.1f} ± {r['std_backlog_cost']:,.1f}",
                "Fill Rate (%)": f"{r['mean_fill_rate']:.2f}% ± {r['std_fill_rate']:.2f}%",
                "Step SL (%)": f"{r['mean_service_level']:.1f}% ± {r['std_service_level']:.1f}%",
                "Stockouts": f"{r['mean_stockouts']:.2f} ± {r['std_stockouts']:.2f}",
                "n": int(r["n"]),
            })
        print(f"\n### {sc_cap} Scenario: Cost & Service Performance\n")
        print(pd.DataFrame(tbl_cost_service).to_markdown(index=False))

        # Table 2: Bullwhip ratios and Order Variances
        tbl_bw_var = []
        for _, r in sub.iterrows():
            tbl_bw_var.append({
                "System": r["system"],
                "E2E Bullwhip": f"{r['mean_e2e_bw']:.2f} ± {r['std_e2e_bw']:.2f}",
                "Ret BW": f"{r['mean_bw_retailer']:.2f} ± {r['std_bw_retailer']:.2f}",
                "Who BW": f"{r['mean_bw_wholesaler']:.2f} ± {r['std_bw_wholesaler']:.2f}",
                "Dis BW": f"{r['mean_bw_distributor']:.2f} ± {r['std_bw_distributor']:.2f}",
                "Mfr BW": f"{r['mean_bw_manufacturer']:.2f} ± {r['std_bw_manufacturer']:.2f}",
                "Var(O_Ret)": f"{r['mean_var_retailer']:.1f} ± {r['std_var_retailer']:.1f}",
                "Var(O_Mfr)": f"{r['mean_var_manufacturer']:.1f} ± {r['std_var_manufacturer']:.1f}",
                "n": int(r["n"]),
            })
        print(f"\n### {sc_cap} Scenario: Bullwhip & Absolute Order Variance\n")
        print(pd.DataFrame(tbl_bw_var).to_markdown(index=False))

    # Generate required plots
    plot_cost_vs_bullwhip_pareto(df_summary)
    plot_shock_echelon_dynamics(shock_trajectories)


def compute_pareto_frontier(points: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """Compute 2D Pareto frontier (minimizing both bullwhip and cost)."""
    # Sort primarily by bullwhip ascending, secondarily by cost ascending
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    frontier = []
    min_cost = float("inf")
    for bw, cost, lbl in sorted_pts:
        if cost < min_cost:
            frontier.append((bw, cost, lbl))
            min_cost = cost
    return frontier


def plot_cost_vs_bullwhip_pareto(df_summary: pd.DataFrame):
    """Plot cost vs end-to-end bullwhip with Pareto frontier."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharey=False)

    colors = {
        "System 0: Constant": "#e74c3c",
        "EOQ Untuned": "#3498db",
        "EOQ Tuned": "#2980b9",
        "(s, S) Untuned": "#9b59b6",
        "(s, S) Tuned": "#8e44ad",
        "System 2b Untuned": "#f39c12",
        "System 2b Tuned": "#d35400",
        "System 2c Tuned": "#16a085",
        "Base-Stock (z=2.0)": "#7f8c8d",
        "Oracle (Reference)": "#27ae60",
    }
    markers = {
        "System 0: Constant": "X",
        "EOQ Untuned": "s",
        "EOQ Tuned": "s",
        "(s, S) Untuned": "o",
        "(s, S) Tuned": "o",
        "System 2b Untuned": "^",
        "System 2b Tuned": "^",
        "System 2c Tuned": "D",
        "Base-Stock (z=2.0)": "v",
        "Oracle (Reference)": "*",
    }

    for idx, sc_name in enumerate(SCENARIOS):
        ax = axes[idx]
        sc_data = df_summary[df_summary["scenario"] == sc_name]

        points = []
        for _, row in sc_data.iterrows():
            lbl = row["system"]
            bw = row["mean_e2e_bw"]
            cost = row["mean_total_cost"]
            bw_err = row["std_e2e_bw"]
            cost_err = row["std_total_cost"]

            col = colors.get(lbl, "gray")
            mkr = markers.get(lbl, "o")
            sz = 140 if ("Tuned" in lbl or "Oracle" in lbl) else 80

            ax.errorbar(
                bw,
                cost,
                xerr=bw_err,
                yerr=cost_err,
                fmt=mkr,
                color=col,
                ecolor=col,
                elinewidth=1.2,
                capsize=3,
                markersize=9 if "Oracle" not in lbl else 13,
                label=lbl if idx == 0 else None,
                alpha=0.9,
                zorder=4,
            )
            ax.annotate(
                lbl.replace("System ", "S").replace("Untuned", "Unt").replace("Tuned", "Tun"),
                (bw, cost),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                alpha=0.85,
            )
            if not np.isnan(bw) and not np.isnan(cost):
                points.append((bw, cost, lbl))

        # Compute and plot Pareto frontier
        pareto_pts = compute_pareto_frontier(points)
        if len(pareto_pts) > 1:
            px = [p[0] for p in pareto_pts]
            py = [p[1] for p in pareto_pts]
            ax.plot(
                px,
                py,
                color="#e67e22",
                linestyle="--",
                linewidth=1.8,
                label="Pareto Frontier" if idx == 0 else None,
                zorder=3,
            )

        ax.set_title(f"Scenario: {sc_name.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("End-to-End Bullwhip Ratio", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Mean Total Cost ($)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Benchmark v2.1: Cost vs. End-to-End Bullwhip with Pareto Frontier (Test Seeds 0..7, n = 8)",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.85, bottom=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.02), fontsize=9)

    out_path = Path("results/cost_vs_bullwhip_pareto.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated plot saved to {out_path}")


def plot_shock_echelon_dynamics(shock_trajectories: Dict[str, Dict[str, List[float]]]):
    """Plot demand vs orders per echelon in shock scenario for key policies."""
    # Policies to visualize
    target_policies = [
        "System 0: Constant",
        "EOQ Untuned",
        "EOQ Tuned",
        "System 2b Tuned",
        "System 2c Tuned",
    ]

    available_policies = [p for p in target_policies if p in shock_trajectories]
    n_plots = len(available_policies)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.2 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    echelon_colors = {
        "retailer_orders": "#3498db",
        "wholesaler_orders": "#2ecc71",
        "distributor_orders": "#f39c12",
        "manufacturer_orders": "#e74c3c",
    }
    echelon_labels = {
        "retailer_orders": "Retailer Orders",
        "wholesaler_orders": "Wholesaler Orders",
        "distributor_orders": "Distributor Orders",
        "manufacturer_orders": "Manufacturer Orders",
    }

    steps = list(range(COMMON_HORIZON))

    for idx, pol_name in enumerate(available_policies):
        ax = axes[idx]
        data = shock_trajectories[pol_name]
        demand = data["demand"]

        # Highlight shock window (steps 15 to 25)
        ax.axvspan(15, 25, color="#fce4ec", alpha=0.6, label="Shock Window (Level Shift)" if idx == 0 else None)

        # Plot external demand
        ax.plot(steps, demand, color="black", linestyle="--", linewidth=2.0, label="Customer Demand" if idx == 0 else None)

        # Plot echelon orders
        for k, col in echelon_colors.items():
            orders = data[k]
            ax.plot(steps, orders, color=col, linewidth=1.4, alpha=0.85, label=echelon_labels[k] if idx == 0 else None)

        ax.set_title(f"Policy: {pol_name}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Quantity (Units)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("Simulation Day (Steps)", fontsize=11)
    fig.suptitle(
        "Shock Scenario Dynamics: Customer Demand vs. Echelon Orders over Time (Seed 0)",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.92, bottom=0.10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.96), ncol=3, fontsize=9)

    out_path = Path("results/shock_demand_vs_orders_echelons.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated plot saved to {out_path}")


if __name__ == "__main__":
    run_eval()
