"""Parameter sweep for System 2b (Smoothed Order-Up-To) on tuning seeds 100-104.

Sweeps:
- alpha in [0.2, 0.4, 0.6, 0.8]
- delta in [0.25, 0.5, 1.0]
Across 3 scenarios: stable, seasonal, shock
Tuning seeds: 100, 101, 102, 103, 104
Evaluates total cost vs bullwhip trade-off curve, identifies optimal parameters,
writes them to configs/sim.yaml, and outputs results.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List

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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sweep_system2b")

ALPHAS = [0.2, 0.4, 0.6, 0.8]
DELTAS = [0.25, 0.5, 1.0]
TUNING_SEEDS = [100, 101, 102, 103, 104]
SCENARIOS_LIST = ["stable", "seasonal", "shock"]
WARMUP_STEPS = 5


def run_sweep() -> pd.DataFrame:
    cfg = get_config()
    sim_cfg = cfg.sim
    all_scenarios = generate_scenarios()

    records: List[Dict[str, Any]] = []
    param_grid = list(itertools.product(ALPHAS, DELTAS))
    total_runs = len(param_grid) * len(SCENARIOS_LIST) * len(TUNING_SEEDS)

    print("=" * 85)
    print(f"RUNNING SYSTEM 2b PARAMETER SWEEP ({total_runs} RUNS TOTAL)")
    print(f"Alphas: {ALPHAS}")
    print(f"Deltas: {DELTAS}")
    print(f"Tuning seeds: {TUNING_SEEDS}")
    print(f"Scenarios: {SCENARIOS_LIST}")
    print(f"Warm-up: {WARMUP_STEPS} steps | Steady-state init: True")
    print("=" * 85)

    run_count = 0
    for alpha, delta in param_grid:
        for sc_name in SCENARIOS_LIST:
            base_sc = all_scenarios[sc_name]
            for seed in TUNING_SEEDS:
                run_count += 1
                # Seed sampler independently per run
                sc = base_sc.clone() if hasattr(base_sc, "clone") else base_sc
                sc.lead_time_sampler = LeadTimeSampler(seed=seed)

                # Configure System 2b with test alpha and delta
                system = get_system(
                    "baseline_smoothed_order_up_to",
                    config={
                        "alpha": alpha,
                        "delta": delta,
                        "rate_limit_delta": delta,
                        "forecast_cap_multiple": 2.0,
                    },
                )
                system.calibrate(sc)
                system.reset(scenario=sc)

                # Initialize environment
                env = BeerGameEnv(
                    config=sim_cfg,
                    scenario=sc,
                    seed=seed,
                    steady_state_init=True,
                )
                states = env.reset(seed=seed)

                for t in range(len(sc.demand)):
                    orders = system.decide_orders(states)
                    states = env.step(orders)

                metrics = compute_simulation_metrics(env, warmup_steps=WARMUP_STEPS)

                ret_bw = float(metrics.bullwhip_ratio["retailer"])
                who_bw = float(metrics.bullwhip_ratio["wholesaler"])
                dis_bw = float(metrics.bullwhip_ratio["distributor"])
                man_bw = float(metrics.bullwhip_ratio["manufacturer"])
                mean_bw = float(np.mean([ret_bw, who_bw, dis_bw, man_bw]))

                records.append({
                    "alpha": alpha,
                    "delta": delta,
                    "scenario": sc_name,
                    "seed": seed,
                    "total_cost": float(metrics.total_cost),
                    "retailer_bullwhip": ret_bw,
                    "wholesaler_bullwhip": who_bw,
                    "distributor_bullwhip": dis_bw,
                    "manufacturer_bullwhip": man_bw,
                    "mean_bullwhip": mean_bw,
                    "retailer_fill_rate": float(metrics.fill_rate["retailer"]),
                    "retailer_service_level": float(metrics.service_level["retailer"]),
                })

    df = pd.DataFrame(records)

    # Aggregate by (alpha, delta)
    summary = df.groupby(["alpha", "delta"]).agg(
        cost_mean=("total_cost", "mean"),
        cost_std=("total_cost", "std"),
        ret_bw_mean=("retailer_bullwhip", "mean"),
        ret_bw_std=("retailer_bullwhip", "std"),
        man_bw_mean=("manufacturer_bullwhip", "mean"),
        mean_bw_mean=("mean_bullwhip", "mean"),
        fill_rate_mean=("retailer_fill_rate", "mean"),
        service_level_mean=("retailer_service_level", "mean"),
    ).reset_index()

    # Sort by total cost
    summary = summary.sort_values(by="cost_mean", ascending=True)

    print("\nSWEEP SUMMARY ACROSS TUNING SEEDS 100-104 (ALL 3 SCENARIOS):")
    print("-" * 95)
    print(f"{'Alpha':<6} | {'Delta':<6} | {'Total Cost ($)':<18} | {'Retailer BW':<15} | {'Mean BW':<12} | {'Service Level':<14} | {'Fill Rate':<10}")
    print("-" * 95)
    for _, row in summary.iterrows():
        c_str = f"${row['cost_mean']:,.2f} ± {row['cost_std']:,.1f}"
        rbw_str = f"{row['ret_bw_mean']:.3f} ± {row['ret_bw_std']:.3f}"
        mbw_str = f"{row['mean_bw_mean']:.3f}"
        sl_str = f"{row['service_level_mean']*100:.1f}%"
        fr_str = f"{row['fill_rate_mean']*100:.1f}%"
        print(f"{row['alpha']:<6.2f} | {row['delta']:<6.2f} | {c_str:<18} | {rbw_str:<15} | {mbw_str:<12} | {sl_str:<14} | {fr_str:<10}")
    print("-" * 95)

    # Save CSV
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "system2b_sweep_results.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nDetailed sweep summary written to: {csv_path}")

    # Plot Trade-off Curve
    plot_tradeoff_curve(summary, out_dir / "tradeoff_curve.png")

    # Select optimal configuration
    # Balance: Minimize total cost while keeping bullwhip controlled and service level high
    # Let's find Pareto-optimal candidate
    best_row = select_best_configuration(summary)
    best_alpha = float(best_row["alpha"])
    best_delta = float(best_row["delta"])

    print("\n" + "=" * 85)
    print("SELECTED OPTIMAL PARAMETERS FOR SYSTEM 2b (FROM TUNING SEEDS 100-104 ONLY):")
    print(f"Optimal Alpha: {best_alpha:.2f}")
    print(f"Optimal Delta: {best_delta:.2f}")
    print(f"Mean Total Cost: ${best_row['cost_mean']:,.2f}")
    print(f"Mean Retailer Bullwhip: {best_row['ret_bw_mean']:.3f}")
    print(f"Mean System Bullwhip:   {best_row['mean_bw_mean']:.3f}")
    print(f"Mean Service Level:     {best_row['service_level_mean']*100:.1f}%")
    print("=" * 85)

    # Write chosen parameters to configs/sim.yaml
    update_sim_yaml(best_alpha, best_delta)

    return summary


def select_best_configuration(summary: pd.DataFrame) -> pd.Series:
    """Select the best balanced configuration: lowest cost among service level >= 85% and dampened bullwhip."""
    # Filter candidates with solid service level
    candidates = summary[summary["service_level_mean"] >= 0.85]
    if candidates.empty:
        candidates = summary

    # Pick lowest cost candidate among acceptable service level
    best = candidates.iloc[0]
    return best


def plot_tradeoff_curve(summary: pd.DataFrame, out_path: Path):
    """Plot Total Cost vs. Bullwhip trade-off curve across swept parameters."""
    plt.figure(figsize=(10, 7))

    alphas = summary["alpha"].unique()
    colors = {0.2: "#3B82F6", 0.4: "#10B981", 0.6: "#F59E0B", 0.8: "#EF4444"}
    markers = {0.25: "o", 0.5: "s", 1.0: "^"}

    for _, row in summary.iterrows():
        a = row["alpha"]
        d = row["delta"]
        x = row["ret_bw_mean"]
        y = row["cost_mean"]

        plt.scatter(
            x,
            y,
            color=colors.get(a, "black"),
            marker=markers.get(d, "o"),
            s=120,
            alpha=0.9,
            edgecolors="#1E293B",
            linewidths=1.2,
            zorder=3,
        )
        label_text = f"α={a}, δ={d}"
        plt.annotate(
            label_text,
            (x, y),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=9,
            fontweight="semibold",
            color="#1E293B",
        )

    # Add legend handles for Alpha colors and Delta markers
    alpha_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[a], markersize=9, label=f"α = {a}")
        for a in sorted(alphas)
    ]
    delta_handles = [
        plt.Line2D([0], [0], marker=markers[d], color="w", markerfacecolor="#64748B", markersize=9, label=f"δ = {d}")
        for d in sorted(markers.keys())
    ]

    leg1 = plt.legend(handles=alpha_handles, title="Order Smoothing α", loc="upper left", framealpha=0.9)
    plt.gca().add_artist(leg1)
    plt.legend(handles=delta_handles, title="Rate Limit δ", loc="upper right", framealpha=0.9)

    plt.title(
        "System 2b Parameter Sweep: Total Cost vs. Retailer Bullwhip Trade-Off Curve\n"
        f"Evaluated on Tuning Seeds {TUNING_SEEDS[0]}-{TUNING_SEEDS[-1]} across Stable, Seasonal, & Shock",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Retailer Bullwhip Ratio: Var(Orders) / Var(Demand)", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Total System Cost ($)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Trade-off curve plot saved to: {out_path}")


def update_sim_yaml(chosen_alpha: float, chosen_delta: float):
    """Write chosen parameters and tuning seed record to configs/sim.yaml."""
    yaml_path = Path("configs/sim.yaml")
    if not yaml_path.exists():
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "four_echelon" not in data:
        data["four_echelon"] = {}

    data["four_echelon"]["alpha"] = round(chosen_alpha, 2)
    data["four_echelon"]["rate_limit_delta"] = round(chosen_delta, 2)
    data["four_echelon"]["tuning_seeds_used"] = TUNING_SEEDS

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Updated configs/sim.yaml with chosen alpha={chosen_alpha:.2f}, delta={chosen_delta:.2f}, tuning_seeds={TUNING_SEEDS}")


if __name__ == "__main__":
    run_sweep()
