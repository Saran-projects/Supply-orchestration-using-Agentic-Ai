"""Final Benchmark v2.1 Evaluation across Test Seeds 0 to 7.

Reruns:
- System 0 (Constant Order)
- System 1 Untuned ((s, S))
- System 1 Tuned ((s, S): s=1.2, S=0.25)
- System 2 Untuned (EOQ)
- System 2 Tuned (EOQ: Q=0.75, ROP=1.2)
- System 2b Untuned (Smoothed OUT: alpha=0.8, beta=0.8, z=2.0, delta=0.5, cap=1.5)
- System 2b Tuned (Smoothed OUT: alpha=0.6, beta=0.6, z=2.5, delta=0.5, cap=1.5)
- Oracle Policy (Labelled Reference)

Across Scenarios: stable, seasonal, shock
Test Seeds: 0 to 7
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sca.config import get_config
from sca.data.scenarios import generate_scenarios
from sca.sim.environment import BeerGameEnv
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import get_system
import sca.baselines

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark_v2_1_final")

INIT_INV = 687.0
TEST_SEEDS = list(range(8))
SCENARIOS = ["stable", "seasonal", "shock"]
COMMON_HORIZON = 40
WARMUP_STEPS = 5
BW_CEILING = 5.756  # 0.75 * 7.675 (untuned EOQ on tuning seeds)


def main():
    cfg = get_config()
    sim_cfg = cfg.sim.model_copy(deep=True)
    sim_cfg.four_echelon.initial_inventory = float(INIT_INV)
    sim_cfg.four_echelon.horizon_steps = COMMON_HORIZON
    sim_cfg.four_echelon.warmup_steps = WARMUP_STEPS
    sim_cfg.four_echelon.backlog_cost_per_unit = 5.0
    sim_cfg.four_echelon.holding_cost_per_unit = 0.50
    sim_cfg.four_echelon.steady_state_init = True
    sim_cfg.four_echelon.independent_demand_paths = True

    all_scenarios = generate_scenarios()

    systems_to_eval = [
        ("System 0: Constant", "baseline_constant", {}),
        ("System 1: (s, S) Untuned", "baseline_s_S", {}),
        ("System 1: (s, S) Tuned", "baseline_s_S", {"s_factor": 1.2, "S_factor": 0.25}),
        ("System 2: EOQ Untuned", "baseline_eoq", {}),
        ("System 2: EOQ Tuned", "baseline_eoq", {"order_quantity_multiplier": 0.75, "reorder_point_factor": 1.2}),
        (
            "System 2b: Smoothed OUT Untuned",
            "baseline_smoothed_order_up_to",
            {
                "alpha": 0.8,
                "beta": 0.8,
                "z": 2.0,
                "delta": 0.5,
                "forecast_cap_multiple": 1.5,
                "include_lead_time_variance": True,
                "db_path": False,
            },
        ),
        (
            "System 2b: Smoothed OUT Tuned",
            "baseline_smoothed_order_up_to",
            {
                "alpha": 0.6,
                "beta": 0.6,
                "z": 2.5,
                "delta": 0.5,
                "forecast_cap_multiple": 1.5,
                "include_lead_time_variance": True,
                "db_path": False,
            },
        ),
        ("Oracle (Reference)", "baseline_oracle", {}),
    ]

    all_results = []
    print("\n" + "=" * 120)
    print("RUNNING FINAL BENCHMARK v2.1 ON TEST SEEDS 0 TO 7 (Initial Inv = 687.0)")
    print("=" * 120)

    for sys_label, sys_id, custom_cfg in systems_to_eval:
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
                e2e_bw = float(metrics.end_to_end_bullwhip) if metrics.end_to_end_bullwhip is not None else np.nan

                all_results.append({
                    "system_label": sys_label,
                    "system_name": sys_id,
                    "scenario": sc_name,
                    "seed": seed,
                    "total_cost": float(metrics.total_cost),
                    "fill_rate": float(metrics.fill_rate.get("retailer", 0.0)) * 100.0,
                    "service_level": float(metrics.service_level.get("retailer", 0.0)) * 100.0,
                    "stockout_steps": int(metrics.stockout_steps.get("retailer", 0)),
                    "mean_bullwhip": mean_bw,
                    "end_to_end_bullwhip": e2e_bw,
                })

    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv("results/benchmark_v2_1_test_seeds0_7_raw.csv", index=False)
    print(f"Saved raw run results ({len(df_raw)} runs) to results/benchmark_v2_1_test_seeds0_7_raw.csv")

    # Aggregate summaries per system and scenario
    summary_rows = []
    for (label, sc_name), group in df_raw.groupby(["system_label", "scenario"]):
        m_cost = group["total_cost"].mean()
        s_cost = group["total_cost"].std()
        m_fr = group["fill_rate"].mean()
        s_fr = group["fill_rate"].std()
        m_sl = group["service_level"].mean()
        s_sl = group["service_level"].std()
        m_bw = group["mean_bullwhip"].mean()
        s_bw = group["mean_bullwhip"].std()
        m_e2e = group["end_to_end_bullwhip"].mean()
        s_e2e = group["end_to_end_bullwhip"].std()
        m_so = group["stockout_steps"].mean()

        # Check constraints
        fr_pass = m_fr >= 99.0
        sl_pass = m_sl >= 85.0
        bw_pass = m_bw <= BW_CEILING

        is_oracle = "Oracle" in label
        if is_oracle:
            status = "Reference"
            c_cost = m_cost
        elif fr_pass and sl_pass and bw_pass:
            status = "Qualified"
            c_cost = m_cost
        else:
            reasons = []
            if not fr_pass:
                reasons.append(f"FR={m_fr:.1f}%<99%")
            if not sl_pass:
                reasons.append(f"SL={m_sl:.1f}%<85%")
            if not bw_pass:
                reasons.append(f"BW={m_bw:.2f}>5.76")
            status = f"Disqualified ({', '.join(reasons)})"
            c_cost = np.nan

        summary_rows.append({
            "system_label": label,
            "scenario": sc_name,
            "mean_cost": m_cost,
            "std_cost": s_cost,
            "mean_fr": m_fr,
            "std_fr": s_fr,
            "mean_sl": m_sl,
            "std_sl": s_sl,
            "mean_bw": m_bw,
            "std_bw": s_bw,
            "mean_e2e": m_e2e,
            "std_e2e": s_e2e,
            "mean_stockout": m_so,
            "constrained_cost": c_cost,
            "status": status,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/benchmark_v2_1_test_seeds0_7_summary.csv", index=False)
    print("Saved summary table to results/benchmark_v2_1_test_seeds0_7_summary.csv\n")

    # Display tables per scenario
    for sc_name in ["stable", "seasonal", "shock"]:
        print(f"\n==================== SCENARIO: {sc_name.upper()} (Test Seeds 0..7) ====================")
        sc_df = df_summary[df_summary["scenario"] == sc_name].copy()
        sc_df["sort_key"] = sc_df["constrained_cost"].fillna(999999999)
        sc_df = sc_df.sort_values("sort_key")

        formatted = []
        for _, row in sc_df.iterrows():
            formatted.append({
                "System": row["system_label"],
                "Cost ($)": f"{row['mean_cost']:,.0f} ± {row['std_cost']:,.0f}",
                "Fill Rate (%)": f"{row['mean_fr']:.2f}%",
                "Service Lvl (%)": f"{row['mean_sl']:.1f}%",
                "Mean BW": f"{row['mean_bw']:.3f}",
                "E2E BW": f"{row['mean_e2e']:.2f}",
                "Constrained Cost": f"${row['constrained_cost']:,.0f}" if not np.isnan(row["constrained_cost"]) else "NaN",
                "Status": row["status"],
            })
        print(pd.DataFrame(formatted).to_markdown(index=False))

    plot_tradeoff(df_summary)


def plot_tradeoff(df_summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    scenarios = ["stable", "seasonal", "shock"]

    colors = {
        "System 0: Constant": "#e74c3c",
        "System 1: (s, S) Untuned": "#9b59b6",
        "System 1: (s, S) Tuned": "#8e44ad",
        "System 2: EOQ Untuned": "#3498db",
        "System 2: EOQ Tuned": "#2980b9",
        "System 2b: Smoothed OUT Untuned": "#f39c12",
        "System 2b: Smoothed OUT Tuned": "#d35400",
        "Oracle (Reference)": "#27ae60",
    }
    markers = {
        "System 0: Constant": "X",
        "System 1: (s, S) Untuned": "o",
        "System 1: (s, S) Tuned": "o",
        "System 2: EOQ Untuned": "s",
        "System 2: EOQ Tuned": "s",
        "System 2b: Smoothed OUT Untuned": "^",
        "System 2b: Smoothed OUT Tuned": "^",
        "Oracle (Reference)": "*",
    }

    for idx, sc_name in enumerate(scenarios):
        ax = axes[idx]
        sc_data = df_summary[df_summary["scenario"] == sc_name]

        ax.axvline(BW_CEILING, color="gray", linestyle="--", alpha=0.7, label=f"BW Ceiling ({BW_CEILING:.2f})")

        for _, row in sc_data.iterrows():
            lbl = row["system_label"]
            col = colors.get(lbl, "black")
            mkr = markers.get(lbl, "o")
            sz = 150 if "Oracle" in lbl else 90

            fill = "full" if "Qualified" in row["status"] or "Reference" in row["status"] else "none"

            ax.scatter(
                row["mean_bw"],
                row["mean_cost"],
                color=col,
                marker=mkr,
                s=sz,
                edgecolors=col,
                facecolors=col if fill == "full" else "none",
                linewidth=2.0 if fill == "none" else 1.0,
                label=lbl if idx == 0 else None,
                zorder=5,
            )
            ax.annotate(
                lbl.replace("System ", "S").replace(": ", "-").replace("Untuned", "U").replace("Tuned", "T"),
                (row["mean_bw"], row["mean_cost"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                alpha=0.85,
            )

        ax.set_title(f"Scenario: {sc_name.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Bullwhip across Echelons", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Mean Total Cost ($)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Benchmark v2.1: Total Cost vs. Bullwhip on Test Seeds 0..7 (Initial Inv = 687.0)",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.85, bottom=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=9)

    out_path = Path("results/benchmark_v2_1_cost_vs_bullwhip_test.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated plot saved to {out_path}")


if __name__ == "__main__":
    main()
