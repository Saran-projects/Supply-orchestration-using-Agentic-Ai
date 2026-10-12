"""Verification script for Bullwhip calculation: System 2 (EOQ) on Shock scenario (Seed 42).

Computes and prints demand variance, order variance, and bullwhip ratio per echelon,
with steady-state initialization and 5-step warm-up period applied.
Generates publication-quality echelon order vs demand time-series plots.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sca.baselines  # noqa: F401 (Registers baseline policies)
from sca.config import get_config
from sca.data.scenarios import LeadTimeSampler, generate_scenarios
from sca.sim.environment import ECHELONS, BeerGameEnv
from sca.sim.metrics import calculate_bullwhip_ratio, compute_simulation_metrics
from sca.systems.base import get_system

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("verify_bullwhip")


def run_verification(seed: int = 42, warmup_steps: int = 5):
    cfg = get_config()
    sim_cfg = cfg.sim

    # 1. Load Shock scenario
    scenarios = generate_scenarios()
    shock_sc = scenarios["shock"]
    # Seed sampler consistently
    shock_sc.lead_time_sampler = LeadTimeSampler(seed=seed)

    # 2. Instantiate and calibrate System 2 (EOQ)
    sys2 = get_system("baseline_eoq")
    sys2.calibrate(shock_sc)
    sys2.reset(scenario=shock_sc)

    # 3. Initialize environment at steady state
    env = BeerGameEnv(
        config=sim_cfg,
        scenario=shock_sc,
        seed=seed,
        steady_state_init=True,
    )
    states = env.reset(seed=seed)

    horizon = len(shock_sc.demand)
    for t in range(horizon):
        orders = sys2.decide_orders(states)
        states = env.step(orders)

    assert env.check_unit_conservation(), "Conservation violated!"

    # 4. Metrics with warmup
    metrics = compute_simulation_metrics(env, warmup_steps=warmup_steps)

    print("=" * 80)
    print(f"BULLWHIP CALCULATION VERIFICATION: SYSTEM 2 (EOQ) | SHOCK SCENARIO | SEED {seed}")
    print(f"Warm-up steps: {warmup_steps} (Evaluated steps: {warmup_steps}..{horizon-1} = {horizon - warmup_steps} steps)")
    print(f"Chain Initialization: Steady-State (in-transit pipelines populated)")
    print("=" * 80)
    print(f"{'Echelon':<15} | {'Var(Demand)':<14} | {'Var(Orders)':<14} | {'Bullwhip Ratio':<15} | {'Total Cost ($)':<14}")
    print("-" * 80)

    echelon_data = {}
    for e in ECHELONS:
        hist = env.history[e]
        dem_full = np.array([r.demand for r in hist], dtype=float)
        ord_full = np.array([r.order_placed for r in hist], dtype=float)

        dem_post = dem_full[warmup_steps:]
        ord_post = ord_full[warmup_steps:]

        var_dem = float(np.var(dem_post, ddof=1))
        var_ord = float(np.var(ord_post, ddof=1))
        bw = float(metrics.bullwhip_ratio[e])
        cost = float(metrics.cost_by_echelon[e])

        echelon_data[e] = {
            "dem_full": dem_full,
            "ord_full": ord_full,
            "var_dem": var_dem,
            "var_ord": var_ord,
            "bullwhip": bw,
            "cost": cost,
        }

        print(f"{e.capitalize():<15} | {var_dem:<14.2f} | {var_ord:<14.2f} | {bw:<15.4f} | ${cost:<13.2f}")

    print("=" * 80)
    print(f"Total Simulation Cost: ${metrics.total_cost:,.2f}")
    print(f"Retailer Service Level: {metrics.service_level['retailer']*100:.1f}%")
    print(f"Retailer Fill Rate:     {metrics.fill_rate['retailer']*100:.1f}%")
    print("=" * 80)

    # 5. Plot Orders and Demand per Echelon
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    steps = np.arange(horizon)

    echelon_colors = {
        "retailer": "#2563EB",
        "wholesaler": "#7C3AED",
        "distributor": "#D97706",
        "manufacturer": "#DC2626",
    }

    for idx, e in enumerate(ECHELONS):
        ax = axes[idx]
        d_series = echelon_data[e]["dem_full"]
        o_series = echelon_data[e]["ord_full"]
        bw = echelon_data[e]["bullwhip"]
        v_d = echelon_data[e]["var_dem"]
        v_o = echelon_data[e]["var_ord"]

        # Highlight warm-up period
        ax.axvspan(0, warmup_steps, color="#E2E8F0", alpha=0.6, label="Warm-up (excluded from metrics)" if idx == 0 else "")

        # Demand and Orders
        ax.plot(steps, d_series, label="Incoming Demand", color="#475569", linestyle="--", linewidth=1.5, alpha=0.85)
        ax.plot(steps, o_series, label="Orders Placed", color=echelon_colors[e], linewidth=2.0)

        title_str = (
            f"{e.capitalize()}: Bullwhip = {bw:.3f} "
            f"(Var(Orders) = {v_o:.1f}, Var(Demand) = {v_d:.1f})"
        )
        ax.set_title(title_str, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel("Units", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9)

    axes[-1].set_xlabel("Simulation Step (Days)", fontsize=11, fontweight="bold")
    plt.suptitle(
        f"System 2 (EOQ) Order vs. Demand Dynamics — Shock Scenario (Seed {seed})\n"
        f"Steady-State Initialization & {warmup_steps}-Step Warm-Up Discarded",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "system2_shock_seed42_bullwhip.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved successfully to: {plot_path}")


if __name__ == "__main__":
    run_verification(seed=42, warmup_steps=5)
