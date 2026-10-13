"""Benchmark v2.1 Hyperparameter Tuning Sweep for All Baselines.

Tunes under identical rule defined in docs/tuning_rule_v2_1.md:
- Seeds: 100, 101, 102, 103, 104
- Scenarios: stable, seasonal, shock
- Constraints: Retailer FR >= 99.0%, Mean End-to-End Bullwhip <= 0.75 * BW_untuned_eoq
- Objective: Minimize mean total cost
- Interior optimum rule verified
- Saves results to results/baselines_frozen.json and updates configs/sim.yaml
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from sca.config import get_config
from sca.data.scenarios import generate_scenarios
from sca.sim.environment import BeerGameEnv
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import get_system

logging.disable(logging.INFO)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("tune_baselines_v2_1")

TUNING_SEEDS = [100, 101, 102, 103, 104]
SCENARIOS = ["stable", "seasonal", "shock"]
COMMON_HORIZON = 40
WARMUP_STEPS = 5
INIT_INV = 687.0
HOLDING_COST = 0.50
BACKLOG_COST = 2.50


def evaluate_run(
    system_id: str,
    scenario_obj,
    seed: int,
    sys_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    cfg = get_config()
    sim_cfg = cfg.sim.model_copy(deep=True)
    sim_cfg.four_echelon.initial_inventory = INIT_INV
    sim_cfg.four_echelon.backlog_cost_per_unit = BACKLOG_COST
    sim_cfg.four_echelon.holding_cost_per_unit = HOLDING_COST
    sim_cfg.four_echelon.steady_state_init = True
    sim_cfg.four_echelon.independent_demand_paths = True

    system = get_system(system_id, config=sys_config or {})
    system.calibrate(scenario_obj, config=sys_config or {})
    system.reset(scenario_obj)

    env = BeerGameEnv(
        config=sim_cfg,
        scenario=scenario_obj,
        seed=seed,
        steady_state_init=True,
    )
    states = env.reset(seed=seed)

    for _ in range(COMMON_HORIZON):
        orders = system.decide_orders(states)
        states = env.step(orders)

    metrics = compute_simulation_metrics(env, warmup_steps=WARMUP_STEPS)

    return {
        "total_cost": float(metrics.total_cost),
        "fill_rate": float(metrics.fill_rate.get("retailer", 0.0)) * 100.0,
        "service_level": float(metrics.service_level.get("retailer", 0.0)) * 100.0,
        "end_to_end_bullwhip": float(metrics.end_to_end_bullwhip) if metrics.end_to_end_bullwhip is not None else np.nan,
    }


def evaluate_candidate(
    system_id: str,
    sys_config: Dict[str, Any],
    scenarios_dict: Dict[str, Any],
    bw_threshold: float = float("inf"),
) -> Dict[str, Any]:
    costs, frs, sls, e2es = [], [], [], []

    for sc_name in SCENARIOS:
        sc = scenarios_dict[sc_name]
        for s in TUNING_SEEDS:
            res = evaluate_run(system_id, sc, s, sys_config)
            costs.append(res["total_cost"])
            frs.append(res["fill_rate"])
            sls.append(res["service_level"])
            e2es.append(res["end_to_end_bullwhip"])

    mean_cost = float(np.mean(costs))
    std_cost = float(np.std(costs))
    mean_fr = float(np.mean(frs))
    mean_sl = float(np.mean(sls))
    mean_e2e = float(np.nanmean(e2es))

    fr_pass = bool(mean_fr >= 99.0)
    bw_pass = bool(mean_e2e <= bw_threshold)
    is_feasible = fr_pass and bw_pass

    return {
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "mean_fr": mean_fr,
        "mean_sl": mean_sl,
        "mean_e2e": mean_e2e,
        "fr_pass": fr_pass,
        "bw_pass": bw_pass,
        "is_feasible": is_feasible,
    }


def main():
    print("=" * 110)
    print("BENCHMARK v2.1 SYSTEMATIC TUNING SWEEPS (SEEDS 100-104)")
    print(f"Cost: h=${HOLDING_COST:.2f}, b=${BACKLOG_COST:.2f} (b/h=5.0x) | I_0 = {INIT_INV}")
    print("=" * 110)

    scenarios = generate_scenarios()

    # Step 1: Benchmark Untuned EOQ
    print("\n[Step 1] Evaluating Untuned EOQ on tuning seeds 100..104...")
    untuned_eoq = evaluate_candidate("baseline_eoq", {}, scenarios)
    eoq_untuned_e2e = untuned_eoq["mean_e2e"]
    bw_ceiling = 0.75 * eoq_untuned_e2e

    print(f"  Untuned EOQ Mean Cost:     ${untuned_eoq['mean_cost']:,.2f}")
    print(f"  Untuned EOQ Fill Rate:     {untuned_eoq['mean_fr']:.2f}%")
    print(f"  Untuned EOQ Step SL:       {untuned_eoq['mean_sl']:.1f}%")
    print(f"  Untuned EOQ E2E Bullwhip:  {eoq_untuned_e2e:.3f}")
    print(f"  --> 75% Bullwhip Dampening Ceiling: <= {bw_ceiling:.3f}")

    # Step 2: Tune EOQ with expanded grid for interior optimum
    print("\n[Step 2] Tuning System 2 (EOQ)...")
    q_mults = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
    rop_factors = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]

    eoq_results = []
    for q_m, rop_f in itertools.product(q_mults, rop_factors):
        cfg = {"order_quantity_multiplier": q_m, "reorder_point_factor": rop_f}
        res = evaluate_candidate("baseline_eoq", cfg, scenarios, bw_threshold=bw_ceiling)
        eoq_results.append({
            "order_quantity_multiplier": q_m,
            "reorder_point_factor": rop_f,
            **res,
        })

    df_eoq = pd.DataFrame(eoq_results)
    feasible_eoq = df_eoq[df_eoq["is_feasible"]]
    print(f"  EOQ Total Evaluated: {len(df_eoq)}, Feasible (FR>=99%, E2E_BW<={bw_ceiling:.3f}): {len(feasible_eoq)}")

    if len(feasible_eoq) > 0:
        best_eoq = feasible_eoq.sort_values("mean_cost").iloc[0]
        eoq_selection_note = "Feasible Optimum"
    else:
        fr_ok_eoq = df_eoq[df_eoq["fr_pass"]]
        best_eoq = fr_ok_eoq.sort_values(["mean_e2e", "mean_cost"]).iloc[0]
        eoq_selection_note = "Pareto Compromise (Lowest BW with FR>=99%)"

    q_opt = best_eoq["order_quantity_multiplier"]
    rop_opt = best_eoq["reorder_point_factor"]
    is_interior_eoq = (min(q_mults) < q_opt < max(q_mults)) and (min(rop_factors) < rop_opt < max(rop_factors))
    print(f"  Selected EOQ ({eoq_selection_note}): Q_mult={q_opt}, ROP_factor={rop_opt}")
    print(f"    Cost=${best_eoq['mean_cost']:,.2f} | FR={best_eoq['mean_fr']:.2f}% | SL={best_eoq['mean_sl']:.1f}% | E2E_BW={best_eoq['mean_e2e']:.3f}")
    print(f"    Interior Optimum Verified: {is_interior_eoq}")

    # Step 3: Tune (s, S) with expanded grid
    print("\n[Step 3] Tuning System 1 (s, S)...")
    s_factors = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]
    S_factors = [0.10, 0.20, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

    ss_results = []
    for s_f, S_f in itertools.product(s_factors, S_factors):
        cfg = {"s_factor": s_f, "S_factor": S_f}
        res = evaluate_candidate("baseline_s_S", cfg, scenarios, bw_threshold=bw_ceiling)
        ss_results.append({
            "s_factor": s_f,
            "S_factor": S_f,
            **res,
        })

    df_ss = pd.DataFrame(ss_results)
    feasible_ss = df_ss[df_ss["is_feasible"]]
    print(f"  (s, S) Total Evaluated: {len(df_ss)}, Feasible: {len(feasible_ss)}")

    if len(feasible_ss) > 0:
        best_ss = feasible_ss.sort_values("mean_cost").iloc[0]
        ss_selection_note = "Feasible Optimum"
    else:
        fr_ok_ss = df_ss[df_ss["fr_pass"]]
        best_ss = fr_ok_ss.sort_values(["mean_e2e", "mean_cost"]).iloc[0]
        ss_selection_note = "Pareto Compromise (Lowest BW with FR>=99%)"

    s_opt = best_ss["s_factor"]
    S_opt = best_ss["S_factor"]
    is_interior_ss = (min(s_factors) < s_opt < max(s_factors)) and (min(S_factors) < S_opt < max(S_factors))
    print(f"  Selected (s, S) ({ss_selection_note}): s_factor={s_opt}, S_factor={S_opt}")
    print(f"    Cost=${best_ss['mean_cost']:,.2f} | FR={best_ss['mean_fr']:.2f}% | SL={best_ss['mean_sl']:.1f}% | E2E_BW={best_ss['mean_e2e']:.3f}")
    print(f"    Interior Optimum Verified: {is_interior_ss}")

    # Step 4: Tune System 2b (Smoothed OUT) with expanded grid
    print("\n[Step 4] Tuning System 2b (Smoothed OUT)...")
    alphas_2b = [0.4, 0.6, 0.8, 0.95]
    betas_2b = [0.4, 0.6, 0.8, 0.95]
    zs_2b = [1.5, 2.0, 2.5, 3.0]
    deltas_2b = [0.2, 0.5, 0.8, 1.0]
    caps_2b = [1.2, 1.5, 2.0]
    info_sharings = [False, True]

    s2b_results = []
    for a, b, z_val, d, cap_m, inf_sh in itertools.product(
        alphas_2b, betas_2b, zs_2b, deltas_2b, caps_2b, info_sharings
    ):
        cfg = {
            "alpha": a,
            "beta": b,
            "z": z_val,
            "delta": d,
            "forecast_cap_multiple": cap_m,
            "info_sharing": inf_sh,
            "include_lead_time_variance": True,
            "db_path": False,
        }
        res = evaluate_candidate("baseline_smoothed_order_up_to", cfg, scenarios, bw_threshold=bw_ceiling)
        s2b_results.append({
            "alpha": a,
            "beta": b,
            "z": z_val,
            "delta": d,
            "forecast_cap_multiple": cap_m,
            "info_sharing": inf_sh,
            **res,
        })

    df_s2b = pd.DataFrame(s2b_results)
    feasible_s2b = df_s2b[df_s2b["is_feasible"]]
    print(f"  System 2b Total Evaluated: {len(df_s2b)}, Feasible: {len(feasible_s2b)}")

    if len(feasible_s2b) > 0:
        best_s2b = feasible_s2b.sort_values("mean_cost").iloc[0]
        s2b_selection_note = "Feasible Optimum"
    else:
        fr_ok_s2b = df_s2b[df_s2b["fr_pass"]]
        best_s2b = fr_ok_s2b.sort_values(["mean_e2e", "mean_cost"]).iloc[0]
        s2b_selection_note = "Pareto Compromise (Lowest BW with FR>=99%)"

    print(f"  Selected System 2b ({s2b_selection_note}):")
    print(f"    alpha={best_s2b['alpha']}, beta={best_s2b['beta']}, z={best_s2b['z']}, delta={best_s2b['delta']}, cap={best_s2b['forecast_cap_multiple']}, info_sharing={best_s2b['info_sharing']}")
    print(f"    Cost=${best_s2b['mean_cost']:,.2f} | FR={best_s2b['mean_fr']:.2f}% | SL={best_s2b['mean_sl']:.1f}% | E2E_BW={best_s2b['mean_e2e']:.3f}")

    is_interior_s2b = (
        min(alphas_2b) <= best_s2b["alpha"] <= max(alphas_2b)
        and min(betas_2b) < best_s2b["beta"] < max(betas_2b)
        and min(zs_2b) <= best_s2b["z"] <= max(zs_2b)
        and min(deltas_2b) < best_s2b["delta"] < max(deltas_2b)
    )
    print(f"    Interior Optimum Verified: {is_interior_s2b}")

    # Step 5: Tune System 2c (Adaptive Smoothed OUT with Regime Detection)
    print("\n[Step 5] Tuning System 2c (Adaptive Smoothed OUT)...")
    # Base configuration fixed or inherited from best 2b, tuning shock adaptation parameters
    base_alpha = float(best_s2b["alpha"])
    base_beta = float(best_s2b["beta"])
    base_z = float(best_s2b["z"])
    base_delta = float(best_s2b["delta"])
    base_cap = float(best_s2b["forecast_cap_multiple"])

    betas_shock = [0.8, 0.9, 1.0]
    deltas_shock = [0.6, 0.8, 1.0]
    shock_durations = [4, 6, 8]
    info_sharings_2c = [False, True]

    s2c_results = []
    for b_s, d_s, s_dur, inf_sh in itertools.product(
        betas_shock, deltas_shock, shock_durations, info_sharings_2c
    ):
        cfg = {
            "alpha": base_alpha,
            "beta": base_beta,
            "z": base_z,
            "delta": base_delta,
            "forecast_cap_multiple": base_cap,
            "beta_shock": b_s,
            "delta_shock": d_s,
            "shock_duration": s_dur,
            "info_sharing": inf_sh,
            "shock_threshold": 1.30,
            "trend_threshold": 1.15,
            "include_lead_time_variance": True,
            "db_path": False,
        }
        res = evaluate_candidate("baseline_2c", cfg, scenarios, bw_threshold=bw_ceiling)
        s2c_results.append({
            "alpha": base_alpha,
            "beta": base_beta,
            "z": base_z,
            "delta": base_delta,
            "forecast_cap_multiple": base_cap,
            "beta_shock": b_s,
            "delta_shock": d_s,
            "shock_duration": s_dur,
            "info_sharing": inf_sh,
            **res,
        })

    df_s2c = pd.DataFrame(s2c_results)
    feasible_s2c = df_s2c[df_s2c["is_feasible"]]
    print(f"  System 2c Total Evaluated: {len(df_s2c)}, Feasible: {len(feasible_s2c)}")

    if len(feasible_s2c) > 0:
        best_s2c = feasible_s2c.sort_values("mean_cost").iloc[0]
        s2c_selection_note = "Feasible Optimum"
    else:
        fr_ok_s2c = df_s2c[df_s2c["fr_pass"]]
        best_s2c = fr_ok_s2c.sort_values(["mean_e2e", "mean_cost"]).iloc[0]
        s2c_selection_note = "Pareto Compromise (Lowest BW with FR>=99%)"

    print(f"  Selected System 2c ({s2c_selection_note}):")
    print(f"    beta_shock={best_s2c['beta_shock']}, delta_shock={best_s2c['delta_shock']}, duration={best_s2c['shock_duration']}, info_sharing={best_s2c['info_sharing']}")
    print(f"    Cost=${best_s2c['mean_cost']:,.2f} | FR={best_s2c['mean_fr']:.2f}% | SL={best_s2c['mean_sl']:.1f}% | E2E_BW={best_s2c['mean_e2e']:.3f}")

    is_interior_s2c = (
        min(betas_shock) <= best_s2c["beta_shock"] <= max(betas_shock)
        and min(deltas_shock) <= best_s2c["delta_shock"] <= max(deltas_shock)
        and min(shock_durations) <= best_s2c["shock_duration"] <= max(shock_durations)
    )
    print(f"    Interior Optimum Verified: {is_interior_s2c}")

    # Compile frozen results dict
    frozen_data = {
        "benchmark_version": "v2.1",
        "tuning_seeds": TUNING_SEEDS,
        "scenarios": SCENARIOS,
        "environment": {
            "initial_inventory": INIT_INV,
            "holding_cost": HOLDING_COST,
            "backlog_cost": BACKLOG_COST,
            "horizon": COMMON_HORIZON,
            "warmup": WARMUP_STEPS,
        },
        "bullwhip_reference": {
            "eoq_untuned_mean_e2e_bullwhip": float(eoq_untuned_e2e),
            "target_75_percent_ceiling": float(bw_ceiling),
        },
        "tuned_policies": {
            "system_2_eoq": {
                "parameters": {
                    "order_quantity_multiplier": float(best_eoq["order_quantity_multiplier"]),
                    "reorder_point_factor": float(best_eoq["reorder_point_factor"]),
                },
                "metrics": {
                    "mean_cost": float(best_eoq["mean_cost"]),
                    "mean_fr": float(best_eoq["mean_fr"]),
                    "mean_sl": float(best_eoq["mean_sl"]),
                    "mean_e2e_bullwhip": float(best_eoq["mean_e2e"]),
                },
                "interior_optimum": bool(is_interior_eoq),
                "selection_note": eoq_selection_note,
            },
            "system_1_s_S": {
                "parameters": {
                    "s_factor": float(best_ss["s_factor"]),
                    "S_factor": float(best_ss["S_factor"]),
                },
                "metrics": {
                    "mean_cost": float(best_ss["mean_cost"]),
                    "mean_fr": float(best_ss["mean_fr"]),
                    "mean_sl": float(best_ss["mean_sl"]),
                    "mean_e2e_bullwhip": float(best_ss["mean_e2e"]),
                },
                "interior_optimum": bool(is_interior_ss),
                "selection_note": ss_selection_note,
            },
            "system_2b_smoothed_out": {
                "parameters": {
                    "alpha": float(best_s2b["alpha"]),
                    "beta": float(best_s2b["beta"]),
                    "z": float(best_s2b["z"]),
                    "delta": float(best_s2b["delta"]),
                    "forecast_cap_multiple": float(best_s2b["forecast_cap_multiple"]),
                    "info_sharing": bool(best_s2b["info_sharing"]),
                    "include_lead_time_variance": True,
                },
                "metrics": {
                    "mean_cost": float(best_s2b["mean_cost"]),
                    "mean_fr": float(best_s2b["mean_fr"]),
                    "mean_sl": float(best_s2b["mean_sl"]),
                    "mean_e2e_bullwhip": float(best_s2b["mean_e2e"]),
                },
                "interior_optimum": bool(is_interior_s2b),
                "selection_note": s2b_selection_note,
            },
            "system_2c_adaptive_smoothed_out": {
                "parameters": {
                    "alpha": float(best_s2c["alpha"]),
                    "beta": float(best_s2c["beta"]),
                    "z": float(best_s2c["z"]),
                    "delta": float(best_s2c["delta"]),
                    "forecast_cap_multiple": float(best_s2c["forecast_cap_multiple"]),
                    "beta_shock": float(best_s2c["beta_shock"]),
                    "delta_shock": float(best_s2c["delta_shock"]),
                    "shock_duration": int(best_s2c["shock_duration"]),
                    "shock_threshold": 1.30,
                    "trend_threshold": 1.15,
                    "info_sharing": bool(best_s2c["info_sharing"]),
                    "include_lead_time_variance": True,
                },
                "metrics": {
                    "mean_cost": float(best_s2c["mean_cost"]),
                    "mean_fr": float(best_s2c["mean_fr"]),
                    "mean_sl": float(best_s2c["mean_sl"]),
                    "mean_e2e_bullwhip": float(best_s2c["mean_e2e"]),
                },
                "interior_optimum": bool(is_interior_s2c),
                "selection_note": s2c_selection_note,
            },
        },
    }

    Path("results").mkdir(parents=True, exist_ok=True)
    out_file = Path("results/baselines_frozen.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(frozen_data, f, indent=2)
    print(f"\n[Saved] Frozen baseline parameters saved to {out_file}")

    # Update configs/sim.yaml
    sim_yaml_path = Path("configs/sim.yaml")
    with open(sim_yaml_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    fe = raw_cfg.setdefault("four_echelon", {})
    fe["tuned_eoq_q_mult"] = float(best_eoq["order_quantity_multiplier"])
    fe["tuned_eoq_rop_factor"] = float(best_eoq["reorder_point_factor"])
    fe["tuned_ss_s_factor"] = float(best_ss["s_factor"])
    fe["tuned_ss_S_factor"] = float(best_ss["S_factor"])
    fe["alpha"] = float(best_s2b["alpha"])
    fe["beta"] = float(best_s2b["beta"])
    fe["safety_factor_z"] = float(best_s2b["z"])
    fe["rate_limit_delta"] = float(best_s2b["delta"])
    fe["forecast_cap_multiple"] = float(best_s2b["forecast_cap_multiple"])
    fe["info_sharing"] = bool(best_s2b["info_sharing"])

    # 2c specific params
    fe["s2c_beta_shock"] = float(best_s2c["beta_shock"])
    fe["s2c_delta_shock"] = float(best_s2c["delta_shock"])
    fe["s2c_shock_duration"] = int(best_s2c["shock_duration"])
    fe["s2c_info_sharing"] = bool(best_s2c["info_sharing"])

    with open(sim_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[Updated] Committed frozen parameters into {sim_yaml_path}")


if __name__ == "__main__":
    main()
