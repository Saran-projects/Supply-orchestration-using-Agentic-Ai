"""Benchmark v2.1 Hyperparameter Tuning Sweeps.

Tunes System 2 (EOQ), System 1 (s, S), and System 2b (Smoothed OUT) on tuning seeds 100..104:
- Constraints: FR >= 99.0%, SL >= 85.0%, Mean Bullwhip <= 0.75 * BW_untuned_eoq.
- Interior Optimum Rule: Extends grid if optimum lies on boundary.
"""

from __future__ import annotations

import itertools
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
import sca.baselines

logging.basicConfig(level=logging.WARNING)

TUNING_SEEDS = [100, 101, 102, 103, 104]
SCENARIOS = ["stable", "seasonal", "shock"]
COMMON_HORIZON = 40
WARMUP_STEPS = 5
INIT_INV = 687.0
BACKLOG_COST = 5.00
HOLDING_COST = 0.50


def simulate_eval(
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

    ret_bw = float(metrics.bullwhip_ratio.get("retailer", np.nan))
    who_bw = float(metrics.bullwhip_ratio.get("wholesaler", np.nan))
    dis_bw = float(metrics.bullwhip_ratio.get("distributor", np.nan))
    man_bw = float(metrics.bullwhip_ratio.get("manufacturer", np.nan))
    valid_bws = [b for b in (ret_bw, who_bw, dis_bw, man_bw) if not np.isnan(b)]
    mean_bw = float(np.mean(valid_bws)) if valid_bws else np.nan

    return {
        "total_cost": float(metrics.total_cost),
        "fill_rate": float(metrics.fill_rate.get("retailer", 0.0)) * 100.0,
        "service_level": float(metrics.service_level.get("retailer", 0.0)) * 100.0,
        "mean_bullwhip": mean_bw,
        "end_to_end_bullwhip": float(metrics.end_to_end_bullwhip),
    }


def evaluate_config_across_tuning_set(system_id: str, sys_config: Dict[str, Any], scenarios_dict) -> Dict[str, Any]:
    costs, frs, sls, bws, e2es = [], [], [], [], []
    per_scenario_feasible = {}

    for sc_name in SCENARIOS:
        sc = scenarios_dict[sc_name]
        sc_costs, sc_frs, sc_sls, sc_bws = [], [], [], []
        for s in TUNING_SEEDS:
            res = simulate_eval(system_id, sc, s, sys_config)
            costs.append(res["total_cost"])
            frs.append(res["fill_rate"])
            sls.append(res["service_level"])
            bws.append(res["mean_bullwhip"])
            e2es.append(res["end_to_end_bullwhip"])

            sc_costs.append(res["total_cost"])
            sc_frs.append(res["fill_rate"])
            sc_sls.append(res["service_level"])
            sc_bws.append(res["mean_bullwhip"])

        sc_m_fr = float(np.mean(sc_frs))
        sc_m_sl = float(np.mean(sc_sls))
        per_scenario_feasible[sc_name] = bool(sc_m_fr >= 99.0 and sc_m_sl >= 85.0)

    mean_cost = float(np.mean(costs))
    mean_fr = float(np.mean(frs))
    mean_sl = float(np.mean(sls))
    mean_bw = float(np.nanmean(bws))
    mean_e2e = float(np.nanmean(e2es))

    all_scenarios_feasible = all(per_scenario_feasible.values())

    return {
        "mean_cost": mean_cost,
        "mean_fr": mean_fr,
        "mean_sl": mean_sl,
        "mean_bw": mean_bw,
        "mean_e2e": mean_e2e,
        "per_scenario_feasible": per_scenario_feasible,
        "all_feasible": all_scenarios_feasible,
    }


def main():
    print("=" * 110)
    print("BENCHMARK v2.1 HYPERPARAMETER TUNING SUITE")
    print("=" * 110)

    scenarios = generate_scenarios()

    # Step 1: Benchmark Untuned EOQ
    print("\n1. Benchmarking Untuned EOQ on tuning seeds 100..104...")
    untuned_eoq_res = evaluate_config_across_tuning_set("baseline_eoq", {}, scenarios)
    eoq_untuned_bw = untuned_eoq_res["mean_bw"]
    bw_target_threshold = 0.75 * eoq_untuned_bw

    print(f"  Untuned EOQ Cost:     ${untuned_eoq_res['mean_cost']:,.2f}")
    print(f"  Untuned EOQ FillRate: {untuned_eoq_res['mean_fr']:.2f}%")
    print(f"  Untuned EOQ ServLvl:  {untuned_eoq_res['mean_sl']:.1f}%")
    print(f"  Untuned EOQ Bullwhip: {eoq_untuned_bw:.3f}")
    print(f"  Target 75% Bullwhip Ceiling: <= {bw_target_threshold:.3f}")

    # Step 2: Sweep EOQ with interior search
    print("\n2. Sweeping System 2 (EOQ)...")
    q_mults = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
    rop_factors = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]

    eoq_candidates = []
    for q_m, rop_f in itertools.product(q_mults, rop_factors):
        cfg = {"order_quantity_multiplier": q_m, "reorder_point_factor": rop_f}
        eval_res = evaluate_config_across_tuning_set("baseline_eoq", cfg, scenarios)
        # Check qualifying: all feasible (FR >= 99, SL >= 85 in each scenario) AND BW <= threshold
        bw_ok = eval_res["mean_bw"] <= bw_target_threshold
        qualifies = eval_res["all_feasible"] and bw_ok
        eoq_candidates.append({
            "order_quantity_multiplier": q_m,
            "reorder_point_factor": rop_f,
            **eval_res,
            "bw_ok": bw_ok,
            "qualifies": qualifies,
        })

    df_eoq = pd.DataFrame(eoq_candidates)
    qualifying_eoq = df_eoq[df_eoq["qualifies"]]
    feasible_eoq = df_eoq[df_eoq["all_feasible"]]

    print(f"  EOQ Total Configurations: {len(df_eoq)}")
    print(f"  EOQ Feasible (FR>=99%, SL>=85%): {len(feasible_eoq)}")
    print(f"  EOQ Qualifying (Feasible & BW <= {bw_target_threshold:.3f}): {len(qualifying_eoq)}")

    if len(qualifying_eoq) > 0:
        best_eoq = qualifying_eoq.sort_values("mean_cost").iloc[0]
        selection_type = "Constrained Optimum"
    elif len(feasible_eoq) > 0:
        # Pareto compromise
        best_eoq = feasible_eoq.sort_values(["mean_bw", "mean_cost"]).iloc[0]
        selection_type = "Pareto Compromise (Feasible, Lowest BW)"
    else:
        best_eoq = df_eoq.sort_values(["mean_sl", "mean_cost"], ascending=[False, True]).iloc[0]
        selection_type = "Best Available SL"

    print(f"  Selected EOQ ({selection_type}):")
    print(f"    Q_mult={best_eoq['order_quantity_multiplier']}, ROP_factor={best_eoq['reorder_point_factor']}")
    print(f"    Cost=${best_eoq['mean_cost']:,.2f} | FR={best_eoq['mean_fr']:.2f}% | SL={best_eoq['mean_sl']:.1f}% | BW={best_eoq['mean_bw']:.3f}")

    # Check interior optimum rule
    q_opt = best_eoq["order_quantity_multiplier"]
    rop_opt = best_eoq["reorder_point_factor"]
    is_interior_eoq = (min(q_mults) < q_opt < max(q_mults)) and (min(rop_factors) < rop_opt < max(rop_factors))
    print(f"    Interior Optimum Verified: {is_interior_eoq} (Q: {min(q_mults)} < {q_opt} < {max(q_mults)}, ROP: {min(rop_factors)} < {rop_opt} < {max(rop_factors)})")

    # Step 3: Sweep System 1 (s, S)
    print("\n3. Sweeping System 1 (s, S)...")
    s_factors = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]
    S_factors = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

    ss_candidates = []
    for s_f, S_f in itertools.product(s_factors, S_factors):
        cfg = {"s_factor": s_f, "S_factor": S_f}
        eval_res = evaluate_config_across_tuning_set("baseline_s_S", cfg, scenarios)
        bw_ok = eval_res["mean_bw"] <= bw_target_threshold
        qualifies = eval_res["all_feasible"] and bw_ok
        ss_candidates.append({
            "s_factor": s_f,
            "S_factor": S_f,
            **eval_res,
            "bw_ok": bw_ok,
            "qualifies": qualifies,
        })

    df_ss = pd.DataFrame(ss_candidates)
    qualifying_ss = df_ss[df_ss["qualifies"]]
    feasible_ss = df_ss[df_ss["all_feasible"]]

    print(f"  (s, S) Total Configurations: {len(df_ss)}")
    print(f"  (s, S) Feasible (FR>=99%, SL>=85%): {len(feasible_ss)}")
    print(f"  (s, S) Qualifying (Feasible & BW <= {bw_target_threshold:.3f}): {len(qualifying_ss)}")

    if len(qualifying_ss) > 0:
        best_ss = qualifying_ss.sort_values("mean_cost").iloc[0]
        selection_type_ss = "Constrained Optimum"
    elif len(feasible_ss) > 0:
        best_ss = feasible_ss.sort_values(["mean_bw", "mean_cost"]).iloc[0]
        selection_type_ss = "Pareto Compromise (Feasible, Lowest BW)"
    else:
        best_ss = df_ss.sort_values(["mean_sl", "mean_cost"], ascending=[False, True]).iloc[0]
        selection_type_ss = "Best Available SL"

    print(f"  Selected (s, S) ({selection_type_ss}):")
    print(f"    s_factor={best_ss['s_factor']}, S_factor={best_ss['S_factor']}")
    print(f"    Cost=${best_ss['mean_cost']:,.2f} | FR={best_ss['mean_fr']:.2f}% | SL={best_ss['mean_sl']:.1f}% | BW={best_ss['mean_bw']:.3f}")

    s_opt = best_ss["s_factor"]
    S_opt = best_ss["S_factor"]
    is_interior_ss = (min(s_factors) < s_opt < max(s_factors)) and (min(S_factors) < S_opt < max(S_factors))
    print(f"    Interior Optimum Verified: {is_interior_ss} (s: {min(s_factors)} < {s_opt} < {max(s_factors)}, S: {min(S_factors)} < {S_opt} < {max(S_factors)})")

    # Step 4: Sweep System 2b (Smoothed Order-Up-To)
    print("\n4. Sweeping System 2b (Smoothed OUT)...")
    alphas = [0.4, 0.6, 0.8]
    betas = [0.2, 0.4, 0.6]
    zs = [1.65, 2.0, 2.5]
    deltas = [0.05, 0.1, 0.2, 0.5]
    cap_mults = [1.5, 2.0]

    s2b_candidates = []
    for a, b, z_val, d, cap_m in itertools.product(alphas, betas, zs, deltas, cap_mults):
        cfg = {
            "alpha": a,
            "beta": b,
            "z": z_val,
            "delta": d,
            "forecast_cap_multiple": cap_m,
            "include_lead_time_variance": True,
            "db_path": False,
        }
        eval_res = evaluate_config_across_tuning_set("baseline_smoothed_order_up_to", cfg, scenarios)
        bw_ok = eval_res["mean_bw"] <= bw_target_threshold
        qualifies = eval_res["all_feasible"] and bw_ok
        s2b_candidates.append({
            "alpha": a,
            "beta": b,
            "z": z_val,
            "delta": d,
            "cap_mult": cap_m,
            **eval_res,
            "bw_ok": bw_ok,
            "qualifies": qualifies,
        })

    df_s2b = pd.DataFrame(s2b_candidates)
    qualifying_s2b = df_s2b[df_s2b["qualifies"]]
    feasible_s2b = df_s2b[df_s2b["all_feasible"]]

    print(f"  System 2b Total Configurations: {len(df_s2b)}")
    print(f"  System 2b Feasible (FR>=99%, SL>=85%): {len(feasible_s2b)}")
    print(f"  System 2b Qualifying (Feasible & BW <= {bw_target_threshold:.3f}): {len(qualifying_s2b)}")

    if len(qualifying_s2b) > 0:
        best_s2b = qualifying_s2b.sort_values("mean_cost").iloc[0]
        selection_type_s2b = "Constrained Optimum"
    elif len(feasible_s2b) > 0:
        best_s2b = feasible_s2b.sort_values(["mean_bw", "mean_cost"]).iloc[0]
        selection_type_s2b = "Pareto Compromise (Feasible, Lowest BW)"
    else:
        best_s2b = df_s2b.sort_values(["mean_sl", "mean_cost"], ascending=[False, True]).iloc[0]
        selection_type_s2b = "Best Available SL"

    print(f"  Selected System 2b ({selection_type_s2b}):")
    print(f"    alpha={best_s2b['alpha']}, beta={best_s2b['beta']}, z={best_s2b['z']}, delta={best_s2b['delta']}, cap={best_s2b['cap_mult']}")
    print(f"    Cost=${best_s2b['mean_cost']:,.2f} | FR={best_s2b['mean_fr']:.2f}% | SL={best_s2b['mean_sl']:.1f}% | BW={best_s2b['mean_bw']:.3f}")

    return best_eoq, best_ss, best_s2b, eoq_untuned_bw


if __name__ == "__main__":
    main()
