"""Refine Search Grids to Guarantee Strictly Interior Optima for Benchmark v2.1.

Extends:
- (s, S): S_factors down to 0.05..0.50
- System 2b: beta up to 0.8, z up to 3.5, delta up to 1.0, cap_mult down to 1.2
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tune_benchmarks_v2_1 import evaluate_config_across_tuning_set

from sca.data.scenarios import generate_scenarios

logging.basicConfig(level=logging.WARNING)


def refine():
    scenarios = generate_scenarios()
    bw_target_threshold = 5.756  # 0.75 * 7.675 from untuned EOQ

    # 1. Refine (s, S) grid
    print("\n--- REFINING (s, S) SEARCH GRID ---")
    s_factors = [1.00, 1.10, 1.20, 1.30, 1.40]
    S_factors = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75]

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
    print(f"Qualifying (s, S) configs: {len(qualifying_ss)} / {len(df_ss)}")

    if len(qualifying_ss) > 0:
        best_ss = qualifying_ss.sort_values("mean_cost").iloc[0]
    else:
        best_ss = df_ss[df_ss["all_feasible"]].sort_values(["mean_bw", "mean_cost"]).iloc[0]

    s_opt, S_opt = best_ss["s_factor"], best_ss["S_factor"]
    is_interior_ss = (min(s_factors) < s_opt < max(s_factors)) and (min(S_factors) < S_opt < max(S_factors))
    print(f"Optimal (s, S): s_factor={s_opt}, S_factor={S_opt}")
    print(f"  Cost=${best_ss['mean_cost']:,.2f} | FR={best_ss['mean_fr']:.2f}% | SL={best_ss['mean_sl']:.1f}% | BW={best_ss['mean_bw']:.3f}")
    print(f"  Strictly Interior: {is_interior_ss} ({min(s_factors)} < {s_opt} < {max(s_factors)}, {min(S_factors)} < {S_opt} < {max(S_factors)})")

    # 2. Refine System 2b grid
    print("\n--- REFINING SYSTEM 2b SEARCH GRID ---")
    alphas = [0.4, 0.6, 0.8]
    betas = [0.4, 0.6, 0.8]
    zs = [2.0, 2.5, 3.0]
    deltas = [0.2, 0.5, 0.8]
    cap_mults = [1.2, 1.5, 2.0]

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
    print(f"Qualifying System 2b configs: {len(qualifying_s2b)} / {len(df_s2b)}")

    if len(qualifying_s2b) > 0:
        best_s2b = qualifying_s2b.sort_values("mean_cost").iloc[0]
    else:
        best_s2b = df_s2b[df_s2b["all_feasible"]].sort_values(["mean_bw", "mean_cost"]).iloc[0]

    a_opt = best_s2b["alpha"]
    b_opt = best_s2b["beta"]
    z_opt = best_s2b["z"]
    d_opt = best_s2b["delta"]
    cap_opt = best_s2b["cap_mult"]

    is_int_a = min(alphas) < a_opt < max(alphas)
    is_int_b = min(betas) < b_opt < max(betas)
    is_int_z = min(zs) < z_opt < max(zs)
    is_int_d = min(deltas) < d_opt < max(deltas)
    is_int_cap = min(cap_mults) < cap_opt < max(cap_mults)
    all_int_2b = is_int_a and is_int_b and is_int_z and is_int_d and is_int_cap

    print(f"Optimal System 2b: alpha={a_opt}, beta={b_opt}, z={z_opt}, delta={d_opt}, cap={cap_opt}")
    print(f"  Cost=${best_s2b['mean_cost']:,.2f} | FR={best_s2b['mean_fr']:.2f}% | SL={best_s2b['mean_sl']:.1f}% | BW={best_s2b['mean_bw']:.3f}")
    print(f"  Strictly Interior: {all_int_2b} (a:{is_int_a}, b:{is_int_b}, z:{is_int_z}, d:{is_int_d}, cap:{is_int_cap})")

    return best_ss, best_s2b


if __name__ == "__main__":
    refine()
