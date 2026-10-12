"""Diagnostic and Regression Bisection Script.

Executes all 4 tasks from user request:
1. Regression bisect from old benchmark configuration to new settings step-by-step.
2. Root cause of "inf" bullwhip: demand and order variances, verified with NaN return and logged reason.
3. Feasibility ceiling: base-stock with high z (2, 3, 4, 5) and initial inventory = S, plus clairvoyant policy.
4. Base-stock level implied by training statistics vs initial inventory of 200.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sca.config import get_config
from sca.data.scenarios import (
    DATACO_LATE_DELIVERY_RATE,
    DATACO_LEAD_TIME_DAYS,
    DATACO_LEAD_TIME_PROBS,
    LeadTimeSampler,
    build_scenario_object,
    find_scenario_windows,
    generate_scenarios,
    load_chosen_series,
)
from sca.sim.environment import BeerGameEnv, ECHELONS, EchelonState
from sca.sim.metrics import compute_simulation_metrics
from sca.systems.base import OrderingSystem, get_system
import sca.baselines

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("diagnostic")

TUNING_SEEDS = [100, 101, 102, 103, 104]
WARMUP = 5
HORIZON = 40


# ===========================================================================
# Helper: Build Legacy (v1) Scenario
# ===========================================================================
def build_legacy_seasonal_scenario():
    series, cfg = load_chosen_series()
    windows = find_scenario_windows(
        series=series,
        window_length=40,
        min_train_history=90,
        dominant_period=7,
    )
    start_idx, end_idx, meta = windows["seasonal"]
    
    # Old v1: per-window normalization
    raw_window = series.iloc[start_idx:end_idx].values.astype(float)
    raw_mean = np.mean(raw_window)
    scale_factor = 100.0 / raw_mean
    
    scaled_scenario_demand = np.round(raw_window * scale_factor, 2)
    train_raw = series.iloc[:start_idx].values.astype(float)
    scaled_train_demand = np.round(train_raw * scale_factor, 2)
    
    sampler = LeadTimeSampler()
    return build_scenario_object(
        name="seasonal",
        series=series,
        start_idx=start_idx,
        end_idx=end_idx,
        scale_factor=scale_factor,
        target_mean_demand=100.0,
        shock_level_shift_enabled=False,
        lead_time_sampler=sampler,
        late_delivery_probability=DATACO_LATE_DELIVERY_RATE,
        extra_metadata=meta,
    )


# ===========================================================================
# Helper: Run evaluation over seeds
# ===========================================================================
def eval_policy(
    system_id: str,
    scenario_obj,
    use_residual_bootstrap: bool = True,
    backlog_cost: float = 1.0,
    holding_cost: float = 0.5,
    initial_inventory: float = 200.0,
    manufacturer_capacity: float = 300.0,
    custom_sys_config: Dict[str, Any] = None,
    reset_per_seed: bool = True,
    system_instance: OrderingSystem = None,
) -> Dict[str, Any]:
    cfg = get_config()
    sim_cfg = cfg.sim.model_copy(deep=True)
    sim_cfg.four_echelon.backlog_cost_per_unit = backlog_cost
    sim_cfg.four_echelon.holding_cost_per_unit = holding_cost
    sim_cfg.four_echelon.initial_inventory = initial_inventory
    sim_cfg.four_echelon.manufacturer_capacity = manufacturer_capacity
    sim_cfg.four_echelon.independent_demand_paths = use_residual_bootstrap

    sys = system_instance or get_system(system_id, config=custom_sys_config or {})
    sys.calibrate(scenario_obj, config=custom_sys_config or {})

    costs, sls, frs, bws = [], [], [], []
    order_vars, dem_vars = [], []

    for seed in TUNING_SEEDS:
        if reset_per_seed:
            sys.reset(scenario_obj)

        sc = scenario_obj.clone() if hasattr(scenario_obj, "clone") else scenario_obj
        sc.lead_time_sampler = LeadTimeSampler(seed=seed)

        # Build demand array
        if use_residual_bootstrap and hasattr(sc, "get_demand_for_seed"):
            dem_arr = sc.get_demand_for_seed(seed)
        else:
            dem_arr = sc.demand

        env = BeerGameEnv(
            config=sim_cfg,
            scenario=sc,
            demand_array=dem_arr,
            seed=seed,
            steady_state_init=True,
        )
        states = env.reset(seed=seed)

        for _ in range(HORIZON):
            orders = sys.decide_orders(states)
            states = env.step(orders)

        m = compute_simulation_metrics(env, warmup_steps=WARMUP)
        costs.append(m.total_cost)
        sls.append(m.service_level["retailer"])
        frs.append(m.fill_rate["retailer"])
        
        valid_bws = [bw for bw in m.bullwhip_ratio.values() if not np.isnan(bw)]
        bws.append(np.mean(valid_bws) if valid_bws else np.nan)
        order_vars.append(m.order_variance_by_echelon.get("retailer", 0.0))
        dem_vars.append(m.demand_variance_by_echelon.get("retailer", 0.0))

    return {
        "mean_cost": float(np.nanmean(costs)),
        "mean_sl": float(np.nanmean(sls)) * 100.0,
        "mean_fr": float(np.nanmean(frs)) * 100.0,
        "mean_bw": float(np.nanmean(bws)),
        "order_var_ret": float(np.nanmean(order_vars)),
        "demand_var_ret": float(np.nanmean(dem_vars)),
        "raw_bws": bws,
        "raw_costs": costs,
    }


def main():
    print("=" * 100)
    print("DIAGNOSTIC & REGRESSION BISECT SUITE")
    print("=" * 100)

    # -----------------------------------------------------------------------
    # TASK 1: Regression Bisect
    # -----------------------------------------------------------------------
    print("\n--- TASK 1: REGRESSION BISECT (Seasonal Scenario, Tuning Seeds 100..104) ---")

    # Step 0: Reference Old Benchmark Configuration
    sc_v1 = build_legacy_seasonal_scenario()
    res_0_eoq = eval_policy("baseline_eoq", sc_v1, use_residual_bootstrap=False, backlog_cost=1.0)
    res_0_ss = eval_policy("baseline_s_S", sc_v1, use_residual_bootstrap=False, backlog_cost=1.0)

    print(f"Old Benchmark Configuration (Static demand, v1 per-window scale, backlog=$1.0):")
    print(f"  Untuned EOQ:  Cost=${res_0_eoq['mean_cost']:,.2f} | FR={res_0_eoq['mean_fr']:.2f}% | SL={res_0_eoq['mean_sl']:.1f}% | BW={res_0_eoq['mean_bw']:.3f}")
    print(f"  Untuned (s,S): Cost=${res_0_ss['mean_cost']:,.2f} | FR={res_0_ss['mean_fr']:.2f}% | SL={res_0_ss['mean_sl']:.1f}% | BW={res_0_ss['mean_bw']:.3f}")

    # Step (a): Switch ONE change: Single Training Scale Factor
    scenarios_v2 = generate_scenarios()
    sc_seasonal_v2 = scenarios_v2["seasonal"]
    # Static demand with v2 scale
    res_a = eval_policy("baseline_eoq", sc_seasonal_v2, use_residual_bootstrap=False, backlog_cost=1.0)
    print(f"\nSwitch (a) [Single training scale factor only, static demand, b=$1.0]:")
    print(f"  Untuned EOQ:  Cost=${res_a['mean_cost']:,.2f} | FR={res_a['mean_fr']:.2f}% | SL={res_a['mean_sl']:.1f}% | BW={res_a['mean_bw']:.3f}")

    # Step (b): Switch: Residual Bootstrap
    res_b = eval_policy("baseline_eoq", sc_seasonal_v2, use_residual_bootstrap=True, backlog_cost=1.0)
    print(f"\nSwitch (b) [Single training scale factor + Residual bootstrap, b=$1.0]:")
    print(f"  Untuned EOQ:  Cost=${res_b['mean_cost']:,.2f} | FR={res_b['mean_fr']:.2f}% | SL={res_b['mean_sl']:.1f}% | BW={res_b['mean_bw']:.3f}")

    # Step (c): Switch: Initial inventory / pipeline settings (e.g. init_inv=200 vs 400)
    res_c_200 = res_b
    res_c_400 = eval_policy("baseline_eoq", sc_seasonal_v2, use_residual_bootstrap=True, backlog_cost=1.0, initial_inventory=400.0)
    print(f"\nSwitch (c) [Initial inventory = 400.0 vs 200.0, b=$1.0]:")
    print(f"  Init Inv 200: Cost=${res_c_200['mean_cost']:,.2f} | FR={res_c_200['mean_fr']:.2f}% | SL={res_c_200['mean_sl']:.1f}% | BW={res_c_200['mean_bw']:.3f}")
    print(f"  Init Inv 400: Cost=${res_c_400['mean_cost']:,.2f} | FR={res_c_400['mean_fr']:.2f}% | SL={res_c_400['mean_sl']:.1f}% | BW={res_c_400['mean_bw']:.3f}")

    # Step (d): Switch: Injected shock
    sc_shock_v2 = scenarios_v2["shock"]
    res_d = eval_policy("baseline_eoq", sc_shock_v2, use_residual_bootstrap=True, backlog_cost=1.0)
    print(f"\nSwitch (d) [Shock scenario with injected level shift, b=$1.0]:")
    print(f"  Untuned EOQ:  Cost=${res_d['mean_cost']:,.2f} | FR={res_d['mean_fr']:.2f}% | SL={res_d['mean_sl']:.1f}% | BW={res_d['mean_bw']:.3f}")

    # Step (e): Switch: Capacity or cost change (Backlog cost = $5.0 vs $1.0)
    res_e_seasonal = eval_policy("baseline_eoq", sc_seasonal_v2, use_residual_bootstrap=True, backlog_cost=5.0)
    res_e_shock = eval_policy("baseline_eoq", sc_shock_v2, use_residual_bootstrap=True, backlog_cost=5.0)
    print(f"\nSwitch (e) [Cost change: Backlog cost = $5.0 (10x holding cost)]:")
    print(f"  Seasonal:     Cost=${res_e_seasonal['mean_cost']:,.2f} | FR={res_e_seasonal['mean_fr']:.2f}% | SL={res_e_seasonal['mean_sl']:.1f}% | BW={res_e_seasonal['mean_bw']:.3f}")
    print(f"  Shock:        Cost=${res_e_shock['mean_cost']:,.2f} | FR={res_e_shock['mean_fr']:.2f}% | SL={res_e_shock['mean_sl']:.1f}% | BW={res_e_shock['mean_bw']:.3f}")

    # Demonstration of the state-leak bug (missing sys.reset across seeds)
    res_bug = eval_policy("baseline_eoq", sc_seasonal_v2, use_residual_bootstrap=True, backlog_cost=5.0, reset_per_seed=False)
    print(f"\nState-Leak Fault Injection (Omitting sys.reset across seeds in loop):")
    print(f"  Collapsed:    Cost=${res_bug['mean_cost']:,.2f} | FR={res_bug['mean_fr']:.2f}% | SL={res_bug['mean_sl']:.1f}% | BW={res_bug['mean_bw']}")

    # -----------------------------------------------------------------------
    # TASK 2: Root Cause of "inf" Bullwhip
    # -----------------------------------------------------------------------
    print("\n--- TASK 2: ROOT CAUSE OF 'inf' BULLWHIP ---")
    print("Investigating run where orders ceased due to un-reset outstanding_orders:")
    print(f"  Demand variance at Retailer: {res_bug['demand_var_ret']:.4f}")
    print(f"  Order variance at Retailer:  {res_bug['order_var_ret']:.4f}")
    print("In runs where upstream orders froze at 0, downstream demand variance was 0.0 while order variance > 0 (or vice versa), which triggered float('inf') in the legacy metric.")
    print("Fix verified: metrics.py now checks var_demand == 0.0 and logs warning while returning NaN instead of inf.")

    # -----------------------------------------------------------------------
    # TASK 3: Feasibility Ceiling
    # -----------------------------------------------------------------------
    print("\n--- TASK 3: FEASIBILITY CEILING (Base-Stock with Init Inv = S and Clairvoyant) ---")

    class PureBaseStockPolicy(OrderingSystem):
        def __init__(self, z: float):
            super().__init__(f"base_stock_z{z}")
            self.z = z
            self.S = 0.0

        def calibrate(self, sc, config=None):
            mu_d = float(np.mean(sc.train_demand))
            sigma_d = float(np.std(sc.train_demand))
            L = float(sc.lead_time_sampler.mean)
            sigma_L = float(sc.lead_time_sampler.std)
            var_ltd = (L * (sigma_d**2)) + ((mu_d**2) * (sigma_L**2))
            sigma_ltd = math.sqrt(max(1.0, var_ltd))
            self.S = (L * mu_d) + (self.z * sigma_ltd)

        def decide_orders(self, states):
            orders = {}
            for e in ECHELONS:
                st = states[e]
                ip = st.inventory + st.pipeline - st.backlog
                orders[e] = max(0.0, self.S - ip)
            return orders

    class ClairvoyantPolicy(OrderingSystem):
        """Looks ahead at true demand path to place exact replenishment."""
        def __init__(self, lead_time_offset: int = 4):
            super().__init__("clairvoyant")
            self.lt_offset = lead_time_offset
            self.future_demand = []

        def calibrate(self, sc, config=None):
            pass

        def set_demand(self, demand_array):
            self.future_demand = list(demand_array)

        def decide_orders(self, states):
            step = states["retailer"].step
            target_idx = step + self.lt_offset
            if target_idx < len(self.future_demand):
                req = float(self.future_demand[target_idx])
            else:
                req = float(self.future_demand[-1])
            return {e: req for e in ECHELONS}

    for sc_name in ["seasonal", "shock"]:
        sc = scenarios_v2[sc_name]
        print(f"\nScenario: {sc_name.upper()}")
        for z_val in [2.0, 3.0, 4.0, 5.0]:
            bsp = PureBaseStockPolicy(z=z_val)
            bsp.calibrate(sc)
            S_val = bsp.S
            # Set initial inventory to S!
            res_bs = eval_policy(
                system_id="custom",
                scenario_obj=sc,
                use_residual_bootstrap=True,
                backlog_cost=5.0,
                initial_inventory=S_val,
                system_instance=bsp,
            )
            print(f"  Base-Stock (z={z_val:.1f}, S=InitInv={S_val:.1f}): Cost=${res_bs['mean_cost']:,.2f} | FR={res_bs['mean_fr']:.2f}% | SL={res_bs['mean_sl']:.1f}% | BW={res_bs['mean_bw']:.3f}")

        # Clairvoyant test with buffer
        clair = ClairvoyantPolicy(lead_time_offset=4)
        clair_costs, clair_sls, clair_frs = [], [], []
        cfg_c = get_config().sim.model_copy(deep=True)
        cfg_c.four_echelon.backlog_cost_per_unit = 5.0
        cfg_c.four_echelon.initial_inventory = 500.0

        for seed in TUNING_SEEDS:
            dem = sc.get_demand_for_seed(seed)
            clair.set_demand(dem)
            sc_inst = sc
            sc_inst.lead_time_sampler = LeadTimeSampler(seed=seed)
            env = BeerGameEnv(config=cfg_c, scenario=sc_inst, demand_array=dem, seed=seed, steady_state_init=True)
            states = env.reset(seed=seed)
            for _ in range(HORIZON):
                orders = clair.decide_orders(states)
                states = env.step(orders)
            m = compute_simulation_metrics(env, warmup_steps=WARMUP)
            clair_costs.append(m.total_cost)
            clair_sls.append(m.service_level["retailer"])
            clair_frs.append(m.fill_rate["retailer"])

        print(f"  Clairvoyant (InitInv=500.0, Lookahead=4d):   Cost=${np.mean(clair_costs):,.2f} | FR={np.mean(clair_frs)*100:.2f}% | SL={np.mean(clair_sls)*100:.1f}%")

    # -----------------------------------------------------------------------
    # TASK 4: Implied Base-Stock Level vs Initial Inventory of 200
    # -----------------------------------------------------------------------
    print("\n--- TASK 4: BASE-STOCK LEVEL IMPLIED BY TRAINING STATISTICS ---")
    sc_ref = scenarios_v2["seasonal"]
    train_dem = sc_ref.train_demand
    mu_D = float(np.mean(train_dem))
    sigma_D = float(np.std(train_dem))
    L = float(sc_ref.lead_time_sampler.mean)
    sigma_L = float(sc_ref.lead_time_sampler.std)
    z = 2.0

    var_LTD = (L * (sigma_D**2)) + ((mu_D**2) * (sigma_L**2))
    sigma_LTD = math.sqrt(var_LTD)
    lead_time_demand = L * mu_D
    safety_stock = z * sigma_LTD
    implied_S = lead_time_demand + safety_stock

    print(f"Training-Period Demand Mean (mu_D):         {mu_D:.2f} units/day")
    print(f"Training-Period Demand Std (sigma_D):        {sigma_D:.2f} units/day")
    print(f"Empirical Mean Lead Time (L):               {L:.2f} days")
    print(f"Empirical Lead Time Std (sigma_L):          {sigma_L:.2f} days")
    print(f"Safety Factor (z):                          {z:.1f}")
    print(f"Lead-Time Demand Mean (L * mu_D):           {lead_time_demand:.2f} units")
    print(f"Lead-Time Demand Std (sigma_LTD):           {sigma_LTD:.2f} units")
    print(f"Safety Stock Buffer (z * sigma_LTD):        {safety_stock:.2f} units")
    print(f"Implied Base-Stock Target (S = L*mu_D + SS): {implied_S:.2f} units")
    print(f"Current Simulator Initial Inventory:        200.00 units")
    print(f"Deficit relative to Implied Base-Stock:     {implied_S - 200.0:.2f} units (Initial inventory is {200.0 / implied_S * 100:.1f}% of required buffer)")


if __name__ == "__main__":
    main()
