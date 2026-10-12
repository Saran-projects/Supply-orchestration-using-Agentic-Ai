# Benchmark v2.1 Protocol Amendment

This amendment formally records the diagnosis, bug resolution, protocol corrections, and updated experimental design for **Benchmark v2.1**, established prior to executing baseline parameter sweeps and test evaluations.

---

## 1. The State Leak Bug and Its Resolution

### Diagnosis & Root Cause
During initial benchmark calibration runs, untuned classical baselines (EOQ and (s, S)) appeared to suffer a catastrophic collapse on tuning seeds 101–104 (costs exceeding \$280,000, retailer service level dropping to 40–50%, and bullwhip ratios evaluating to infinite / NaN).

A systematic regression bisection revealed that this collapse was **not** caused by scenario generation v2, circular block bootstrap noise, or cost asymmetry ratios. Rather, it was an artifact of a state leakage across simulation runs:
- In `EoqPolicy` and `SsPolicy`, the internal dictionary `self.outstanding_orders` tracks unfulfilled orders placed upstream.
- In diagnostic loops where `system.reset()` was omitted between seeds, `outstanding_orders` accumulated across runs.
- By seed 101, `self.outstanding_orders` exceeded 4,000 units. Consequently, the inventory position $IP = \text{inventory} + \text{on\_order} - \text{backlog}$ permanently exceeded the reorder threshold ($4,200 \gg ROP = 627$), causing the policy to place zero orders for all 40 simulation steps.
- At upstream tiers, placing zero orders resulted in zero demand variance ($\text{Var}(\text{demand}) = 0$), causing division-by-zero errors that manifested as `inf` bullwhip ratios.

### Comprehensive Fixes Applied
1. **Explicit Policy Reset**:
   - Every policy class (`EoqPolicy`, `SsPolicy`, `SmoothedOrderUpToPolicy`, `ConstantOrderPolicy`, `BaseStockPolicy`, `OraclePolicy`) implements a strict `reset(scenario)` method clearing all state dictionaries (`outstanding_orders = {e: 0.0}`, smoothed demand history, last orders, oracle lookahead buffers).
2. **Environment Re-Seeding & Isolation**:
   - `BeerGameEnv.reset(seed)` now dynamically refreshes `self.external_demand` using `scenario.get_demand_for_seed(seed)` and instantiates a clean `LeadTimeSampler(seed=seed)` whenever a seed is passed, ensuring complete episode independence even when reusing environment instances.
3. **Metrics Robustness**:
   - Division-by-zero cases where $\text{Var}(\text{demand}) = 0$ or $\text{Var}(\text{orders}) = 0$ return `np.nan` with a logged warning, replacing legacy `inf` values.
4. **Automated Run Isolation Test**:
   - Added `tests/test_run_isolation.py` covering all 10 registered systems (`baseline_constant`, `baseline_s_S`, `baseline_eoq`, `baseline_smoothed_order_up_to`, `baseline_base_stock`, `baseline_oracle`, `single_agent`, `multi_agent_info`, `multi_agent_tools`, `orchestrated`). The test asserts that running a target `(scenario, seed)` in isolation produces bitwise-identical metrics to executing it at the end of a multi-seed batch. This test is an enforced component of `make test`.

---

## 2. Initial Inventory Buffering: Base-Stock Target Calibration

### Theoretical Motivation
The legacy simulator default initialized each echelon with an arbitrary `initial_inventory = 200.0` units. Under DataCo empirical logistics:
- Mean transit lead time $L \approx 3.52$ days (with late deliveries: $p_{\text{late}} = 0.548$, delay $= +1$ day).
- Scaled training demand mean $\mu_D = 100.00$ units/day, $\sigma_D = 20.21$ units/day.
- Mean lead-time demand is $L \cdot \mu_D \approx 352$ units.

Starting an echelon with only 200 units creates an immediate structural deficit of $\approx 152$ units during the initial pipeline transit delay before step 1 replenishments can physically arrive, forcing unavoidable transient stockouts regardless of policy intelligence.

### Parameter Formulation ($z = 2.0$)
To ensure a rigorous, fair, and physically feasible benchmark, all four echelons are initialized at their theoretical base-stock order-up-to level $S$ derived strictly from training-period statistics ($t \in [0, 467]$) with zero test lookahead:

$$S = L \cdot \mu_D + z \cdot \sigma_{\text{LTD}}$$

Where:
$$\sigma_{\text{LTD}} = \sqrt{L \cdot \sigma_D^2 + \mu_D^2 \cdot \sigma_L^2}$$

Empirical values:
- $\mu_D = 100.00$ units/day
- $\sigma_D = 20.21$ units/day
- $L = 3.52$ days
- $\sigma_L = 1.62$ days
- $\sigma_{\text{LTD}} = \sqrt{3.52 \cdot (20.21)^2 + (100.0)^2 \cdot (1.62)^2} = \sqrt{1,437.7 + 26,244.0} \approx 166.38 \text{ units}$
- Safety Stock ($z = 2.0$): $SS = 2.0 \times 166.38 = 332.76 \text{ units}$
- Base-Stock Level: $S = 352.00 + 332.76 = \mathbf{686.56 \approx 687.0 \text{ units}}$

**Specification**:
- Benchmark v2.1 baseline initial inventory: $I_0 = \mathbf{687.0 \text{ units}}$ per echelon.
- Sensitivity evaluation: $I_0 = \mathbf{400.0 \text{ units}}$ per echelon.

---

## 3. Cost Model & Asymmetry Ratio

| Configuration Parameter | Benchmark v2.1 Value | Sensitivity Value |
|:---|:---:|:---:|
| Holding Cost per Unit ($h$) | **$0.50 / unit-step** | **$0.50 / unit-step** |
| Backlog Penalty per Unit ($b$) | **$5.00 / unit-step** | **$2.50 / unit-step** |
| Cost Asymmetry Ratio ($b/h$) | **10.0x** | **5.0x** |

### Justification
In supply chain management (Silver-Pyke-Peterson, Sterman 1989), stockouts incur severe compounding consequences (lost gross margins, expediting freight, customer churn, contractual service penalties). An asymmetry ratio $b/h \ge 10.0\times$ properly penalizes unfulfilled demand and prevents degenerate zero-inventory starving strategies from winning on cost.

---

## 4. Unchanged Pre-Registered Constraints

The pre-registered feasibility constraints defined in `docs/benchmark_v2_rule.md` remain strictly enforced:

1. **Retailer Fill Rate**: $\overline{FR}(\theta) \ge 99.0\%$ on tuning seeds $\{100 \dots 104\}$.
2. **Retailer Step Service Level**: $\overline{SL}(\theta) \ge 85.0\%$ on tuning seeds $\{100 \dots 104\}$.
3. **Bullwhip Dampening Ceiling (75% Rule)**:
   $$\overline{BW}(\theta) \le 0.75 \times \overline{BW}_{\text{EOQ\_untuned}}$$
   evaluated on tuning seeds $\{100 \dots 104\}$.
4. **Interior Optimum Rule**: The minimum-cost feasible configuration $\theta^*$ must be strictly interior to the search grid ($\theta_{\min} < \theta^* < \theta_{\max}$).

---

## 5. System 0 Contingency Rule

System 0 orders a constant $D_0 \approx 100.0$ units every step, completely blind to inventory state.

### Contingency Protocol
- Evaluated strictly on tuning seeds $\{100, 101, 102, 103, 104\}$.
- If System 0 satisfies $\mathcal{F}$ ($FR \ge 99\%, SL \ge 85\%$) and achieves lower total cost than calibrated baselines in Seasonal or Shock scenarios:
  1. The benchmark run halts immediately.
  2. The run does **not** proceed to test seeds $0 \dots 7$.
  3. Report scenario characteristics (demand CV, peak-to-median ratio, persistent regime shift duration) and await explicit user review.
