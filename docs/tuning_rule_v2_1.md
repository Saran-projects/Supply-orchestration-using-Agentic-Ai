# Benchmark v2.1 Hyperparameter Tuning Rule & Protocol

**Document Version**: 2.1.0  
**Status**: Pre-registered and Frozen prior to baseline tuning sweeps  
**Reference Document**: [benchmark_v2_1.md](file:///C:/Users/saran/smartsupply/docs/benchmark_v2_1.md)

---

## 1. Scope & Objective

This document formalizes the exact hyperparameter tuning protocol, parameter domains, constraints, and objective function for **Benchmark v2.1** baseline policies:
- **System 2 (EOQ)**: Economic Order Quantity with Reorder Point
- **System 1 (s, S)**: Min-Max Inventory Policy
- **System 2b (Smoothed OUT)**: Exponentially smoothed demand forecast and order-up-to target with rate-limited smoothing
- **System 2c (Adaptive Smoothed OUT)**: System 2b combined with a statistical regime detector (CUSUM / short-to-long ratio) that dynamically adapts $\beta$ and $\delta$ during detected shock regimes, with optional echelon information sharing (`info_sharing`).

All policies are tuned **exclusively on tuning seeds** $\mathcal{S}_{\text{tune}} = \{100, 101, 102, 103, 104\}$ across the three canonical evaluation scenarios: **Stable**, **Seasonal**, and **Shock**. Test seeds $\{0, 1, 2, 3, 4, 5, 6, 7\}$ are strictly reserved for post-tuning evaluation and are never accessed during the hyperparameter search.

---

## 2. Simulation Environment Calibration

All tuning simulations execute under identical physical logistics and cost environments:
1. **Initial Inventory Buffering**:
   - Initial on-hand inventory per echelon: $I_0 = \mathbf{687.0 \text{ units}}$ ($z = 2.0$ base-stock target derived from training statistics: $\mu_D = 100.0, \sigma_D = 20.21, L = 3.52, \sigma_L = 1.62$).
   - Initial pipeline shipments: $100.0$ units per transit step (steady-state flow conservation).
2. **Cost Parameters**:
   - Holding cost: $h = \$0.50$ per unit-day.
   - Stockout backlog cost: $b = \$2.50$ per unit-day ($b/h = 5.0\times$ primary ratio).
   - Sensitivity benchmarks: $b/h = 2.0\times$ ($b = \$1.00$) and $b/h = 10.0\times$ ($b = \$5.00$).
3. **Horizon & Warmup**:
   - Total simulation horizon: $T = 40$ steps.
   - Metric warmup window: $W = 5$ steps (steps $0 \dots 4$ excluded from metrics calculation).
4. **Demand Regimes**:
   - Single global scale factor applied to raw DataCo series: $\text{scale} = 100.0 / \mu_{D, \text{train\_raw}}$.
   - **Stable**: Low-CV historical window + circular block bootstrap.
   - **Seasonal**: High weekly autocorrelation window ($p = 7$) + circular block bootstrap.
   - **Shock**: Empirical window + injected level shift ($1.5\times$ surge from step 15 to step 25).
5. **State Isolation**:
   - `env.reset(seed)` and `system.reset(scenario)` are invoked before every simulation run.

---

## 3. Mathematical Selection Rule

For any policy configuration candidate $\theta \in \Theta$:

### 3.1 Metrics Aggregation
The configuration is evaluated across all scenarios $\mathcal{C} = \{\text{stable}, \text{seasonal}, \text{shock}\}$ and tuning seeds $s \in \mathcal{S}_{\text{tune}}$ ($3 \times 5 = 15$ runs per candidate):

$$\overline{\text{TotalCost}}(\theta) = \frac{1}{|\mathcal{C}| \cdot |\mathcal{S}_{\text{tune}}|} \sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}_{\text{tune}}} \text{TotalCost}(c, s; \theta)$$

$$\overline{FR}(\theta) = \frac{1}{|\mathcal{C}| \cdot |\mathcal{S}_{\text{tune}}|} \sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}_{\text{tune}}} \text{FillRate}_{\text{retailer}}(c, s; \theta)$$

$$\overline{BW}_{\text{e2e}}(\theta) = \frac{1}{|\mathcal{C}| \cdot |\mathcal{S}_{\text{tune}}|} \sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}_{\text{tune}}} BW_{\text{e2e}}(c, s; \theta)$$

$$\overline{SL}(\theta) = \frac{1}{|\mathcal{C}| \cdot |\mathcal{S}_{\text{tune}}|} \sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}_{\text{tune}}} \text{ServiceLevel}_{\text{retailer}}(c, s; \theta)$$

### 3.2 Feasibility Constraint Set $\mathcal{F}$
A candidate configuration $\theta$ is **feasible** ($\theta \in \mathcal{F}$) if and only if:
1. **Retailer Unit Fill Rate**:
   $$\overline{FR}(\theta) \ge 99.0\%$$
2. **End-to-End Bullwhip Dampening Ceiling (75% Rule)**:
   $$\overline{BW}_{\text{e2e}}(\theta) \le \text{Threshold}_{BW} = 0.75 \times \overline{BW}_{\text{e2e, EOQ\_untuned}}$$
   where $\overline{BW}_{\text{e2e, EOQ\_untuned}}$ is measured across tuning seeds $\{100 \dots 104\}$.

### 3.3 Step Service Level Reporting
- **Step Service Level ($\overline{SL}$)** (fraction of non-warmup steps with zero unfulfilled backlog at the retailer) is strictly **reported** in summary tables, but is **not constrained** in the formal selection filter.

### 3.4 Optimization Objective
$$\theta^* = \arg\min_{\theta \in \mathcal{F}} \overline{\text{TotalCost}}(\theta)$$

If no configuration satisfies both constraints, the Pareto compromise with lowest $\overline{BW}_{\text{e2e}}$ among candidates with $\overline{FR} \ge 99.0\%$ is reported with explicit caveat documentation.

---

## 4. Interior Optimum Rule & Search Grids

To prevent boundary clipping and ensure robust local or global optimality:
- If an optimal parameter $\theta_j^*$ lies on the minimum or maximum bound of its discrete evaluation grid ($\theta_j^* \in \{\min(\Theta_j), \max(\Theta_j)\}$), the grid MUST be expanded outward until the parameter is strictly **interior**:
  $$\min(\Theta_j) < \theta_j^* < \max(\Theta_j)$$

### Search Grids

#### 1. System 2 (EOQ)
- `order_quantity_multiplier` ($Q_{\text{mult}}$): $[0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]$
- `reorder_point_factor` ($ROP_{\text{factor}}$): $[0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]$

#### 2. System 1 (s, S)
- `s_factor` ($s_{\text{factor}}$): $[0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]$
- `S_factor` ($S_{\text{factor}}$): $[0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]$

#### 3. System 2b (Smoothed OUT)
- `alpha` ($\alpha$ - forecast smoothing): $[0.4, 0.6, 0.8]$
- `beta` ($\beta$ - order smoothing): $[0.2, 0.4, 0.6, 0.8]$
- `z` ($z$ - safety factor): $[1.5, 2.0, 2.5]$
- `delta` ($\delta$ - rate-limiting band): $[0.2, 0.3, 0.5, 0.8]$
- `forecast_cap_multiple`: $[1.5, 2.0]$
- `info_sharing`: $[\text{False}, \text{True}]$

#### 4. System 2c (Adaptive Smoothed OUT with Regime Detection)
- Steady-state parameters: inherited from System 2b
- `beta_shock` ($\beta_{\text{shock}}$): $[0.8, 0.9, 1.0]$
- `delta_shock` ($\delta_{\text{shock}}$): $[0.6, 0.8, 1.0]$
- `shock_duration` (steps): $[4, 6, 8]$
- `shock_threshold`: $1.30$
- `info_sharing`: $[\text{False}, \text{True}]$

---

## 5. Artifact Freezing Protocol

Once tuning completes:
1. All selected parameters $(\theta_{\text{EOQ}}^*, \theta_{s,S}^*, \theta_{2b}^*, \theta_{2c}^*)$ are recorded in `results/baselines_frozen.json` with full tuning metrics.
2. The parameters are committed to `configs/sim.yaml` under `four_echelon`.
3. The frozen configurations are then evaluated once on test seeds $\{0 \dots 7\}$ without subsequent modification.
