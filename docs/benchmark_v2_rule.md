# Benchmark v2 Tuning Protocol & Selection Rule

## Overview & Scope
This document specifies the pre-registered evaluation, hyperparameter tuning protocol, and selection rules for **Benchmark v2** across all supply chain policies:
1. **System 0: Constant Order Baseline** (zero-state benchmark ordering training-period mean demand)
2. **System 1: (s, S) Min-Max Policy** (untuned and tuned)
3. **System 2: EOQ / ROP Policy** (untuned and tuned)
4. **System 2b: Smoothed Order-Up-To Policy** (untuned and tuned)

This specification supersedes Benchmark v1 and is formally committed **prior** to executing tuning sweeps.

---

## 1. Cost Configuration & Asymmetry Ratio

Cost parameters are defined in `configs/sim.yaml` (`four_echelon`) and enforced in `FourEchelonConfig` and `BeerGameEnv`:

| Parameter | Configuration Key | Value | Description |
|:---|:---|:---:|:---|
| **Holding Cost ($h$)** | `four_echelon.holding_cost_per_unit` | **$0.50 / unit-step** | Capital and storage cost per unit of inventory on hand per step |
| **Backlog Cost ($b$)** | `four_echelon.backlog_cost_per_unit` | **$5.00 / unit-step** | Unfulfilled downstream demand penalty per unit per step |
| **Backlog Multiplier ($b/h$)** | `four_echelon.backlog_cost_multiplier` | **10.0x** | Ratio of backlog penalty to holding cost |

### Rationale
In real-world multi-echelon supply networks and classic inventory theory (Silver-Pyke-Peterson, Sterman 1989), stockout and backlog penalties significantly exceed holding cost (typically $5\times$ to $20\times$). A low penalty ratio (such as the legacy $2.0\times$) enables naive policies like System 0 to run out of stock almost continuously while appearing artificially competitive on total cost due to zero holding expense. The $10.0\times$ penalty ratio, combined with formal service-level constraints, ensures that unfulfilled demand incurs realistic penalties.

---

## 2. Selection Rule v2 (Constrained Optimization)

For each candidate policy configuration $\theta$ evaluated over tuning seeds $\mathcal{S}_{\text{tune}} = \{100, 101, 102, 103, 104\}$ across all three scenarios (Stable, Seasonal, Shock):

### A. Feasibility Constraints Set $\mathcal{F}$
A configuration $\theta$ is **feasible** ($\theta \in \mathcal{F}$) if and only if it simultaneously satisfies all three conditions on tuning seeds:

1. **Retailer Fill Rate Constraint**:
   $$\overline{FR}(\theta) = \frac{1}{|\mathcal{S}_{\text{tune}}|} \sum_{s \in \mathcal{S}_{\text{tune}}} \text{FillRate}_{\text{retailer}}(s; \theta) \ge 99.0\%$$
   *(Measures the share of total requested demand units actually shipped).*

2. **Retailer Step-Level Service Level Constraint**:
   $$\overline{SL}(\theta) = \frac{1}{|\mathcal{S}_{\text{tune}}|} \sum_{s \in \mathcal{S}_{\text{tune}}} \text{ServiceLevel}_{\text{retailer}}(s; \theta) \ge 85.0\%$$
   *(Measures the share of simulation time steps with zero unfulfilled backlog).*

3. **Bullwhip Dampening Ceiling (75% Rule)**:
   $$\overline{BW}(\theta) = \frac{1}{|\mathcal{S}_{\text{tune}}|} \sum_{s \in \mathcal{S}_{\text{tune}}} \frac{1}{4}\sum_{e \in \text{Echelons}} BW_e(s; \theta) \le \text{Threshold}_{BW}$$
   where:
   $$\text{Threshold}_{BW} = 0.75 \times \overline{BW}_{\text{EOQ\_untuned}}$$
   evaluated on the same tuning seeds $\{100 \dots 104\}$.

### B. Constrained Cost Definition
$$\text{Cost}_{\text{constrained}}(\theta) = \begin{cases} 
\overline{\text{TotalCost}}(\theta), & \text{if } \theta \in \mathcal{F} \\
\text{NaN (disqualified / n/a)}, & \text{if } \theta \notin \mathcal{F}
\end{cases}$$

### C. Optimization Decision Rule
The selected tuned parameter vector $\theta^*$ is:
$$\theta^* = \arg\min_{\theta \in \mathcal{F}} \overline{\text{TotalCost}}(\theta)$$

If no configuration satisfies $\mathcal{F}$ ($\mathcal{F} = \emptyset$), the algorithm constructs the empirical Pareto frontier between $(\overline{\text{TotalCost}}, \overline{BW})$ subject to $FR \ge 99\%$ and $SL \ge 85\%$, selecting the minimum-cost frontier compromise.

---

## 3. The Acceptance Test

Before committing tuned parameters or running test evaluation on seeds $0 \dots 7$, the benchmark must pass the **Acceptance Test Gate**:

> **Acceptance Criterion**:
> System 0 (Constant Baseline ordering training-period mean demand) **must lose** to the best tuned baseline on **constrained cost** in both the **Seasonal** and **Shock** scenarios on tuning seeds $\{100, 101, 102, 103, 104\}$.

### Test Protocol:
1. System 0 is evaluated on tuning seeds $\{100, 101, 102, 103, 104\}$ across Stable, Seasonal, and Shock.
2. In Seasonal and Shock:
   - System 0 orders a constant $D_0 \approx 100$ every step.
   - Customer demand fluctuates with cyclical peaks (Seasonal) and sustained surge (Shock).
   - System 0 fails the step service level ($\overline{SL} < 85\%$) and/or fill rate ($\overline{FR} < 99\%$), resulting in $\text{Cost}_{\text{constrained}} = \text{NaN}$.
   - The tuned baselines satisfy $\mathcal{F}$ with finite constrained cost.
3. **Decision Gate**:
   - **PASS**: System 0 loses on constrained cost in Seasonal and Shock. Proceed to freeze parameters and evaluate test seeds $0 \dots 7$.
   - **FAIL**: If System 0 still achieves lower constrained cost in Seasonal or Shock, the test halts immediately. Report scenario difficulty statistics (CV, peak/median ratio, backlog step count) and stop for user review.

---

## 4. Interior Optimum Rule for Search Grids

To ensure that hyperparameter tuning does not settle on boundary artifacts:

1. **Boundary Prohibition**: If the minimum-cost feasible configuration $\theta^*$ lies on the boundary (i.e. at $\min(\Theta)$ or $\max(\Theta)$ for any search dimension):
   $$\theta^* \in \partial \Theta$$
   the configuration is considered **provisional**.
2. **Mandatory Grid Extension**: The grid for that parameter must be extended outward in the direction of the boundary until the optimum is strictly **interior**:
   $$\theta_{\min} < \theta^* < \theta_{\max}$$
3. This applies to:
   - **EOQ**: $Q_{\text{mult}}$ and $ROP_{\text{factor}}$
   - **(s, S)**: $s_{\text{factor}}$ and $S_{\text{factor}}$
   - **System 2b**: $\alpha, \beta, z, \delta, \text{cap\_mult}$

---

## 5. Test Evaluation Protocol (Seeds 0 to 7)

Once the Acceptance Test passes and parameters are frozen:
1. Frozen parameters are committed to `configs/sim.yaml`.
2. All policies are evaluated **once** across test seeds $\{0, 1, 2, 3, 4, 5, 6, 7\}$ (completely disjoint from tuning seeds $\{100 \dots 104\}$).
3. The evaluation reports:
   - Per-scenario summary tables (Mean $\pm$ Std) for Total Cost, Holding Cost, Backlog Cost, Bullwhip Ratios, Order & Demand Variances, Fill Rate (%), Step Service Level (%), and Constrained Cost ($).
   - Undefined ratios ($0/0$) rendered as `"n/a"` and excluded from numerical averages.
   - Cost vs Bullwhip scatter plots per scenario marking baseline points, sweep configurations, and Pareto frontiers.
