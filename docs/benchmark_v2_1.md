# Benchmark v2.1 Protocol and Evaluation Specification

## Overview & Scope
This document specifies the pre-registered evaluation, hyperparameter tuning protocol, cost models, scenario mechanics, and acceptance criteria for **Benchmark v2.1** across all supply chain ordering policies.

This protocol supersedes Benchmark v1 and Benchmark v2 and is formally established **prior to running any baseline tuning sweeps or test-seed evaluations**.

---

## 1. Initial Inventory Buffering: Base-Stock Calibration ($z = 2.0$)

### Theoretical Motivation
In multi-echelon supply chain logistics (Sterman 1989; Silver, Pyke & Peterson 1998), initialising echelons with arbitrary or deficit inventory (such as legacy $I_0 = 200.0$ units) induces artificial startup transients:
- Under DataCo empirical shipping logistics, mean transit lead time is $L \approx 3.52$ days ($\sigma_L \approx 1.62$ days).
- With scaled training demand mean $\mu_D = 100.0$ units/day, the expected lead-time demand is $L \cdot \mu_D \approx 352$ units.
- Starting with 200 units creates an immediate structural deficit of $\approx 152$ units before step 1 replenishment shipments can arrive, generating unavoidable pipeline starvation regardless of ordering policy intelligence.

### Theoretical Formulation
To ensure a physically grounded, fair, and stable initialisation, every echelon in the 4-echelon serial supply chain is initialized at the theoretical base-stock order-up-to target $S$ calculated strictly from training-period statistics ($t \in [0, 467]$) with zero test lookahead:

$$S = L \cdot \mu_D + z \cdot \sigma_{\text{LTD}}$$

Where the standard deviation of lead-time demand $\sigma_{\text{LTD}}$ accounts for both demand variance and stochastic lead-time variance:

$$\sigma_{\text{LTD}} = \sqrt{L \cdot \sigma_D^2 + \mu_D^2 \cdot \sigma_L^2}$$

### Empirical Derivation
From the DataCo training split:
- Scaled training demand mean: $\mu_D = 100.00$ units/day
- Training demand standard deviation: $\sigma_D = 20.21$ units/day
- Mean transit lead time: $L = 3.52$ days
- Lead time standard deviation: $\sigma_L = 1.62$ days
- Lead-time demand variance:
  $$\text{Var}_{\text{LTD}} = 3.52 \cdot (20.21)^2 + (100.00)^2 \cdot (1.62)^2 = 1,437.7 + 26,244.0 = 27,681.7$$
- Standard deviation of lead-time demand:
  $$\sigma_{\text{LTD}} = \sqrt{27,681.7} \approx 166.38 \text{ units}$$
- Safety stock at $z = 2.0$:
  $$SS = 2.0 \times 166.38 = 332.76 \text{ units}$$
- Base-stock order-up-to target:
  $$S = (3.52 \times 100.00) + 332.76 = 352.00 + 332.76 = 684.76 \approx \mathbf{687.0 \text{ units}}$$

### Specification Rule
- **Primary initial inventory**: $I_0 = \mathbf{687.0 \text{ units}}$ per echelon, identical for every system.
- Initial pipeline shipments are initialized at steady state ($Q = 100.0$ units in transit) to maintain conservation of flow.

---

## 2. Cost Model & Asymmetry Ratios

The cost parameters represent physical inventory holding and stockout backlogging penalties:

$$\text{Cost}_{\text{step}}(e) = h \cdot I_e + b \cdot B_e$$

where $I_e$ is on-hand inventory (never negative) and $B_e$ is unfulfilled backlog (never negative).

| Cost Configuration | Holding Cost ($h$) | Backlog Cost ($b$) | Backlog/Holding Ratio ($b/h$) | Description |
|:---|:---:|:---:|:---:|:---|
| **Primary Baseline** | **$0.50** / unit-day | **$2.50** / unit-day | **5.0x** | Primary Benchmark v2.1 cost regime |
| **Sensitivity Low** | **$0.50** / unit-day | **$1.00** / unit-day | **2.0x** | Low stockout penalty regime |
| **Sensitivity High** | **$0.50** / unit-day | **$5.00** / unit-day | **10.0x** | High stockout penalty regime |

### Rationale
In modern supply chain management, customer stockouts incur severe compounding costs (lost customer goodwill, expediting freight, cancellation fees, lost gross margin). An asymmetry ratio $b/h = 5.0\times$ properly balances stockout avoidance against working capital holding expense. Sensitivities at $b/h = 2.0\times$ and $b/h = 10.0\times$ test the robustness of policy ordering behavior under varying penalties.

---

## 3. Scenario Generation v2 Protocol

### 3.1 Global Training Scaling
- To prevent artificial normalization leaks, a **single global scale factor** is derived strictly from the training period ($t \in [0, 467]$) such that the training mean equals $\mu_D = 100.0$ units/day:
  $$\text{Scale Factor} = \frac{100.0}{\mu_{D, \text{train\_raw}}}$$
- This identical scale factor scales all scenario windows and training histories without per-window renormalisation.

### 3.2 Scenario Regimes
All scenarios evaluate across 40 simulation steps (with the first 5 steps treated as warm-up for metrics):
1. **Stable Scenario**:
   - Selected from the historical DataCo window exhibiting the lowest coefficient of variation ($CV$).
   - Reconstructed via real window trend + seasonal component plus circular block-bootstrapped residuals across seeds.
2. **Seasonal Scenario**:
   - Selected from the DataCo window exhibiting the highest autocorrelation at dominant weekly period ($p = 7$).
   - Retains fixed weekly seasonal pattern and trend from the real window, plus block-bootstrapped residuals across seeds.
3. **Shock Scenario**:
   - Built from real window plus an injected level shift:
     - Multiplier: $1.5\times$ ($+50\%$ demand surge).
     - Start Step: step 15.
     - Duration: 10 steps (steps 15 to 25).
     - Provenance label: `dataco_window_plus_injected_level_shift_v2`.
   - After step 25, demand returns to baseline levels.

### 3.3 System 0 Contingency Rule
System 0 is a zero-state baseline ordering constant mean demand ($D_0 = 100.0$ units) every step regardless of inventory or backlog.
- If System 0 achieves lower constrained cost than calibrated baselines on tuning seeds $\{100 \dots 104\}$:
  - **Contingency Trigger**: Make the shock level shift persistent:
    - Injected shift multiplier: $1.5\times$.
    - Start Step: step 15.
    - Duration: 25 steps (persisting through the end of the 40-step horizon, steps 15 to 40).
  - Rerun the tuning evaluation once under the persistent shift contingency.

---

## 4. Selection Rule for Tuned Policies

For each candidate policy configuration $\theta$ evaluated over tuning seeds $\mathcal{S}_{\text{tune}} = \{100, 101, 102, 103, 104\}$ across all three scenarios:

### A. Feasibility Constraint Set $\mathcal{F}$
A candidate configuration $\theta$ is feasible ($\theta \in \mathcal{F}$) if and only if it satisfies both conditions:
1. **Retailer Unit Fill Rate Constraint**:
   $$\overline{FR}(\theta) = \frac{1}{|\mathcal{S}_{\text{tune}}|} \sum_{s \in \mathcal{S}_{\text{tune}}} \text{FillRate}_{\text{retailer}}(s; \theta) \ge 99.0\%$$
   *(Proportion of cumulative requested demand units fulfilled).*
2. **End-to-End Bullwhip Dampening Ceiling (75% Rule)**:
   $$\overline{BW}_{\text{e2e}}(\theta) = \frac{1}{|\mathcal{S}_{\text{tune}}|} \sum_{s \in \mathcal{S}_{\text{tune}}} BW_{\text{e2e}}(s; \theta) \le \text{Threshold}_{BW}$$
   where:
   $$\text{Threshold}_{BW} = 0.75 \times \overline{BW}_{\text{e2e, EOQ\_untuned}}$$
   evaluated on tuning seeds $\{100 \dots 104\}$.

### B. Reporting vs Constraint
- **Step Service Level ($\overline{SL}$)**: The proportion of steps with zero backlog is **reported** in all summary tables, but is **not constrained** in the formal selection rule.

### C. Interior Optimum Rule for Search Grids
To eliminate boundary artifacts:
- If the lowest-cost feasible configuration $\theta^*$ lies on the edge of the search grid ($\theta^* \in \partial \Theta$), the grid must be expanded outward in that direction until the optimum is strictly **interior**:
  $$\theta_{\min} < \theta^* < \theta_{\max}$$
- Applicable parameters:
  - **EOQ**: $Q_{\text{mult}} \in [0.25, 2.00]$, $ROP_{\text{factor}} \in [0.80, 1.40]$
  - **(s, S)**: $s_{\text{factor}} \in [0.80, 1.40]$, $S_{\text{factor}} \in [0.25, 2.00]$
  - **System 2b**: $\alpha, \beta \in [0.1, 0.9]$, $z \in [1.0, 3.0]$, $\delta \in [0.2, 0.8]$, $\text{cap} \in [1.2, 3.0]$

### D. Objective Decision Rule
$$\theta^* = \arg\min_{\theta \in \mathcal{F}} \overline{\text{TotalCost}}(\theta)$$

---

## 5. Acceptance Test Protocol

Before committing tuned parameters or running test-seed evaluations ($0 \dots 7$), the benchmark must pass the **Acceptance Gate**:

> **Acceptance Criterion**:
> System 0 (ordering constant training-period mean demand $D_0 = 100.0$) **must lose** to the best tuned baseline on **constrained cost** in both the **Seasonal** and **Shock** scenarios on tuning seeds $\{100, 101, 102, 103, 104\}$.

### Systems Under Test & Reference Rows
The acceptance evaluation includes:
1. **Reference Rows**:
   - **System 0: Constant** ($D_0 = 100.0$ units)
   - **Untuned EOQ** ($Q_{\text{mult}} = 1.0, ROP_{\text{factor}} = 1.0$)
   - **Untuned (s, S)** ($s_{\text{factor}} = 1.0, S_{\text{factor}} = 1.0$)
   - **Base-Stock ($z = 2.0$)** ($S = 687.0$ units)
   - **Oracle (Clairvoyant 4-day lookahead)** (Labelled: **Reference Only**)
2. **Tuned Policies**:
   - **Tuned EOQ**
   - **Tuned (s, S)**
   - **Tuned System 2b (Smoothed OUT)**

### Status Classification:
- **PASS**: Meets all feasibility constraints ($\overline{FR} \ge 99.0\%$, $\overline{BW}_{\text{e2e}} \le 0.75 \times \overline{BW}_{\text{e2e, EOQ\_untuned}}$).
- **DISQUALIFIED (Reason)**: Fails fill rate or bullwhip ceiling constraint ($\text{Cost}_{\text{constrained}} = \text{NaN}$).
- **REFERENCE ONLY**: Reference benchmark (Oracle).
