# Baseline Tuning Protocol & Selection Rule (System 1 & System 2)

## Objective and Scope
This document formalizes the pre-registered parameter tuning protocol and decision rules for:
1. **System 1: (s, S) Min-Max Policy**
2. **System 2: Economic Order Quantity (EOQ / ROP) Policy**

This protocol mirrors the selection rule established for System 2b in [docs/system2b_selection_rule.md](file:///c:/Users/saran/smartsupply/docs/system2b_selection_rule.md).

---

## 1. Untuned Baseline Reference & Threshold Definition
1. **Untuned System 2 (EOQ)** is evaluated across tuning seeds $\{100, 101, 102, 103, 104\}$ on all three scenarios (Stable, Seasonal, Shock).
2. The grand average mean bullwhip ratio across all four echelons is established:
   $$\overline{BW}_{\text{EOQ\_untuned}} = \frac{1}{15} \sum_{r=1}^{15} \frac{1}{4} \left( BW_{\text{retailer}}^{(r)} + BW_{\text{wholesaler}}^{(r)} + BW_{\text{distributor}}^{(r)} + BW_{\text{manufacturer}}^{(r)} \right)$$
3. **Target Bullwhip Ceiling (75% Rule)**:
   $$\text{Threshold}_{BW} = 0.75 \times \overline{BW}_{\text{EOQ\_untuned}}$$

A tuned configuration must dampen mean network bullwhip by at least 25% relative to the untuned EOQ baseline to be eligible for selection.

---

## 2. Search Grids on Tuning Seeds (100 to 104)

### A. System 2 (EOQ) Tuning Grid
- **Order Quantity Multiplier ($Q_{\text{mult}}$)**: Scale factor applied to Wilson's batch size $EOQ = \sqrt{2 \mu_D K / h}$.
  $$Q_{\text{tuned}} = Q_{\text{mult}} \times EOQ$$
  Grid: $Q_{\text{mult}} \in \{0.50, 0.75, 1.00, 1.25, 1.50, 2.00\}$
- **Reorder Point Factor ($ROP_{\text{factor}}$)**: Scale factor applied to $ROP = \mu_L \mu_D + z \sigma_{\text{LTD}}$.
  $$ROP_{\text{tuned}} = ROP_{\text{factor}} \times ROP$$
  Grid: $ROP_{\text{factor}} \in \{0.80, 0.90, 1.00, 1.10, 1.20\}$
- **Total Configurations**: $6 \times 5 = 30$ configurations ($450$ tuning runs).

### B. System 1 (s, S) Tuning Grid
- **Reorder Point Factor ($s_{\text{factor}}$)**: Scale factor applied to lower threshold level $s$.
  $$s_{\text{tuned}} = s_{\text{factor}} \times s_{\text{base}}$$
  Grid: $s_{\text{factor}} \in \{0.80, 0.90, 1.00, 1.10, 1.20\}$
- **Order-Up-To Spread Factor ($S_{\text{factor}}$)**: Scale factor applied to the target buffer increment $S - s = S_{\text{factor}} \times EOQ$.
  $$S_{\text{tuned}} = s_{\text{tuned}} + S_{\text{factor}} \times EOQ$$
  Grid: $S_{\text{factor}} \in \{0.50, 0.75, 1.00, 1.25, 1.50, 2.00\}$
- **Total Configurations**: $5 \times 6 = 30$ configurations ($450$ tuning runs).

---

## 3. Mathematical Selection Rule
For each system $k \in \{\text{EOQ}, (s, S)\}$:
1. Identify the qualifying set $\mathcal{Q}_k$:
   $$\mathcal{Q}_k = \left\{ \theta \;\middle|\; \mu(\overline{BW}_k(\theta)) \le \text{Threshold}_{BW} \right\}$$
2. If $\mathcal{Q}_k \ne \emptyset$, select the configuration that minimizes grand average total supply chain cost:
   $$\theta_k^* = \arg\min_{\theta \in \mathcal{Q}_k} \mu(\text{TotalCost}_k(\theta))$$
3. If $\mathcal{Q}_k = \emptyset$, identify the non-dominated Pareto frontier between Total Cost and Mean Bullwhip on the tuning seeds and select the frontier compromise point with lowest cost.

---

## 4. Parameter Freezing & Reporting
1. The chosen parameters $\theta_{\text{EOQ}}^*$ and $\theta_{s\_S}^*$ are written to `configs/sim.yaml`.
2. Evaluated **once** on test seeds $\{0, 1, 2, 3, 4, 5, 6, 7\}$.
3. The untuned baselines are retained as explicitly labelled reference rows:
   - `System 1: (s, S)` (Untuned baseline)
   - `System 1-tuned: (s, S) Tuned` (Selected parameters)
   - `System 2: EOQ` (Untuned baseline)
   - `System 2-tuned: EOQ Tuned` (Selected parameters)
   - `System 2b: Smoothed OUT` (Tuned parameters)
   - `System 0: Constant` (Zero-state constant benchmark)
