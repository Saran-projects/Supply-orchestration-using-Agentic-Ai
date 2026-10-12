# System 2b Parameter Selection Rule

## Objective and Protocol
This document formalizes the decision rule for tuning and freezing parameters of **System 2b: Smoothed Order-Up-To** prior to examining sweep outputs, adhering strictly to pre-registered evaluation standards.

---

## 1. Tuning Grid and Evaluation Scope
- **Target Parameters**:
  - Demand forecast smoothing factor: $\alpha \in \{0.2, 0.5, 0.8\}$
  - Order replenishment smoothing factor: $\beta \in \{0.2, 0.5, 0.8\}$
  - Safety stock factor: $z \in \{1.28, 1.65, 2.0\}$ (corresponding to ~90%, 95%, 97.7% target service levels)
  - Rate limiting bound: $\delta \in \{0.05, 0.10, 0.20, 0.50\}$ (fraction of reference demand)
  - Forecast cap multiple: $\text{cap\_multiple} \in \{1.5, 2.0\}$
  - Lead-time variance handling: $\text{include\_lead\_time\_variance} \in \{\text{True}, \text{False}\}$
- **Tuning Seeds**: $\{100, 101, 102, 103, 104\}$ ($N = 5$ seeds).
- **Scenarios**: Stable, Seasonal, Shock ($3$ scenarios).
- **Total Tuning Runs**: Evaluated across all $15$ scenario-seed pairs per configuration.
- **Horizon**: Common $40$ steps, with initial $5$ steps discarded as warm-up ($35$ evaluated steps).
- **Initialization**: Steady-state pipeline initialization.

---

## 2. Benchmark Reference: System 2 (EOQ)
1. System 2 (EOQ) is evaluated across the identical tuning seeds $\{100, 101, 102, 103, 104\}$ across all 3 scenarios ($15$ runs).
2. For each run $r$, the mean bullwhip across the four echelons is calculated:
   $$\overline{BW}_{\text{EOQ}}^{(r)} = \frac{1}{4} \left( BW_{\text{retailer}}^{(r)} + BW_{\text{wholesaler}}^{(r)} + BW_{\text{distributor}}^{(r)} + BW_{\text{manufacturer}}^{(r)} \right)$$
3. The grand benchmark average across all tuning runs is established:
   $$\mu(\overline{BW}_{\text{EOQ}}) = \frac{1}{15} \sum_{r=1}^{15} \overline{BW}_{\text{EOQ}}^{(r)}$$

---

## 3. Bullwhip Reduction Constraint (75% Threshold)
A candidate parameter configuration $\theta = (\alpha, \beta, z, \delta, \dots)$ is **qualifying** if and only if its grand mean bullwhip across echelons over all tuning seeds and scenarios satisfies:
$$\mu(\overline{BW}_{S2b}(\theta)) \le 0.75 \times \mu(\overline{BW}_{\text{EOQ}})$$

That is, System 2b must achieve at least a **25% bullwhip reduction** compared to EOQ across the entire supply chain network.

---

## 4. Cost Minimization Selection
Among the subset of qualifying configurations $\mathcal{Q}$:
$$\mathcal{Q} = \left\{ \theta \;\middle|\; \mu(\overline{BW}_{S2b}(\theta)) \le 0.75 \times \mu(\overline{BW}_{\text{EOQ}}) \right\}$$

The optimal configuration $\theta^*$ is selected as the one that minimizes the grand average total supply chain cost:
$$\theta^* = \arg\min_{\theta \in \mathcal{Q}} \mu(\text{TotalCost}_{S2b}(\theta))$$

---

## 5. Contingency / Fallback Rule
If no parameter combination qualifies ($\mathcal{Q} = \emptyset$):
1. Compute the empirical **Pareto frontier** between Mean Bullwhip and Total Cost on the tuning seeds.
2. Report the Pareto frontier configurations.
3. Halt execution and report to the user without selecting an arbitrary parameter set.

---

## 6. Post-Selection Freezing
Once $\theta^*$ is identified according to Rules 3 and 4:
1. $\theta^*$ will be written and frozen into `configs/sim.yaml`.
2. The frozen configuration will be evaluated **once** on the test seeds $\{0, 1, 2, 3, 4, 5, 6, 7\}$.
3. Parameters will NOT be modified or retuned afterwards.
