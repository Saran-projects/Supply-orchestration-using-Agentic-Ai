# Empirical Study Protocol: LLM Multi-Agent Supply Chain Orchestration (v1.0)

**Document Status**: Pre-Registered Scientific Protocol  
**Date**: October 2026  
**Git Tag**: `llm-protocol-v1`  
**Applies To**: System 3 (`single_agent`), System 4 (`multi_agent_info`), System 5 (`multi_agent_tools`), System 6 (`orchestrated`) vs. Tuned Classical Baselines  
**Enforced Budget Hard Cap**: **$3.60 USD** ($4.00 account credit minus $0.40 reserve)

---

## 1. Executive Summary & Research Questions

This document formalizes the pre-registered scientific protocol governing all empirical evaluations of Large Language Model (LLM) ordering policies in the 4-echelon serial supply chain simulation (Benchmark v2.1).

### Core Research Questions

- **(RQ1) Orchestration vs. Monolithic Agent**:  
  Does multi-agent orchestration with role division and negotiation (System 6) reduce end-to-end bullwhip and total supply chain cost compared to an uncoordinated single LLM agent per echelon (System 3)?
- **(RQ2) LLM Orchestration vs. Tuned Classical & Heuristic Baselines**:  
  How does System 6 compare against mathematically tuned classical inventory policies (Tuned EOQ) and the deterministic regime-adaptive heuristic (System 2c)?
- **(RQ3) Value of Architecture Components (Ablations)**:  
  What do the individual architectural mechanisms contribute to performance?
  - Information sharing across echelons without tools (System 4 vs. System 3)
  - Deterministic calculation tools without negotiation (System 5 vs. System 4)
  - Negotiation rounds and supervisor arbitration (System 6 vs. System 5)
- **(RQ4) Behavioral Reliability, Groundedness, and Operational Efficiency**:  
  How reliable, robust, and token-efficient are autonomous agent systems?
  - Do systems achieve zero fallbacks and $\ge 99\%$ structured JSON output validity?
  - What is the empirical token consumption, monetary cost in USD, and decision latency per step?

---

## 2. Primary Confirmatory Comparisons

To eliminate post-hoc hypothesis generation and multiple testing inflation ($p$-hacking), we strictly delineate **Primary Confirmatory Comparisons** from **Exploratory Analyses**.

### Confirmatory Primary Comparisons (Only)
All confirmatory evaluations are conducted separately within each scenario (**Stable**, **Seasonal**, **Shock**) on two primary target metrics:
1. **Total System Cost ($)** (sum of holding and backlog costs across all 4 echelons over 40 steps, $h = \$0.50, b = \$2.50$)
2. **End-to-End Bullwhip Ratio** ($\frac{\text{Var}(\text{Manufacturer Orders})}{\text{Var}(\text{Customer Demand})}$)
3. **Constraint**: Feasibility requires Retailer Unit Fill Rate $\ge 99.0\%$.

The **three pre-registered primary comparisons** are:
1. **System 6 (`orchestrated`) vs. System 3 (`single_agent`)**:  
   *Hypothesis $H_1$*: Multi-agent coordination with tools and negotiation suppresses order amplification and lowers total holding/backlog cost compared to independent uncoordinated echelon prompts.
2. **System 6 (`orchestrated`) vs. Tuned EOQ (`baseline_eoq` tuned)**:  
   *Hypothesis $H_2$*: Orchestrated agents achieve comparable supply chain cost ($\le 1.15\times$ Tuned EOQ) while dampening end-to-end bullwhip to at or below Tuned EOQ levels.
3. **System 6 (`orchestrated`) vs. Deterministic System 2c (`baseline_2c` tuned)**:  
   *Hypothesis $H_3$*: Semantic reasoning combined with deterministic tools matches or outperforms the rule-based regime-adaptive smoothed order-up-to policy in non-stationary and shock regimes.

### Exploratory Analyses (Formally Labeled)
All other comparisons are formally labeled as **exploratory**:
- S4 vs. S3 (Value of shared echelon visibility)
- S5 vs. S4 (Value of pure calculation tool outputs)
- S6 vs. S5 (Value of conflict detection and supervisor negotiation rounds)
- S6 vs. System 0 (Constant order baseline)
- S6 vs. Oracle (Clairvoyant 4-day lookahead reference)
- Cross-scenario aggregations and token spend efficiency ratios

---

## 3. Pre-Registered Success Criteria

An LLM policy configuration (evaluated on tuning seeds 100–102, or on test seeds) is judged successful if and only if it satisfies all of the following criteria simultaneously:

1. **Zero Fallbacks**: Exactly **0 fallbacks** ($0.0\%$ fallback rate; no unhandled exceptions falling back to previous orders).
2. **High Structured-Output Validity**: $\ge \mathbf{99.0\%}$ first-pass valid JSON schema compliance.
3. **Bullwhip Dampening**: End-to-end bullwhip ratio $\le$ Tuned EOQ's end-to-end bullwhip in the corresponding scenario.
4. **Cost Competitiveness**: Total supply chain cost $\le \mathbf{1.15\times}$ Tuned EOQ's cost in the corresponding scenario.
5. **Customer Service Feasibility**: Retailer unit fill rate $\ge \mathbf{99.0\%}$.

---

## 4. Statistical Methodology

### Paired Experimental Design
Because customer demand and stochastic lead times are deterministic functions of random seeds, all system comparisons are strictly **paired by seed** ($s \in \mathcal{S}$):
$$D_s = X_{S6, s} - X_{\text{Baseline}, s}$$

### Hypothesis Testing & Sample Size Rules
1. **Sample Size Threshold ($n \ge 6$)**:
   - Two-sided **Wilcoxon signed-rank test** with **Holm-Bonferroni correction** across the 3 primary comparisons is applied **only if $n \ge 6$** per scenario.
2. **Small-Sample Protocol ($n < 6$)**:
   - If $n < 6$ (such as under Plan B where $n = 4$ test seeds 0–3):
     - **No formal $p$-values or statistical significance claims are asserted**.
     - Analysis reports descriptive per-seed differences ($D_s$), mean paired difference ($\overline{D}$), and median paired difference.
3. **Bootstrap Confidence Intervals**:
   - Non-parametric **95% Bootstrap Confidence Intervals** generated with $B = 2,000$ paired resamples for both cost and bullwhip differences.
4. **Exploratory Labeling**:
   - All ablation contrasts (S4 vs S3, S5 vs S4, S6 vs S5) and secondary metrics are explicitly designated as exploratory.

---

## 5. Hyperparameter & Prompt Tuning Protocol

Prompt engineering and system tuning must strictly follow these rules:

1. **Iteration Budget**: At most **3 paid tuning iterations** total.
2. **Single-Variable Rule**: Exactly **ONE logged change per iteration** (e.g. tightening word limit, modifying negotiation round threshold, or adjusting tool recommendation framing). Every change must be recorded in `docs/prompt_changelog.md` with rationale and result before proceeding.
3. **Strict Seed Quarantining**:
   - Tuning occurs exclusively on tuning seeds $\mathcal{S}_{\text{tune}} = \{100, 101, 102\}$.
   - Test seeds $\{0, 1, 2, 3, 4, 5, 6, 7\}$ are strictly firewalled and never evaluated during prompt tuning.
4. **Dry-Run Safety Pre-Check**:
   - Before launching any paid tuning iteration, a dry-run projection must verify:
     - Projected single iteration cost $\le \$0.35$ USD.
     - Cumulative tuning spend $\le \$1.50$ USD.
     - Total cumulative spend plus projection $\le \$3.60$ USD hard cap.
   - If any limit is breached, the run halts immediately.

---

## 6. Execution Branching: Plan A vs. Plan B

Execution scope on test seeds is governed by the empirical cost of evaluating one single S3 + S6 pair (40 steps):

$$\text{Pair Cost} = \text{Cost}_{S3}(40 \text{ steps}) + \text{Cost}_{S6}(40 \text{ steps})$$

### Branching Rule

```
                                 [ Pair Cost Check ]
                                          |
                   +----------------------+----------------------+
                   |                                             |
          Pair Cost <= $0.15 USD                        Pair Cost > $0.15 USD
                   |                                             |
             [ PLAN A ]                                    [ PLAN B ]
   - S3 & S6: 3 scenarios x seeds 0-7            - S3 & S6: Shock & Seasonal x seeds 0-3
     (2 x 3 x 8 = 48 runs, n = 8)                  (2 x 2 x 4 = 16 runs, n = 4)
   - S4 & S5: Shock x seeds 0-2                  - S4 & S5: Shock x seed 0
     (2 x 3 = 6 runs)                              (2 x 1 = 2 runs)
   - Wilcoxon + Holm tests (n = 8 >= 6)          - Descriptive per-seed differences &
                                                   Bootstrap CIs (n = 4 < 6)
```

### Plan Determination based on Empirical Rate Card
- Under the confirmed Isoquant GLM rate card ($0.027 in, $0.018 cached in, $0.490 out per 1M tokens) with Prompt v3.1:
  - Empirical S3 run cost: **$0.0664 USD**
  - Empirical S6 run cost: **$0.2377 USD**
  - **Empirical S3 + S6 Pair Cost**: **$0.3041 USD**
- Since **$0.3041 USD > $0.15 USD**, **PLAN B is formally triggered**.
- **Plan B Financial Audit**:
  - S3 + S6: 8 pairs $\times \$0.3041 = \$2.433$ USD
  - S4 + S5: 1 run each $\approx \$0.193$ USD
  - Total Plan B test cost: $\approx \mathbf{\$2.626 \text{ USD}}$
  - Cumulative project spend: $\$0.8067 \text{ (current)} + \$2.626 \approx \mathbf{\$3.433 \text{ USD}} \le \mathbf{\$3.600 \text{ USD Hard Cap}}$.
  - Fully honors the **$3.60 USD Hard Cap** while preserving the **$0.40 USD reserve**.

---

## 7. Baseline Reference Target Table (Frozen Benchmark v2.1)

All primary comparisons evaluate against the frozen benchmark distributions on test seeds ($0 \dots 7$, $n = 8$, $h = \$0.50, b = \$2.50$, $I_0 = 687.0$):

| Scenario | Metric | Tuned EOQ ($Q=0.25, ROP=1.2$) | Deterministic 2c ($\beta_{\text{shk}}=1.0, \delta_{\text{shk}}=0.8$) | Reference Oracle | Primary Target Thresholds for System 6 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Stable** | Total Cost ($) | $20,670.7 \pm 2,041.5$ | $21,170.2 \pm 1,898.5$ | $37,007.7 \pm 3,721.5$ | Cost $\le \$23,771.3$ ($1.15\times$ EOQ) |
| | E2E Bullwhip | $10.40 \pm 2.29$ | $14.19 \pm 3.51$ | $0.95 \pm 0.19$ | Bullwhip $\le 10.40$ |
| | Fill Rate (%) | $100.00\% \pm 0.00\%$ | $100.00\% \pm 0.00\%$ | $100.00\% \pm 0.00\%$ | Fill Rate $\ge 99.0\%$ |
| **Seasonal** | Total Cost ($) | $19,863.7 \pm 1,932.3$ | $21,842.2 \pm 1,629.4$ | $36,530.3 \pm 2,988.7$ | Cost $\le \$22,843.3$ ($1.15\times$ EOQ) |
| | E2E Bullwhip | $6.98 \pm 1.94$ | $8.40 \pm 2.49$ | $0.93 \pm 0.13$ | Bullwhip $\le 6.98$ |
| | Fill Rate (%) | $100.00\% \pm 0.00\%$ | $100.00\% \pm 0.00\%$ | $100.00\% \pm 0.00\%$ | Fill Rate $\ge 99.0\%$ |
| **Shock** | Total Cost ($) | $29,214.8 \pm 8,114.2$ | $35,642.8 \pm 12,476.5$ | $36,470.8 \pm 5,597.6$ | Cost $\le \$33,597.0$ ($1.15\times$ EOQ) |
| | E2E Bullwhip | $2.68 \pm 0.81$ | $3.83 \pm 0.56$ | $0.89 \pm 0.16$ | Bullwhip $\le 2.68$ |
| | Fill Rate (%) | $97.23\% \pm 5.51\%$ | $97.36\% \pm 5.03\%$ | $100.00\% \pm 0.00\%$ | Fill Rate $\ge 99.0\%$ |

---

## 8. Failure Handling & Mandatory Reporting

1. **Strict Fallback Accounting**:
   - Any invalid order (negative, non-numeric, or malformed JSON) triggers previous-period fallback $O_t(e) = O_{t-1}(e)$.
   - Every fallback increments `agent_calls.fallback = TRUE` and is reported in `behavioural_metrics.fallback_rate`.
   - The success criterion demands **0 fallbacks**.
2. **Unconditional Publication**:
   - All runs, whether conforming or null/negative results, are logged in `runs_registry` and DuckDB telemetry.
   - No selective reporting or seed omission is permitted.

---

## Amendment A1 (dated 2026-10-14): design v5 is the third and last paid tuning iteration

1. **Agent design v5** (`docs/agent_design_v5.md`, prompt version `v5.0`) replaces free-form order quantities with a
   fixed action space. The LLM picks follow / boost_small / boost_large / trim, and code computes the order inside the shared safe range.
   Design v5 counts as the **third and last paid tuning iteration** under section 5. The two earlier paid
   iterations were v4.0 and v4.1 (`docs/prompt_changelog.md`).
2. **Bug fixes do not count as tuning iterations.** A bug fix corrects code so it does what the documented design says,
   without changing prompts, design or tuned parameters (e.g. the fixes listed in `docs/diagnosis_phase5.md`).
   Each fix is listed in `docs/prompt_changelog.md` with its date.
3. **Any later design change needs a new dated amendment** to this protocol before it is run. A design change covers
   prompts, action space, references, band, multipliers, information or tools available to a system, or orchestration.
   Such an amendment must state why the change is made, and whether results obtained before it remain comparable.
4. The v5 action multipliers were tuned without any LLM on seeds 100–104 under `docs/tuning_rule_v2_1.md`, and frozen in
   `configs/agent_v5.yaml` (follow 1.0, boost_small 1.2, boost_large 1.8, trim 0.9). No candidate met fill rate ≥ 99%,
   so the rule's caveat fallback applied, with trim constrained to ≤ 0.95 (user decision, 2026-10-14). They are not
   re-tuned on paid runs.

---

## Amendment A2 (dated 2026-10-14): design v5.1 (LLM tool selection and S6 negotiation) and a model comparison

Written before any v5.1 run. It does not change Amendment A1 or any v5 result.

1. **Why.** In v5 the tools are deterministic and System 6 does not negotiate, so tool-choice accuracy, parameter
   accuracy and negotiation efficiency cannot be measured. The user asked for these metrics for every model.
2. **Design v5.1** (`tool_mode: llm_tools`; Systems 5 and 6 only; Systems 3 and 4 have no tools and stay on v5.0):
   - **Tool request:** before each decision, the model receives the state and a catalogue of five tools
     (`forecast_demand`, `regime_detector`, `smoothed_order_up_to`, `eoq_quantity`, `safe_order_range`).
     It returns which tools to call and their inputs, as JSON. Code runs the requested tools with the model's inputs and
     shows the results in the decision prompt.
   - **Tool-choice accuracy:** a decision counts as correct when the model requests both required tools
     (`regime_detector`, `smoothed_order_up_to`) and no tool outside the relevant set
     (required tools plus `forecast_demand`).
   - **Parameter accuracy:** the share of requested relevant calls whose inputs match the observed state
     (inventory, pipeline, backlog and last order within ±0.5 or 1%; demand history equal to the received demand shown).
   - **Orders are unchanged.** The order is still `clip(reference × multiplier)` with the reference computed by code from
     the true state, so outcomes stay comparable with v5. Tool calls are scored, but they do not move the order.
   - **System 6 negotiation:** after the orchestrator's decision, the three deterministic specialists respond.
     A specialist objects when its nearest action differs from the chosen action and the chosen order is more than 15%
     from its proposal. If any specialist objects, the model decides again with the objections shown, for at most
     2 rounds. Rounds and messages per decision are logged.
3. **Model comparison (exploratory, descriptive only):**
   - Systems 3–6 × {glm-5.3-flash (Isoquant), `deepseek/deepseek-v4.1-flash`, `google/gemini-3.8-flash` (OpenRouter)}
     × stable, seasonal and shock × tuning seed 102.
   - One run per cell: no statistical claims, and no tuning on these results.
   - Multipliers, prompts v5.0/v5.1, reference and band stay frozen.
4. **Safeguards:**
   - Budget hard stop at $2.00 across all models for this task.
   - A dry-run projection must fit before any paid call.
   - Abort on HTTP 401/402/403.
   - A run with fallbacks above 5%, or with LLM calls fewer than decisions, is invalid and stops the batch.
5. **Iterations:** v5.1 adds a measurement layer and negotiation for the requested metrics. It is not a tuning iteration:
   nothing is tuned on its results. Any change after seeing them needs a new dated amendment.
6. **Dated note (2026-10-14, before any comparison run):** the dry run, using output tokens measured on one probe call
   per model, projected $2.94. The user raised the budget hard stop for this task from $2.00 to **$3.50** and kept
   `google/gemini-3.8-flash`. Nothing else changed.
