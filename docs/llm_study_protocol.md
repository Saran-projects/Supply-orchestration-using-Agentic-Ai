# Empirical Study Protocol: LLM Multi-Agent Supply Chain Orchestration (v1.0)

**Document Status**: Pre-Registered Experimental Protocol  
**Date**: October 2026  
**Git Tag**: `llm-protocol-v1`  
**Applies To**: System 3 (`single_agent`), System 4 (`multi_agent_info`), System 5 (`multi_agent_tools`), System 6 (`orchestrated`) vs Classical Baselines  

---

## 1. Executive Summary & Research Questions

This document establishes the pre-registered scientific protocol governing all empirical evaluations of Large Language Model (LLM) ordering policies in the 4-echelon supply chain simulation (Benchmark v2.1).

### Core Research Questions

- **(RQ1) Orchestration vs. Monolithic Agent**:  
  Do tool-augmented orchestrated agents (System 6) achieve lower total supply chain cost and reduced end-to-end bullwhip compared to an un-orchestrated single agent (System 3)?
- **(RQ2) LLMs vs. Classical Optimization Baselines**:  
  How do autonomous LLM policies compare against mathematically tuned classical baselines (System 2: EOQ-Tuned, System 1: (s, S)-Tuned) and heuristic smoothing (System 2b: Smoothed Order-Up-To)?
- **(RQ3) Component Ablation & Value of Architecture**:  
  What do the individual components of the multi-agent architecture contribute?
  - Information sharing without tools (System 4 vs. System 3)
  - Tool access without orchestration/negotiation (System 5 vs. System 4)
  - Conflict detection and negotiation rounds (System 6 vs. System 5)
- **(RQ4) Behavioral Reliability, Groundedness, and Operational Cost**:  
  How robust, grounded, and economically viable are autonomous agent systems? Specifically:
  - What are their structured-output success and fallback rates?
  - What is the empirical monetary cost in LLM API tokens required to manage the supply chain, and does it justify any performance advantage?

---

## 2. Hypotheses & Primary Comparisons

To prevent post-hoc hypothesis generation and multiple testing inflation ($p$-hacking), we strictly delineate **Primary Confirmatory Comparisons** from **Exploratory Analyses**.

### Confirmatory Primary Comparisons
All confirmatory evaluations are evaluated separately within each of the three pre-registered demand regimes (**Stable**, **Seasonal**, **Shock**) on two target metrics:
1. **Total System Cost ($)** (sum of holding and backlog costs across all 4 echelons over 40 days)
2. **End-to-End Bullwhip Ratio** ($\frac{\text{Var}(\text{Manufacturer Orders})}{\text{Var}(\text{Customer Demand})}$)

The **three pre-registered primary comparisons** are:
1. **System 6 (`orchestrated`) vs. System 3 (`single_agent`)**:  
   *Hypothesis $H_1$*: Multi-agent coordination with tools and negotiation suppresses order amplification and lowers total holding/backlog cost compared to a single prompt managing all tiers.
2. **System 6 (`orchestrated`) vs. System 2b (`baseline_smoothed_order_up_to`)**:  
   *Hypothesis $H_2$*: Semantic reasoning and cross-echelon negotiation match or outperform classical heuristic order smoothing in non-stationary and shock regimes.
3. **System 6 (`orchestrated`) vs. System 2-Tuned (`baseline_eoq` tuned)**:  
   *Hypothesis $H_3$*: Orchestrated agents achieve comparable cost ($\le +15\%$) while adhering to pre-registered bullwhip ceilings ($\le 5.756$).

```
+---------------------------------------------------------------------------------------+
|                              BENCHMARK EVALUATION TAXONOMY                            |
+---------------------------------------------------------------------------------------+
|  PRIMARY CONFIRMATORY (Holm-corrected):                                               |
|    - S6 vs. S3 (Orchestration vs. Monolith)                                           |
|    - S6 vs. S2b (LLM vs. Classical Smoothed OUT)                                      |
|    - S6 vs. S2-Tuned (LLM vs. Tuned EOQ)                                              |
|                                                                                       |
|  EXPLORATORY (Formally Labeled in all reports):                                       |
|    - S4 vs. S3 (Value of distributed echelon agents)                                  |
|    - S5 vs. S4 (Value of pure calculation tools)                                      |
|    - S6 vs. S5 (Value of conflict detection & negotiation rounds)                     |
|    - S6 vs. System 0 (Constant Order)                                                 |
|    - S6 vs. Oracle (Clairvoyant Base-Stock Reference)                                 |
|    - Cross-scenario aggregations, token spend efficiency ratios                       |
+---------------------------------------------------------------------------------------+
```

---

## 3. Pre-Registered Success Criteria for Prompt & System Tuning

Before promoting an agent prompt configuration from the tuning phase to the frozen evaluation grid, the following criteria must be satisfied on **tuning seeds 100 to 104**:

1. **Deterministic Structured-Output Reliability**:
   - **Zero fallbacks** across all evaluated tuning steps ($0.0\%$ fallback rate).
   - $\ge \mathbf{99.0\%}$ first-pass valid JSON structured-output adherence.
2. **Bullwhip Suppression**:
   - System 6 mean bullwhip across echelons $\le \mathbf{75\%}$ of untuned EOQ ($\le \mathbf{5.756}$).
3. **Cost Competitiveness**:
   - System 6 total cost on tuning seeds must fall within **$+15\%$ of Tuned EOQ** in Stable and Seasonal scenarios.
4. **Operational Feasibility Floor**:
   - Retail fill rate $\ge \mathbf{99.0\%}$.
   - Retail step service level $\ge \mathbf{85.0\%}$.
   *(Note: Performance against these feasibility floors must be reported transparently regardless of whether they are met.)*

---

## 4. Statistical Methodology & Significance Thresholds

### Paired Experimental Design
Because external customer demand series and lead-time stochasticity are generated via pseudorandom seed streams, runs are strictly **paired by random seed** ($i \in \{0, \dots, 7\}$):
$$D_i = X_{\text{System 6}, i} - X_{\text{Baseline}, i}$$

### Hypothesis Testing
- **Non-Parametric Test**: Two-sided **Wilcoxon signed-rank test** on paired differences $D_i$ for each scenario.
- **Family-Wise Error Rate Control**: **Holm-Bonferroni step-down correction** applied across the family of 3 primary comparisons for each metric within each scenario:
  $$p_{(k)} \le \frac{\alpha}{m - k + 1},\quad \text{where } \alpha = 0.05,\; m = 3$$
- **Effect Size**: Non-parametric effect size reported via **rank-biserial correlation** ($r_{\text{rb}}$):
  $$r_{\text{rb}} = \frac{W^+ - W^-}{W^+ + W^-}$$
- **Confidence Intervals**: Median paired difference reported alongside non-parametric **95% bootstrap confidence intervals** generated with $B = 10,000$ paired resamples.
- **Small-Sample Guardrail**:
  > **No claims of statistical significance ($p < 0.05$) will be asserted when effective sample size $n < 6$** (e.g., in the presence of tied pairs or zero-difference outcomes). With $n = 8$ test seeds, statistical power is reserved for moderate-to-large effect sizes ($|r_{\text{rb}}| > 0.6$).

---

## 5. Seed Partitioning & Firewall

To eliminate data snooping, prompt overfitting, and test set contamination:

| Dataset / Partition | Seed Range | Purpose | Permitted Operations |
|:---|:---:|:---|:---|
| **Tuning Partition** | `100` to `104` (5 seeds) | Prompt engineering, role instructions, schema iteration, negotiation threshold tuning | Iterative trial-and-error, error logging, parameter sweeps |
| **Test Partition** | `0` to `7` (8 seeds) | Final frozen evaluation grid | **Single-pass evaluation only**. Zero prompt modifications permitted. |

**Firewall Enforcement**:
- The test seed range ($0 \dots 7$) is strictly forbidden from being queried or evaluated during prompt iteration.
- The `results/benchmark_manifest.json` configuration hash must match the active simulation environment before executing test seeds.

---

## 6. Tuning Budget & Prompt Freezing

Prompt engineering and hyperparameter adjustments must adhere to strict iteration budgets:

1. **Primary Systems (S3 & S6)**:
   - At most **4 prompt versions** evaluated per system on tuning seeds.
   - S3 (`single_agent`) and S6 (`orchestrated`) receive equal tuning iteration budgets to prevent unequal prompt optimization bias.
2. **Ablation Baselines (S4 & S5)**:
   - At most **1 prompt iteration** evaluated per system on tuning seeds.
   - S4 (`multi_agent_info`) and S5 (`multi_agent_tools`) inherit the role prompts and schema conventions finalized during S6 tuning to ensure direct ablation comparability.
3. **Prompt Freezing Contract**:
   - Final prompts are saved to `src/sca/agents/prompts/` and tagged with their version identifiers (`single_agent_v2.txt`, `orchestrated_v2.txt`).
   - Prompt files are checked into git prior to running the test grid.
   - **No prompt edits, system prompt additions, or temperature tweaks are allowed once execution on test seeds has started.**

---

## 7. Mandatory Reporting & Publication Integrity Rule

1. **Unconditional Reporting**:
   - All evaluated systems, scenarios, and metrics must be published in primary summary tables.
   - Null results (no statistically significant difference between S6 and S3) and negative results (S6 cost exceeding classical baselines, or S6 failing bullwhip constraints) **must be explicitly reported and discussed**.
2. **No Selective Scenario Suppression**:
   - Even if System 6 performs poorly in the Shock scenario due to negotiation latency or reasoning failures, all 8 seeds of the Shock scenario must be reported.

---

## 8. Failure Handling & Fallback Contract

LLM agents operating in live simulation loops can encounter parse errors, schema validation rejections, or rate limits.

### Fallback Protocol
1. **Execution Guard**:
   - When an LLM agent produces an invalid order (negative, non-numeric, or malformed JSON), the system logs an error event and immediately executes the fallback policy:
     $$O_t(e) = O_{t-1}(e)$$
     *(Repeat the previous period's order quantity for echelon $e$)*.
2. **Telemetry & Accounting**:
   - Every fallback event is recorded in DuckDB `agent_calls` (`fallback = TRUE`) and aggregated into `behavioural_metrics.fallback_rate`.
   - Fallback rate is defined as:
     $$\text{Fallback Rate} = \frac{\text{Total Fallback Invocations}}{\text{Total Agent Invocations}} \times 100\%$$
   - Any run where $\text{Fallback Rate} > 0\%$ on test seeds is flagged in summary reports.

---

## 9. Baseline Comparison Reference Table (Seeds 0 to 7)

For direct comparison, the frozen benchmark v2.1 performance targets established across test seeds 0 to 7 are:

| Scenario | EOQ Tuned (Cost / BW) | (s, S) Tuned (Cost / BW) | System 2b Tuned (Cost / BW) | Oracle Reference (Cost / BW) |
|:---|:---:|:---:|:---:|:---:|
| **Stable** | $22,648 / 6.944 | **$21,524 / 1.564** | $42,627 / 3.537 | $36,949 / 1.069 |
| **Seasonal** | $22,678 / 4.564 | **$20,620 / 1.394** | $54,414 / 3.205 | $36,436 / 0.871 |
| **Shock** | $37,044 / 1.843 | $38,834 / 1.226 | $216,119 / 2.378 | **$40,109 / 1.107** |

*All primary and exploratory LLM comparisons will evaluate against these exact frozen empirical distributions.*
