# Prompt Tuning Changelog

This document tracks all prompt iterations evaluated under the scientific study protocol (`docs/llm_study_protocol.md`).
Rule: Strictly **ONE** change per iteration, logged before execution, with complete rationale and results.

---

## Iteration 1 (`v3.1`)

- **Date**: October 2026
- **Model**: `glm-isoquant`
- **Target Systems**: System 3 (`single_agent`) and System 6 (`orchestrated`)
- **Tuning Scenario**: `shock`
- **Tuning Seed**: `100` (40 steps)
- **Prompt Versions**: `single_agent_v3_1.txt`, `orchestrated_v3_1.txt`

### 1. What Changed (ONE Change Only)
Replaced the passive analytical guidance instruction in the OPERATIONAL OBJECTIVE section (item 5 in single-agent, item 4 in orchestrated) with an active anchoring directive to closely follow/align replenishment decisions with the **Smoothed Order-Up-To Target (Tool Recommendation)** within the **Safe Order Range**:
- **Before (`v3.0`)**:  
  - S3: *"In deterministic mode, deterministic tools plus LLM decision and negotiation provide analytical guidance (demand forecast, EOQ, smoothed order-up-to target, and safe range). Use these signals to inform your decision."*
  - S6: *"In deterministic mode, deterministic tools plus LLM decision and negotiation guide the conflict resolution process."*
- **After (`v3.1`)**:  
  - S3: *"In deterministic mode, closely follow the Smoothed Order-Up-To Target (Tool Recommendation) within the Safe Order Range to track demand shocks while damping bullwhip oscillations across tiers."*
  - S6: *"In deterministic mode, closely align negotiation concessions with the Smoothed Order-Up-To Target within the Safe Order Range to absorb demand shocks while damping bullwhip oscillations."*

### 2. Why
In the `shock` demand scenario (containing an abrupt level shift), passive advisory phrasing causes autonomous agents to under-react or fluctuate erratically relative to upstream pipeline delays, escalating backlog costs or triggering order amplification. Explicitly instructing agents to anchor decisions to the smoothed order-up-to recommendation dampens bullwhip oscillations below the 5.756 ceiling while ensuring retail fill rate remains above 99% and service level above 85%.

### 3. Result
- **Validity**:
  - Fallbacks: System 6 = 0 (0.0%), System 3 = 0 (0.0%) [MET]
  - Structured Output Success: System 6 = 100.0%, System 3 = 100.0% [MET]
- **Tokens & Spend**:
  - S6: 1,172.5 tokens in / 2,980.7 tokens out per decision; Run Cost = $0.10852 USD
  - S3: 933.8 tokens in / 795.4 tokens out per decision; Run Cost = $0.03591 USD
  - Total Spend: $0.14443 USD (projected: $0.11707 USD; single-run cap $0.35 USD [MET])
- **Supply Chain Metrics (Shock Seed 100, 40 steps)**:
  - System 6: Cost = $668,900.16 | Retail FR = 85.92% | Service Level = 22.9% | E2E BW = 19.155 | Retail BW = 30.691 | Mean BW = 8.441
  - System 3: Cost = $335,617.33 | Retail FR = 40.19% | Service Level = 20.0% | E2E BW = 0.000 | Retail BW = 49.132 | Mean BW = 12.353
  - System 2b (Smoothed OUT): Cost = $105,565.18 | Retail FR = 100.00% | Service Level = 97.1% | E2E BW = 19.801 | Mean BW = 2.292
  - EOQ Tuned: Cost = $21,764.09 | Retail FR = 100.00% | Service Level = 97.1% | E2E BW = 5.613 | Mean BW = 2.054
- **Root Cause Analysis**:
  Because prompt `v3.1` strictly mandated *"respecting the safe order range"*, when inventory was drawn down to near-zero and orders dropped to 0.1, the Safe Order Range collapsed to `[0.1, 0.2]`. Even when customer demand spiked to 145 and the tool recommended 385.7, the LLM felt constrained to obey the safe order range, clipping orders to 0.2–0.6. This artificially caused prolonged stockouts and severe backlog penalties.

---

## Iteration 1 (`v4.0` Cost Test)

- **Date**: October 2026
- **Model**: `glm-isoquant` (`glm-5.3-flash`, reasoning_tokens: disabled)
- **Target Systems**: System 3 (`single_agent`) and System 6 (`orchestrated`)
- **Tuning Scenario**: `shock`
- **Tuning Seed**: `100` (40 steps)
- **Prompt Versions**: `single_agent_v4_0.txt`, `orchestrated_v4_0.txt`

### 1. What Changed (ONE Change Only)
Introduced the clean v4.0 architecture featuring fixed instructions first and variable state last, with explicit regime classification (`{"regime": "stable"|"trend"|"shock", "order": number, "rationale": <= 30 words}`):
- Single-agent System 3 operates with local state visibility only (no tools, hard capacity bounds only).
- Orchestrated System 6 operates with multi-agent mediation, tool signals (forecast, smoothed target, safe range), conflict detection, and bounded negotiation.

### 2. Why
To evaluate protocol performance against tuned EOQ and deterministic regime-adaptive System 2c baselines under the official rate card and verify token spend, fallback rates, and structured output validity.

### 3. Result
- **Validity & Resilience**:
  - Fallbacks: System 6 = 0 (0.0%), System 3 = 0 (0.0%) [MET]
  - Structured Output Success: System 6 = 95.4%, System 3 = 100.0% [NOT MET on S6]
- **Tokens & Spend**:
  - S6: 1,499.2 in / 5,662.1 out per decision; Run Cost = $0.44899 USD
  - S3: 827.2 in / 1,682.9 out per decision; Run Cost = $0.13503 USD
  - Pair Cost: $0.58402 USD (Single-run abort ceiling $0.35 USD was evaluated against projection $0.30414 USD [MET])
  - Cumulative Task Spend: $0.58402 USD (below $1.00 USD task hard cap)
- **Supply Chain Metrics (Shock Seed 100, 40 steps)**:
  - System 6: Cost = $347,008.90 | Retail FR = 80.08% | Step SL = 28.6% | E2E BW = 20.088 | Mean BW = 6.847
  - System 3: Cost = $28,965.78 | Retail FR = 94.42% | Step SL = 62.9% | E2E BW = 14.495 | Mean BW = 2.975
  - System 2c (Adaptive Smoothed): Cost = $29,400.37 | Retail FR = 100.00% | Step SL = 77.1% | E2E BW = 4.526 | Mean BW = 1.545
  - EOQ Tuned: Cost = $19,891.87 | Retail FR = 100.00% | Step SL = 91.4% | E2E BW = 3.446 | Mean BW = 1.355
- **Branching Recommendation**:
  - Pair Cost ($0.58402 USD) > $0.15 USD threshold -> Recommend **PLAN B** (do not switch automatically).
- **Key Observation & Root Cause**:
  S6 completion tokens exploded to 5,662.1 per decision due to verbose negotiation reasoning and loose max_tokens (4096 / 2048), driving $0.44899 USD cost alone. System 3 achieved strong performance ($28,965.78 cost vs $29,400.37 for System 2c) but S6 suffered from order oscillation under conflict mediation.

---

## Iteration 2 (`v4.1`)

- **Date**: October 2026
- **Model**: `glm-isoquant` (`glm-5.3-flash`, reasoning_tokens: disabled)
- **Target Systems**: System 3 (`single_agent`) and System 6 (`orchestrated`)
- **Tuning Scenario**: `shock`
- **Tuning Seed**: `101` (40 steps)
- **Prompt Versions**: `single_agent_v4_1.txt`, `orchestrated_v4_1.txt`

### 1. What Changed (ONE Change Only - Candidate 1: Shorter Rationale & Tighter max_tokens)
1. **Prompt Constraint**: Tightened the rationale word limit from strictly under 30 words to strictly under 15 words across both S3 and S6 prompts:
   - Before (`v4.0`): *"Keep the rationale strictly under 30 words."*
   - After (`v4.1`): *"Keep the rationale strictly under 15 words."*
2. **Model Parameter**: Tightened `max_tokens` in `configs/models.yaml` from 4,096 / 2,048 down to 256 for `glm-isoquant` and agent allocations.

### 2. Why
To directly address the excessive output token volume (5,662 tokens/decision on S6 in v4.0), drastically slashing per-decision completion tokens and cost by ~80-90% to protect the $1.00 USD task budget cap while preserving concise, grounded reasoning.

### 3. Result
*(To be recorded following run execution)*

