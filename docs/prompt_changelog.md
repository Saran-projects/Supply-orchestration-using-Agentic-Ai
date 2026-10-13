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

## Iteration 2 (`v3.2`)

- **Date**: October 2026
- **Model**: `glm-isoquant`
- **Target Systems**: System 3 (`single_agent`) and System 6 (`orchestrated`)
- **Tuning Scenario**: `shock`
- **Tuning Seed**: `101` (alternating seed, 40 steps)
- **Prompt Versions**: `single_agent_v3_2.txt`, `orchestrated_v3_2.txt`

### 1. What Changed (ONE Change Only)
Relaxed the rigid constraint requiring strict adherence to the Safe Order Range during stockouts by instructing agents to prioritize matching the **Smoothed Order-Up-To Target** to prevent stockouts whenever on-hand inventory drops, treating the Safe Order Range as guidance for normal steady-state operation:
- **Before (`v3.1`)**:  
  - S3: *"The replenishment order quantity must be a non-negative number between 0.0 and your echelon's maximum allowed capacity, respecting the safe order range."*
  - S6: *"Concessions must prioritize stability and smooth replenishment."*
- **After (`v3.2`)**:  
  - S3: *"The replenishment order quantity must be a non-negative number between 0.0 and your echelon's maximum allowed capacity; prioritize matching the Smoothed Order-Up-To Target to prevent stockouts whenever on-hand inventory drops, using the Safe Order Range for steady-state smoothing."*
  - S6: *"While order stability dampens bullwhip oscillations, concessions must prioritize matching the Smoothed Order-Up-To Target to prevent stockouts whenever inventory is depleted."*

### 2. Why
To prevent the ratchet effect where depleted inventory locks orders to miniature safe ranges (e.g. 0.1–0.4), starving downstream tiers and causing severe backlog cost explosions. Prioritizing the smoothed target when inventory drops allows the agent to replenish properly during demand shifts without getting trapped.

### 3. Result
*(To be recorded following run execution)*
