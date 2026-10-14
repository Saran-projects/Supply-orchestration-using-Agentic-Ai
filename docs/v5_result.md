# Design v5 result — shock, seed 102 (tuning seed), 40 steps, prompt v5.0, glm-isoquant

Date: 2026-10-14. Source: run output in `results/v5_paid_run_seed102_attempt{1,2,3}_*.log`.
**Caveat:** `scripts/v5_paid_run.py` used `db_path=False`, so these decisions were **not** written to DuckDB.
The numbers below come from the captured run output, not from DuckDB.

## Attempts

| Attempt | Request setting | S3 fallbacks | Outcome |
|---|---|---|---|
| 1 | max_tokens 1024, thinking "disabled" (ignored upstream) | 31/160 | DEFECT (truncation); S6 skipped |
| 2 | max_tokens 2048 | 13/160 | DEFECT (truncation); S6 skipped |
| 3 | max_tokens 1024, reasoning_effort "low" | 0/160 | valid (reported below) |

Attempts 1–3 are bug fixes under protocol Amendment A1; no tuning iteration was used (see `docs/prompt_changelog.md`).

## Validity (attempt 3)

- System 3: 160 LLM calls for 160 decisions; 0 fallbacks; 100% first-pass valid JSON. **Valid.**
- System 6: 160 LLM calls for 160 decisions; 0 fallbacks; 100% first-pass valid JSON. **Valid.**

## System 6 against the other policies (same seed)

| Policy | Cost | Fill rate % | E2E bullwhip | S6 Δcost | S6 ΔBW |
|---|---|---|---|---|---|
| **System 6 (glm, v5.0)** | **26,214** | 100 | **4.27** | – | – |
| Always-follow, tool reference (floor) | 24,969 | 100 | 4.15 | +5.0% | +2.7% |
| System 2c tuned | 25,064 | 100 | 4.32 | +4.6% | −1.2% |
| Tuned EOQ | 26,576 | 100 | 3.89 | −1.4% | +9.6% |
| System 3 (glm, v5.0) | 131,748 | 82.0 | 15.00 | | |

Per-echelon bullwhip for S6: retailer 4.40, wholesaler 0.98, distributor 0.82, manufacturer 0.79.

## LLM behaviour

- **Non-follow choices:** S6 chose something other than "follow" in 41% of decisions: trim 30–35%, boost_small 5–12%,
  boost_large 0%. For S3 the figure was 46%.
- **Were those choices right?** Not on this seed. Always-follow beat S6 on both cost and bullwhip, so the departures
  cost more than they saved. The oracle with next-step demand also scored worse than always-follow (27,647; BW 4.50),
  so one-step foresight does not identify better actions here. Per-decision agreement with the oracle was not computed,
  because the decisions were not persisted.
- **Regime accuracy against the detector:** S6 73.8%, S3 39.4%.
- **Cost per run:** S3 $0.0067; S6 $0.0099. Total task spend (3 attempts + checks): $0.140 of the $0.45 cap.

## Protocol success criteria (section 3), S6

Zero fallbacks: met. First-pass validity ≥ 99%: met. Cost ≤ 1.15 × EOQ: met. Fill rate ≥ 99%: met.
**E2E bullwhip ≤ tuned EOQ: not met** (4.27 vs 3.89).

## Classification

**WORSE_THAN_FLOOR.** On this seed, S6 is worse than the always-follow floor on both cost (+5.0%) and end-to-end
bullwhip (+2.7%). The margins are small and this is a single tuning seed (n = 1), so it is not a confirmatory result.
The offline design study (`docs/v5_s6_design_study.md`) found no v5 design that lets action choice materially reduce bullwhip.
