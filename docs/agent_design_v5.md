# Agent design v5 (prompt v5.0)

Date: 2026-10-14. Status: implemented and tested with MockLLM only; no paid calls made under v5.
Code: `src/sca/systems/v5.py`. Config: `configs/agent_v5.yaml`. Prompts: `src/sca/agents/prompts/*_v5_0.txt`.
Scripts: `scripts/v5_scripted_tuning.py`, `scripts/v5_mock_comparison.py`. Tests: `tests/test_v5.py`, `tests/test_run_isolation.py`.

## 1. Why

In the phase-5 diagnosis (`docs/diagnosis_phase5.md`), free-form LLM order quantities failed in two ways: truncated or empty
output, and a mediator that turned failures into zero orders. v5 removes the order quantity from the LLM entirely.
The LLM picks one of four actions. Code computes the order, so a bad answer can at worst move the order inside a range that code controls.

## 2. Decision contract

The LLM answers with JSON only (`ActionDecision`):

```json
{"regime": "stable|trend|shock", "action": "follow|boost_small|boost_large|trim", "rationale": "<= 20 words"}
```

Code then computes:

```
order = clip(reference × multiplier[action], band_low, band_high)
```

- **Multipliers** live in config (`multipliers`, frozen; see section 5).
- **Band** is the shared `safe_order_bounds()`, the same function `validate_order` enforces. It takes the last order,
  the demand estimate, delta 0.8, a forecast cap of 1.2 × the demand estimate, and capacity (500, or 300 for the manufacturer).
  These are the frozen 2c values.
- **Fallback:** an invalid answer (malformed JSON, unknown action, rationale over 20 words, truncation) is not retried by
  the system. The client already retries a schema failure once. The order falls back to the previous order clipped into
  the band (protocol section 8), and the decision counts as a fallback.

## 3. References (computed in code and shown to the model)

| Systems | Reference |
|---|---|
| 3, 4 | `max(0, m + (S_basic − IP) / H)`, where m is the trailing mean of received demand over the last W steps, `S_basic = m × mean lead time`, and IP = inventory + pipeline − backlog. W = 5 and H = 4 (config). No safety stock, no smoothing, no regime flag. |
| 5, 6 | The smoothed order-up-to tool's order (`smoothed_order_up_to`, frozen 2c parameters: α = β = 0.95, z = 2.5, β_shock = 1.0, customer-demand information sharing). |

## 4. Regime: detector acts, LLM is scored

- The deterministic `regime_detector` (short window 5, long window 15, thresholds 1.30 and 1.15) runs for every echelon.
  A shock signal keeps a **shock phase** active for 4 steps, followed by 4 **after-shock** steps.
- **Only the detector's phase can widen the band**, and only for Systems 5 and 6 (`delta_shock`). The LLM's regime label
  is logged and scored against the detector's raw label, never acted on. `test_llm_regime_never_widens_band` checks this.
- Note: the frozen 2c values make `delta_shock` equal to `delta` (0.8), so widening is implemented but currently changes nothing.

## 5. Systems differ only in information, tools/band and orchestration

| System | Information | Reference | Band widens in shock | Orchestration |
|---|---|---|---|---|
| 3 `single_agent_v5` | own echelon | basic | no | – |
| 4 `multi_agent_info_v5` | + all echelons' state | basic | no | – |
| 5 `multi_agent_tools_v5` | + tool outputs (forecast, order-up-to level, detector) | tool | yes | – |
| 6 `orchestrated_v5` | same as 5 | tool | yes | deterministic demand, inventory and procurement proposals, each mapped to its nearest action, plus a conflict flag; the LLM arbitrates |

All four make exactly one LLM call per echelon-step, with the same action space and the same output schema. Tools are deterministic.

## 6. Logging (per decision; `decision_log` in memory and DuckDB table `v5_decisions`)

Each decision row records: action, LLM regime, detector regime and phase, reference, multiplier, order before and after
clipping, clipped flag, band limits, band widened, fallback, error type and message, tokens, USD, and prompt version `v5.0`.
`llm_stats()` feeds the run-validity rules: a run is invalid if more than 5% of decisions fell back, or if LLM calls < decisions.

## 7. Multiplier tuning (no LLM; seeds 100–104; `docs/tuning_rule_v2_1.md`)

- **Method:** the detector-mapped scripted policy (stable → follow, trend → boost_small, shock → boost_large,
  after-shock → trim), pooled over both reference families, 3 scenarios and 5 seeds (30 runs per candidate).
  Feasibility: fill rate ≥ 99% and end-to-end bullwhip ≤ 19.149 (0.75 × untuned EOQ). Objective: minimum mean cost.
  Grid values on an edge were expanded until interior.
- **Result:** **no candidate is feasible.** The basic reference has no safety stock, so pooled fill rate never reaches 99%.
  The rule's caveat fallback (highest fill rate, then lowest cost) applies.
- **Dated note, 2026-10-14 (user decision):** the unconstrained fallback chose trim = 1.0, at its hard limit, which would
  make trim identical to follow. Trim is therefore constrained to ≤ 0.95 so all four actions stay distinct.
- **Frozen:** follow 1.0, **boost_small 1.2, boost_large 1.8, trim 0.9**. All three are interior after expansion (280 candidates).
  Pooled cost 38,675, fill rate 97.82%, bullwhip 12.71. Logs: `results/v5_multiplier_tuning.csv` and the `multipliers_tuning_log` field in `configs/agent_v5.yaml`.

## 8. MockLLM results (seeds 100–102, 3 scenarios, frozen multipliers; `results/v5_mock_comparison_seeds100_102.csv`)

| Policy | Invalid runs | Follow / boost_small / boost_large / trim / fallback | Clip rate | Regime accuracy | Cost | Fill rate % | E2E bullwhip |
|---|---|---|---|---|---|---|---|
| S3 = S4, mock valid (random) | 0/9 | .25/.28/.21/.26/0 | .37 | .34 | 64,607 | 92.6 | 17.53 |
| S3 = S4, mock detector-copy | 0/9 | .41/.04/.48/.07/0 | .35 | 1.00 | 52,018 | 95.1 | 15.42 |
| S3 = S4, mock malformed or bad action | 9/9 | 0/0/0/0/1.0 | .00 | – | 44,665 | 91.5 | 0.00 |
| S5 = S6, mock valid (random) | 0/9 | .25/.28/.21/.26/0 | .67 | .36 | 23,570 | 100.0 | 8.17 |
| S5 = S6, mock always-follow | 0/9 | 1/0/0/0/0 | .65 | .94 | 23,598 | 99.9 | 8.18 |
| S5 = S6, mock detector-copy | 0/9 | .92/.05/.02/.01/0 | .65 | 1.00 | 23,636 | 99.9 | 8.18 |
| S5 = S6, mock malformed or bad action | 9/9 | 0/0/0/0/1.0 | .04 | – | 53,568 | 83.9 | 0.07 |
| scripted follow [basic] | – | | .34 | – | 72,670 | 93.0 | 18.70 |
| scripted detector-mapped [basic] | – | | .35 | – | 52,018 | 95.1 | 15.42 |
| scripted oracle [basic] | – | | .32 | – | 64,919 | 93.1 | 14.53 |
| scripted follow [tool] | – | | .65 | – | 23,598 | 99.9 | 8.18 |
| scripted detector-mapped [tool] | – | | .65 | – | 23,636 | 99.9 | 8.18 |
| scripted oracle [tool] | – | | .64 | – | 24,286 | 99.8 | 8.22 |
| System 2c tuned | – | | – | – | 23,316 | 100.0 | 8.26 |
| EOQ tuned | – | | – | – | 21,713 | 99.9 | 6.10 |

Checks: mock detector-copy equals scripted detector-mapped, and mock always-follow equals scripted follow.
S3 equals S4 and S5 equals S6 under MockLLM because the mock ignores prompt content; information and orchestration can
only change results with a real model.

## 9. What this means before any paid run

1. **S3/S4 cannot meet the protocol's success criteria with any action sequence tested.** Even the next-step-demand
   oracle reaches only 93% fill rate, because the basic reference has no safety stock (by design). Expect S3/S4 to fail
   criteria 4–5 regardless of LLM quality.
2. **For S5/S6, the action barely matters.** The frozen 2c band clips about 65% of orders. A random action choice
   costs within 1% of always-follow, which is itself within 1.3% of 2c. A paid S6 run therefore tests reliability
   (validity, fallbacks) far more than decision quality.
3. Neither S5/S6 nor 2c beats tuned EOQ on bullwhip on these seeds (8.2 vs 6.1), so protocol criterion 3 is unlikely to be met.
4. A paid run also needs `max_tokens` large enough for the model's hidden reasoning (diagnosis phase 5). The current
   setting of 256 truncates every call.
