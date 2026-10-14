# Phase 5 diagnosis — shock, seed 101, prompt v4.1

Date: 2026-10-14. Read-only: no network calls, no code changes. Sources: `results/glm_experiment.duckdb`
(runs `s3_glm_shock_seed_101_v4_1_aea845`, `s6_glm_shock_seed_101_v4_1_793cfe`), `configs/models.yaml`, git history,
offline replays using the scratchpad scripts `replay.py` and `rebuild_s3.py` (only scratchpad files were written),
and `results/benchmark_v2_1_test_seeds0_7_raw.csv`.

## Findings at a glance

1. **v4.1 sent no LLM decisions.** System 3: 160/160 decisions fell back. System 6: 542/542 LLM calls failed.
   Cause: v4.1 cut `max_tokens` from 4096 to 256 while the model still produces hidden reasoning tokens.
2. **System 6 v4.1 is a deterministic mediator.** An offline replay with every LLM call failing reproduces all 160
   logged final orders exactly. Its poor results come from the mediator rule (median of proposals), not from the LLM.
3. **The S6 prompt asks the model to follow a target it never receives.** `orchestrated_v4_1.txt` says to align with
   the "Smoothed Order-Up-To Target", but the prompt has no placeholder for that value.
4. **Q3:** see the section below.
5. **Q4: No.** Tuned 2c is worse than tuned EOQ on end-to-end bullwhip in 24/24 paired runs. Neither keeps fill rate ≥ 99% on shock.

## 1. System 3: the error behind the 160 fallbacks

- `agent_calls` holds 160 rows, all `fallback=true` and `error_type=StructuredOutputError`, each ordering 117.38
  (the initial `last_order`). Exact message, the same in all 160 rows:
  `Invalid JSON: EOF while parsing a value at line 1 column 0 [type=json_invalid, input_value='', input_type=str]`.
- So the HTTP call succeeded (latency 5–19 s, no 4xx/5xx) and returned **empty `content`**, twice per decision
  (first try plus one retry). Status code, response body and `finish_reason` were **not logged**.
- Evidence that this was truncation (`finish_reason=length`):
  - Same model and prompt family, `max_tokens` 4096: completion tokens averaged 795 (v3.1), 1,683 (v4.0) and
    1,135 (v3.2), with a maximum of 6,659, even with `reasoning_tokens: "disabled"`. The `thinking:{type:disabled}` flag is ignored.
  - System 6 v4.1 logged exactly **512 completion tokens on every call** (2 × 256, i.e. both attempts hit the cap).
  - The `max_tokens: 4096 → 256` change is in commit `a1bf36c` and `docs/prompt_changelog.md` ("Iteration 2 (v4.1)").
    The 256-token output budget is spent on hidden reasoning before any JSON is written.
  - Code path: `beta.chat.completions.parse` raises on a length finish. The adapter silently falls back to
    `chat.completions.create`, whose `message.content` is `None`, so `content or ""` gives `''`.
- Request rebuilt offline (step 0, retailer):
  `POST https://api.isoquant.ai/v1/chat/completions` with `model=glm-5.3-flash`, `temperature=0.2`, `seed=101`,
  **`max_tokens=256`**, `timeout=60`, `extra_body={"thinking":{"type":"disabled"}}`, and `response_format` set to a
  json_schema of `OrderDecision` (regime/order/quantity/rationale/reasoning). There is one user message with the v4.1 prompt.
- Side effects:
  - Logged cost for S3 v4.1 is **$0.00**, because the `StructuredOutputError` path in `llm_base` drops token counts.
    Real spend was about 320 calls × (~1k input + 256 output) ≈ $0.03, not recorded.
  - The tuning script's 401/402/403 check never fires, because `llm_base` turns every exception into a fallback.
  - The changelog calls v4.1 "ONE change only" but changed two things (rationale length and `max_tokens`).

## 2. System 6 vs System 2c (shock, seed 101)

There is no LLM order in v4.1. "Order" below is S6's final order (mediator plus validation). "Target" is S6's own
smoothed order-up-to tool (`orchestrated.py` rule 4, alpha 0.2). Steps 5–39, after warmup.

| Echelon | order/target median [p10, p90] | within 5% | > 25% away | mean backlog S6 / 2c | order var S6 / 2c | bullwhip S6 / 2c |
|---|---|---|---|---|---|---|
| retailer | 0.86 [0.49, 1.27] | 3% | 46% | 277.6 / 23.0 | 31,115 / 3,776 | 24.36 / 2.96 |
| wholesaler | 0.71 [0.38, 2.28] | 0% | 83% | 802.5 / 25.0 | 38,502 / 3,859 | 1.31 / 1.05 |
| distributor | 0.61 [0.00, 1.91] | 3% | 89% | 607.0 / 14.4 | 37,120 / 4,917 | 1.01 / 1.21 |
| manufacturer | 0.62 [0.00, 2.08] | 3% | 89% | 311.9 / 0.0 | 12,858 / 5,547 | 0.38 / 1.11 |

Totals: S6 cost 199,324, retailer fill rate 88.3%, end-to-end bullwhip 10.07. 2c: 29,291, 100%, 4.34.

- **Backlog source: the wholesaler (802 mean) and the distributor (607).** Early on, the inventory and procurement
  agents propose 0 (high steady-state stock). The median rule halves orders (58.7 → 29.3 at steps 0–1), which starves
  the pipeline. Upstream, distributor and manufacturer order zero in 10/35 and 13/35 steps. The shortage then reaches
  the retailer, whose backlog peaks at steps 29–39 (704 at t=31) even though it then orders close to target (376 vs 438).
- **Order variance source: the retailer.** Its bullwhip is 24.4 vs 2.96 for 2c. Biggest swings: retailer at t=25 (|Δq|=141),
  wholesaler at t=29–30, distributor at t=32, manufacturer at t=33. This is the shock wave moving upstream about one step per echelon.
- 39/40 retailer decisions went to negotiation. Every negotiation fell back to the capacity-bounded median because the LLM failed.

## 3. No-LLM runs on seeds 100-102, all scenarios

Policy: each echelon orders S6's smoothed target, clipped to [0, capacity]. Noise is multiplicative Gaussian
(σ = 5/15/30%, seeded). The comparison is with tuned 2c (`baselines_frozen.json`).

Before the fixes, S6's own target (rule 4: alpha 0.2, z 1.65, stateless) matched 2c within 5% in **0/9** runs.
Mean Δcost was +1,416% and mean Δe2e +503% at noise 0. Noise of 5/15/30% barely changed this (+1,416% / +1,415% / +1,517%),
and the minimum fill rate was 90.1 / 89.6 / 88.8 / 82.5%. So the gap is plumbing (a different target tool), not the LLM.
After the 2026-10-14 fixes, S6 with `order_source: tool_target` reproduces 2c exactly (Δ 0.00%) in 9/9 runs
(`tests/test_phase5_fixes.py`). The noise variants have not been re-run against the fixed target.

## 4. Seeds 0-7: tuned 2c vs tuned EOQ (from `benchmark_v2_1_test_seeds0_7_raw.csv`, frozen params)

| Scenario | e2e BW EOQ | e2e BW 2c | 2c lower in | fill rate EOQ / 2c (min) |
|---|---|---|---|---|
| stable | 10.40 | 14.19 | 0/8 | 100 / 100 |
| seasonal | 6.98 | 8.40 | 0/8 | 100 / 100 |
| shock | 2.68 | 3.83 | 0/8 | 97.2 (85.2) / 97.4 (87.2) |

**Answer: no.** 2c has higher end-to-end bullwhip in 24/24 paired runs (mean +2.12, Wilcoxon p < 0.001). The fill-rate
condition (≥ 99%) also fails for both on shock, 2 of 8 seeds each. 2c also costs more in every scenario.
Note: `scripts/benchmark_v2_1_final_test.py` uses different "tuned" parameters (EOQ multiplier 0.75, backlog cost 5.0)
than `baselines_frozen.json` (0.25, 2.5). Only the frozen-parameter rerun is used here.

## Recommended fixes, ranked by impact

1. **Restore an adequate output budget** (e.g. `max_tokens` ≥ 2048 for glm, or a provider-verified way to really turn
   thinking off). Treat `finish_reason=length` and empty content as their own error type, not as a JSON failure.
   Without this, v4.1 results are invalid for both systems. Record this as a dated amendment, since v4.1 is pre-registered.
2. **Log the raw failure:** `finish_reason`, HTTP status, and the first 500 characters of the body or content for every
   failed call. Keep token counts and cost on the `StructuredOutputError` path (S3 currently records $0).
3. **Pass the smoothed target value into the S6 prompt** (add a `{smoothed_target}` field). Today the model is told to
   follow a number it never sees. Consider turning on `enable_smoothed_target_conflict`.
4. **Fix the S6 mediator fallback.** When the LLM fails, use the tool target instead of the median of {demand, 0, 0, …}.
   The median is what produced the zero orders and the 600–800-unit backlogs. Count these as fallbacks: today S6 logs
   `fallback_used=false` for them.
5. **Enforce run validity** (the fallback-rate > 5% and calls < decisions rules now in `LLMClient`/runner). Wire it into
   `scripts/run_prompt_tuning.py`, which has its own loop and would have marked both v4.1 runs invalid.
6. **Re-examine 2c as the reference policy.** On test seeds it is dominated by tuned EOQ on bullwhip and cost.
   This is only a report: any change to the comparison set must follow a pre-registered rule.
