# Final evaluation plan — Systems 1–6, design v5 (pre-registered 2026-10-14, before any test-seed run)

Dated note to `docs/llm_study_protocol.md` (section 6). The v5 pair cost (S3 + S6) measured on seed 102 is about
$0.017, at or below the $0.15 threshold, so **Plan A** applies. At the user's request, the evaluation extends Plan A
to **all four LLM systems** on all scenarios. The budget cap is raised to **$1.20** cumulative for this task, including
$0.140 already spent on seed 102. Nothing below changes after test results are seen.

## Runs

- **Scenarios** stable, seasonal and shock × **test seeds 0–7**, giving 24 runs per system. 40 steps, 5 warmup steps.
  Environment as in `docs/tuning_rule_v2_1.md`: I₀ 687, h = 0.5, b = 2.5, steady-state start.
- **Systems** (frozen configurations, nothing re-tuned):
  1. (s, S) tuned: `baseline_s_S`, `results/baselines_frozen.json`.
  2. EOQ tuned: `baseline_eoq`, frozen.
  3. `single_agent_v5`
  4. `multi_agent_info_v5`
  5. `multi_agent_tools_v5`
  6. `orchestrated_v5`

  Systems 3–6 use glm-isoquant (glm-5.3-flash), `reasoning_effort: low`, `max_tokens` 1024, prompt v5.0, and the
  frozen multipliers.
- **References** (reported, not part of the six): System 2c tuned, and the scripted always-follow floor for each reference family.

## Rules

- **Validity:** a run is invalid if fallbacks exceed 5% or if LLM calls are fewer than decisions (`src/sca/eval/validity.py`).
  An invalid run stops the batch; the error is reported and nothing is re-tuned.
- **Abort and budget:** abort on HTTP 401/402/403. Hard stop before any call whose worst case would push cumulative spend over $1.20.
- **Statistics:** confirmatory tests are the three pre-registered comparisons only:
  - S6 vs S3
  - S6 vs tuned EOQ
  - S6 vs 2c

  Each is tested per scenario on total cost and end-to-end bullwhip, with a two-sided paired Wilcoxon test (n = 8)
  and Holm correction across the three comparisons. Everything else is exploratory.
- **Success criteria:** protocol section 3, applied per scenario.

## Metrics

- **Supply chain performance:**
  - Total cost (holding + backlog, all echelons).
  - Bullwhip per echelon, plus end-to-end bullwhip (the primary bullwhip metric).
  - Stockout steps.
  - Retailer fill rate.
  - Response time per decision (ms).
- **Agent behaviour:**
  - **Instruction following:** share of first-pass valid, in-schema outputs.
  - **Groundedness:** share of numbers quoted in rationales that match a number shown in the prompt, within ±0.5 or 1%.
  - **Fallback and error rate.**
  - **Token usage and USD.**
  - **Tool-choice and parameter accuracy:** n/a (tools are deterministic in v5).
  - **Negotiation:** n/a (v5 System 6 makes one orchestrator call per decision and has no negotiation rounds).
