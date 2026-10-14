# CLAUDE.md

Rules for working in this repo. They are not optional.

1. **Config-driven code.** Parameters, thresholds, models and paths live in `configs/*.yaml` (loaded via `sca.config`). Do not hard-code them in `src/` or `scripts/`.
2. **Determinism.** The same `(scenario, seed)` must give the same results. Seed every source of randomness from the run seed.
3. **All LLM calls go through `LLMClient`** (`src/sca/llm/client.py`). No direct provider SDK or HTTP calls anywhere else.
4. **Run isolation.** Call `reset()` on the environment and on every policy inside the per-run loop. `tests/test_run_isolation.py` must pass.
5. **Seed split.** Seeds `100-104` are for tuning only. Seeds `0-7` are for final results only. Never tune on `0-7`.
6. **Pre-registration.** Write the rules for an experiment in `docs/` before running it. Never change them after seeing test results unless you add a dated note explaining the change.
7. **Main bullwhip metric** is end-to-end bullwhip (`end_to_end_bullwhip`). Per-echelon ratios are secondary.
8. **Never run any command that spends API money.** Real network calls are blocked unless `ALLOW_PAID_CALLS=1` is set. Never set it yourself; only the user does.

## Paid-call guard and run validity

- `LLMClient` refuses any real network call (`PaidCallsDisabledError`) unless `ALLOW_PAID_CALLS=1`. Mock and cached calls always work.
- HTTP 401, 402 and 403 raise `FatalAPIError`: the run stops at once, with no retry and no fallback order.
- A run is marked `invalid`, and the batch stops, if more than `max_fallback_rate` (5%, `configs/experiment.yaml`) of decisions used the fallback, or if LLM calls are fewer than decisions (`src/sca/eval/validity.py`).
