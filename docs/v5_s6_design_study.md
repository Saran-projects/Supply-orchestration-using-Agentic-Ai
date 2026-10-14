# System 6 offline design study (2026-10-14)

No LLM, no network calls. Tuning seeds 100–104 × 3 scenarios, 15 runs per candidate. Script: `scripts/v5_s6_design_study.py`.
Raw results: `results/v5_s6_design_study.csv`.

## Question

Can any System 5/6 design meet all three protocol targets at once, against tuned EOQ on the tuning seeds?

| Target | Value |
|---|---|
| End-to-end bullwhip ≤ tuned EOQ | ≤ 6.64 |
| Fill rate | ≥ 99% |
| Cost ≤ 1.15 × tuned EOQ | ≤ 25,056 |

## Design levers varied (120 candidates)

- **Band width** `delta`: 0.1, 0.2, 0.3, 0.5, 0.8 (the current design uses 0.8).
- **Forecast cap**: 1.2 or 1.5.
- **Order smoothing in the reference** (β of the smoothed order-up-to tool): 0.2, 0.4, 0.6, 0.95. Current: 0.95.
- **Chooser:** each design was scored with three scripted choosers as a proxy for the LLM:
  - `follow`: always use the reference.
  - `detector_map`: rule-based actions from the regime detector.
  - `oracle`: knows next-step demand.

## Results

| Design | Cost | Fill rate % | Bullwhip |
|---|---|---|---|
| Current, follow | 24,376 | 99.70 | 8.69 |
| Current, oracle | 24,618 | 99.87 | 8.71 |
| Lowest bullwhip with fill rate ≥ 99% (δ 0.3, cap 1.2, β 0.2, detector_map) | 32,031 | 99.31 | 6.88 |
| Tuned EOQ (target) | 21,788 | 99.93 | 6.64 |

1. **0 of 120 candidates meet all three targets.** Smoothing orders lowers bullwhip but raises cost well above the cap.
   Even the lowest-bullwhip feasible design (6.88) stays above EOQ, at 32k cost.
2. **The action choice barely matters.** Holding the design fixed and changing only the chooser moved bullwhip by
   +0.02 (detector_map) and +0.20 (oracle) on average, compared with always-follow. An action chooser that knows
   next-step demand does not reduce bullwhip, so an LLM picking among these four actions has almost no room to.
3. **In this architecture, bullwhip is set by the reference and band, not by the LLM's choices.** The levers that cut
   bullwhip are deterministic (β, δ), and they trade bullwhip against cost.

## Conclusion

No design change within the v5 architecture is supported by this study, so no amendment is drafted. Under
Amendment A1, any further change needs a new dated amendment. This study gives no evidence that such a change would
let System 6 beat tuned EOQ on bullwhip within the cost and service constraints.

**Limitation:** the oracle is myopic (one step ahead, nearest-to-ideal action). This study does not rule out a
multi-step planner beating it, but no LLM in this design plans beyond one step either.
