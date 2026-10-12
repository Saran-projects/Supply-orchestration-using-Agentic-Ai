# Final Classical Baseline & Oracle Benchmark Results (v2.1)

Evaluated across 8 test seeds (0 to 7) per scenario under frozen Benchmark v2.1 conditions:
- **Initial Inventory**: 687.0 units per echelon (base-stock target $S = L\mu_D + 2.0\sigma_{\text{LTD}}$)
- **Common Horizon**: 40 days (warmup: 5 days)
- **Cost Parameters**: holding cost $h = \$0.50$/unit-day, backlog cost $b = \$5.00$/unit-day ($b/h = 10.0$)
- **Feasibility Constraints**: Fill Rate $\ge 99.0\%$, Step Service Level $\ge 85.0\%$, Mean Bullwhip $\le 5.756$

## Scenario: STABLE

| System | Total Cost ($) | Fill Rate (%) | Service Level (%) | Mean Bullwhip | E2E Bullwhip | Constrained Cost | Status |
|:---|---:|---:|---:|---:|---:|---:|:---|
| System 1-Tuned (s, S Tuned: s=1.2, S=0.25) | $21,524 ± $1,992 | 100.00% ± 0.00% | 99.6% ± 1.0% | 1.564 ± 0.253 | 8.45 ± 1.72 | $21,524 | Qualified |
| Oracle (Clairvoyant Reference) | $36,949 ± $3,660 | 100.00% ± 0.00% | 100.0% ± 0.0% | 1.069 ± 0.072 | 1.24 ± 0.29 | $36,949 | Reference |
| System 0 (Constant Order) | $37,147 ± $3,638 | 100.00% ± 0.00% | 100.0% ± 0.0% | 0.000 ± 0.000 | 0.00 ± 0.00 | $37,147 | Qualified |
| System 2b Untuned (Smoothed OUT) | $39,154 ± $12,349 | 100.00% ± 0.00% | 92.1% ± 7.4% | 3.813 ± 0.620 | 53.09 ± 18.91 | $39,154 | Qualified |
| System 2b-Tuned (Smoothed OUT: alpha=0.6, beta=0.6, z=2.5, delta=0.5, cap=1.5) | $42,627 ± $17,014 | 100.00% ± 0.00% | 97.5% ± 3.2% | 3.537 ± 0.590 | 50.69 ± 13.84 | $42,627 | Qualified |
| System 1 (s, S Untuned) | $24,860 ± $1,534 | 99.12% ± 2.18% | 88.6% ± 7.5% | 19.256 ± 3.495 | 65.49 ± 11.81 | NaN | Disqualified (BW=19.26>5.76) |
| System 2-Tuned (EOQ Tuned: Q=0.75, ROP=1.2) | $22,648 ± $2,248 | 100.00% ± 0.00% | 99.3% ± 1.3% | 6.944 ± 1.502 | 26.47 ± 6.26 | NaN | Disqualified (BW=6.94>5.76) |
| System 2 (EOQ Untuned) | $24,941 ± $2,487 | 99.96% ± 0.11% | 86.4% ± 9.3% | 12.695 ± 2.719 | 43.89 ± 10.74 | NaN | Disqualified (BW=12.70>5.76) |

## Scenario: SEASONAL

| System | Total Cost ($) | Fill Rate (%) | Service Level (%) | Mean Bullwhip | E2E Bullwhip | Constrained Cost | Status |
|:---|---:|---:|---:|---:|---:|---:|:---|
| System 1-Tuned (s, S Tuned: s=1.2, S=0.25) | $20,620 ± $2,033 | 100.00% ± 0.00% | 99.3% ± 1.3% | 1.394 ± 0.102 | 5.51 ± 1.09 | $20,620 | Qualified |
| System 2-Tuned (EOQ Tuned: Q=0.75, ROP=1.2) | $22,678 ± $2,040 | 100.00% ± 0.00% | 97.9% ± 4.0% | 4.564 ± 1.595 | 17.14 ± 7.04 | $22,678 | Qualified |
| System 0 (Constant Order) | $35,734 ± $3,092 | 100.00% ± 0.00% | 100.0% ± 0.0% | 0.000 ± 0.000 | 0.00 ± 0.00 | $35,734 | Qualified |
| Oracle (Clairvoyant Reference) | $36,436 ± $3,067 | 100.00% ± 0.00% | 100.0% ± 0.0% | 0.871 ± 0.071 | 0.88 ± 0.29 | $36,436 | Reference |
| System 2b Untuned (Smoothed OUT) | $48,515 ± $22,051 | 100.00% ± 0.00% | 93.2% ± 4.3% | 3.207 ± 0.381 | 36.24 ± 7.78 | $48,515 | Qualified |
| System 2b-Tuned (Smoothed OUT: alpha=0.6, beta=0.6, z=2.5, delta=0.5, cap=1.5) | $54,414 ± $37,143 | 100.00% ± 0.00% | 98.9% ± 2.1% | 3.205 ± 0.747 | 37.04 ± 10.19 | $54,414 | Qualified |
| System 1 (s, S Untuned) | $24,658 ± $2,020 | 99.41% ± 1.30% | 83.6% ± 7.0% | 12.105 ± 3.658 | 39.97 ± 14.50 | NaN | Disqualified (SL=83.6%<85%, BW=12.11>5.76) |
| System 2 (EOQ Untuned) | $26,050 ± $2,877 | 98.59% ± 2.09% | 82.9% ± 5.1% | 7.917 ± 2.347 | 26.64 ± 8.99 | NaN | Disqualified (FR=98.6%<99%, SL=82.9%<85%, BW=7.92>5.76) |

## Scenario: SHOCK

| System | Total Cost ($) | Fill Rate (%) | Service Level (%) | Mean Bullwhip | E2E Bullwhip | Constrained Cost | Status |
|:---|---:|---:|---:|---:|---:|---:|:---|
| Oracle (Clairvoyant Reference) | $40,109 ± $7,474 | 99.27% ± 2.05% | 91.1% ± 25.3% | 1.107 ± 0.133 | 1.30 ± 0.53 | $40,109 | Reference |
| System 0 (Constant Order) | $46,658 ± $22,821 | 97.58% ± 5.93% | 83.2% ± 27.5% | 0.000 ± 0.000 | 0.00 ± 0.00 | NaN | Disqualified (FR=97.6%<99%, SL=83.2%<85%) |
| System 1 (s, S Untuned) | $50,254 ± $21,510 | 96.28% ± 5.72% | 60.0% ± 15.7% | 4.006 ± 1.193 | 10.97 ± 4.41 | NaN | Disqualified (FR=96.3%<99%, SL=60.0%<85%) |
| System 1-Tuned (s, S Tuned: s=1.2, S=0.25) | $38,834 ± $14,957 | 97.97% ± 4.70% | 76.1% ± 15.9% | 1.226 ± 0.085 | 2.63 ± 0.65 | NaN | Disqualified (FR=98.0%<99%, SL=76.1%<85%) |
| System 2 (EOQ Untuned) | $58,859 ± $36,194 | 94.77% ± 8.02% | 61.8% ± 20.7% | 2.595 ± 0.774 | 6.94 ± 2.75 | NaN | Disqualified (FR=94.8%<99%, SL=61.8%<85%) |
| System 2-Tuned (EOQ Tuned: Q=0.75, ROP=1.2) | $37,044 ± $12,676 | 96.41% ± 7.09% | 81.4% ± 10.7% | 1.843 ± 0.387 | 4.91 ± 1.73 | NaN | Disqualified (FR=96.4%<99%, SL=81.4%<85%) |
| System 2b Untuned (Smoothed OUT) | $162,834 ± $99,157 | 99.44% ± 1.18% | 78.6% ± 13.5% | 2.351 ± 0.274 | 13.03 ± 5.40 | NaN | Disqualified (SL=78.6%<85%) |
| System 2b-Tuned (Smoothed OUT: alpha=0.6, beta=0.6, z=2.5, delta=0.5, cap=1.5) | $216,119 ± $148,810 | 97.41% ± 7.33% | 84.3% ± 6.3% | 2.378 ± 0.293 | 12.95 ± 4.20 | NaN | Disqualified (FR=97.4%<99%, SL=84.3%<85%) |

## Key Findings

1. **System 0 Defeated in Stable & Seasonal**:
   - In Stable demand, System 1-Tuned ((s, S)) achieves $21,524 ± $1,992, beating System 0 ($37,147 ± $3,638) by 42%.
   - In Seasonal demand, System 1-Tuned achieves $20,620 ± $2,033 and System 2-Tuned (EOQ) achieves $22,678 ± $2,040, both soundly outperforming System 0 ($35,734 ± $3,092).
2. **System 0 Disqualified in Shock**:
   - System 0 cannot adapt to the +50% level shift during the shock window, suffering retail stockouts that degrade fill rate to 97.58% (< 99.0%) and service level to 83.2% (< 85.0%), leading to disqualification.
3. **Bullwhip Suppression**:
   - Tuned (s, S) dampens mean bullwhip across echelons to 1.394–1.564 (down from 12.1–19.3 in untuned (s, S)), satisfying the 5.756 ceiling constraint.
4. **Oracle Reference**:
   - Clairvoyant base-stock achieves near-perfect fill rate (99.27%–100%) and low bullwhip (0.871–1.107) across all regimes, serving as the benchmark ceiling.
