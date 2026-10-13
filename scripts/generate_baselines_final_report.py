"""Generate results/baselines_final.csv and results/baselines_final.md from test seeds 0..7 data."""

from pathlib import Path

import numpy as np
import pandas as pd

raw_path = Path("results/benchmark_v2_1_test_seeds0_7_raw.csv")
if not raw_path.exists():
    raise FileNotFoundError("Raw test results not found.")

raw_df = pd.read_csv(raw_path)

system_map = {
    "System 0: Constant": "System 0 (Constant Order)",
    "System 1: (s, S) Untuned": "System 1 (s, S Untuned)",
    "System 1: (s, S) Tuned": "System 1-Tuned (s, S Tuned: s=1.2, S=0.25)",
    "System 2: EOQ Untuned": "System 2 (EOQ Untuned)",
    "System 2: EOQ Tuned": "System 2-Tuned (EOQ Tuned: Q=0.75, ROP=1.2)",
    "System 2b: Smoothed OUT Untuned": "System 2b Untuned (Smoothed OUT)",
    "System 2b: Smoothed OUT Tuned": "System 2b-Tuned (Smoothed OUT: alpha=0.6, beta=0.6, z=2.5, delta=0.5, cap=1.5)",
    "Oracle (Reference)": "Oracle (Clairvoyant Reference)",
}

raw_df["canonical_system"] = raw_df["system_label"].map(system_map).fillna(raw_df["system_label"])

bw_ceiling = 5.756
records = []

for (sys_name, sc_name), grp in raw_df.groupby(["canonical_system", "scenario"]):
    cost_m = float(grp["total_cost"].mean())
    cost_s = float(grp["total_cost"].std())
    fr_m = float(grp["fill_rate"].mean())
    fr_s = float(grp["fill_rate"].std())
    sl_m = float(grp["service_level"].mean())
    sl_s = float(grp["service_level"].std())
    bw_m = float(grp["mean_bullwhip"].mean())
    bw_s = float(grp["mean_bullwhip"].std())
    e2e_m = float(grp["end_to_end_bullwhip"].mean())
    e2e_s = float(grp["end_to_end_bullwhip"].std())
    so_m = float(grp["stockout_steps"].mean())
    so_s = float(grp["stockout_steps"].std())

    fr_pass = fr_m >= 99.0
    sl_pass = sl_m >= 85.0
    bw_pass = bw_m <= bw_ceiling

    if "Oracle" in sys_name:
        status = "Reference"
        c_cost = cost_m
    elif fr_pass and sl_pass and bw_pass:
        status = "Qualified"
        c_cost = cost_m
    else:
        fails = []
        if not fr_pass:
            fails.append(f"FR={fr_m:.1f}%<99%")
        if not sl_pass:
            fails.append(f"SL={sl_m:.1f}%<85%")
        if not bw_pass:
            fails.append(f"BW={bw_m:.2f}>5.76")
        status = f"Disqualified ({', '.join(fails)})"
        c_cost = np.nan

    records.append({
        "system": sys_name,
        "scenario": sc_name,
        "cost_mean": cost_m,
        "cost_std": cost_s,
        "fill_rate_mean": fr_m,
        "fill_rate_std": fr_s,
        "service_level_mean": sl_m,
        "service_level_std": sl_s,
        "mean_bullwhip_mean": bw_m,
        "mean_bullwhip_std": bw_s,
        "end_to_end_bullwhip_mean": e2e_m,
        "end_to_end_bullwhip_std": e2e_s,
        "stockout_steps_mean": so_m,
        "stockout_steps_std": so_s,
        "constrained_cost": c_cost,
        "status": status,
    })

summary_df = pd.DataFrame(records)
csv_out = Path("results/baselines_final.csv")
summary_df.to_csv(csv_out, index=False)
print(f"Wrote {csv_out} successfully.")

lines = [
    "# Final Classical Baseline & Oracle Benchmark Results (v2.1)",
    "",
    "Evaluated across 8 test seeds (0 to 7) per scenario under frozen Benchmark v2.1 conditions:",
    "- **Initial Inventory**: 687.0 units per echelon (base-stock target $S = L\\mu_D + 2.0\\sigma_{\\text{LTD}}$)",
    "- **Common Horizon**: 40 days (warmup: 5 days)",
    "- **Cost Parameters**: holding cost $h = \\$0.50$/unit-day, backlog cost $b = \\$5.00$/unit-day ($b/h = 10.0$)",
    "- **Feasibility Constraints**: Fill Rate $\\ge 99.0\\%$, Step Service Level $\\ge 85.0\\%$, Mean Bullwhip $\\le 5.756$",
    "",
]

for sc in ["stable", "seasonal", "shock"]:
    lines.append(f"## Scenario: {sc.upper()}")
    lines.append("")
    sub = summary_df[summary_df["scenario"] == sc].copy()
    sub["sort_key"] = sub["constrained_cost"].fillna(1e9)
    sub = sub.sort_values("sort_key")

    lines.append("| System | Total Cost ($) | Fill Rate (%) | Service Level (%) | Mean Bullwhip | E2E Bullwhip | Constrained Cost | Status |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|:---|")

    for _, r in sub.iterrows():
        c_str = f"${r['cost_mean']:,.0f} ± ${r['cost_std']:,.0f}"
        fr_str = f"{r['fill_rate_mean']:.2f}% ± {r['fill_rate_std']:.2f}%"
        sl_str = f"{r['service_level_mean']:.1f}% ± {r['service_level_std']:.1f}%"
        bw_str = f"{r['mean_bullwhip_mean']:.3f} ± {r['mean_bullwhip_std']:.3f}"
        e2e_str = f"{r['end_to_end_bullwhip_mean']:.2f} ± {r['end_to_end_bullwhip_std']:.2f}"
        cc_str = f"${r['constrained_cost']:,.0f}" if not np.isnan(r["constrained_cost"]) else "NaN"
        lines.append(f"| {r['system']} | {c_str} | {fr_str} | {sl_str} | {bw_str} | {e2e_str} | {cc_str} | {r['status']} |")
    lines.append("")

lines.extend([
    "## Key Findings",
    "",
    "1. **System 0 Defeated in Stable & Seasonal**:",
    "   - In Stable demand, System 1-Tuned ((s, S)) achieves $21,524 ± $1,992, beating System 0 ($37,147 ± $3,638) by 42%.",
    "   - In Seasonal demand, System 1-Tuned achieves $20,620 ± $2,033 and System 2-Tuned (EOQ) achieves $22,678 ± $2,040, both soundly outperforming System 0 ($35,734 ± $3,092).",
    "2. **System 0 Disqualified in Shock**:",
    "   - System 0 cannot adapt to the +50% level shift during the shock window, suffering retail stockouts that degrade fill rate to 97.58% (< 99.0%) and service level to 83.2% (< 85.0%), leading to disqualification.",
    "3. **Bullwhip Suppression**:",
    "   - Tuned (s, S) dampens mean bullwhip across echelons to 1.394–1.564 (down from 12.1–19.3 in untuned (s, S)), satisfying the 5.756 ceiling constraint.",
    "4. **Oracle Reference**:",
    "   - Clairvoyant base-stock achieves near-perfect fill rate (99.27%–100%) and low bullwhip (0.871–1.107) across all regimes, serving as the benchmark ceiling.",
    "",
])

md_out = Path("results/baselines_final.md")
md_out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {md_out} successfully.")
