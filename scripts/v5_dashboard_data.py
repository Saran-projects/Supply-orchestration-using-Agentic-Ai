"""Aggregate results/v5_final_runs.csv (+ results/v5_final.duckdb) into the dashboard's datasets.

Writes results/dashboard/*.json (arrays of row objects). Pre-registered stats (docs/v5_final_eval_plan.md):
S6 vs S3, S6 vs tuned EOQ, S6 vs 2c; per scenario; total cost and e2e bullwhip; two-sided paired Wilcoxon, Holm across
the three comparisons (within scenario x metric).
"""

from __future__ import annotations

import json
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from sca.config import find_project_root

ROOT = find_project_root()
OUT = ROOT / "results/dashboard"
ORDER = ["S1 (s,S) tuned", "S2 EOQ tuned", "S3 single agent", "S4 multi-agent info", "S5 multi-agent tools",
         "S6 orchestrated", "ref 2c tuned", "ref follow [basic]", "ref follow [tool]"]
LABEL = {
    "S1 (s,S) tuned": "S1 (s,S) policy", "S2 EOQ tuned": "S2 EOQ", "S3 single agent": "S3 Single LLM agent",
    "S4 multi-agent info": "S4 Multi-agent + info sharing", "S5 multi-agent tools": "S5 Multi-agent + tools",
    "S6 orchestrated": "S6 Orchestrated (proposed)", "ref 2c tuned": "Ref: 2c adaptive heuristic",
    "ref follow [basic]": "Ref: always-follow (basic ref.)", "ref follow [tool]": "Ref: always-follow (tool ref.)",
}
LLM = ["S3 single agent", "S4 multi-agent info", "S5 multi-agent tools", "S6 orchestrated"]
ECH = ["retailer", "wholesaler", "distributor", "manufacturer"]


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def main() -> int:
    df = pd.read_csv(ROOT / "results/v5_final_runs.csv")
    df = df.drop_duplicates(["system", "scenario", "seed"], keep="last")
    df = df[df.system.isin(ORDER)]
    OUT.mkdir(parents=True, exist_ok=True)
    n_llm = df[df.system.isin(LLM)].groupby("system").size().to_dict()

    # 1. Summary per system (means over all scenarios x seeds)
    summary = []
    for i, s in enumerate(ORDER):
        g = df[df.system == s]
        if g.empty:
            continue
        row = {
            "system": LABEL[s], "rank": i + 1, "kind": "LLM" if s in LLM else ("Reference" if s.startswith("ref") else "Traditional"),
            "runs": int(len(g)),
            "total_cost": r(g.total_cost.mean(), 0), "holding_cost": r(g.holding_cost.mean(), 0),
            "backlog_cost": r(g.backlog_cost.mean(), 0),
            "fill_rate": r(g.fill_rate.mean() / 100, 4), "e2e_bullwhip": r(g.e2e_bullwhip.mean(), 2),
            **{f"bw_{e}": r(g[f"bw_{e}"].mean(), 2) for e in ECH},
            "stockout_steps_retailer": r(g.stockout_steps_retailer.mean(), 1),
            "stockout_steps_total": r(g.stockout_steps_total.mean(), 1),
            "latency_ms": r(g.llm_latency_ms.mean() if s in LLM else g.latency_ms.mean(), 1),
        }
        if s in LLM:
            row.update({
                "instruction_following": r(g.instruction_following.mean()),
                "groundedness": r(g.groundedness.mean()),
                "rationales_with_numbers": r(g.rationales_with_numbers.mean()),
                "fallback_rate": r(g.fallback_rate.mean()), "error_rate": r(g.error_rate.mean()),
                "tokens_in": r(g.tokens_in.mean(), 0), "tokens_out": r(g.tokens_out.mean(), 0),
                "usd_per_run": r(g.usd.mean(), 4), "usd_total": r(g.usd.sum(), 4),
                "regime_accuracy": r(g.regime_accuracy.mean()), "clip_rate": r(g.clip_rate.mean()),
                "invalid_runs": int(g.invalid_reason.notna().sum()),
                "tool_choice_accuracy": "n/a: tools are deterministic in v5",
                "parameter_accuracy": "n/a: tools are deterministic in v5",
                "negotiation": "n/a: 1 orchestrator call, 0 rounds" if s == "S6 orchestrated" else "n/a: no negotiation",
            })
        summary.append(row)

    # 2. By scenario
    by_sc = (df.groupby(["system", "scenario"]).agg(total_cost=("total_cost", "mean"), fill_rate=("fill_rate", "mean"),
                                                   e2e_bullwhip=("e2e_bullwhip", "mean"),
                                                   stockout_steps_retailer=("stockout_steps_retailer", "mean"),
                                                   runs=("seed", "count")).reset_index())
    by_sc["fill_rate"] /= 100
    by_sc["system"] = by_sc.system.map(LABEL)
    by_sc["key"] = by_sc.system + "|" + by_sc.scenario
    by_sc = by_sc.round({"total_cost": 0, "fill_rate": 4, "e2e_bullwhip": 2, "stockout_steps_retailer": 1})

    # 3. Action mix (LLM systems), from the decision log
    mix = []
    db = ROOT / "results/v5_final.duckdb"
    if db.exists():
        con = duckdb.connect(str(db), read_only=True)
        q = con.execute("select system, action, count(*) n from v5_decisions group by 1, 2").df()
        con.close()
        sys_map = {"single_agent_v5": "S3 single agent", "multi_agent_info_v5": "S4 multi-agent info",
                   "multi_agent_tools_v5": "S5 multi-agent tools", "orchestrated_v5": "S6 orchestrated"}
        q["system"] = q.system.map(sys_map).map(LABEL)
        tot = q.groupby("system").n.transform("sum")
        q["share"] = (q.n / tot).round(4)
        mix = q[["system", "action", "share"]].to_dict("records")

    # 4. Confirmatory tests
    tests = []
    piv = {m: df.pivot_table(index=["scenario", "seed"], columns="system", values=m) for m in ("total_cost", "e2e_bullwhip")}
    for sc in ("stable", "seasonal", "shock"):
        for metric, p in piv.items():
            rows = []
            for base in ("S3 single agent", "S2 EOQ tuned", "ref 2c tuned"):
                if "S6 orchestrated" not in p or base not in p:
                    continue
                d = (p.loc[sc]["S6 orchestrated"] - p.loc[sc][base]).dropna()
                if len(d) < 6:
                    pval = None
                else:
                    pval = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                rows.append({"scenario": sc, "metric": metric, "comparison": f"S6 vs {LABEL[base]}", "n": int(len(d)),
                             "mean_diff": r(d.mean(), 2), "median_diff": r(d.median(), 2),
                             "s6_better_seeds": int((d < 0).sum()), "p_raw": pval})
            # Holm across the three comparisons
            valid = sorted([x for x in rows if x["p_raw"] is not None], key=lambda x: x["p_raw"])
            m = len(valid)
            running = 0.0
            for k, x in enumerate(valid):
                running = max(running, min(1.0, (m - k) * x["p_raw"]))
                x["p_holm"] = r(running, 4)
            for x in rows:
                x.setdefault("p_holm", None)
                x["p_raw"] = r(x["p_raw"], 4)
                x["verdict"] = ("n < 6: descriptive only" if x["p_holm"] is None else
                                ("S6 better" if x["p_holm"] < 0.05 and x["mean_diff"] < 0 else
                                 "S6 worse" if x["p_holm"] < 0.05 else "no significant difference"))
                x["key"] = f"{sc}|{metric}|{x['comparison']}"
            tests.extend(rows)

    for name, data in (("summary", summary), ("by_scenario", by_sc.to_dict("records")), ("action_mix", mix),
                       ("tests", tests)):
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}: LLM runs per system {n_llm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
