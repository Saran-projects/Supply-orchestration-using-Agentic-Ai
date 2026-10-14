"""Offline System 6 design study (no LLM, no network), tuning seeds 100-104 only.

Question: can any S5/S6 design change make bullwhip <= tuned EOQ while keeping fill rate >= 99% and
cost <= 1.15 x tuned EOQ (protocol section 3)? Each candidate varies band width (delta), forecast cap,
and the reference's order smoothing (beta of the smoothed order-up-to tool), and is scored with
scripted choosers (follow, detector_map, oracle) as a proxy for what an LLM choosing actions could do.
Pooled over 3 scenarios x 5 seeds per candidate (as in docs/tuning_rule_v2_1.md).
"""

from __future__ import annotations

import itertools
import json
import logging
import sys

import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

from sca.config import find_project_root, get_config, load_agent_v5_config  # noqa: E402
from sca.data.scenarios import generate_scenarios  # noqa: E402
from sca.sim.environment import BeerGameEnv  # noqa: E402
from sca.sim.metrics import compute_simulation_metrics  # noqa: E402
from sca.systems.base import get_system  # noqa: E402

SEEDS = [100, 101, 102, 103, 104]
SCENARIOS = ["stable", "seasonal", "shock"]
ROOT = find_project_root()
SC = generate_scenarios()
GRID = {
    "delta": [0.1, 0.2, 0.3, 0.5, 0.8],
    "forecast_cap_multiple": [1.2, 1.5],
    "beta": [0.2, 0.4, 0.6, 0.95],
}
CHOOSERS = ["follow", "detector_map", "oracle"]


def run(chooser, delta, cap, beta, scenario, seed):
    base = load_agent_v5_config()
    band = base.band.model_copy(update={"delta": delta, "delta_shock": delta, "forecast_cap_multiple": cap})
    cfg = base.model_copy(update={"band": band})
    system = get_system("orchestrated_v5", v5_config=cfg,
                        config={"db_path": False, "chooser": chooser, "tool_params_override": {"beta": beta}})
    sc = SC[scenario]
    system.calibrate(sc)
    system.reset(sc)
    env = BeerGameEnv(config=get_config().sim, scenario=sc, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)
    system.attach_env(env)
    for _ in range(40):
        states = env.step(system.decide_orders(states))
    m = compute_simulation_metrics(env, warmup_steps=5)
    return m.total_cost, 100 * m.fill_rate["retailer"], m.end_to_end_bullwhip, np.mean([r["clipped"] for r in system.decision_log])


def main() -> int:
    frozen = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))["tuned_policies"]
    eoq = frozen["system_2_eoq"]["metrics"]
    s2c = frozen["system_2c_adaptive_smoothed_out"]["metrics"]
    cost_cap = 1.15 * eoq["mean_cost"]
    print(f"Targets (tuning seeds 100-104): EOQ cost {eoq['mean_cost']:.0f}, BW {eoq['mean_e2e_bullwhip']:.2f}; "
          f"2c cost {s2c['mean_cost']:.0f}, BW {s2c['mean_e2e_bullwhip']:.2f}; cost cap {cost_cap:.0f}")
    rows = []
    for delta, cap, beta in itertools.product(GRID["delta"], GRID["forecast_cap_multiple"], GRID["beta"]):
        for chooser in CHOOSERS:
            res = np.array([run(chooser, delta, cap, beta, sc, s) for sc in SCENARIOS for s in SEEDS])
            rows.append({"delta": delta, "cap": cap, "beta": beta, "chooser": chooser,
                         "cost": res[:, 0].mean(), "fill_rate": res[:, 1].mean(), "min_fill_rate": res[:, 1].min(),
                         "e2e_bw": res[:, 2].mean(), "clip_rate": res[:, 3].mean()})
    df = pd.DataFrame(rows)
    df["meets_all"] = (df.fill_rate >= 99) & (df.cost <= cost_cap) & (df.e2e_bw <= eoq["mean_e2e_bullwhip"])
    df.to_csv(ROOT / "results/v5_s6_design_study.csv", index=False)
    pd.set_option("display.width", 200)
    cur = df[(df.delta == 0.8) & (df.cap == 1.2) & (df.beta == 0.95)]
    print("\nCurrent design (delta 0.8, cap 1.2, beta 0.95):\n" + cur.round(3).to_string(index=False))
    ok = df[df.meets_all]
    print(f"\nCandidates meeting FR>=99, cost<=cap, BW<=EOQ: {len(ok)} of {len(df)}")
    print(ok.sort_values("e2e_bw").head(15).round(3).to_string(index=False))
    print("\nLowest bullwhip with FR>=99 per chooser:")
    fr = df[df.fill_rate >= 99]
    print(fr.loc[fr.groupby("chooser").e2e_bw.idxmin()].round(3).to_string(index=False))
    print("\nLLM headroom (detector_map/oracle minus follow, same design), mean over designs with FR>=99:")
    piv = df.pivot_table(index=["delta", "cap", "beta"], columns="chooser", values=["e2e_bw", "cost"])
    for ch in ("detector_map", "oracle"):
        d_bw = (piv[("e2e_bw", ch)] - piv[("e2e_bw", "follow")]).mean()
        d_cost = (piv[("cost", ch)] - piv[("cost", "follow")]).mean()
        print(f"  {ch}: mean dBW {d_bw:+.3f}, mean dCost {d_cost:+.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
