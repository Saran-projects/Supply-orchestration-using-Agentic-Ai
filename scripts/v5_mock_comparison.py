"""Design v5: all four systems with MockLLM (no network) on tuning seeds 100-102, next to scripted
policies, System 2c and tuned EOQ. Prints validity, action mix, clip rate, regime accuracy, cost,
fill rate and end-to-end bullwhip (means over 3 scenarios x 3 seeds).
"""

from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

logging.disable(logging.WARNING)
os.environ.pop("ALLOW_PAID_CALLS", None)  # mock only

from sca.config import find_project_root, get_config  # noqa: E402
from sca.data.scenarios import generate_scenarios  # noqa: E402
from sca.eval.validity import check_run_validity  # noqa: E402
from sca.sim.environment import BeerGameEnv  # noqa: E402
from sca.sim.metrics import compute_simulation_metrics  # noqa: E402
from sca.systems.base import get_system  # noqa: E402
from sca.systems.v5 import ACTIONS, V5_SYSTEM_IDS, summarize_decisions  # noqa: E402

SEEDS = [100, 101, 102]
SCENARIOS = ["stable", "seasonal", "shock"]
MOCK_MODES = ["valid", "always_follow", "detector_copy", "malformed", "bad_action"]
ROOT = find_project_root()
SC = generate_scenarios()


def simulate(system, scenario, seed):
    sc = SC[scenario]
    system.calibrate(sc)
    system.reset(sc)
    env = BeerGameEnv(config=get_config().sim, scenario=sc, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)
    if hasattr(system, "attach_env"):
        system.attach_env(env)
    for _ in range(40):
        states = env.step(system.decide_orders(states))
    return compute_simulation_metrics(env, warmup_steps=5)


def row(label, system, m):
    out = {"policy": label, "cost": m.total_cost, "fill_rate": 100 * m.fill_rate["retailer"],
           "e2e_bw": m.end_to_end_bullwhip}
    if hasattr(system, "decision_log"):
        s = summarize_decisions(system.decision_log)
        out.update({f"mix_{a}": s["action_mix"][a] for a in (*ACTIONS, "fallback")})
        out.update(clip_rate=s["clip_rate"], regime_acc=s["regime_accuracy"],
                   invalid=check_run_validity(system.llm_stats()) is not None)
    return out


def main() -> int:
    frozen = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))["tuned_policies"]
    rows = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for sid in V5_SYSTEM_IDS:
                for mode in MOCK_MODES:
                    system = get_system(sid, config={"db_path": False, "seed": seed, "mock_v5_mode": mode})
                    rows.append(row(f"{sid} [mock:{mode}]", system, simulate(system, scenario, seed)))
            for sid, fam in (("single_agent_v5", "basic"), ("multi_agent_tools_v5", "tool")):
                for chooser in ("follow", "detector_map", "oracle"):
                    system = get_system(sid, config={"db_path": False, "chooser": chooser})
                    rows.append(row(f"scripted {chooser} [{fam}]", system, simulate(system, scenario, seed)))
            for label, sid, key in (("System 2c tuned", "baseline_2c", "system_2c_adaptive_smoothed_out"),
                                    ("EOQ tuned", "baseline_eoq", "system_2_eoq")):
                params = {**frozen[key]["parameters"], "db_path": False}
                system = get_system(sid, config=params)
                system.calibrate(SC[scenario], config=params)
                rows.append(row(label, system, simulate_baseline(system, scenario, seed)))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/v5_mock_comparison_seeds100_102.csv", index=False)
    agg = {c: "mean" for c in df.columns if c not in ("policy", "invalid")}
    if "invalid" in df:
        agg["invalid"] = lambda x: f"{int(np.nansum(x.astype(float)))}/{x.notna().sum()}" if x.notna().any() else "-"
    out = df.groupby("policy", sort=False).agg(agg)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["invalid", "mix_follow", "mix_boost_small", "mix_boost_large", "mix_trim", "mix_fallback",
            "clip_rate", "regime_acc", "cost", "fill_rate", "e2e_bw"]
    print(out[cols].round(3).to_string())
    return 0


def simulate_baseline(system, scenario, seed):
    sc = SC[scenario]
    system.reset(sc)
    env = BeerGameEnv(config=get_config().sim, scenario=sc, seed=seed, steady_state_init=True)
    states = env.reset(seed=seed)
    for _ in range(40):
        states = env.step(system.decide_orders(states))
    return compute_simulation_metrics(env, warmup_steps=5)


if __name__ == "__main__":
    sys.exit(main())
