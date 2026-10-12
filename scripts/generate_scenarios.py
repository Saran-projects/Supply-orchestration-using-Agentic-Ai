#!/usr/bin/env python3
"""CLI utility to generate, inspect, and plot simulation scenarios."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sca.config import get_config
from sca.data.scenarios import generate_scenarios, plot_all_scenarios


def main():
    parser = argparse.ArgumentParser(description="Generate simulation scenarios from DataCo series")
    parser.add_argument("--category", type=str, default=None, help="Category name (e.g. Cleats)")
    parser.add_argument("--market", type=str, default=None, help="Market name (e.g. Europe, LATAM)")
    parser.add_argument("--window", type=int, default=None, help="Window length in steps")
    parser.add_argument("--target-mean", type=float, default=None, help="Target mean demand")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plot figures")
    args = parser.parse_args()

    cfg = get_config().data
    if args.category:
        cfg.category = args.category
    if args.market:
        cfg.market = args.market
    if args.window:
        cfg.scenario_window_length = args.window
    if args.target_mean:
        cfg.target_mean_demand = args.target_mean

    print(f"[+] Generating scenarios for category='{cfg.category}', market='{cfg.market or 'ALL'}'...")
    scenarios = generate_scenarios(data_config=cfg)

    print("\n--- Generated Scenarios ---")
    for name, sc in scenarios.items():
        print(f"\n[{name.upper()}]")
        print(f"  - Dates:         {sc.dates[0].strftime('%Y-%m-%d')} to {sc.dates[-1].strftime('%Y-%m-%d')} ({sc.horizon} days)")
        print(f"  - Scaled Mean:   {sc.mean_demand:.2f} (Target: {cfg.target_mean_demand})")
        print(f"  - Scaled Std:    {sc.std_demand:.2f}")
        print(f"  - CV:            {sc.cv:.4f}")
        print(f"  - Scale Factor:  x{sc.scale_factor:.4f} (Raw Mean: {sc.metadata['raw_mean']:.2f})")
        print(f"  - Train Steps:   {len(sc.train_demand)} days (Max train: {sc.train_dates[-1].strftime('%Y-%m-%d')})")
        print(f"  - Lead Time:     {sc.lead_time_sampler.mean:.2f} days (Empirical distribution fitted)")
        print(f"  - Late Delivery: {sc.late_delivery_probability*100:.2f}%")
        if "acf_dominant" in sc.metadata:
            print(f"  - Weekly ACF:    {sc.metadata['acf_dominant']:.4f}")
        if "max_spike_ratio" in sc.metadata:
            print(f"  - Peak Spike:    {sc.metadata['max_spike_ratio']:.2f}x rolling median")

    if not args.no_plot:
        print(f"\n[+] Saving scenario plots to: {cfg.scenarios_output_dir}")
        plots = plot_all_scenarios(scenarios, output_dir=cfg.scenarios_output_dir)
        for k, p in plots.items():
            print(f"  - Plot [{k}]: {p}")


if __name__ == "__main__":
    main()
