"""Benchmark Manifest and Cryptographic Configuration Hashing.

Computes a canonical SHA-256 hash over:
- simulation environment parameters
- data preprocessing & series selection parameters
- scenario generation settings
- baseline parameters (EOQ, (s, S), System 2b, System 0, Oracle)
- seeds (tuning seeds 100..104, test seeds 0..7)
- horizon and warmup steps
- cost parameters (holding cost, backlog cost, backlog multiplier)

Provides manifest generation, persistence to `results/benchmark_manifest.json`,
and validation against unauthorized configuration drift for real-LLM runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sca.config import AppConfig, get_config

logger = logging.getLogger("sca.eval.manifest")

DEFAULT_MANIFEST_PATH = Path("results/benchmark_manifest.json")
DEFAULT_GIT_TAG = "benchmark-v2-1-frozen"


def get_git_commit_hash() -> str:
    """Retrieve current Git HEAD commit hash, or return fallback."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception as e:
        logger.warning("Could not determine git commit: %s", e)
        return "unknown_commit"


def get_canonical_benchmark_config(app_cfg: Optional[AppConfig] = None) -> Dict[str, Any]:
    """Extract canonical benchmark configuration dictionary across all components."""
    cfg = app_cfg or get_config()
    sim = cfg.sim.four_echelon
    data = cfg.data

    return {
        "sim": {
            "initial_inventory": float(sim.initial_inventory),
            "initial_backlog": float(sim.initial_backlog),
            "manufacturer_capacity": float(sim.manufacturer_capacity),
            "default_lead_time": int(sim.default_lead_time),
            "late_delivery_delay": int(sim.late_delivery_delay),
            "steady_state_init": bool(sim.steady_state_init),
            "disruption_probability_per_step": float(cfg.sim.disruption_probability_per_step),
            "disruption_duration_days_max": int(cfg.sim.disruption_duration_days_max),
        },
        "data": {
            "category": data.category,
            "granularity": data.granularity,
            "market": data.market,
            "rolling_window": int(data.rolling_window),
            "dataset_filename": data.dataset_filename,
            "train_split": float(data.train_split),
            "val_split": float(data.val_split),
            "test_split": float(data.test_split),
            "random_seed": int(data.random_seed),
        },
        "scenario": {
            "scenario_window_length": int(data.scenario_window_length),
            "min_train_history": int(data.min_train_history),
            "target_mean_demand": float(data.target_mean_demand),
            "dominant_period": int(data.dominant_period),
            "shock_level_shift_enabled": bool(data.shock_level_shift_enabled),
            "shock_level_shift_multiplier": float(data.shock_level_shift_multiplier),
            "shock_level_shift_duration": int(data.shock_level_shift_duration),
            "shock_level_shift_start": int(data.shock_level_shift_start),
            "shock_provenance_label": str(data.shock_provenance_label),
            "scenarios": ["stable", "seasonal", "shock"],
        },
        "baseline_parameters": {
            "system_0_constant": {"source": "training_period_mean"},
            "system_1_ss_untuned": {"s_factor": 1.0, "S_factor": 1.0},
            "system_1_ss_tuned": {
                "s_factor": float(sim.tuned_ss_s_factor),
                "S_factor": float(sim.tuned_ss_S_factor),
            },
            "system_2_eoq_untuned": {
                "order_quantity_multiplier": 1.0,
                "reorder_point_factor": 1.0,
            },
            "system_2_eoq_tuned": {
                "order_quantity_multiplier": float(sim.tuned_eoq_q_mult),
                "reorder_point_factor": float(sim.tuned_eoq_rop_factor),
            },
            "system_2b_smoothed_out_untuned": {
                "alpha": 0.8,
                "beta": 0.8,
                "z": 2.0,
                "delta": 0.5,
                "forecast_cap_multiple": 1.5,
            },
            "system_2b_smoothed_out_tuned": {
                "alpha": float(sim.alpha),
                "beta": float(sim.beta),
                "z": float(sim.safety_factor_z),
                "delta": float(sim.rate_limit_delta),
                "forecast_cap_multiple": float(sim.forecast_cap_multiple),
            },
            "oracle": {
                "policy": "clairvoyant_base_stock",
                "lead_time_offset": 4,
            },
        },
        "seeds": {
            "tuning_seeds": [100, 101, 102, 103, 104],
            "test_seeds": [0, 1, 2, 3, 4, 5, 6, 7],
        },
        "horizon": {
            "horizon_steps": int(sim.horizon_steps),
            "warmup_steps": int(sim.warmup_steps),
        },
        "cost_parameters": {
            "holding_cost_per_unit": float(sim.holding_cost_per_unit),
            "backlog_cost_multiplier": float(sim.backlog_cost_multiplier),
            "backlog_cost_per_unit": float(sim.backlog_cost_per_unit),
        },
    }


def compute_benchmark_config_hash(app_cfg: Optional[AppConfig] = None) -> Tuple[str, Dict[str, Any]]:
    """Compute deterministic SHA-256 hash of canonical benchmark configuration."""
    canonical_dict = get_canonical_benchmark_config(app_cfg)
    canonical_json = json.dumps(canonical_dict, sort_keys=True, indent=2)
    config_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return config_hash, canonical_dict


def write_benchmark_manifest(
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    git_commit: Optional[str] = None,
    git_tag: str = DEFAULT_GIT_TAG,
    app_cfg: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    """Generate and write the frozen benchmark manifest JSON file."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_hash, canonical_dict = compute_benchmark_config_hash(app_cfg)
    commit = git_commit or get_git_commit_hash()

    manifest_data = {
        "benchmark_version": "2.1",
        "benchmark_tag": git_tag,
        "config_hash": config_hash,
        "git_commit": commit,
        "canonical_config": canonical_dict,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=True)

    logger.info("Saved benchmark manifest to %s with config_hash=%s", path, config_hash)
    return manifest_data


def load_benchmark_manifest(manifest_path: Path | str = DEFAULT_MANIFEST_PATH) -> Optional[Dict[str, Any]]:
    """Load benchmark manifest from disk if it exists."""
    path = Path(manifest_path)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_config_matches_manifest(
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    app_cfg: Optional[AppConfig] = None,
) -> Tuple[bool, str, Optional[str]]:
    """Verify whether current configuration matches the frozen benchmark manifest.

    Returns:
        (is_match, current_hash, manifest_hash)
    """
    current_hash, _ = compute_benchmark_config_hash(app_cfg)
    manifest = load_benchmark_manifest(manifest_path)
    if manifest is None:
        return False, current_hash, None
    manifest_hash = manifest.get("config_hash")
    return (current_hash == manifest_hash), current_hash, manifest_hash
