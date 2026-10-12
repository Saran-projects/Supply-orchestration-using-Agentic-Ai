"""Scenario generation from real DataCo demand windows: stable, seasonal, and shock.

Provides:
- LeadTimeSampler: Fitted empirical distribution of shipping lead times.
- Scenario: Container for simulation demand, training history, lead-time sampler, and metrics.
- generate_scenarios: Extracts stable, seasonal, and shock windows without data leakage.
- plot_scenario & plot_all_scenarios: Visualizations of scenario windows and training splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sca.config import DataConfig, get_config
from sca.data.aggregation import extract_single_series

logger = logging.getLogger(__name__)

# Default empirical parameters fitted from full DataCo dataset (180,519 records)
DATACO_LEAD_TIME_DAYS = np.array([0, 1, 2, 3, 4, 5, 6])
DATACO_LEAD_TIME_PROBS = np.array([
    0.028141,  # 0 days (Same Day deliveries)
    0.025798,  # 1 day (Same Day next-day deliveries)
    0.313640,  # 2 days (First Class fixed 2d + Second/Standard)
    0.159346,  # 3 days
    0.157950,  # 4 days
    0.156011,  # 5 days
    0.159113,  # 6 days
])
DATACO_LATE_DELIVERY_RATE = 0.54829


class LeadTimeSampler:
    """Empirical lead-time sampler fitted from DataCo shipping distributions."""

    def __init__(
        self,
        days: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        name: str = "DataCo_Empirical_LeadTime",
    ):
        self.days = np.array(days if days is not None else DATACO_LEAD_TIME_DAYS, dtype=int)
        raw_probs = np.array(probabilities if probabilities is not None else DATACO_LEAD_TIME_PROBS, dtype=float)
        self.probabilities = raw_probs / raw_probs.sum()  # Normalize
        self.name = name
        self.rng = np.random.default_rng(seed)

    def sample(self, size: int = 1) -> np.ndarray:
        """Sample discrete lead time in days according to empirical probability distribution."""
        samples = self.rng.choice(self.days, size=size, p=self.probabilities)
        return samples if size > 1 else int(samples[0])

    def __call__(self, size: int = 1) -> Union[int, np.ndarray]:
        return self.sample(size=size)

    @property
    def mean(self) -> float:
        return float(np.sum(self.days * self.probabilities))

    @property
    def std(self) -> float:
        var = np.sum(((self.days - self.mean) ** 2) * self.probabilities)
        return float(np.sqrt(var))

    @property
    def distribution(self) -> Dict[int, float]:
        return {int(d): float(p) for d, p in zip(self.days, self.probabilities, strict=True)}

    @classmethod
    def fit(
        cls,
        df: Optional[pd.DataFrame] = None,
        shipping_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> "LeadTimeSampler":
        """Fit lead-time empirical distribution directly from DataCo DataFrame."""
        if df is None:
            return cls(seed=seed)

        sub = df
        if shipping_mode and "Shipping Mode" in df.columns:
            sub = df[df["Shipping Mode"] == shipping_mode]

        col = "Days for shipping (real)"
        if col in sub.columns:
            counts = sub[col].value_counts().sort_index()
            days = counts.index.to_numpy(dtype=int)
            probs = counts.to_numpy(dtype=float) / counts.sum()
            name = f"DataCo_{shipping_mode or 'AllModes'}_LeadTime"
            return cls(days=days, probabilities=probs, seed=seed, name=name)

        return cls(seed=seed)

    def __repr__(self) -> str:
        return f"LeadTimeSampler(name='{self.name}', mean={self.mean:.2f}d, std={self.std:.2f}d, days={list(self.days)})"


@dataclass
class Scenario:
    """Simulation scenario container with demand array, fitted samplers, and train/test split."""

    name: str                                # "stable" | "seasonal" | "shock"
    demand: np.ndarray                       # Scaled demand array for simulator
    raw_demand: np.ndarray                   # Original unscaled demand values
    dates: List[pd.Timestamp]                # Dates in the scenario window
    train_demand: np.ndarray                 # Scaled training history strictly BEFORE window
    train_raw_demand: np.ndarray             # Raw training history before window
    train_dates: List[pd.Timestamp]          # Training dates strictly BEFORE window
    lead_time_sampler: LeadTimeSampler       # Callable sampler for lead times
    late_delivery_probability: float         # Fitted late delivery probability
    scale_factor: float                      # Global scale factor from training period ONLY
    fitted_trend_seasonal: Optional[np.ndarray] = None  # Fixed trend + seasonal component
    residuals: Optional[np.ndarray] = None   # Residuals for circular block bootstrap
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Strict validation: verify no data leakage between train and test/scenario sets
        if len(self.train_dates) > 0 and len(self.dates) > 0:
            max_train_date = max(self.train_dates)
            min_scenario_date = min(self.dates)
            if max_train_date >= min_scenario_date:
                raise ValueError(
                    f"Data leakage detected! Max train date ({max_train_date}) >= "
                    f"Min scenario date ({min_scenario_date})"
                )
            train_set = set(self.train_dates)
            scenario_set = set(self.dates)
            if not train_set.isdisjoint(scenario_set):
                raise ValueError("Data leakage detected! Overlapping timestamps between train and scenario sets.")

        if self.fitted_trend_seasonal is None or self.residuals is None:
            # Auto-decompose demand into trend/seasonal + residuals if not provided
            y = np.array(self.demand, dtype=float)
            p = 7 if len(y) >= 7 else max(1, len(y))
            if p > 1:
                means_dow = np.array([np.mean([y[i] for i in range(len(y)) if i % p == d]) for d in range(p)])
                s_comp = np.array([means_dow[i % p] - np.mean(means_dow) for i in range(len(y))])
                deseas = y - s_comp
                t = np.arange(len(y))
                slope, intercept = np.polyfit(t, deseas, 1)
                self.fitted_trend_seasonal = intercept + slope * t + s_comp
                self.residuals = y - self.fitted_trend_seasonal
            else:
                self.fitted_trend_seasonal = y.copy()
                self.residuals = np.zeros_like(y)

    @property
    def horizon(self) -> int:
        return len(self.demand)

    @property
    def mean_demand(self) -> float:
        return float(np.mean(self.demand))

    @property
    def std_demand(self) -> float:
        return float(np.std(self.demand))

    @property
    def cv(self) -> float:
        return float(self.std_demand / (self.mean_demand + 1e-9))

    def get_demand_for_seed(self, seed: int, block_size: int = 7) -> np.ndarray:
        """Generate an independent, reproducible demand trajectory for a given seed via circular block bootstrap of residuals.

        Preserves the fixed trend/seasonal component from the real window plus block-bootstrapped residuals,
        while maintaining independent sample variation across seeds.
        Keeps tuning paths (seeds 100 to 104) disjoint from test paths (seeds 0 to 7).
        """
        rng = np.random.default_rng(seed)
        n = len(self.demand)
        if self.residuals is None or np.allclose(self.residuals, 0.0):
            return self.demand.copy()

        num_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, n, size=num_blocks)
        blocks = [self.residuals.take(range(s, s + block_size), mode="wrap") for s in starts]
        boot_res = np.concatenate(blocks)[:n]
        reconstructed = self.fitted_trend_seasonal + boot_res
        return np.round(np.maximum(0.0, reconstructed), 2)

    def summary(self) -> str:
        prov = self.metadata.get("provenance", "real")
        return (
            f"Scenario[{self.name.upper()}]: horizon={self.horizon} days, "
            f"scaled_mean={self.mean_demand:.1f} (global scale={self.scale_factor:.3f}), "
            f"raw_mean={np.mean(self.raw_demand):.1f}, CV={self.cv:.3f}, "
            f"provenance={prov}, "
            f"window=[{self.dates[0].strftime('%Y-%m-%d')} -> {self.dates[-1].strftime('%Y-%m-%d')}], "
            f"train_len={len(self.train_demand)} days"
        )


def load_chosen_series(data_config: Optional[DataConfig] = None) -> Tuple[pd.Series, DataConfig]:
    """Load the processed demand time-series chosen in configuration."""
    cfg = data_config or get_config().data

    # Select daily or weekly parquet
    parquet_filename = (
        cfg.processed_weekly_filename
        if cfg.granularity.lower() == "weekly"
        else cfg.processed_daily_filename
    )
    parquet_path = cfg.processed_data_dir / parquet_filename

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed demand parquet not found at: {parquet_path.resolve()}. "
            "Please run 'make data' or 'python scripts/process_data.py' first."
        )

    df_agg = pd.read_parquet(parquet_path)

    # Determine category to use: if null in config, pick highest total volume category
    category = cfg.category
    if category is None or category not in df_agg["category"].values:
        top_cat = (
            df_agg.groupby("category")["demand"]
            .sum()
            .idxmax()
        )
        logger.info("Category not specified in config; falling back to top category: '%s'", top_cat)
        category = top_cat

    market = cfg.market
    freq = "W-MON" if cfg.granularity.lower() == "weekly" else "D"

    # Extract continuous single series
    series_df = extract_single_series(
        df_agg=df_agg,
        category=category,
        market=market,
        fill_zeros=True,
        freq=freq,
    )

    if series_df.empty:
        raise ValueError(f"No series data found for category='{category}' and market='{market}'.")

    series = series_df.set_index("date")["demand"]
    logger.info(
        "Loaded series '%s' (market=%s, granularity=%s): %d steps from %s to %s",
        category,
        market or "ALL",
        cfg.granularity,
        len(series),
        series.index.min().strftime("%Y-%m-%d"),
        series.index.max().strftime("%Y-%m-%d"),
    )
    return series, cfg


def find_scenario_windows(
    series: pd.Series,
    window_length: int = 90,
    min_train_history: int = 90,
    dominant_period: int = 7,
    min_window_separation: int = 30,
) -> Dict[str, Tuple[int, int, Dict[str, Any]]]:
    """Identify stable, seasonal, and shock windows from the series.

    Args:
        series: Continuous demand series with DatetimeIndex.
        window_length: Length of evaluation window in steps.
        min_train_history: Minimum training steps before window start.
        dominant_period: Lag period for autocorrelation calculation.
        min_window_separation: Minimum step separation between window starts
                               to ensure distinct regimes (set 0 for unconstrained).

    Returns:
        Dict mapping scenario name to (start_idx, end_idx, metrics_dict).
    """
    values = series.values.astype(float)
    dates = series.index
    n = len(values)

    if n < min_train_history + window_length:
        raise ValueError(
            f"Series length ({n}) is too short for min_train_history ({min_train_history}) "
            f"+ window_length ({window_length})."
        )

    # Rolling median for shock ratio (14-day window centered)
    rolling_med = pd.Series(values).rolling(14, min_periods=3, center=True).median()
    rolling_med = rolling_med.fillna(np.median(values)).values
    spike_ratios = values / np.maximum(rolling_med, 1.0)

    candidate_stats = []

    # Slide window
    for s in range(min_train_history, n - window_length + 1):
        e = s + window_length
        w = values[s:e]
        mean_val = float(np.mean(w))
        std_val = float(np.std(w))
        cv_val = float(std_val / (mean_val + 1e-9)) if mean_val > 0 else 999.0

        # Autocorrelation at dominant period (e.g. lag 7)
        if std_val > 1e-6 and len(w) > dominant_period:
            centered = w - mean_val
            denom = float(np.sum(centered**2))
            numer = float(np.sum(centered[:-dominant_period] * centered[dominant_period:]))
            acf_val = float(numer / denom) if denom > 0 else 0.0
        else:
            acf_val = 0.0

        max_spike = float(np.max(spike_ratios[s:e]))
        spike_pos = s + int(np.argmax(spike_ratios[s:e]))

        candidate_stats.append({
            "start_idx": s,
            "end_idx": e,
            "mean": mean_val,
            "std": std_val,
            "cv": cv_val,
            "acf": acf_val,
            "max_spike_ratio": max_spike,
            "spike_idx": spike_pos,
        })

    # 1. Stable: Lowest Coefficient of Variation (CV)
    valid_cv = [c for c in candidate_stats if c["mean"] > 10.0]  # Avoid degenerate near-zero windows
    best_stable = min(valid_cv or candidate_stats, key=lambda x: x["cv"])
    selected_starts = [best_stable["start_idx"]]

    # 2. Seasonal: Highest Autocorrelation at Dominant Period (separated from stable if requested)
    def _is_separated(start_idx: int) -> bool:
        return all(abs(start_idx - prev) >= min_window_separation for prev in selected_starts)

    seasonal_candidates = [c for c in candidate_stats if _is_separated(c["start_idx"])]
    best_seasonal = max(seasonal_candidates or candidate_stats, key=lambda x: x["acf"])
    selected_starts.append(best_seasonal["start_idx"])

    # 3. Shock: Contains the maximum demand spike ratio vs rolling median
    shock_candidates = [c for c in candidate_stats if _is_separated(c["start_idx"])]
    best_shock = max(shock_candidates or candidate_stats, key=lambda x: x["max_spike_ratio"])

    results = {
        "stable": (
            best_stable["start_idx"],
            best_stable["end_idx"],
            {
                "criterion": "lowest_coefficient_of_variation",
                "cv": round(best_stable["cv"], 4),
                "mean": round(best_stable["mean"], 2),
                "std": round(best_stable["std"], 2),
                "acf_dominant": round(best_stable["acf"], 4),
                "max_spike_ratio": round(best_stable["max_spike_ratio"], 2),
            },
        ),
        "seasonal": (
            best_seasonal["start_idx"],
            best_seasonal["end_idx"],
            {
                "criterion": "strongest_autocorrelation_at_dominant_period",
                "dominant_period": dominant_period,
                "acf_dominant": round(best_seasonal["acf"], 4),
                "cv": round(best_seasonal["cv"], 4),
                "mean": round(best_seasonal["mean"], 2),
                "std": round(best_seasonal["std"], 2),
                "max_spike_ratio": round(best_seasonal["max_spike_ratio"], 2),
            },
        ),
        "shock": (
            best_shock["start_idx"],
            best_shock["end_idx"],
            {
                "criterion": "highest_demand_spike_ratio_vs_rolling_median",
                "max_spike_ratio": round(best_shock["max_spike_ratio"], 2),
                "peak_date": str(dates[best_shock["spike_idx"]].date()),
                "cv": round(best_shock["cv"], 4),
                "mean": round(best_shock["mean"], 2),
                "std": round(best_shock["std"], 2),
                "acf_dominant": round(best_shock["acf"], 4),
            },
        ),
    }

    return results


def build_scenario_object(
    name: str,
    series: pd.Series,
    start_idx: int,
    end_idx: int,
    scale_factor: float = 1.0,
    target_mean_demand: float = 100.0,
    shock_level_shift_enabled: bool = False,
    shock_level_shift_multiplier: float = 1.5,
    shock_level_shift_duration: int = 10,
    shock_level_shift_start: int = 15,
    shock_provenance_label: str = "dataco_window_plus_injected_level_shift_v2",
    lead_time_sampler: Optional[LeadTimeSampler] = None,
    late_delivery_probability: Optional[float] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Scenario:
    """Build a validated Scenario object under Scenario Generation v2 protocol.

    - Uses one global scaling factor derived strictly from the training period (no per-window renormalisation).
    - Preserves fixed trend/seasonal component from the real window and extracts residuals for bootstrap.
    - Injects level-shift option for the shock scenario with configurable multiplier and duration.
    - Records provenance label in metadata.
    """
    dates = list(series.index)
    values = series.values.astype(float)

    # Train / Test Split: Forecaster trains only on data strictly before window
    train_dates = dates[:start_idx]
    train_raw = values[:start_idx]

    scenario_dates = dates[start_idx:end_idx]
    scenario_raw = values[start_idx:end_idx].copy()

    # Injected level-shift for shock scenario if configured
    is_injected_shift = False
    provenance = "dataco_historical_real_window"
    if name.lower() == "shock" and shock_level_shift_enabled:
        horizon = len(scenario_raw)
        st = min(shock_level_shift_start, horizon - 1)
        en = min(st + shock_level_shift_duration, horizon)
        scenario_raw[st:en] = scenario_raw[st:en] * shock_level_shift_multiplier
        is_injected_shift = True
        provenance = shock_provenance_label

    # Single scaling factor from training period applied to both scenario and training history
    scaled_scenario_demand = np.round(scenario_raw * scale_factor, 2)
    scaled_train_demand = np.round(train_raw * scale_factor, 2)

    # Decompose scaled window into fixed trend/seasonal + residuals
    y = scaled_scenario_demand
    p = 7 if len(y) >= 7 else max(1, len(y))
    if p > 1:
        means_dow = np.array([np.mean([y[i] for i in range(len(y)) if i % p == d]) for d in range(p)])
        s_comp = np.array([means_dow[i % p] - np.mean(means_dow) for i in range(len(y))])
        deseas = y - s_comp
        t = np.arange(len(y))
        slope, intercept = np.polyfit(t, deseas, 1)
        fitted_trend_seasonal = intercept + slope * t + s_comp
        residuals = y - fitted_trend_seasonal
    else:
        fitted_trend_seasonal = y.copy()
        residuals = np.zeros_like(y)

    sampler = lead_time_sampler or LeadTimeSampler()
    late_prob = late_delivery_probability if late_delivery_probability is not None else DATACO_LATE_DELIVERY_RATE

    raw_window_mean = float(np.mean(scenario_raw))
    meta = {
        "start_date": scenario_dates[0].strftime("%Y-%m-%d"),
        "end_date": scenario_dates[-1].strftime("%Y-%m-%d"),
        "horizon_steps": len(scenario_dates),
        "train_steps": len(train_dates),
        "raw_mean": round(raw_window_mean, 2),
        "scaled_mean": round(float(np.mean(scaled_scenario_demand)), 2),
        "target_mean": target_mean_demand,
        "scale_factor": round(scale_factor, 4),
        "provenance": provenance,
        "is_injected_shift": is_injected_shift,
        "shock_level_shift_multiplier": shock_level_shift_multiplier if is_injected_shift else 1.0,
        "shock_level_shift_duration": shock_level_shift_duration if is_injected_shift else 0,
        "shock_level_shift_start": shock_level_shift_start if is_injected_shift else 0,
        **(extra_metadata or {}),
    }

    return Scenario(
        name=name,
        demand=scaled_scenario_demand,
        raw_demand=scenario_raw,
        dates=scenario_dates,
        train_demand=scaled_train_demand,
        train_raw_demand=train_raw,
        train_dates=train_dates,
        lead_time_sampler=sampler,
        late_delivery_probability=late_prob,
        scale_factor=scale_factor,
        fitted_trend_seasonal=fitted_trend_seasonal,
        residuals=residuals,
        metadata=meta,
    )


def generate_scenarios(
    data_config: Optional[DataConfig] = None,
    series: Optional[pd.Series] = None,
) -> Dict[str, Scenario]:
    """Generate all three scenarios: stable, seasonal, and shock under Scenario Generation v2.

    Args:
        data_config: DataConfig containing window length, scaling, and series choices.
        series: Optional preloaded demand series.

    Returns:
        Dict mapping 'stable', 'seasonal', 'shock' to Scenario objects.
    """
    if series is None:
        series, cfg = load_chosen_series(data_config)
    else:
        cfg = data_config or get_config().data

    window_len = cfg.scenario_window_length
    min_train = cfg.min_train_history
    dominant_period = cfg.dominant_period
    target_mean = cfg.target_mean_demand

    logger.info(
        "Extracting scenario windows (window_length=%d, min_train=%d, dominant_period=%d, target_mean=%.1f)",
        window_len,
        min_train,
        dominant_period,
        target_mean,
    )

    windows = find_scenario_windows(
        series=series,
        window_length=window_len,
        min_train_history=min_train,
        dominant_period=dominant_period,
    )

    # Compute ONE global scaling factor from the training period ONLY (no per-window renormalisation)
    min_train_start = min(s for s, e, _ in windows.values())
    train_slice_raw = series.iloc[:min_train_start].values.astype(float)
    train_raw_mean = float(np.mean(train_slice_raw))
    global_scale_factor = float(target_mean / train_raw_mean) if train_raw_mean > 0 else 1.0

    logger.info(
        "Global scale factor from training period (steps 0 to %d, raw_mean=%.2f): %.4f (target_mean=%.1f)",
        min_train_start,
        train_raw_mean,
        global_scale_factor,
        target_mean,
    )

    lead_sampler = LeadTimeSampler()
    scenarios = {}

    for name, (start_idx, end_idx, meta) in windows.items():
        meta["category"] = cfg.category
        meta["market"] = cfg.market or "ALL"
        meta["granularity"] = cfg.granularity
        meta["train_raw_mean"] = round(train_raw_mean, 2)

        sc = build_scenario_object(
            name=name,
            series=series,
            start_idx=start_idx,
            end_idx=end_idx,
            scale_factor=global_scale_factor,
            target_mean_demand=target_mean,
            shock_level_shift_enabled=getattr(cfg, "shock_level_shift_enabled", True),
            shock_level_shift_multiplier=getattr(cfg, "shock_level_shift_multiplier", 1.5),
            shock_level_shift_duration=getattr(cfg, "shock_level_shift_duration", 10),
            shock_level_shift_start=getattr(cfg, "shock_level_shift_start", 15),
            shock_provenance_label=getattr(cfg, "shock_provenance_label", "dataco_window_plus_injected_level_shift_v2"),
            lead_time_sampler=lead_sampler,
            late_delivery_probability=DATACO_LATE_DELIVERY_RATE,
            extra_metadata=meta,
        )
        scenarios[name] = sc
        logger.info("Generated scenario: %s", sc.summary())

    return scenarios


# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------

def plot_scenario(
    scenario: Scenario,
    output_dir: Union[str, Path] = "results/scenarios",
) -> Path:
    """Plot training history and scenario evaluation window for a single scenario."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "stable": "#16a34a",     # Emerald green
        "seasonal": "#8b5cf6",   # Purple
        "shock": "#dc2626",      # Crimson red
    }
    color = colors.get(scenario.name, "#2563eb")

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=False)

    # 1. Full Series: Train + Scenario Window
    all_train_dates = scenario.train_dates
    ax1.plot(
        all_train_dates,
        scenario.train_demand,
        color="#94a3b8",
        linewidth=1.0,
        alpha=0.7,
        label=f"Forecaster Train History ({len(all_train_dates)} steps)",
    )
    ax1.plot(
        scenario.dates,
        scenario.demand,
        color=color,
        linewidth=2.0,
        label=f"Scenario Evaluation Window ({scenario.horizon} steps)",
    )
    ax1.axvline(
        x=scenario.dates[0],
        color="#0f172a",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Train/Test Cutoff (No Leakage)",
    )
    ax1.set_title(
        f"Scenario [{scenario.name.upper()}]: Full Timeline & Train/Test Split",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("Scaled Demand (Units)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 2. Zoomed Scenario Window
    rolling_7 = pd.Series(scenario.demand).rolling(7, min_periods=1).mean()
    ax2.plot(scenario.dates, scenario.demand, color="#64748b", alpha=0.4, linewidth=1.2, label="Scaled Daily Demand")
    ax2.plot(scenario.dates, rolling_7, color=color, linewidth=2.2, label="7-Day Rolling Trend")
    ax2.axhline(
        scenario.mean_demand,
        color="#334155",
        linestyle=":",
        linewidth=1.4,
        label=f"Window Mean ({scenario.mean_demand:.1f})",
    )

    # Annotation Box with Scenario Metrics
    meta_text = (
        f"Window: {scenario.dates[0].strftime('%Y-%m-%d')} to {scenario.dates[-1].strftime('%Y-%m-%d')}\n"
        f"Horizon: {scenario.horizon} days | Scaling: x{scenario.scale_factor:.3f}\n"
        f"Scaled Mean: {scenario.mean_demand:.1f} | Scaled Std: {scenario.std_demand:.1f}\n"
        f"CV: {scenario.cv:.3f}\n"
    )
    if "acf_dominant" in scenario.metadata:
        meta_text += f"Weekly ACF (Lag 7): {scenario.metadata['acf_dominant']:.3f}\n"
    if "max_spike_ratio" in scenario.metadata:
        meta_text += f"Peak Spike Ratio: {scenario.metadata['max_spike_ratio']:.2f}x median\n"
    meta_text += f"Lead Time: {scenario.lead_time_sampler.mean:.1f}d | Late Rate: {scenario.late_delivery_probability*100:.1f}%"

    ax2.text(
        0.02,
        0.95,
        meta_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
    )

    ax2.set_title(
        f"Detailed Evaluation Window: {scenario.name.capitalize()} Demand Dynamics",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Demand Units", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout()

    plot_path = out_dir / f"scenario_{scenario.name}.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scenario plot to: %s", plot_path)
    return plot_path


def plot_all_scenarios(
    scenarios: Dict[str, Scenario],
    output_dir: Union[str, Path] = "results/scenarios",
) -> Dict[str, Path]:
    """Plot individual scenario figures and a side-by-side comparison figure."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = {}

    for name, sc in scenarios.items():
        p = plot_scenario(sc, output_dir=out_dir)
        generated_plots[name] = p

    # Multi-panel Comparison Plot
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 11), sharey=True)
    colors = {"stable": "#16a34a", "seasonal": "#8b5cf6", "shock": "#dc2626"}

    for idx, (name, sc) in enumerate(scenarios.items()):
        ax = axes[idx]
        c = colors.get(name, "#2563eb")
        rolling = pd.Series(sc.demand).rolling(7, min_periods=1).mean()
        ax.plot(sc.dates, sc.demand, color="#94a3b8", alpha=0.5, linewidth=1.0, label="Demand")
        ax.plot(sc.dates, rolling, color=c, linewidth=2.0, label="7-Day Rolling Mean")
        ax.axhline(sc.mean_demand, color="#334155", linestyle=":", linewidth=1.2)

        title_info = f"[{name.upper()}] Dates: {sc.dates[0].strftime('%Y-%m-%d')} to {sc.dates[-1].strftime('%Y-%m-%d')} | CV={sc.cv:.3f}"
        if "max_spike_ratio" in sc.metadata and name == "shock":
            title_info += f" | Spike={sc.metadata['max_spike_ratio']}x"
        if "acf_dominant" in sc.metadata and name == "seasonal":
            title_info += f" | ACF7={sc.metadata['acf_dominant']}"

        ax.set_title(title_info, fontsize=11, fontweight="bold")
        ax.set_ylabel("Demand Units", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.suptitle("Comparison of Simulation Scenarios (Target Mean = 100.0 Units)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    comp_path = out_dir / "all_scenarios_comparison.png"
    fig.savefig(comp_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated_plots["comparison"] = comp_path
    logger.info("Saved all scenarios comparison plot to: %s", comp_path)

    return generated_plots
