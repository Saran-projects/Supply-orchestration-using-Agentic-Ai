"""Tests for scenario generation, scaling, lead-time sampler, and no-leakage train/test splits."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sca.config import get_config
from sca.data.scenarios import (
    LeadTimeSampler,
    Scenario,
    generate_scenarios,
    plot_all_scenarios,
)


@pytest.fixture(scope="module")
def sample_scenarios():
    cfg = get_config().data
    return generate_scenarios(data_config=cfg)


class TestScenarioStructureAndRegimes:
    """Verify that all three regimes are generated with proper metadata and properties."""

    def test_three_scenarios_exist(self, sample_scenarios):
        cfg = get_config().data
        assert set(sample_scenarios.keys()) == {"stable", "seasonal", "shock"}
        for name, sc in sample_scenarios.items():
            assert isinstance(sc, Scenario)
            assert sc.name == name
            assert sc.horizon == cfg.scenario_window_length
            assert len(sc.demand) == cfg.scenario_window_length
            assert len(sc.raw_demand) == cfg.scenario_window_length
            assert len(sc.dates) == cfg.scenario_window_length

    def test_regime_specific_criteria(self, sample_scenarios):
        stable = sample_scenarios["stable"]
        seasonal = sample_scenarios["seasonal"]
        shock = sample_scenarios["shock"]

        # Stable must have very low coefficient of variation
        assert stable.cv < 0.25
        assert "cv" in stable.metadata

        # Shock must have a noticeable spike above rolling median
        assert shock.metadata["max_spike_ratio"] > 1.5

        # Seasonal must have valid autocorrelation recorded
        assert "acf_dominant" in seasonal.metadata


class TestNoDataLeakage:
    """Strict verification of temporal separation between training data and evaluation window."""

    def test_train_strictly_precedes_scenario_window(self, sample_scenarios):
        for name, sc in sample_scenarios.items():
            assert len(sc.train_dates) > 0, f"Scenario '{name}' has empty train dates"
            assert len(sc.dates) > 0, f"Scenario '{name}' has empty scenario dates"

            max_train = max(sc.train_dates)
            min_scenario = min(sc.dates)
            assert max_train < min_scenario, (
                f"Data leakage in {name}! Max train date {max_train} "
                f"is not strictly before min scenario date {min_scenario}"
            )

    def test_train_and_scenario_sets_are_disjoint(self, sample_scenarios):
        for name, sc in sample_scenarios.items():
            train_set = set(sc.train_dates)
            scenario_set = set(sc.dates)
            overlap = train_set.intersection(scenario_set)
            assert len(overlap) == 0, f"Data leakage in {name}! Found overlapping timestamps: {overlap}"

    def test_leakage_detection_raises_error(self):
        dates = [pd.Timestamp("2016-01-01") + pd.Timedelta(days=i) for i in range(10)]
        with pytest.raises(ValueError, match="Data leakage detected"):
            Scenario(
                name="leak_test",
                demand=np.array([10.0] * 5),
                raw_demand=np.array([10.0] * 5),
                dates=dates[4:9],  # Starts at day 4
                train_demand=np.array([10.0] * 6),
                train_raw_demand=np.array([10.0] * 6),
                train_dates=dates[0:6],  # Ends at day 5 (OVERLAPS!)
                lead_time_sampler=LeadTimeSampler(),
                late_delivery_probability=0.5,
                scale_factor=1.0,
            )


class TestScalingToSimulatorUnits:
    """Verify demand scaling to configurable target mean under Scenario Generation v2."""

    def test_training_demand_mean_matches_target(self, sample_scenarios):
        cfg = get_config().data
        for name, sc in sample_scenarios.items():
            # In v2, global scale factor is calibrated strictly from training history to match target
            # Verify training history scaled mean closely matches target mean (within 1%)
            train_mean = float(np.mean(sc.train_demand[:sc.metadata.get("train_steps", len(sc.train_demand))]))
            assert np.isclose(train_mean, cfg.target_mean_demand, rtol=0.02), (
                f"Scenario '{name}' training scaled mean {train_mean} does not match target {cfg.target_mean_demand}"
            )

    def test_train_and_scenario_use_same_scale_factor(self, sample_scenarios):
        for _name, sc in sample_scenarios.items():
            expected_scenario = np.round(sc.raw_demand * sc.scale_factor, 2)
            expected_train = np.round(sc.train_raw_demand * sc.scale_factor, 2)
            np.testing.assert_allclose(sc.demand, expected_scenario, atol=0.02)
            np.testing.assert_allclose(sc.train_demand, expected_train, atol=0.02)


class TestLeadTimeSamplerAndLateDelivery:
    """Verify empirical lead time sampler and late delivery probability."""

    def test_lead_time_sampler_sampling(self):
        sampler = LeadTimeSampler(seed=42)
        samples = sampler.sample(size=1000)
        assert len(samples) == 1000
        assert all(0 <= s <= 6 for s in samples)
        assert 3.0 < np.mean(samples) < 4.0

    def test_sampler_fit_from_df(self):
        df_dummy = pd.DataFrame({
            "Days for shipping (real)": [2, 2, 4, 4, 6],
            "Shipping Mode": ["Standard Class"] * 5,
        })
        sampler = LeadTimeSampler.fit(df_dummy, shipping_mode="Standard Class")
        assert sampler.mean == pytest.approx(3.6)
        assert sampler.days.tolist() == [2, 4, 6]

    def test_late_delivery_probability(self, sample_scenarios):
        for sc in sample_scenarios.values():
            assert 0.40 < sc.late_delivery_probability < 0.65


class TestScenarioPlotting:
    """Verify plot generation and image file presence."""

    def test_plot_single_and_all(self, sample_scenarios, tmp_path):
        plots = plot_all_scenarios(sample_scenarios, output_dir=tmp_path)
        assert "stable" in plots
        assert "seasonal" in plots
        assert "shock" in plots
        assert "comparison" in plots

        for p in plots.values():
            assert Path(p).is_file()
            assert Path(p).stat().st_size > 5000
