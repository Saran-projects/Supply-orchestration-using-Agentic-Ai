"""Unit tests for demand forecasters (Linear Regression and Gradient Boosting)."""

import numpy as np
import pytest

from sca.data.scenarios import LeadTimeSampler, Scenario
from sca.forecast import (
    ForecastEvaluationResult,
    GradientBoostingForecaster,
    LinearRegressionForecaster,
    get_forecaster,
)


@pytest.fixture
def synthetic_scenario():
    """Create synthetic scenario with known trend and seasonality."""
    t = np.arange(200, dtype=float)
    # Sine seasonality (period 7) + small linear trend
    full_dem = np.round(100.0 + 15.0 * np.sin(2 * np.pi * t / 7) + 0.05 * t, 2)

    train = full_dem[:150]
    test = full_dem[150:]

    train_dates = [np.datetime64("2020-01-01") + np.timedelta64(int(i), "D") for i in range(150)]
    test_dates = [np.datetime64("2020-05-30") + np.timedelta64(int(i), "D") for i in range(50)]

    return Scenario(
        name="synthetic_seasonal",
        demand=test,
        raw_demand=test,
        dates=test_dates,
        train_demand=train,
        train_raw_demand=train,
        train_dates=train_dates,
        lead_time_sampler=LeadTimeSampler(seed=42),
        late_delivery_probability=0.1,
        scale_factor=1.0,
    )


def test_linear_regression_forecaster(synthetic_scenario):
    """Test LinearRegressionForecaster fitting and evaluation."""
    model = LinearRegressionForecaster()
    assert not model.is_fitted

    res: ForecastEvaluationResult = model.evaluate(synthetic_scenario)
    assert model.is_fitted
    assert res.mae >= 0.0
    assert res.rmse >= 0.0
    assert res.rmse >= res.mae  # RMSE >= MAE is mathematical property
    assert res.horizon == len(synthetic_scenario.demand)
    assert len(res.y_pred) == len(synthetic_scenario.demand)


def test_gradient_boosting_forecaster(synthetic_scenario):
    """Test GradientBoostingForecaster fitting and evaluation."""
    model = GradientBoostingForecaster(n_estimators=30, max_depth=2, random_state=42)
    assert not model.is_fitted

    res: ForecastEvaluationResult = model.evaluate(synthetic_scenario)
    assert model.is_fitted
    assert res.mae > 0.0
    assert res.rmse > 0.0
    assert res.rmse >= res.mae
    assert res.horizon == len(synthetic_scenario.demand)


def test_get_forecaster_factory():
    """Verify factory returns appropriate subclass."""
    lr = get_forecaster("linear")
    assert isinstance(lr, LinearRegressionForecaster)

    gb = get_forecaster("gradient_boosting")
    assert isinstance(gb, GradientBoostingForecaster)

    with pytest.raises(ValueError, match="Unknown forecaster"):
        get_forecaster("non_existent_model")


def test_multi_step_prediction():
    """Verify recursive horizon prediction produces non-negative outputs."""
    history = np.full(50, 100.0)
    forecaster = LinearRegressionForecaster()
    forecaster.fit(history)

    preds = forecaster.predict_horizon(history, horizon=7)
    assert len(preds) == 7
    assert np.all(preds >= 0.0)
    # On flat constant 100 series, prediction should remain ~100
    assert np.allclose(preds, 100.0, atol=1.0)


def test_no_data_leakage(synthetic_scenario):
    """Strictly verify the forecaster's internal dataset has length <= len(train_demand)."""
    forecaster = LinearRegressionForecaster()
    forecaster.fit(synthetic_scenario.train_demand)

    # Features must only be built from train_demand
    max_hist = forecaster.max_history_needed
    expected_samples = len(synthetic_scenario.train_demand) - max_hist
    X, y = forecaster._build_dataset(synthetic_scenario.train_demand)
    assert len(X) == expected_samples
