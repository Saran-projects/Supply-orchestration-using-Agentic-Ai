"""Time-series demand forecasting models using lag features: Linear Regression and Gradient Boosting.

All models train strictly on pre-scenario window history to guarantee zero lookahead data leakage.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from sca.data.scenarios import Scenario, generate_scenarios

logger = logging.getLogger(__name__)

DEFAULT_LAGS = [1, 2, 3, 7, 14]
DEFAULT_ROLLING = [7, 14]


@dataclass
class ForecastEvaluationResult:
    """Metrics and predictions resulting from evaluating a forecaster on a scenario window."""

    model_name: str
    scenario_name: str
    mae: float
    rmse: float
    mape: float
    horizon: int
    mean_true: float
    mean_pred: float
    y_true: np.ndarray
    y_pred: np.ndarray

    def summary(self) -> str:
        return (
            f"[{self.model_name.upper()}] on '{self.scenario_name}' ({self.horizon} steps): "
            f"MAE={self.mae:.2f}, RMSE={self.rmse:.2f}, MAPE={self.mape:.1f}%, "
            f"True Mean={self.mean_true:.1f}, Pred Mean={self.mean_pred:.1f}"
        )


class BaseDemandForecaster(ABC):
    """Abstract base class for supply chain demand forecasters."""

    def __init__(
        self,
        name: str = "DemandForecaster",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        self.name = name
        self.lags = sorted(lags or DEFAULT_LAGS)
        self.rolling_windows = sorted(rolling_windows or DEFAULT_ROLLING)
        self.max_history_needed = max(max(self.lags), max(self.rolling_windows))
        self.is_fitted = False
        self.train_mean: float = 100.0

    def _extract_features_row(self, history: np.ndarray) -> np.ndarray:
        """Extract a single feature vector from the trailing slice of historical demand."""
        arr = np.asarray(history, dtype=float)
        n = len(arr)

        if n < self.max_history_needed:
            # Pad with training mean or available mean if history is too short
            pad_val = self.train_mean if self.is_fitted else (float(np.mean(arr)) if n > 0 else 100.0)
            padding = np.full(self.max_history_needed - n, pad_val)
            arr = np.concatenate([padding, arr])
            n = len(arr)

        feats: List[float] = []
        # Lag features
        for lag in self.lags:
            feats.append(float(arr[n - lag]))

        # Rolling statistics (excluding current target step)
        for w in self.rolling_windows:
            window_slice = arr[n - w : n]
            feats.append(float(np.mean(window_slice)))

        return np.array(feats, dtype=float)

    def _build_dataset(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct feature matrix X and target vector y from a 1D training time-series."""
        arr = np.asarray(series, dtype=float)
        X, y = [], []
        for i in range(self.max_history_needed, len(arr)):
            row = []
            for lag in self.lags:
                row.append(arr[i - lag])
            for w in self.rolling_windows:
                row.append(float(np.mean(arr[i - w : i])))
            X.append(row)
            y.append(arr[i])
        return np.array(X, dtype=float), np.array(y, dtype=float)

    @abstractmethod
    def fit(self, train_demand: Union[List[float], np.ndarray]) -> "BaseDemandForecaster":
        """Fit model strictly on pre-window training history."""
        pass

    @abstractmethod
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw model prediction on feature matrix."""
        pass

    def predict_step(self, recent_history: Union[List[float], np.ndarray]) -> float:
        """Predict next single-step demand given recent demand history."""
        if not self.is_fitted:
            raise RuntimeError(f"Forecaster '{self.name}' must be fitted before predict_step.")

        feat_row = self._extract_features_row(np.asarray(recent_history, dtype=float))
        pred = float(self._predict_raw(feat_row.reshape(1, -1))[0])
        return float(max(0.0, pred))

    def predict_horizon(
        self,
        history: Union[List[float], np.ndarray],
        horizon: int = 1,
    ) -> np.ndarray:
        """Autoregressive multi-step demand forecast rolling predictions forward."""
        if horizon <= 0:
            return np.array([], dtype=float)

        curr_hist = list(np.asarray(history, dtype=float))
        predictions: List[float] = []

        for _ in range(horizon):
            p = self.predict_step(curr_hist)
            predictions.append(p)
            curr_hist.append(p)

        return np.array(predictions, dtype=float)

    def evaluate(self, scenario: Scenario) -> ForecastEvaluationResult:
        """Evaluate forecaster on scenario window using strict pre-window training."""
        # 1. Fit on pre-window training demand ONLY
        self.fit(scenario.train_demand)

        # 2. Step-by-step rolling evaluation on the scenario window
        full_series = np.concatenate([scenario.train_demand, scenario.demand])
        test_start = len(scenario.train_demand)
        horizon = len(scenario.demand)

        y_true = np.asarray(scenario.demand, dtype=float)
        y_pred = []

        for t in range(horizon):
            # Known history up to step t of scenario
            history_t = full_series[: test_start + t]
            pred = self.predict_step(history_t)
            y_pred.append(pred)

        y_pred_arr = np.array(y_pred, dtype=float)

        # 3. Metrics: MAE, RMSE, MAPE
        mae = float(np.mean(np.abs(y_true - y_pred_arr)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred_arr) ** 2)))
        mape = float(np.mean(np.abs((y_true - y_pred_arr) / np.maximum(y_true, 1.0)))) * 100.0

        res = ForecastEvaluationResult(
            model_name=self.name,
            scenario_name=scenario.name,
            mae=round(mae, 3),
            rmse=round(rmse, 3),
            mape=round(mape, 2),
            horizon=horizon,
            mean_true=round(float(np.mean(y_true)), 2),
            mean_pred=round(float(np.mean(y_pred_arr)), 2),
            y_true=y_true,
            y_pred=y_pred_arr,
        )
        logger.info(res.summary())
        return res


class LinearRegressionForecaster(BaseDemandForecaster):
    """Linear Regression demand forecaster utilizing lag and rolling mean features."""

    def __init__(
        self,
        name: str = "LinearRegressionForecaster",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        fit_intercept: bool = True,
    ):
        super().__init__(name=name, lags=lags, rolling_windows=rolling_windows)
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def fit(self, train_demand: Union[List[float], np.ndarray]) -> "LinearRegressionForecaster":
        arr = np.asarray(train_demand, dtype=float)
        self.train_mean = float(np.mean(arr)) if len(arr) > 0 else 100.0

        X, y = self._build_dataset(arr)
        if len(X) == 0:
            raise ValueError(
                f"Training demand length ({len(arr)}) is too short for max history ({self.max_history_needed})."
            )

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class GradientBoostingForecaster(BaseDemandForecaster):
    """Gradient Boosting demand forecaster with decision trees over lag features."""

    def __init__(
        self,
        name: str = "GradientBoostingForecaster",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        n_estimators: int = 60,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        super().__init__(name=name, lags=lags, rolling_windows=rolling_windows)
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(self, train_demand: Union[List[float], np.ndarray]) -> "GradientBoostingForecaster":
        arr = np.asarray(train_demand, dtype=float)
        self.train_mean = float(np.mean(arr)) if len(arr) > 0 else 100.0

        X, y = self._build_dataset(arr)
        if len(X) == 0:
            raise ValueError(
                f"Training demand length ({len(arr)}) is too short for max history ({self.max_history_needed})."
            )

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def get_forecaster(
    model_type: str = "linear",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    **kwargs: Any,
) -> BaseDemandForecaster:
    """Instantiate a demand forecaster by name/type ('linear' or 'gradient_boosting')."""
    m = model_type.lower().strip()
    if m in ("linear", "linear_regression", "lr", "ols"):
        return LinearRegressionForecaster(lags=lags, rolling_windows=rolling_windows, **kwargs)
    elif m in ("gradient_boosting", "gbr", "gbm", "boosting"):
        return GradientBoostingForecaster(lags=lags, rolling_windows=rolling_windows, **kwargs)
    else:
        raise ValueError(
            f"Unknown forecaster model_type '{model_type}'. Expected 'linear' or 'gradient_boosting'."
        )


def evaluate_all_scenarios(
    model_type: str = "linear",
    scenarios: Optional[Dict[str, Scenario]] = None,
    **kwargs: Any,
) -> Dict[str, ForecastEvaluationResult]:
    """Evaluate a forecaster on all scenarios ('stable', 'seasonal', 'shock')."""
    sc_dict = scenarios or generate_scenarios()
    results: Dict[str, ForecastEvaluationResult] = {}

    for name, sc in sc_dict.items():
        forecaster = get_forecaster(model_type=model_type, **kwargs)
        eval_res = forecaster.evaluate(sc)
        results[name] = eval_res

    return results
