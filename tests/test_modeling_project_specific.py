import pytest
import numpy as np
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from smartcheck.modeling_project_specific import (
    compute_metrics,
    plot_predictions,
    compute_residuals_plot,
    interpret_model,
    SARIMAXWrapper,
)


class TestComputeMetrics:
    """Unit tests for compute_metrics"""

    # == Fixtures ==
    @pytest.fixture
    def dummy_data(self):
        y_true = pd.Series(np.random.rand(100))
        y_pred = y_true + np.random.normal(0, 0.1, size=100)
        return y_true, y_pred

    # == Tests ==
    def test_returns_expected_keys(self, dummy_data):
        y_true, y_pred = dummy_data
        metrics = compute_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {"R2", "RMSE", "MAE"}
        assert all(isinstance(v, float) for v in metrics.values())


class TestPlotPredictions:
    """Unit tests for plot_predictions"""

    # == Fixtures ==
    @pytest.fixture
    def dummy_data(self):
        dates = pd.DataFrame({
            "date_et_heure_de_comptage_local": pd.date_range(
                start="2024-01-01", periods=100, freq="h"
            )
        })
        y_true = pd.Series(np.random.rand(100))
        y_pred = y_true + np.random.normal(0, 0.1, size=100)
        return dates, y_true, y_pred

    # == Tests ==
    def test_without_period(self, dummy_data):
        dates, y_true, y_pred = dummy_data
        fig = plot_predictions("TestCounter", dates, y_true, y_pred)
        assert fig is not None

    def test_with_period(self, dummy_data):
        dates, y_true, y_pred = dummy_data
        periode = ("2024-01-01", "2024-01-03")
        fig = plot_predictions("TestCounter", dates, y_true, y_pred, periode)
        assert fig is not None


class TestComputeResidualsPlot:
    """Unit tests for compute_residuals_plot"""

    # == Fixtures ==
    @pytest.fixture
    def dummy_data(self):
        dates = pd.DataFrame({
            "date_et_heure_de_comptage_local": pd.date_range(
                start="2024-01-01", periods=100, freq="h"
            )
        })
        y_true = pd.Series(np.random.rand(100))
        y_pred = y_true + np.random.normal(0, 0.1, size=100)
        return dates, y_true, y_pred

    # == Tests ==
    def test_without_period(self, dummy_data):
        dates, y_true, y_pred = dummy_data
        fig1, fig2, slope = compute_residuals_plot("TestCounter", dates, y_true, y_pred)
        assert fig1 is not None
        assert fig2 is not None
        assert isinstance(slope, float)

    def test_with_period(self, dummy_data):
        dates, y_true, y_pred = dummy_data
        periode = ("2024-01-01", "2024-01-03")
        fig1, fig2, slope = compute_residuals_plot("TestCounter", dates, y_true,
                                                   y_pred, periode)
        assert fig1 is not None
        assert fig2 is not None
        assert isinstance(slope, float)


class TestInterpretModel:
    """Unit tests for interpret_model"""

    # == Tests ==
    @pytest.mark.parametrize("model_class", [
        LinearRegression,
        KNeighborsRegressor,
        RandomForestRegressor,
        XGBRegressor,
    ])
    def test_model_interpretation(self, model_class):
        X = pd.DataFrame({"x1": np.random.rand(40), "x2": np.random.rand(40)})
        y = X["x1"] * 0.4 + X["x2"] * 0.6 + np.random.normal(0, 0.01, size=40)

        transformer = ColumnTransformer([
            ("prep", StandardScaler(), ["x1", "x2"])
        ])

        pipe = Pipeline([
            ("prep", transformer),
            ("reg", model_class())
        ])

        pipe.fit(X, y)
        model_results = {
            "pipe": pipe,
            "X_test": X,
            "y_test_pred": pipe.predict(X),
        }

        figs = interpret_model(model_results)
        assert figs is None or all(fig is not None for fig in figs)

    def test_unrecognized_model_returns_none(self):
        class DummyModel:
            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.ones(len(X))

        X = pd.DataFrame({"x1": np.random.rand(10)})
        y = np.random.rand(10)

        pipe = Pipeline([
            ("prep", StandardScaler()),
            ("reg", DummyModel())
        ])
        pipe.fit(X, y)

        model_results = {
            "pipe": pipe,
            "X_test": X,
            "y_test_pred": np.ones(len(X))
        }

        result = interpret_model(model_results)
        assert result is None


class TestSARIMAXWrapper:
    """Unit tests for SARIMAXWrapper"""

    # == Fixtures ==
    @pytest.fixture
    def dummy_series(self):
        np.random.seed(42)
        t = np.arange(100)
        y = pd.Series(
            np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.1, size=len(t))
        )
        X = pd.DataFrame({"temp": np.random.rand(len(t))})
        return X, y

    def test_fit_and_predict_with_exog(self, dummy_series):
        X, y = dummy_series
        model = SARIMAXWrapper(
            order=(1, 0, 0),
            seasonal_order=(1, 1, 0, 24),
            use_exo=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X, y)
        assert hasattr(model, "results_")
        preds = model.predict(X)
        assert isinstance(preds, (np.ndarray, pd.Series))
        assert len(preds) == len(X)

    def test_fit_and_predict_without_exog(self, dummy_series):
        _, y = dummy_series
        model = SARIMAXWrapper(order=(1, 0, 0), use_exo=False)
        model.fit(None, y)
        assert hasattr(model, "results_")
        preds = model.predict(n_periods=10)
        assert isinstance(preds, (np.ndarray, pd.Series))
        assert len(preds) == 10

    def test_predict_in_sample(self, dummy_series):
        X, y = dummy_series
        model = SARIMAXWrapper(order=(1, 0, 0))
        model.fit(X, y)
        fitted = model.predict_in_sample()
        assert isinstance(fitted, (np.ndarray, pd.Series))
        assert len(fitted) == len(y)

    def test_predict_without_fit_raises(self):
        model = SARIMAXWrapper()
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(n_periods=5)

    def test_in_sample_without_fit_raises(self):
        model = SARIMAXWrapper()
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_in_sample()

    def test_missing_y_raises(self, dummy_series):
        X, _ = dummy_series
        model = SARIMAXWrapper()
        with pytest.raises(ValueError, match="y .* must be provided"):
            model.fit(X, None)

    def test_missing_X_and_n_periods_raises(self, dummy_series):
        _, y = dummy_series
        model = SARIMAXWrapper()
        model.fit(None, y)
        with pytest.raises(ValueError, match="You must provide either X or n_periods"):
            model.predict()
