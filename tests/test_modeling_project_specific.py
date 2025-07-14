import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from smartcheck.modeling_project_specific import (
    compute_metrics,
    plot_predictions,
    compute_residuals_plot,
    interpret_model,
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
    @pytest.mark.parametrize("model_class", [LinearRegression, KNeighborsRegressor])
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

        figs = interpret_model("TestCounter", model_results)
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
            ("preprocessing_column_transformation", StandardScaler()),
            ("dummy", DummyModel())
        ])
        pipe.fit(X, y)

        model_results = {
            "pipe": pipe,
            "X_test": X,
            "y_test_pred": np.ones(len(X))
        }

        result = interpret_model("no_match", model_results)
        assert result is None

    def test_knn_low_variance_triggers_warning_and_none(self, caplog):
        X = pd.DataFrame({
            "x1": np.random.rand(50),
            "x2": np.random.rand(50)
        })
        y = np.random.rand(50)

        transformer = ColumnTransformer([
            ("prep", StandardScaler(), ["x1", "x2"])
        ])

        pipe = Pipeline([
            ("prep", transformer),
            ("reg", KNeighborsRegressor(n_neighbors=1))
        ])
        pipe.fit(X, y)

        model_results = {
            "pipe": pipe,
            "X_test": X,
            "y_test_pred": pipe.predict(X),
        }

        with patch("smartcheck.modeling_project_specific.PCA") as pca_mock:
            pca_instance = MagicMock()
            pca_instance.fit_transform.return_value = np.random.rand(50, 2)
            pca_instance.explained_variance_ratio_ = np.array([0.2, 0.3])
            pca_mock.return_value = pca_instance

            with caplog.at_level("WARNING"):
                result = interpret_model("lowvar", model_results)

            assert result is None
            assert "PCA explained variance < 90%" in caplog.text
