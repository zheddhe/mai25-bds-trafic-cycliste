import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, MagicMock
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from smartcheck.modeling_project_specific import (
    compute_metrics,
    plot_predictions,
    compute_residuals_plot,
    interpret_model,
    SARIMAXWrapper,
    train_timeseries_model,
    recursive_forecast_model,
)
from smartcheck.modeling_project_specific import AutoregressiveFeaturesTransformer


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

    def test_with_null_period(self, dummy_data):
        dates, y_true, y_pred = dummy_data
        periode = ("2024-01-01", "2024-01-01")
        fig1, fig2, slope = compute_residuals_plot("TestCounter", dates, y_true, y_pred,
                                                   periode)
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
        # BayesSearchCV,
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


class TestTrainTimeseriesModel:
    """Unit tests for train_timeseries_model"""

    # == Fixtures ==
    @pytest.fixture
    def dummy_df(self):
        """Small DataFrame with numeric and categorical data"""
        n = 48
        timestamps = pd.date_range(
            start="2024-01-01", periods=n, freq="h"
        )
        return pd.DataFrame({
            "date_et_heure_de_comptage": timestamps,
            "comptage_horaire": np.linspace(100, 200, n),
            "temperature": np.random.uniform(5, 15, n),
            "jour_semaine": ["lundi", "mardi"] * (n // 2),
        })

    @pytest.fixture(params=[
        "KNN",
        "RandomForest",
        "XGBoost",
        "Lasso",
        "Ridge",
        "LinearRegression"
    ])
    def base_model_type(self, request):
        return request.param

    @pytest.fixture
    def simple_df(self):
        n = 40
        return pd.DataFrame({
            "date_et_heure_de_comptage": pd.date_range(
                start="2024-01-01", periods=n, freq="h"
            ),
            "comptage_horaire": np.random.rand(n) * 100,
            "temperature": np.random.uniform(0, 10, n),
            "jour_semaine": ["lun", "mar"] * (n // 2),
        })

    # == Tests ==
    @patch("smartcheck.modeling_project_specific.Pipeline.fit")
    @patch("smartcheck.modeling_project_specific.Pipeline.predict")
    def test_train_knn_model_without_ar(
        self, mock_predict, mock_fit, dummy_df
    ):
        mock_fit.return_value = None
        mock_predict.side_effect = lambda X: np.ones(len(X))

        results = train_timeseries_model(
            df_compteur=dummy_df,
            model_type="KNN",
            scaler_type="StandardScaler",
            temp_feats=[0, 0, 1],
            forecast=False
        )

        assert isinstance(results, dict)
        assert set(results.keys()) >= {
            "pipe", "X_train", "y_train_pred", "y_test_pred"
        }
        assert results["X_train"].shape[0] > 0
        assert all(np.array(results["y_test_pred"]) == 1.0)

    @patch("smartcheck.modeling_project_specific.Pipeline.fit")
    @patch("smartcheck.modeling_project_specific.Pipeline.predict")
    def test_train_with_ar_features(
        self, mock_predict, mock_fit, dummy_df
    ):
        mock_fit.return_value = None
        mock_predict.side_effect = lambda X: np.zeros(len(X))

        results = train_timeseries_model(
            df_compteur=dummy_df,
            model_type="LinearRegression",
            temp_feats=[2, 2, 3],
            forecast=False
        )

        assert results["ar_transformer"] is not None
        assert all(np.array(results["y_train_pred"]) == 0.0)

    @patch("smartcheck.modeling_project_specific.Pipeline.fit")
    @patch("smartcheck.modeling_project_specific.Pipeline.predict")
    def test_forecast_mode_enabled(
        self, mock_predict, mock_fit, dummy_df
    ):
        mock_fit.return_value = None
        mock_predict.side_effect = lambda X: np.full(len(X), 42.0)

        with patch(
            "smartcheck.modeling_project_specific.recursive_forecast_model"
        ) as mock_recursive:
            mock_recursive.return_value = np.full(
                shape=(len(dummy_df) // 5,), fill_value=99.0
            )

            results = train_timeseries_model(
                df_compteur=dummy_df,
                model_type="Ridge",
                temp_feats=[1, 1, 3],
                forecast=True
            )

            assert mock_recursive.called
            assert (results["y_test_pred"] == 99.0).all()

    @patch("smartcheck.modeling_project_specific.Pipeline.fit")
    @patch("smartcheck.modeling_project_specific.Pipeline.predict")
    def test_supported_models_run(
        self, mock_predict, mock_fit, base_model_type, simple_df
    ):
        mock_fit.return_value = None
        mock_predict.side_effect = lambda X: np.ones(len(X))

        result = train_timeseries_model(
            df_compteur=simple_df,
            model_type=base_model_type,
            scaler_type="RobustScaler",
            temp_feats=[0, 0, 1]
        )
        assert isinstance(result["pipe"].named_steps["reg"], (
            KNeighborsRegressor,
            RandomForestRegressor,
            XGBRegressor,
            Lasso,
            Ridge,
            LinearRegression
        ))

    @patch("smartcheck.modeling_project_specific.Pipeline.fit")
    @patch("smartcheck.modeling_project_specific.Pipeline.predict")
    def test_elasticnet_search_logs_warning(
        self, mock_predict, mock_fit, simple_df, caplog
    ):
        mock_fit.return_value = None
        mock_predict.return_value = np.zeros(len(simple_df) // 5)

        mock_best_estimator = MagicMock()
        mock_best_estimator.alpha = 0.005
        mock_best_estimator.l1_ratio = 0.1

        mock_bayes_search = MagicMock()
        mock_bayes_search.best_estimator_ = mock_best_estimator

        with patch(
            "smartcheck.modeling_project_specific.BayesSearchCV",
            return_value=mock_bayes_search
        ):
            result = train_timeseries_model(
                df_compteur=simple_df,
                model_type="ElasticNet (*)",
                temp_feats=[0, 0, 1]
            )

        assert "Best alpha" in caplog.text
        assert "Best l1_ratio" in caplog.text
        assert "Low regularization detected" in caplog.text
        assert isinstance(
            result["pipe"].named_steps["reg"], MagicMock
        )


class TestRecursiveForecastModel:
    """Unit tests for recursive_forecast_model"""

    # == Fixtures ==
    @pytest.fixture
    def dummy_last_window_df(self):
        n_train = 12
        n_test = 3
        values = np.arange(n_train, dtype=float)  # y known
        nan_part = [np.nan] * n_test  # y unknown (to predict)
        y_all = np.concatenate([values, nan_part])
        return pd.DataFrame({
            "date_et_heure_de_comptage": pd.date_range(
                start="2024-01-01", periods=n_train + n_test, freq="h"
            ),
            "comptage_horaire": y_all,
            "feature1": np.random.rand(n_train + n_test),
            "feature2": np.random.randint(1, 5, n_train + n_test)
        })

    @pytest.fixture
    def mock_ar_transformer(self):
        transformer = MagicMock(spec=AutoregressiveFeaturesTransformer)
        transformer.nb_ar = 3
        transformer.nb_mm = 1
        transformer.roll_wind = 2
        transformer.transform_recursive_step.side_effect = lambda x, y: x.copy()
        return transformer

    @pytest.fixture
    def mock_pipe(self):
        prep = MagicMock()
        reg = MagicMock()
        prep.transform.side_effect = lambda X: X[["feature1", "feature2"]].values
        reg.predict.side_effect = lambda X: np.full(X.shape[0], 42.0)

        pipe = MagicMock()
        pipe.named_steps = {"prep": prep, "reg": reg}
        return pipe

    # == Tests ==
    def test_forecast_runs_and_returns_values(
        self, dummy_last_window_df, mock_pipe, mock_ar_transformer
    ):
        result = recursive_forecast_model(
            pipe=mock_pipe,
            ar_transformer=mock_ar_transformer,
            last_window_df=dummy_last_window_df,
            horizon=3,
            target_col="comptage_horaire"
        )
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(y == 42.0 for y in result)

    def test_raises_if_history_insufficient(
        self, dummy_last_window_df, mock_pipe, mock_ar_transformer
    ):
        # Force besoin d'un historique plus long que disponible
        mock_ar_transformer.nb_ar = 20
        with pytest.raises(ValueError, match="Insufficient history"):
            recursive_forecast_model(
                pipe=mock_pipe,
                ar_transformer=mock_ar_transformer,
                last_window_df=dummy_last_window_df,
                horizon=3,
                target_col="comptage_horaire"
            )

    def test_stops_loop_on_transform_failure(
        self, dummy_last_window_df, mock_pipe, caplog
    ):
        transformer = MagicMock(spec=AutoregressiveFeaturesTransformer)
        transformer.nb_ar = 2
        transformer.nb_mm = 1
        transformer.roll_wind = 1

        transformer.transform_recursive_step.side_effect = [
            dummy_last_window_df.iloc[[0]],
            Exception("step failure")  # fail at step 1
        ]

        result = recursive_forecast_model(
            pipe=mock_pipe,
            ar_transformer=transformer,
            last_window_df=dummy_last_window_df,
            horizon=3,
            target_col="comptage_horaire"
        )

        assert len(result) == 1
        assert "Failed to create AR features" in caplog.text
