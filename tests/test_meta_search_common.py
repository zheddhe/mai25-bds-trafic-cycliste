import pytest
import logging
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from smartcheck.meta_search_common import (
    compare_search_methods,
)


# === Test class for compare_search_methods ===
class TestCompareSearchMethods:

    # === Data Fixtures ===
    @pytest.fixture
    def classification_data(self):
        X, y = make_classification(n_samples=100, n_features=5,
                                   n_classes=2, random_state=42)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    @pytest.fixture
    def regression_data(self):
        X, y, _ = make_regression(n_samples=100, n_features=5, noise=0.1,
                                  random_state=42, coef=True)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    @pytest.fixture
    def clf_model(self):
        return LogisticRegression(solver="liblinear")

    @pytest.fixture
    def reg_model(self):
        return LinearRegression()

    @pytest.fixture
    def clf_param_grid(self):
        return {
            "C": [0.01, 0.1, 1.0],
            "penalty": ["l2", "l1"],
        }

    @pytest.fixture
    def reg_param_grid(self):
        return {
            "fit_intercept": [True],
            "n_jobs": [1, 2, 3, 4, 5],
        }

    # === Tests ===

    def test_classification_runs_fast(self, clf_model, clf_param_grid,
                                      classification_data, caplog):
        caplog.set_level(logging.INFO)
        X_train, X_test, y_train, y_test = classification_data
        results = {}

        compare_search_methods("LogisticRegression", clf_model, clf_param_grid,
                               X_train, X_test, y_train, y_test, results)

        res = results["LogisticRegression"]
        assert all("best_params" in r for r in res.values())
        assert all("test_f1_score" in r for r in res.values())

    def test_regression_runs_fast(self, reg_model, reg_param_grid,
                                  regression_data, caplog):
        caplog.set_level(logging.INFO)
        X_train, X_test, y_train, y_test = regression_data
        results = {}

        compare_search_methods("LinearRegression", reg_model, reg_param_grid,
                               X_train, X_test, y_train, y_test, results)

        res = results["LinearRegression"]
        assert all("best_params" in r for r in res.values())
        assert all("test_mean_squared_error" in r for r in res.values())

    def test_unsupported_model_raises(self, classification_data):
        class FakeEstimator(BaseEstimator):
            def fit(self, X, y): pass
            def predict(self, X): return X

        X_train, X_test, y_train, y_test = classification_data
        with pytest.raises(ValueError, match="neither a classifier nor a regressor"):
            compare_search_methods("FakeModel", FakeEstimator(), {},
                                   X_train, X_test, y_train, y_test, {})

    def test_bad_param_grid_does_not_crash(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        bad_grid = {"does_not_exist": [1, 2, 3, 4, 5]}
        results = {}

        compare_search_methods("Dummy", DummyClassifier(), bad_grid,
                               X_train, X_test, y_train, y_test, results)

        assert "Dummy" in results
        assert isinstance(results["Dummy"], dict)
        assert all(isinstance(v, dict) for v in results["Dummy"].values())

    def test_not_fitted_predict_is_handled_gracefully(self, classification_data,
                                                      clf_model, clf_param_grid,
                                                      caplog):
        X_train, X_test, y_train, y_test = classification_data
        results = {}

        caplog.set_level(logging.WARNING)

        # Patch GridSearchCV.predict to raise NotFittedError
        with patch.object(GridSearchCV, "predict",
                          side_effect=NotFittedError("Not fitted")):
            compare_search_methods("LogisticRegression", clf_model,
                                   clf_param_grid, X_train, X_test,
                                   y_train, y_test, results)

        # Check that the warning was logged
        assert "Model not fitted with GridSearchCV" in caplog.text

        # Check that the test metric is None
        test_metrics = results["LogisticRegression"]["GridSearchCV"]
        assert test_metrics["test_f1_score"] is None
