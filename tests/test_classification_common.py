import pandas as pd
import numpy as np
import pytest
import logging
from sklearn.linear_model import LogisticRegression

from smartcheck.classification_common import (
    logit_analysis,
    cross_validation_with_resampling,
    cross_validation_with_resampling_and_threshold
)


# === Test class for logit_analysis ===
class TestLogitAnalysis:

    # === Data Fixtures ===
    @pytest.fixture
    def dummy_data(self):
        np.random.seed(0)
        n = 50
        return pd.DataFrame({
            "Response": np.random.randint(0, 2, size=n),
            "Gender": np.random.randint(0, 2, size=n),
            "Age": np.random.randint(20, 85, size=n),
            "Driving_License": np.random.randint(0, 2, size=n),
            "Region_Code": np.random.randint(1, 10, size=n),
            "Previously_Insured": np.random.randint(0, 2, size=n),
            "Vehicle_Damage": np.random.randint(0, 2, size=n),
            "Annual_Premium": np.random.randint(24421, 39411, size=n),
            "Policy_Sales_Channel": np.random.randint(1, 5, size=n),
            "Vintage": np.random.randint(50, 400, size=n),
            "Year": np.random.choice([2021, 2022, 2023], size=n)
        })

    @pytest.fixture
    def config(self):
        return {
            "target": "Response",
            "features": [
                "Gender", "Age", "Driving_License", "Region_Code",
                "Previously_Insured", "Vehicle_Damage", "Annual_Premium",
                "Policy_Sales_Channel", "Vintage", "Year"
            ]
        }

    @pytest.fixture
    def adjustments(self):
        return {
            "Age": {
                "type": "normalize",
                "range": (20, 85)
            },
            "Annual_Premium": {
                "type": "normalize",
                "range": (24421, 39411)
            },
            "Gender": {
                "type": "inverse"
            },
            "Previously_Insured": {
                "type": "inverse"
            }
        }

    # === Tests ===
    def test_logit_analysis_runs_without_error(
        self, dummy_data, config, adjustments, caplog
    ):
        caplog.set_level(logging.INFO)
        logit_analysis(dummy_data, config, adjustments)

        # Check that summary was logged
        assert "Results: Logit" in caplog.text
        assert "Model:" in caplog.text
        assert "Odds Ratios" in caplog.text

    def test_missing_variable_in_adjustments_ignored_gracefully(
        self, dummy_data, config, adjustments, caplog
    ):
        # Add non-existing variable to adjustments
        adjustments["NonExistent"] = {"type": "inverse"}

        caplog.set_level(logging.INFO)
        logit_analysis(dummy_data, config, adjustments)

        assert "Results: Logit" in caplog.text
        assert "Odds Ratios" in caplog.text

    def test_both_branches_of_var_in_params(
        self, dummy_data, config, caplog
    ):
        caplog.set_level(logging.INFO)

        adjustments = {
            "Age": {
                "type": "custom",
                "func": lambda x: x * 1.1
            },
            "NotInModel": {
                "type": "normalize",
                "range": (0, 100)
            }
        }

        logit_analysis(dummy_data, config, adjustments)
        assert "Odds Ratios" in caplog.text


# === Test class for cross_validation_with_resampling ===
class TestCrossValidationWithResampling:

    # === Data Fixtures ===
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "target": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            "feature1": np.random.rand(n),
            "feature2": np.random.randint(0, 100, size=n),
            "feature3": np.random.choice([0, 1], size=n)
        })

    @pytest.fixture
    def features(self):
        return ["feature1", "feature2", "feature3"]

    @pytest.fixture
    def model(self):
        return LogisticRegression(solver="liblinear")

    # === Tests ===
    def test_cross_validation_resampling_runs_and_logs(
        self, sample_data, features, model, caplog
    ):
        caplog.set_level(logging.INFO)

        X = sample_data[features]
        y = sample_data["target"]

        # Run the function
        cross_validation_with_resampling(X, y, model)

        # Assertions on logs
        assert "=== SMOTE ===" in caplog.text
        assert "=== Oversampling ===" in caplog.text
        assert "=== Undersampling ===" in caplog.text
        assert "Scores:" in caplog.text
        assert "Mean F1-score" in caplog.text

    def test_f1_scores_are_in_valid_range(self, sample_data, features, model, caplog):
        caplog.set_level(logging.INFO)

        X = sample_data[features]
        y = sample_data["target"]

        cross_validation_with_resampling(X, y, model)

        scores_lines = [
            line for line in caplog.text.splitlines()
            if "Scores:" in line
        ]

        # Extract and validate each score list from log
        for line in scores_lines:
            # Example log line: "Scores: [0.67, 0.73, 0.58, 0.61, 0.69]"
            score_str = line.split("Scores:")[-1].strip()
            score_str = score_str.strip("[]")
            scores = [float(s) for s in score_str.split(",")]

            for score in scores:
                assert 0.0 <= score <= 1.0

    def test_empty_input_raises_error(self, model):
        empty_X = pd.DataFrame(columns=["feature1", "feature2", "feature3"])
        empty_y = pd.Series([], dtype=int)

        with pytest.raises(ValueError):
            cross_validation_with_resampling(empty_X, empty_y, model)


# === Test class for cross_validation_with_resampling_and_threshold ===
class TestCrossValidationWithResamplingAndThreshold:

    # === Data Fixtures ===
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "target": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            "feature1": np.random.rand(n),
            "feature2": np.random.randint(0, 100, size=n),
            "feature3": np.random.choice([0, 1], size=n)
        })

    @pytest.fixture
    def features(self):
        return ["feature1", "feature2", "feature3"]

    @pytest.fixture
    def model(self):
        return LogisticRegression(solver="liblinear")

    @pytest.fixture
    def thresholds_light(self):
        # Reduced for test speed
        return [0.3, 0.5, 0.7]

    # === Tests ===
    def test_cross_validation_with_threshold_runs_and_logs(
        self, sample_data, features, model, thresholds_light, caplog
    ):
        caplog.set_level(logging.INFO)

        X = sample_data[features]
        y = sample_data["target"]

        cross_validation_with_resampling_and_threshold(X, y, model, thresholds_light)

        assert "=== SMOTE ===" in caplog.text
        assert "=== Oversampling ===" in caplog.text
        assert "=== Undersampling ===" in caplog.text
        assert "F1 Scores:" in caplog.text
        assert "Average threshold" in caplog.text
        assert "Average F1-score" in caplog.text

    def test_threshold_and_f1_scores_valid_range(
        self, sample_data, features, model, thresholds_light, caplog
    ):
        caplog.set_level(logging.INFO)

        X = sample_data[features]
        y = sample_data["target"]

        cross_validation_with_resampling_and_threshold(X, y, model, thresholds_light)

        # Extract thresholds and scores from log lines
        thresholds = []
        f1s = []
        for line in caplog.text.splitlines():
            if "Average threshold" in line:
                thresholds.append(float(line.split("=")[-1]))
            if "Average F1-score" in line:
                f1s.append(float(line.split("=")[-1]))

        for t in thresholds:
            assert 0.0 <= t <= 1.0
        for score in f1s:
            assert 0.0 <= score <= 1.0

    def test_empty_input_raises_error(self, model):
        empty_X = pd.DataFrame(columns=["feature1", "feature2", "feature3"])
        empty_y = pd.Series([], dtype=int)

        with pytest.raises(ValueError):
            cross_validation_with_resampling_and_threshold(empty_X, empty_y, model)
