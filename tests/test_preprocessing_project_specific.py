import pytest
import pandas as pd
from smartcheck.preprocessing_project_specific import (
    DatetimePreprocessingTransformer,
    ColumnFilterTransformer,
)


# === Test Class for DatetimePreprocessingTransformer ===
class TestDatetimePreprocessingTransformer:

    # === Data Fixtures ===
    @pytest.fixture
    def naive_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": [
                "2025-03-30T01:30:00+00:00",
                "2025-03-30T03:45:00+00:00"
            ]
        })

    @pytest.fixture
    def aware_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": [
                "2025-03-30T01:30:00+00:00",
                "2025-03-30T03:45:00+00:00"
            ]
        })

    @pytest.fixture
    def malformed_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": ["invalid", "2025-03-30T01:30:00"]
        })

    @pytest.fixture
    def missing_col_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "not_timestamp": ["2025-03-30T01:30:00"]
        })

    # === Tests ===
    def test_transform_naive_datetime(self, naive_df):
        transformer = DatetimePreprocessingTransformer("timestamp")
        result = transformer.transform(naive_df)

        expected_cols = {
            "timestamp_utc",
            "timestamp_local",
            "timestamp_year",
            "timestamp_month",
            "timestamp_day",
            "timestamp_day_of_year",
            "timestamp_day_of_week",
            "timestamp_hour",
            "timestamp_week",
            "timestamp_dayname",
            "timestamp_monthname"
        }

        assert set(result.columns) == expected_cols
        assert str(result["timestamp_local"].dt.tz) == "Europe/Paris"
        assert str(result["timestamp_utc"].dt.tz) == "UTC"

    def test_transform_aware_datetime(self, aware_df):
        transformer = DatetimePreprocessingTransformer("timestamp")
        result = transformer.transform(aware_df)

        assert str(result["timestamp_local"].dt.tz) == "Europe/Paris"
        assert str(result["timestamp_utc"].dt.tz) == "UTC"

    def test_transform_column_missing_raises(self, missing_col_df):
        transformer = DatetimePreprocessingTransformer("timestamp")
        with pytest.raises(KeyError, match="timestamp"):
            transformer.transform(missing_col_df)

    def test_transform_invalid_datetime_raises(self, malformed_df):
        transformer = DatetimePreprocessingTransformer("timestamp")
        with pytest.raises(Exception):
            transformer.transform(malformed_df)

    def test_fit_returns_self(self, naive_df):
        transformer = DatetimePreprocessingTransformer("timestamp")
        result = transformer.fit(naive_df)
        assert result is transformer
        assert transformer.timestamp_col == "timestamp"

    def test_get_feature_names_out(self):
        transformer = DatetimePreprocessingTransformer("my_time")
        expected = [
            "my_time_utc",
            "my_time_local",
            "my_time_year",
            "my_time_month",
            "my_time_day",
            "my_time_day_of_year",
            "my_time_day_of_week",
            "my_time_hour",
            "my_time_week",
            "my_time_dayname",
            "my_time_monthname"
        ]
        assert transformer.get_feature_names_out() == expected


class TestColumnFilterTransformer:

    # === Data Fixtures ===
    @pytest.fixture
    def full_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "col1": [1, 2],
            "col2": [3, 4],
            "col3": [5, 6]
        })

    @pytest.fixture
    def incomplete_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "col1": [1, 2],
            "col3": [5, 6]
        })

    # === Tests ===
    def test_transform_filters_expected_columns(self, full_df):
        transformer = ColumnFilterTransformer(columns_to_keep=["col1", "col3"])
        result = transformer.transform(full_df)

        assert list(result.columns) == ["col1", "col3"]
        pd.testing.assert_frame_equal(result, full_df[["col1", "col3"]])

    def test_transform_raises_for_missing_columns(self, incomplete_df):
        transformer = ColumnFilterTransformer(columns_to_keep=["col1", "col2"])
        with pytest.raises(ValueError, match="Missing expected columns"):
            transformer.transform(incomplete_df)

    def test_fit_returns_self(self, full_df):
        transformer = ColumnFilterTransformer(columns_to_keep=["col1"])
        result = transformer.fit(full_df)
        assert result is transformer
        assert transformer.columns_to_keep == ["col1"]

    def test_get_feature_names_out(self):
        cols = ["a", "b", "c"]
        transformer = ColumnFilterTransformer(columns_to_keep=cols)
        assert transformer.get_feature_names_out() == cols
