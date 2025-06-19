import pytest
import pandas as pd
from smartcheck.preprocessing_project_specific import (
    DatetimePreprocessingTransformer,
    ColumnFilterTransformer,
    MeteoCodePreprocessingTransformer
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


# === Test class for MeteoCodePreprocessingTransformer ===
class TestMeteoCodePreprocessingTransformer:

    # === Fixtures ===
    @pytest.fixture
    def df_valid_codes(self):
        return pd.DataFrame({
            "meteo_code": [-1, 0, 4, 5, 8, 10, 13, 15, 17,
                           20, 31, 40, 52, 65, 72, 84, 91]
        })

    @pytest.fixture
    def expected_categories(self):
        return [
            "unknown",                    # -1
            "cloud_variation",            # 0
            "smoke_or_pollution",         # 4
            "dry_haze",                   # 5
            "dust_or_sandstorm",          # 8
            "drizzle_or_local_fog",       # 10
            "lightning_no_thunder",       # 13
            "visible_precipitation",      # 15
            "storm_or_gusts",             # 17
            "non_violent_rain_or_fog",    # 20
            "evolving_fog",               # 31
            "light_to_heavy_rain",        # 40
            "freezing_drizzle_or_rain",   # 52
            "rain_snow_combination",      # 65
            "snow_or_sleet",              # 72
            "showers_or_small_hail",      # 84
            "thunderstorm_or_hail"        # 91
        ]

    # === Tests ===
    def test_transform_maps_codes_correctly(self, df_valid_codes, expected_categories):
        transformer = MeteoCodePreprocessingTransformer(code_col="meteo_code")
        result = transformer.transform(df_valid_codes)

        assert list(result.columns) == ["meteo_code_category"]
        assert result.shape[0] == len(expected_categories)
        assert result["meteo_code_category"].tolist() == expected_categories

    def test_get_feature_names_out(self):
        transformer = MeteoCodePreprocessingTransformer(code_col="meteo_code")
        assert transformer.get_feature_names_out() == ["meteo_code_category"]

    def test_fit_returns_self(self, df_valid_codes):
        transformer = MeteoCodePreprocessingTransformer(code_col="meteo_code")
        result = transformer.fit(df_valid_codes)
        assert result is transformer
