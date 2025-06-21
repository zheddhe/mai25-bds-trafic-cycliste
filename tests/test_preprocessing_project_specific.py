import pytest
import pandas as pd
from unittest.mock import patch
from smartcheck.preprocessing_project_specific import (
    DatetimePreprocessingTransformer,
    ColumnFilterTransformer,
    MeteoCodePreprocessingTransformer,
    WeatherDataEnrichmentTransformer,
    ColumnNameNormalizerTransformer,
    SchoolHolidayTransformer,
    HolidayFromDatetimeTransformer
)


# === Test Class for DatetimePreprocessingTransformer ===
class TestDatetimePreprocessingTransformer:

    # === Data fixtures ===
    @pytest.fixture
    def naive_df(self) -> pd.DataFrame:
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
    def test_transform_extracts_expected_fields(self, naive_df):
        transformer = DatetimePreprocessingTransformer("timestamp")
        result = transformer.transform(naive_df)
        expected_cols = {
            "timestamp",
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


# === Test Class for ColumnFilterTransformer ===
class TestColumnFilterTransformer:

    # === Data fixtures ===
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

    def test_transform_raises_for_missing_columns(self, incomplete_df):
        transformer = ColumnFilterTransformer(columns_to_keep=["col1", "col2"])
        with pytest.raises(ValueError, match="Missing expected columns"):
            transformer.transform(incomplete_df)

    def test_fit_returns_self(self, full_df):
        transformer = ColumnFilterTransformer(columns_to_keep=["col1"])
        result = transformer.fit(full_df)
        assert result is transformer


# === Test class for MeteoCodePreprocessingTransformer ===
class TestMeteoCodePreprocessingTransformer:

    # === Data fixtures ===
    @pytest.fixture
    def df_valid_codes(self):
        return pd.DataFrame({
            "meteo_code": [0, 4, 5, 9, 12, 13, 15, 18, 22,
                           32, 42, 55, 65, 75, 82, 92, -1]
        })

    @pytest.fixture
    def expected_categories(self):
        return [
            "cloud_variation",
            "smoke_or_pollution",
            "dry_haze",
            "dust_or_sandstorm",
            "drizzle_or_local_fog",
            "lightning_no_thunder",
            "visible_precipitation",
            "storm_or_gusts",
            "non_violent_rain_or_fog",
            "evolving_fog",
            "light_to_heavy_rain",
            "freezing_drizzle_or_rain",
            "rain_snow_combination",
            "snow_or_sleet",
            "showers_or_small_hail",
            "thunderstorm_or_hail",
            "unknown"
        ]

    # === Tests ===
    def test_transform_category_mapping(self, df_valid_codes, expected_categories):
        transformer = MeteoCodePreprocessingTransformer(code_col="meteo_code")
        result = transformer.transform(df_valid_codes)
        assert result["meteo_code_category"].tolist() == expected_categories

    def test_fit_returns_self(self, df_valid_codes):
        transformer = MeteoCodePreprocessingTransformer(code_col="meteo_code")
        assert transformer.fit(df_valid_codes) is transformer


# === Test class for WeatherDataEnrichmentTransformer ===
class TestWeatherDataEnrichmentTransformer:

    # === Data fixtures ===
    @pytest.fixture
    def df_invalid_coords(self):
        return pd.DataFrame({
            "latitude": [999],
            "longitude": [999],
            "date_et_heure_de_comptage_utc": [
                pd.Timestamp("2023-06-01 12:00:00", tz="UTC")
            ]
        })

    # === Tests ===
    @patch(
        "smartcheck.preprocessing_project_specific."
        "fetch_weather_data_from_dataframe"
    )
    def test_transform_api_failure(self, mock_fetch, df_invalid_coords):
        mock_fetch.return_value = pd.DataFrame()
        transformer = WeatherDataEnrichmentTransformer(
            lat_col="latitude",
            lon_col="longitude",
            datetime_col="date_et_heure_de_comptage_utc"
        )
        result = transformer.transform(df_invalid_coords)
        pd.testing.assert_frame_equal(result, df_invalid_coords)

    @patch(
        "smartcheck.preprocessing_project_specific."
        "fetch_weather_data_from_dataframe"
    )
    def test_transform_successful_merge(self, mock_fetch, df_invalid_coords):
        mock_weather = df_invalid_coords.copy()
        mock_weather["temperature_2m"] = [20.5]
        mock_fetch.return_value = mock_weather

        transformer = WeatherDataEnrichmentTransformer(
            lat_col="latitude",
            lon_col="longitude",
            datetime_col="date_et_heure_de_comptage_utc"
        )
        result = transformer.transform(df_invalid_coords)
        assert "temperature_2m" in result.columns
        assert result.shape[0] == 1

    def test_fit_returns_self(self, df_invalid_coords):
        transformer = WeatherDataEnrichmentTransformer(
            lat_col="latitude",
            lon_col="longitude",
            datetime_col="date_et_heure_de_comptage_utc"
        )
        assert transformer.fit(df_invalid_coords) is transformer

    @patch(
        "smartcheck.preprocessing_project_specific."
        "fetch_weather_data_from_dataframe"
    )
    def test_transform_merge_adds_columns_correctly(self, mock_fetch,
                                                    df_invalid_coords):
        # Données météo fictives avec des colonnes supplémentaires
        mock_weather = pd.DataFrame({
            "latitude": [999],
            "longitude": [999],
            "date_et_heure_de_comptage_utc": [
                pd.Timestamp("2023-06-01 12:00:00", tz="UTC")
            ],
            "wind_speed": [5.2],
            "humidity": [78],
        })
        mock_fetch.return_value = mock_weather

        transformer = WeatherDataEnrichmentTransformer(
            lat_col="latitude",
            lon_col="longitude",
            datetime_col="date_et_heure_de_comptage_utc"
        )
        result = transformer.transform(df_invalid_coords)

        # Vérifie que les nouvelles colonnes sont bien présentes après le merge
        assert "wind_speed" in result.columns
        assert "humidity" in result.columns
        assert result.loc[0, "wind_speed"] == 5.2
        assert result.loc[0, "humidity"] == 78


# === Test class for ColumnNameNormalizerTransformer ===
class TestColumnNameNormalizerTransformer:

    # === Data Fixtures ===
    @pytest.fixture
    def df_raw(self):
        return pd.DataFrame({
            "Date d'Heure": [1],
            "Vélos Total (%)": [2]
        })

    # === Tests ===
    def test_transform_normalizes(self, df_raw):
        transformer = ColumnNameNormalizerTransformer()
        result = transformer.transform(df_raw)
        assert "date_d_heure" in result.columns
        assert "velos_total" in result.columns

    def test_fit_returns_self(self, df_raw):
        transformer = ColumnNameNormalizerTransformer()
        assert transformer.fit(df_raw) is transformer


# === Test class for SchoolHolidayTransformer ===
class TestSchoolHolidayTransformer:

    # === Data Fixtures ===
    @pytest.fixture
    def df_known_holiday(self):
        return pd.DataFrame({
            "date_et_heure_de_comptage_local": [
                pd.Timestamp("2023-02-20 12:00:00", tz="Europe/Paris")
            ]
        })

    @pytest.fixture
    def df_no_holiday(self):
        return pd.DataFrame({
            "date_et_heure_de_comptage_local": [
                pd.Timestamp("2023-03-01 12:00:00", tz="Europe/Paris")
            ]
        })

    # === Tests ===
    @patch("smartcheck.preprocessing_project_specific.add_school_vacation_column")
    def test_transform_adds_column(self, mock_add, df_known_holiday):
        mock_result = df_known_holiday.copy()
        mock_result["vacances_scolaires"] = [1]
        mock_add.return_value = mock_result

        transformer = SchoolHolidayTransformer(
            datetime_col="date_et_heure_de_comptage_local",
            location="Paris",
            zone="Zone C"
        )
        result = transformer.transform(df_known_holiday)
        assert "vacances_scolaires" in result.columns
        assert result["vacances_scolaires"].iloc[0] == 1

    @patch("smartcheck.preprocessing_project_specific.add_school_vacation_column")
    def test_transform_no_holiday(self, mock_add, df_no_holiday):
        mock_result = df_no_holiday.copy()
        mock_result["vacances_scolaires"] = [0]
        mock_add.return_value = mock_result

        transformer = SchoolHolidayTransformer(
            datetime_col="date_et_heure_de_comptage_local",
            location="Paris",
            zone="Zone C"
        )
        result = transformer.transform(df_no_holiday)
        assert result["vacances_scolaires"].iloc[0] == 0

    def test_fit_returns_self(self, df_known_holiday):
        transformer = SchoolHolidayTransformer(
            datetime_col="date_et_heure_de_comptage_local"
        )
        assert transformer.fit(df_known_holiday) is transformer


# === Test class for HolidayFromDatetimeTransformer ===
class TestHolidayFromDatetimeTransformer:

    # === Data Fixtures ===
    @pytest.fixture
    def df_existing_column(self):
        return pd.DataFrame({
            "datetime": [pd.Timestamp("2023-07-14 10:00:00", tz="Europe/Paris")],
            "jour_ferie": [42]
        })

    @pytest.fixture
    def df_missing_col(self):
        return pd.DataFrame({"other": [1]})

    # === Tests ===
    @patch("smartcheck.preprocessing_project_specific.add_holiday_column_from_datetime")
    def test_transform_overwrites_existing_column(self, mock_add, df_existing_column):
        mock_add.return_value = df_existing_column.copy()
        transformer = HolidayFromDatetimeTransformer(datetime_col="datetime")
        result = transformer.transform(df_existing_column)
        assert "jour_ferie" in result.columns

    @patch("smartcheck.preprocessing_project_specific.add_holiday_column_from_datetime")
    def test_transform_adds_column_success(self, mock_add, df_existing_column):
        mock_result = df_existing_column.copy()
        mock_result["jour_ferie"] = [1]
        mock_add.return_value = mock_result

        transformer = HolidayFromDatetimeTransformer(datetime_col="datetime")
        result = transformer.transform(df_existing_column)
        assert result["jour_ferie"].iloc[0] == 1

    def test_transform_missing_datetime_col(self, df_missing_col):
        transformer = HolidayFromDatetimeTransformer(datetime_col="missing")
        with pytest.raises(KeyError):
            transformer.transform(df_missing_col)

    def test_fit_returns_self(self, df_existing_column):
        transformer = HolidayFromDatetimeTransformer(datetime_col="datetime")
        assert transformer.fit(df_existing_column) is transformer
