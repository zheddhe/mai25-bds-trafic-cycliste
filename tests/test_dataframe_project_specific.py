import json
import pytest
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from requests.exceptions import HTTPError, RequestException
from unittest.mock import patch, Mock
from smartcheck.dataframe_project_specific import (
    extract_datetime_features,
    get_commune_from_coordinates,
    assign_communes_to_df,
    _load_communes_geojson,
    load_communes_from_config,
    fetch_weather_data_from_dataframe,
    parse_open_meteo_composite_csv,
    add_holiday_column_from_datetime,
    add_school_vacation_column,
    extract_datetime_periodic_features,
    train_test_split_time_aware,
    train_test_split_time_aware_sarimax,
)


class TestExtractDatetimeFeatures:
    """Unit tests for extract_datetime_features"""

    # === Fixtures ===
    @pytest.fixture
    def sample(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date_et_heure_de_comptage": [
                "2025-03-29T02:30:00+0100",
                "2025-03-30T03:30:00+0200",
            ]
        })

    # === Tests ===
    def test_datetime_features_extraction(self, sample):
        result = extract_datetime_features(
            sample,
            timestamp_col="date_et_heure_de_comptage",
            tz_local="Europe/Paris"
        )
        assert result["date_et_heure_de_comptage_local"].dt.hour.tolist() == [2, 3]
        assert result["date_et_heure_de_comptage_year"].tolist() == [2025, 2025]
        assert result["date_et_heure_de_comptage_day_of_year"].tolist() == [88, 89]

    def test_datetime_features_extraction_sarimax(self, sample):
        result = extract_datetime_features(
            sample,
            timestamp_col="date_et_heure_de_comptage",
            tz_local="Europe/Paris",
            for_sarimax=True
        )
        assert result["date_et_heure_de_comptage_local"].dt.hour.tolist() == [2, 3]
        assert "date_et_heure_de_comptage_day_of_year" not in result.columns

    def test_invalid_datetime_format_raises(self):
        df = pd.DataFrame({"bad_ts": ["invalid-timestamp"]})
        with pytest.raises(Exception):
            extract_datetime_features(df, "bad_ts", tz_local="Europe/Paris")


class TestLoadCommunesGeojsonRaw:
    """Unit tests for _load_communes_geojson"""

    # === Fixtures ===
    @pytest.fixture
    def dummy_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({"commune": ["01"]})

    # === Tests ===
    def test_geojson_load_failure_raises_and_logs(self, caplog):
        with caplog.at_level("ERROR"):
            with pytest.raises(Exception):
                _load_communes_geojson("invalid_path.geojson")
        assert "Failed to load commune GeoJSON" in caplog.text

    def test_geojson_load_success_from_local_file(self, tmp_path: Path):
        geojson_data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2.34, 48.85], [2.36, 48.85],
                                     [2.36, 48.87], [2.34, 48.87], [2.34, 48.85]]]
                },
                "properties": {"commune": "01"}
            }]
        }
        path = tmp_path / "communes.geojson"
        path.write_text(json.dumps(geojson_data), encoding="utf-8")
        gdf = _load_communes_geojson(str(path))
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "commune" in gdf.columns
        assert len(gdf) == 1

    @patch("smartcheck.dataframe_project_specific.gpd.read_file")
    @patch("smartcheck.dataframe_common._download_google_drive_file", return_value='{}')
    @patch("smartcheck.dataframe_common._extract_google_drive_file_id",
           return_value="fake_id")
    def test_geojson_load_success_from_google_drive(
        self, mock_id, mock_dl, mock_read, dummy_gdf
    ):
        mock_read.return_value = dummy_gdf
        gdf = _load_communes_geojson("https://drive.google.com/file/d/fake_id/view")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "commune" in gdf.columns
        assert len(gdf) == 1
        mock_read.assert_called_once()

    @patch("smartcheck.dataframe_common._extract_google_drive_file_id",
           return_value=None)
    def test_google_drive_invalid_url_raises(self, mock_extract):
        with pytest.raises(ValueError, match="Could not extract file ID"):
            _load_communes_geojson("https://drive.google.com/file/d//view")


class TestLoadCommunesFromConfig:
    """Unit tests for load_communes_from_config"""

    # === Fixtures ===
    @pytest.fixture
    def dummy_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({"commune": ["01"]})

    # === Tests ===
    @patch("smartcheck.dataframe_project_specific._load_communes_geojson")
    @patch("smartcheck.dataframe_project_specific.load_config")
    def test_load_from_config_success(self, mock_config, mock_loader, dummy_gdf):
        mock_config.return_value = {
            "data": {
                "input": {
                    "communes_geo_data": "dummy/path.geojson"
                }
            }
        }
        mock_loader.return_value = dummy_gdf
        gdf = load_communes_from_config()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "commune" in gdf.columns
        mock_loader.assert_called_once_with("dummy/path.geojson")

    @patch("smartcheck.dataframe_project_specific.load_config")
    def test_load_from_config_missing_key_raises(self, mock_config):
        mock_config.return_value = {"data": {"input": {}}}
        with pytest.raises(ValueError, match="Missing config entry:"):
            load_communes_from_config()


class TestGetCommuneFromCoordinates:
    """Unit tests for get_commune_from_coordinates"""

    # === Fixtures ===
    @pytest.fixture
    def dummy_communes(self) -> gpd.GeoDataFrame:
        polygon = Polygon([
            (2.34, 48.85), (2.36, 48.85),
            (2.36, 48.87), (2.34, 48.87)
        ])
        return gpd.GeoDataFrame({"commune": ["01"], "geometry": [polygon]},
                                crs="EPSG:4326")

    # === Tests ===
    def test_point_inside_polygon(self, dummy_communes):
        result = get_commune_from_coordinates(2.35, 48.86, dummy_communes)
        assert result == "01"

    def test_point_outside_polygon(self, dummy_communes):
        result = get_commune_from_coordinates(2.10, 48.50, dummy_communes)
        assert result is None


class TestAssignCommunesToDf:
    """Unit tests for assign_communes_to_df"""

    # === Fixtures ===
    @pytest.fixture
    def polygon_commune(self) -> gpd.GeoDataFrame:
        polygon = Polygon([
            (2.34, 48.85), (2.36, 48.85),
            (2.36, 48.87), (2.34, 48.87)
        ])
        return gpd.GeoDataFrame(
            {"commune": ["Paris 01"], "geometry": [polygon]},
            crs="EPSG:4326"
        )

    @pytest.fixture
    def polygon_commune_lambert(self) -> gpd.GeoDataFrame:
        polygon = Polygon([
            (651000, 6862000), (653000, 6862000),
            (653000, 6864000), (651000, 6864000)
        ])
        return gpd.GeoDataFrame(
            {"commune": ["Paris L93"], "geometry": [polygon]},
            crs="EPSG:2154"
        )

    @pytest.fixture
    def df_inside(self) -> pd.DataFrame:
        return pd.DataFrame({"lon": [2.35], "lat": [48.86]})

    @pytest.fixture
    def df_outside(self) -> pd.DataFrame:
        return pd.DataFrame({"lon": [2.10], "lat": [48.50]})

    # === Tests ===
    def test_assign_within_success(self, df_inside, polygon_commune):
        result = assign_communes_to_df(
            df_inside, "lon", "lat", polygon_commune,
            commune_column="commune", output_column="result_commune"
        )
        assert "result_commune" in result.columns
        assert result.loc[0, "result_commune"] == "Paris 01"

    def test_assign_within_fallback_to_intersects(self, df_inside):
        polygon = Polygon([
            (2.35, 48.86), (2.36, 48.86),
            (2.36, 48.87), (2.35, 48.87)
        ])
        gdf = gpd.GeoDataFrame(
            {"commune": ["Paris Border"], "geometry": [polygon]},
            crs="EPSG:4326"
        )
        result = assign_communes_to_df(
            df_inside, "lon", "lat", gdf,
            commune_column="commune", output_column="assigned"
        )
        assert result["assigned"].iloc[0] == "Paris Border"

    def test_assign_reprojects_crs(self, df_inside, polygon_commune_lambert):
        result = assign_communes_to_df(
            df_inside, "lon", "lat", polygon_commune_lambert,
            commune_column="commune", output_column="assigned"
        )
        assert isinstance(result.loc[0, "assigned"], str)

    def test_missing_column_raises(self, df_inside):
        polygon = Polygon([
            (2.34, 48.85), (2.36, 48.85),
            (2.36, 48.87), (2.34, 48.87)
        ])
        gdf = gpd.GeoDataFrame(
            {"wrong_col": ["X"], "geometry": [polygon]},
            crs="EPSG:4326"
        )
        with pytest.raises(ValueError, match="not found in communes"):
            assign_communes_to_df(
                df_inside, "lon", "lat", gdf,
                commune_column="commune", output_column="assigned"
            )

    def test_no_match_returns_none(self, df_outside, polygon_commune):
        result = assign_communes_to_df(
            df_outside, "lon", "lat", polygon_commune,
            commune_column="commune", output_column="assigned"
        )
        assert result["assigned"].isna().all()


class TestFetchWeatherDataFromDataFrame:
    """Unit tests for fetch_weather_data_from_dataframe"""

    # === Fixtures ===
    @pytest.fixture
    def minimal_df(self):
        return pd.DataFrame({
            "lat": [48.8566],
            "lon": [2.3522],
            "timestamp": [pd.Timestamp("2024-06-01 00:00:00", tz="UTC")]
        })

    @pytest.fixture
    def mock_response(self):
        return Mock(
            status_code=200,
            text="mock csv",
            raise_for_status=Mock()
        )

    # === Tests ===
    def test_missing_columns(self, caplog):
        df = pd.DataFrame({
            "lat": [48.85],  # missing 'lon' and 'timestamp'
        })
        result = fetch_weather_data_from_dataframe(df, "lat", "lon", "timestamp")
        assert result.empty
        assert "Missing required columns" in caplog.text

    @patch("smartcheck.dataframe_common.requests.get")
    @patch("smartcheck.dataframe_project_specific.parse_open_meteo_composite_csv")
    def test_valid_response(self, mock_parse, mock_get, minimal_df, mock_response):
        mock_get.return_value = mock_response
        mock_parse.return_value = pd.DataFrame({"temperature_2m": [20.0]})

        result = fetch_weather_data_from_dataframe(
            minimal_df, "lat", "lon", "timestamp"
        )
        assert not result.empty
        assert "temperature_2m" in result.columns
        mock_get.assert_called_once()
        mock_parse.assert_called_once()

    @patch("smartcheck.dataframe_common.requests.get")
    def test_http_error(self, mock_get, minimal_df, caplog):
        response = Mock()
        response.text = "error page"
        error = HTTPError("HTTP error occurred", response=response)
        mock_get.side_effect = error

        result = fetch_weather_data_from_dataframe(
            minimal_df, "lat", "lon", "timestamp"
        )
        assert result.empty
        assert "HTTP error while fetching weather data" in caplog.text
        assert "Response: error page" in caplog.text

    @patch("smartcheck.dataframe_common.requests.get")
    def test_request_exception(self, mock_get, minimal_df, caplog):
        mock_get.side_effect = RequestException("Connection failed")

        result = fetch_weather_data_from_dataframe(
            minimal_df, "lat", "lon", "timestamp"
        )
        assert result.empty
        assert "Non-HTTP error while fetching weather data" in caplog.text


class TestParseOpenMeteoCompositeCsv:
    """Unit tests for parse_open_meteo_composite_csv"""

    # === Fixtures ===
    @pytest.fixture
    def valid_csv_content(self):
        return (
            "location_id,elevation\n"
            "0,35\n"
            "1,42\n"
            "location_id,time,temperature_2m,weather_code\n"
            "0,2024-06-01T00:00,20.0,1\n"
            "1,2024-06-01T00:00,21.5,2\n"
        )

    @pytest.fixture
    def malformed_csv_content(self):
        return (
            "location_id,elevation\n"
            "0,35\n"
            "1,42\n"
            "no second block here"
        )

    @pytest.fixture
    def coord_tuples(self):
        return [(48.85, 2.35), (48.86, 2.36)]

    # === Tests ===
    def test_parses_valid_csv(self, valid_csv_content, coord_tuples):
        df = parse_open_meteo_composite_csv(
            content=valid_csv_content,
            coord_tuples=coord_tuples,
            datetime_col="timestamp"
        )

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"elevation", "temperature_2m", "weather_code",
                                   "timestamp", "latitude", "longitude"}
        assert df.shape[0] == 2
        assert df["latitude"].iloc[0] == coord_tuples[0][0]
        assert df["longitude"].iloc[1] == coord_tuples[1][1]

    def test_drops_na_datetimes(self, coord_tuples):
        content = (
            "location_id,elevation\n"
            "0,35\n"
            "1,42\n"
            "location_id,time,temperature_2m,weather_code\n"
            "0,,20.0,1\n"
            "1,2024-06-01T00:00,21.5,2\n"
        )
        df = parse_open_meteo_composite_csv(
            content=content,
            coord_tuples=coord_tuples,
            datetime_col="timestamp"
        )
        assert df.shape[0] == 1
        assert df["temperature_2m"].iloc[0] == 21.5


class TestAddHolidayColumnFromDatetime:
    """Unit tests for add_holiday_column_from_datetime"""

    # === Fixtures ===
    @pytest.fixture
    def df_14juillet(self):
        return pd.DataFrame({
            "datetime": [pd.Timestamp("2023-07-14 10:00:00", tz="Europe/Paris")]
        })

    @pytest.fixture
    def df_non_ferie(self):
        return pd.DataFrame({
            "datetime": [pd.Timestamp("2023-07-15 10:00:00", tz="Europe/Paris")]
        })

    @pytest.fixture
    def df_multi_annees(self):
        return pd.DataFrame({
            "datetime": [
                pd.Timestamp("2022-11-01 10:00:00", tz="Europe/Paris"),
                pd.Timestamp("2023-07-14 10:00:00", tz="Europe/Paris"),
            ]
        })

    # === Tests ===
    def test_error_if_not_datetime(self):
        df = pd.DataFrame({"datetime": ["not a datetime"]})
        with pytest.raises(ValueError):
            add_holiday_column_from_datetime(df, "datetime")

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_adds_jour_ferie_if_date_matches(self, mock_get, df_14juillet):
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {
            "2023-07-14": "Fête nationale"
        }

        result = add_holiday_column_from_datetime(df_14juillet, "datetime")
        assert result["jour_ferie"].iloc[0] == 1

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_adds_zero_if_not_holiday(self, mock_get, df_non_ferie):
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {
            "2023-07-14": "Fête nationale"
        }

        result = add_holiday_column_from_datetime(df_non_ferie, "datetime")
        assert result["jour_ferie"].iloc[0] == 0

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_api_called_for_each_year(self, mock_get, df_multi_annees):
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {}

        add_holiday_column_from_datetime(df_multi_annees, "datetime")
        called_urls = [call.args[0] for call in mock_get.call_args_list]
        assert "2022" in called_urls[0]
        assert "2023" in called_urls[1]
        assert len(called_urls) == 2

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_handles_api_failure_gracefully(self, mock_get, df_14juillet):
        mock_get.side_effect = RequestException("API failure")

        result = add_holiday_column_from_datetime(df_14juillet, "datetime")
        assert "jour_ferie" in result.columns
        assert result["jour_ferie"].iloc[0] == 0


class TestAddSchoolVacationColumn:
    """Unit tests for add_school_vacation_column"""

    # === Fixtures ===
    @pytest.fixture
    def df_inside_holiday(self):
        return pd.DataFrame({
            "datetime": [pd.Timestamp("2024-02-15 10:00:00", tz="UTC")]
        })

    @pytest.fixture
    def df_outside_holiday(self):
        return pd.DataFrame({
            "datetime": [pd.Timestamp("2024-03-15 10:00:00", tz="Europe/Paris")]
        })

    def test_raises_on_non_datetime(self):
        df = pd.DataFrame({"datetime": ["not a datetime"]})
        with pytest.raises(ValueError):
            add_school_vacation_column(df, "datetime")

    # === Tests ===
    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_detects_vacation_period(self, mock_get, df_inside_holiday):
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = [
            {
                "location": "Paris",
                "zones": "Zone C",
                "start_date": "2024-06-10",
                "end_date": "2024-06-25",
                "description": "Vacances d'été"
            },
            {
                "location": "Paris",
                "zones": "Zone C",
                "start_date": "2024-02-10",
                "end_date": "2024-02-25",
                "description": "Vacances d'hiver"
            }
        ]

        df_out = add_school_vacation_column(df_inside_holiday, "datetime")
        assert df_out["vacances_scolaires"].iloc[0] == "Vacances d'hiver"

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_returns_aucune_if_not_in_period(self, mock_get, df_outside_holiday):
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = [
            {
                "location": "Paris",
                "zones": "Zone C",
                "start_date": "2024-02-10",
                "end_date": "2024-02-25",
                "description": "Vacances d'hiver"
            }
        ]

        df_out = add_school_vacation_column(df_outside_holiday, "datetime")
        assert df_out["vacances_scolaires"].iloc[0] == "aucune"

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_api_failure_sets_erreur_api(self, mock_get, df_inside_holiday):
        mock_get.side_effect = RequestException("API down")

        df_out = add_school_vacation_column(df_inside_holiday, "datetime")
        assert df_out["vacances_scolaires"].iloc[0] == "erreur_api"

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_malformed_entry_is_ignored(self, mock_get, df_inside_holiday, caplog):
        mock_get.return_value = Mock(status_code=200)
        # end_date invalide
        mock_get.return_value.json.return_value = [
            {
                "location": "Paris",
                "zones": "Zone C",
                "start_date": "2024-02-10",
                "end_date": "not-a-date",
                "description": "Vacances"
            }
        ]

        df_out = add_school_vacation_column(df_inside_holiday, "datetime")
        assert df_out["vacances_scolaires"].iloc[0] == "aucune"
        assert "Failed to parse record" in caplog.text

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_ignores_wrong_zone_or_location(self, mock_get, df_inside_holiday):
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = [
            {
                "location": "Lyon",
                "zones": "Zone A",
                "start_date": "2024-02-10",
                "end_date": "2024-02-25",
                "description": "Vacances"
            }
        ]

        df_out = add_school_vacation_column(df_inside_holiday, "datetime")
        assert df_out["vacances_scolaires"].iloc[0] == "aucune"

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_api_crash_sets_erreur_api(self, mock_get, df_inside_holiday, caplog):
        mock_get.side_effect = Exception("Boom!")

        df_out = add_school_vacation_column(df_inside_holiday, "datetime")

        assert df_out["vacances_scolaires"].iloc[0] == "erreur_api"
        assert "Error fetching school holiday data" in caplog.text

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_parse_error_logged_and_ignored(self, mock_get, df_inside_holiday, caplog):
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = [
            {
                "location": "Paris",
                "zones": "Zone C",
                "start_date": "2024-02-10",
                "end_date": "XXX",  # date invalide
                "description": "Vacances"
            }
        ]

        df_out = add_school_vacation_column(df_inside_holiday, "datetime")
        assert df_out["vacances_scolaires"].iloc[0] == "aucune"
        assert "Failed to parse record" in caplog.text

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_returns_aucune_when_no_match(self, mock_get, df_outside_holiday):
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = [
            {
                "location": "Paris",
                "zones": "Zone C",
                "start_date": "2024-02-10",
                "end_date": "2024-02-25",
                "description": "Vacances d'hiver"
            }
        ]

        df_out = add_school_vacation_column(df_outside_holiday, "datetime")
        assert df_out["vacances_scolaires"].iloc[0] == "aucune"

    @patch("smartcheck.dataframe_project_specific.requests.get")
    def test_return_aucune_branch_is_covered(self, mock_get):
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2025-04-15 10:00:00", tz="UTC")]
        })

        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = [
            {
                "location": "Paris",
                "zones": "Zone C",
                "start_date": "2025-01-01",
                "end_date": "2025-01-10",
                "description": "Vacances"
            }
        ]

        df_out = add_school_vacation_column(df, "datetime")
        assert df_out["vacances_scolaires"].iloc[0] == "aucune"


class TestExtractDatetimePeriodicFeatures:
    """Unit tests for extract_datetime_periodic_features"""

    # === Fixtures ===
    @pytest.fixture
    def df_timestamps(self):
        return pd.DataFrame({
            "ts": [
                "2024-01-01T08:00:00+0000",
                "2024-06-15T18:30:00+0000"
            ]
        })

    # === Tests ===
    def test_extracted_columns_exist(self, df_timestamps):
        enriched = extract_datetime_periodic_features(df_timestamps, "ts")

        expected_cols = [
            "ts_utc", "ts_local", "ts_year", "ts_month", "ts_day",
            "ts_day_of_year", "ts_day_of_week", "ts_hour", "ts_week",
            "ts_sin_hour", "ts_cos_hour", "ts_sin_day_of_week",
            "ts_cos_day_of_week", "ts_sin_month", "ts_cos_month",
            "ts_sin_week", "ts_cos_week"
        ]

        for col in expected_cols:
            assert col in enriched.columns

    def test_values_are_correct_shape_and_type(self, df_timestamps):
        enriched = extract_datetime_periodic_features(df_timestamps, "ts")

        assert enriched.shape[0] == 2
        assert pd.api.types.is_datetime64_any_dtype(enriched["ts_utc"])
        assert pd.api.types.is_datetime64_any_dtype(enriched["ts_local"])
        assert np.isclose(enriched["ts_sin_hour"] ** 2 +
                          enriched["ts_cos_hour"] ** 2, 1).all()

    def test_invalid_timestamp_raises(self):
        df_invalid = pd.DataFrame({
            "ts": ["not_a_timestamp"]
        })
        with pytest.raises(Exception):
            extract_datetime_periodic_features(df_invalid, "ts")


class TestTrainTestSplitTimeAware:
    """Unit tests for train_test_split_time_aware"""

    # === Fixtures ===
    @pytest.fixture
    def base_df(self):
        return pd.DataFrame({
            "timestamp_local": pd.date_range("2022-01-01", periods=10, freq="D"),
            "timestamp_utc": pd.date_range("2022-01-01", periods=10, freq="D"),
            "identifiant_compteur": ["A"] * 5 + ["B"] * 5,
            "volume": np.arange(10),
            "target": [0, 1] * 5
        })

    # === Tests ===
    def test_split(self, base_df):
        X_tr, X_tr_d, X_te, X_te_d, y_tr, y_te = train_test_split_time_aware(
            df=base_df,
            timestamp_cols=["timestamp_utc", "timestamp_local"],
            target_col="target",
            test_size=0.3
        )

        n_rows = len(base_df)
        n_test = int(n_rows * 0.3)
        n_train = n_rows - n_test

        assert len(X_tr) == n_train
        assert len(X_te) == n_test
        assert len(X_tr_d) == n_train
        assert len(X_te_d) == n_test
        assert len(y_tr) == n_train
        assert len(y_te) == n_test

    def test_removed_columns(self, base_df):
        X_tr, _, _, _, _, _ = train_test_split_time_aware(
            df=base_df,
            timestamp_cols=["timestamp_utc", "timestamp_local"],
            target_col="target"
        )
        assert "timestamp_utc" not in X_tr.columns
        assert "timestamp_local" not in X_tr.columns
        assert "target" not in X_tr.columns

    def test_invalid_timestamp_column(self, base_df):
        with pytest.raises(KeyError):
            train_test_split_time_aware(
                df=base_df,
                timestamp_cols=["nonexistent"],
                target_col="target"
            )


class TestTrainTestSplitTimeAwareSarimax:
    """Unit tests for train_test_split_time_aware_sarimax"""

    # === Fixtures ===
    @pytest.fixture
    def df_regular_hourly(self):
        timestamps = pd.date_range("2024-01-01", periods=24, freq="h")
        return pd.DataFrame({
            "ts": timestamps,
            "y": np.arange(24),
            "x1": np.random.randn(24)
        })

    @pytest.fixture
    def df_with_missing(self):
        timestamps = pd.date_range("2024-01-01", periods=30, freq="h")
        timestamps = timestamps.delete([5, 6, 20])  # introduce gaps
        return pd.DataFrame({
            "ts": timestamps,
            "y": np.arange(len(timestamps)),
            "x1": np.random.randn(len(timestamps))
        })

    # === Test ===
    def test_split_returns_correct_shapes(self, df_regular_hourly):
        X_tr, X_te, y_tr, y_te, gaps = train_test_split_time_aware_sarimax(
            df_regular_hourly,
            timestamp_col="ts",
            target_col="y",
            test_size=0.25
        )

        assert isinstance(X_tr, pd.DataFrame)
        assert isinstance(X_te, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)
        assert isinstance(y_te, pd.Series)
        assert isinstance(gaps, dict)

        n_total = df_regular_hourly.shape[0]
        n_test = int(n_total * 0.25)
        assert len(X_te) == n_test
        assert len(X_tr) + len(X_te) == n_total

    def test_missing_data_detected(self, df_with_missing):
        _, _, _, _, gaps = train_test_split_time_aware_sarimax(
            df_with_missing,
            timestamp_col="ts",
            target_col="y",
            test_size=0.2
        )

        assert isinstance(gaps, dict)
        assert len(gaps) > 0
        assert all("nb_missing" in v for v in gaps.values())

    def test_interpolation_if_gap_allowed(self, df_with_missing):
        _, _, _, _, gaps = train_test_split_time_aware_sarimax(
            df_with_missing,
            timestamp_col="ts",
            target_col="y",
            test_size=0.2,
            interpol_max=3
        )

        assert all(v["nb_missing"] <= 3 for v in gaps.values())

    def test_invalid_frequency_raises(self):
        df = pd.DataFrame({
            "ts": ["not_a_datetime"] * 5,
            "y": [1, 2, 3, 4, 5],
            "x1": [0] * 5
        })
        df["ts"] = pd.Series(df["ts"])

        with pytest.raises(ValueError, match="Cannot apply frequency"):
            train_test_split_time_aware_sarimax(
                df,
                timestamp_col="ts",
                target_col="y",
                test_size=0.2
            )
