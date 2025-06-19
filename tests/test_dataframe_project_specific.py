import json
import pytest
from pathlib import Path
import pandas as pd
import geopandas as gpd
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
    parse_open_meteo_composite_csv
)


# === Test class for extract_datetime_features ===
class TestExtractDatetimeFeatures:

    # === Data Fixtures ===
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

    def test_invalid_datetime_format_raises(self):
        df = pd.DataFrame({"bad_ts": ["invalid-timestamp"]})
        with pytest.raises(Exception):
            extract_datetime_features(df, "bad_ts", tz_local="Europe/Paris")


# === Test class for _load_communes_geojson ===
class TestLoadCommunesGeojsonRaw:

    # === Data Fixtures ===
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


# === Test class for load_communes_from_config ===
class TestLoadCommunesFromConfig:

    # === Data Fixtures ===
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


# === Test class for get_commune_from_coordinates ===
class TestGetCommuneFromCoordinates:

    # === Data Fixtures ===
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


# === Test class for assign_communes_to_df ===
class TestAssignCommunesToDf:

    # === Data Fixtures ===
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


# === Test class for fetch_weather_data_from_dataframe ===
class TestFetchWeatherDataFromDataFrame:

    # === Data Fixtures ===
    @pytest.fixture
    def minimal_df(self):
        return pd.DataFrame({
            "lat": [48.8566],
            "lon": [2.3522],
            "timestamp": [pd.Timestamp("2024-06-01 00:00:00", tz="UTC")]
        })

    # === Mock Fixtures ===
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


# === Test class for parse_open_meteo_composite_csv ===
class TestParseOpenMeteoCompositeCsv:

    # === Data Fixtures ===
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
