import json
import pytest
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from unittest.mock import patch
from smartcheck.dataframe_project_specific import (
    extract_datetime_features,
    get_commune_from_coordinates,
    assign_communes_to_df,
    _load_communes_geojson,
    load_communes_from_config,
)


# === Test class for extract_datetime_features ===
class TestExtractDatetimeFeatures:

    @pytest.fixture
    def sample(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date_et_heure_de_comptage": [
                "2025-03-29T02:30:00+0100",
                "2025-03-30T03:30:00+0200",
            ]
        })

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

    @pytest.fixture
    def dummy_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({"commune": ["01"]})

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

    @pytest.fixture
    def dummy_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({"commune": ["01"]})

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

    @pytest.fixture
    def dummy_communes(self) -> gpd.GeoDataFrame:
        polygon = Polygon([
            (2.34, 48.85), (2.36, 48.85),
            (2.36, 48.87), (2.34, 48.87)
        ])
        return gpd.GeoDataFrame({"commune": ["01"], "geometry": [polygon]},
                                crs="EPSG:4326")

    def test_point_inside_polygon(self, dummy_communes):
        result = get_commune_from_coordinates(2.35, 48.86, dummy_communes)
        assert result == "01"

    def test_point_outside_polygon(self, dummy_communes):
        result = get_commune_from_coordinates(2.10, 48.50, dummy_communes)
        assert result is None


# === Test class for assign_communes_to_df ===
class TestAssignCommunesToDf:

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
