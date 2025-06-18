import io
import logging
from typing import Optional, Union

import pytz
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from smartcheck.paths import load_config
from smartcheck.dataframe_common import (
    _extract_google_drive_file_id,
    _download_google_drive_file,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s]-[%(asctime)s] %(message)s'
)


def extract_datetime_features(
    df: pd.DataFrame,
    timestamp_col: str,
    tz_local: str = "Europe/Paris"
) -> pd.DataFrame:
    """
    Parse ISO8601 timestamps with offset in `timestamp_col`, convert to UTC
    then to `tz_local`, and extract common calendar features.

    Returns a copy of the DataFrame with additional columns.

    Args:
        df: Input DataFrame.
        timestamp_col: Column with ISO8601 timestamp strings.
        tz_local: Local timezone (default: Europe/Paris).

    Returns:
        pd.DataFrame: Copy with time components extracted.
    """
    df = df.copy()
    try:
        df[f"{timestamp_col}_utc"] = pd.to_datetime(
            df[timestamp_col],
            format="%Y-%m-%dT%H:%M:%S%z",
            utc=True
        )
        df[f"{timestamp_col}_local"] = (
            df[f"{timestamp_col}_utc"]
            .dt.tz_convert(pytz.timezone(tz_local))
        )
        ts = df[f"{timestamp_col}_local"]
        df[f"{timestamp_col}_year"] = ts.dt.year
        df[f"{timestamp_col}_month"] = ts.dt.month
        df[f"{timestamp_col}_day"] = ts.dt.day
        df[f"{timestamp_col}_day_of_year"] = ts.dt.dayofyear
        df[f"{timestamp_col}_day_of_week"] = ts.dt.dayofweek
        df[f"{timestamp_col}_hour"] = ts.dt.hour
        df[f"{timestamp_col}_week"] = ts.dt.isocalendar().week
        df[f"{timestamp_col}_dayname"] = ts.dt.day_name()
        df[f"{timestamp_col}_monthname"] = ts.dt.month_name()

        return df

    except Exception as exc:
        logger.error(
            "Error extracting datetime features from column '%s': %s",
            timestamp_col, exc
        )
        raise


def _load_communes_geojson(path: str) -> gpd.GeoDataFrame:
    """
    Internal loader for commune (or equivalent) GeoJSON file
    from a local path or Google Drive.

    Args:
        path: File path or Google Drive URL.

    Returns:
        gpd.GeoDataFrame
    """
    try:
        if path.startswith("https://drive.google.com"):
            file_id = _extract_google_drive_file_id(path)
            if not file_id:
                raise ValueError("Could not extract file ID from Google Drive URL.")
            logger.info("Downloading commune GeoJSON from Google Drive.")
            content = _download_google_drive_file(file_id)
            return gpd.read_file(io.StringIO(content))
        else:
            logger.info("Loading commune GeoJSON from local path.")
            return gpd.read_file(path)
    except Exception as e:
        logger.error("Failed to load commune GeoJSON: %s", e)
        raise


def load_communes_from_config(
    config_key: str = "communes_geo_data"
) -> gpd.GeoDataFrame:
    """
    Load commune GeoJSON using a configuration key.

    Args:
        config_key: Key in config['data']['input'] to locate the file.

    Returns:
        gpd.GeoDataFrame: Geometries of communes or equivalent.
    """
    config = load_config()
    file_path = config["data"]["input"].get(config_key)

    if not file_path:
        logger.error("GeoJSON config key '%s' not found in configuration.", config_key)
        raise ValueError(f"Missing config entry: {config_key}")

    logger.info("Resolved geojson path from config: %s", file_path)
    return _load_communes_geojson(file_path)


def get_commune_from_coordinates(
    lon: float,
    lat: float,
    communes_gdf: gpd.GeoDataFrame,
    commune_column: str = "commune"
) -> Optional[Union[str, int]]:
    """
    Find the geographical unit (e.g., commune) containing the given point.

    Args:
        lon: Longitude.
        lat: Latitude.
        communes_gdf: GeoDataFrame with polygons.
        commune_column: Label column to retrieve.

    Returns:
        Label of matching commune, or None.
    """
    point = Point(lon, lat)
    matching = communes_gdf[communes_gdf.contains(point)]

    if not matching.empty:
        result = matching.iloc[0][commune_column]
        logger.debug("Point (%f, %f) found in commune: %s", lon, lat, result)
        return result

    logger.warning("Point (%f, %f) not found in any commune.", lon, lat)
    return None


def assign_communes_to_df(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    communes_gdf: gpd.GeoDataFrame,
    commune_column: str = "commune",
    output_column: str = "commune"
) -> pd.DataFrame:
    """
    Assign a geographical unit (e.g., commune) to each point based on spatial join.

    Args:
        df: DataFrame with coordinates.
        lon_col: Longitude column name.
        lat_col: Latitude column name.
        communes_gdf: GeoDataFrame with geometry + label.
        commune_column: Name of label column in `communes_gdf`.
        output_column: Name of column to add to `df`.

    Returns:
        pd.DataFrame: Copy with `output_column` added.
    """
    df = df.copy()
    geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
    df_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    if communes_gdf.crs != "EPSG:4326":
        logger.info("Reprojecting communes GeoDataFrame to EPSG:4326")
        communes_gdf = communes_gdf.to_crs("EPSG:4326")

    if commune_column not in communes_gdf.columns:
        raise ValueError(
            f"Column '{commune_column}' not found in communes GeoDataFrame."
        )

    joined = gpd.sjoin(
        df_gdf,
        communes_gdf[[commune_column, "geometry"]],
        how="left",
        predicate="within"
    )

    if joined[commune_column].isna().all():
        logger.warning(
            "No matches found with predicate='within'. Trying predicate='intersects'."
        )
        joined = gpd.sjoin(
            df_gdf,
            communes_gdf[[commune_column, "geometry"]],
            how="left",
            predicate="intersects"
        )

    joined = joined.loc[~joined.index.duplicated(keep="first")]
    df[output_column] = joined[commune_column].reindex(df.index).values

    return df
