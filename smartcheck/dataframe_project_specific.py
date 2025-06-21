import io
import logging
import requests
from typing import Optional, Union
from requests.exceptions import HTTPError, RequestException

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


def fetch_weather_data_from_dataframe(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    datetime_col: str
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API using aligned lat/lon pairs.

    Args:
        df (pd.DataFrame): DataFrame with lat, lon, and datetime columns.
        lat_col (str): Column name for latitude.
        lon_col (str): Column name for longitude.
        datetime_col (str): Column name for datetime (tz-aware, Etc/UTC).

    Returns:
        pd.DataFrame: Hourly weather data per coordinate and timestamp.
    """
    required_cols = {lat_col, lon_col, datetime_col}
    if not required_cols.issubset(df.columns):
        logger.error(
            f"Missing required columns: {required_cols - set(df.columns)}"
        )
        return pd.DataFrame()

    try:
        df_unique = df[[lat_col, lon_col, datetime_col]].drop_duplicates()
        df_unique["date"] = pd.to_datetime(df_unique[datetime_col]).dt.date

        start_date = df_unique["date"].min().strftime("%Y-%m-%d")
        end_date = df_unique["date"].max().strftime("%Y-%m-%d")

        coord_pairs = (
            df_unique[[lat_col, lon_col]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        lat_list = coord_pairs[lat_col].tolist()
        lon_list = coord_pairs[lon_col].tolist()

        latitudes = ",".join(map(str, lat_list))
        longitudes = ",".join(map(str, lon_list))

        url = (
            "https://historical-forecast-api.open-meteo.com/v1/forecast?"
            f"latitude={latitudes}"
            f"&longitude={longitudes}"
            f"&start_date={start_date}"
            f"&end_date={end_date}"
            "&hourly=temperature_2m,weather_code,rain,snowfall"
            "&timezone=Etc%2FUTC&format=csv"
        )

        logger.info(
            f"🌤 Fetching weather data from Open-Meteo "
            f"for {len(lat_list)} coordinate pairs."
        )
        logger.debug(f"URL [{url}]")

        response = requests.get(url)
        response.raise_for_status()

        coord_tuples = list(zip(lat_list, lon_list))
        df_weather = parse_open_meteo_composite_csv(
            response.text,
            coord_tuples,
            datetime_col
        )
        return df_weather

    except HTTPError as exc:
        logger.error(f"HTTP error while fetching weather data: {exc}")
        logger.error(f"Response: {exc.response.text}")
        return pd.DataFrame()

    except RequestException as exc:
        logger.error(f"Non-HTTP error while fetching weather data: {exc}")
        return pd.DataFrame()


def parse_open_meteo_composite_csv(
    content: str,
    coord_tuples: list[tuple[float, float]],
    datetime_col: str
) -> pd.DataFrame:
    """
    Parse Open-Meteo CSV with weather data and elevation per coordinate.

    Args:
        content (str): Raw CSV response from Open-Meteo API.
        coord_tuples (list): Ordered list of (lat, lon) used in request.
        datetime_col (str): Column name for datetime (tz-aware, Etc/UTC).

    Returns:
        pd.DataFrame: Weather data enriched with elevation and lat/lon.
    """
    lines = content.strip().splitlines()

    split_idx = [
        idx for idx, line in enumerate(lines)
        if line.startswith("location_id") and idx > 0
    ]

    first_block = lines[:split_idx[0]]
    second_block = lines[split_idx[0]:]

    df_meta = pd.read_csv(io.StringIO("\n".join(first_block)))
    df_weather = pd.read_csv(io.StringIO("\n".join(second_block)))

    df_meta = df_meta[["location_id", "elevation"]].copy()
    df_meta["latitude"] = [lat for lat, _ in coord_tuples]
    df_meta["longitude"] = [lon for _, lon in coord_tuples]

    df_weather[datetime_col] = pd.to_datetime(df_weather["time"]).dt.tz_localize(
        "Etc/UTC"
    )
    df_weather = df_weather.dropna(subset=[datetime_col])

    df_final = pd.merge(
        df_weather.drop(columns=["time"]),
        df_meta,
        on="location_id",
        how="left"
    )
    df_final = df_final.drop(columns=['location_id'])

    return df_final


def add_holiday_column_from_datetime(df: pd.DataFrame,
                                     datetime_col: str,
                                     country_code: str = "metropole") -> pd.DataFrame:
    """
    Add a 'jour_ferie' binary column indicating if a datetime falls on a French holiday.

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime column (timezone-aware).
        datetime_col (str): Name of the datetime column (must be timezone-aware).
        country_code (str): Code used in the API (e.g., 'metropole', 'alsace-moselle').

    Returns:
        pd.DataFrame: DataFrame with added 'jour_ferie' column (0 if not, 1 if holiday).
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        raise ValueError(f"Column '{datetime_col}' must be a datetime64 dtype.")

    local_dates = df[datetime_col].dt.tz_convert("Europe/Paris").dt.date
    min_year, max_year = local_dates.min().year, local_dates.max().year

    holiday_dates = set()
    for year in range(min_year, max_year + 1):
        url = f"https://calendrier.api.gouv.fr/jours-feries/{country_code}/{year}.json"
        try:
            logger.info(f"🏛️ Fetching public holidays for year {year} from API.")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            raw_dates = list(response.json().keys())
            parsed_dates = pd.to_datetime(raw_dates)
            holiday_dates.update(d.date() for d in parsed_dates)
        except requests.RequestException as e:
            logger.error(f"❌ Error fetching holidays for {year}: {e}")

    df["jour_ferie"] = local_dates.apply(lambda d: int(d in holiday_dates))

    return df


def add_school_vacation_column(
    df: pd.DataFrame,
    datetime_col: str,
    location: str = "Paris",
    zone: str = "Zone C"
) -> pd.DataFrame:
    """
    Add a 'vacances_scolaires' column with the type of school holiday (or 'aucune').

    Args:
        df (pd.DataFrame): Input DataFrame with timezone-aware datetime.
        datetime_col (str): Name of the datetime column.
        location (str): Académie de référence (e.g., "Paris")
        zones (str): Zone scolaire (e.g., "Zone C")

    Returns:
        pd.DataFrame: Enriched DataFrame with 'vacances_scolaires' column.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        raise ValueError(f"Column '{datetime_col}' must be datetime64 dtype.")

    local_dates = df[datetime_col].dt.tz_convert("Europe/Paris").dt.date
    date_min = local_dates.min()
    date_max = local_dates.max()

    api_url = (
        "https://data.education.gouv.fr/api/v2/catalog/datasets/"
        "fr-en-calendrier-scolaire/exports/json"
    )
    try:
        logger.info("🏫 Fetching all school holidays from API.")
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        records = response.json()
    except Exception as e:
        logger.error(f"❌ Error fetching school holiday data: {e}")
        df["vacances_scolaires"] = "erreur_api"
        return df

    vacances = []
    for item in records:
        if item.get("location") != location or item.get("zones") != zone:
            continue
        try:
            start = pd.to_datetime(item["start_date"]).date()
            end = pd.to_datetime(item["end_date"]).date()
            if end < date_min or start > date_max:
                continue
            description = item["description"]
            vacances.append((start, end, description))
        except Exception as e:
            logger.warning(f"⛔ Failed to parse record: {e}")

    logger.debug(f"Holidays retained (filtered): {vacances}")

    def get_vacances_type(date):
        matched = [desc for start, end, desc in vacances if start <= date <= end]
        if matched:
            return matched[0]
        else:
            return "aucune"

    df["vacances_scolaires"] = local_dates.map(get_vacances_type)
    return df
