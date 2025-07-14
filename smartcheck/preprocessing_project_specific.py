import pandas as pd
import logging
# from typing import Optional, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from smartcheck.dataframe_project_specific import (
    extract_datetime_features,
    add_holiday_column_from_datetime,
    fetch_weather_data_from_dataframe,
    add_school_vacation_column,
    extract_datetime_periodic_features,
)
from smartcheck.dataframe_common import (
    normalize_column_names,
)


logger = logging.getLogger(__name__)


class DatetimePreprocessingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for datetime enrichment from a single timestamp column.

    Outputs:
        - <input_column>_utc
        - <input_column>_local
        - <input_column>_year
        - <input_column>_month
        - <input_column>_day
        - <input_column>_day_of_year
        - <input_column>_day_of_week
        - <input_column>_hour
        - <input_column>_week
        - <input_column>_dayname
        - <input_column>_monthname

    Parameters:
        timestamp_col (str): Name of input timestamp column.
    """

    def __init__(self, timestamp_col: str, for_sarimax: bool = False):
        self.timestamp_col = timestamp_col
        self.for_sarimax = for_sarimax

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_t = extract_datetime_features(
            X,
            timestamp_col=self.timestamp_col,
            for_sarimax=self.for_sarimax
        )
        if self.for_sarimax:
            cols_to_drop = [self.timestamp_col]
            X_t = X_t.drop(columns=cols_to_drop, errors="ignore")
        return X_t


class DatetimePeriodicsTransformer(BaseEstimator, TransformerMixin):
    """
    scikit-learn transformer that extracts datetime components and periodic features
    from a timestamp column, and drops the original timestamp col.

    Parameters:
        timestamp_col (str): name of the timestamp column in ISO8601 format.
    """

    def __init__(self, timestamp_col: str):
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_t = extract_datetime_periodic_features(X, timestamp_col=self.timestamp_col)
        cols_to_drop = [self.timestamp_col]
        return X_t.drop(columns=cols_to_drop, errors="ignore")


class ColumnFilterTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to select a predefined subset of columns from a DataFrame.

    Parameters:
        columns_to_keep (list of str): list of column names to retain.
    """

    def __init__(self, columns_to_keep: list[str]):
        self.columns_to_keep = columns_to_keep

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [col for col in self.columns_to_keep if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns in input: {missing_cols}")
        return X[self.columns_to_keep]


class MeteoCodePreprocessingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for mapping meteorological codes (00–99) to categorical weather
    phenomena.

    Parameters:
        code_col (str): Name of the column containing the weather code
        (integer in [0, 99]).

    Output:
        - <code_col>_category: Categorical variable with 16 weather categories
        based on code ranges.
    """

    def __init__(self, code_col: str):
        self.code_col = code_col
        self._category_col = f"{code_col}_category"

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self._category_col] = X[self.code_col].apply(self._map_code_to_category)
        return X

    @staticmethod
    def _map_code_to_category(code: int) -> str:
        if 0 <= code <= 3:
            return "cloud_variation"
        elif code == 4:
            return "smoke_or_pollution"
        elif code == 5:
            return "dry_haze"
        elif 6 <= code <= 9:
            return "dust_or_sandstorm"
        elif 10 <= code <= 12:
            return "drizzle_or_local_fog"
        elif code == 13:
            return "lightning_no_thunder"
        elif 14 <= code <= 16:
            return "visible_precipitation"
        elif 17 <= code <= 19:
            return "storm_or_gusts"
        elif 20 <= code <= 29:
            return "non_violent_rain_or_fog"
        elif 30 <= code <= 39:
            return "evolving_fog"
        elif 40 <= code <= 49:
            return "light_to_heavy_rain"
        elif 50 <= code <= 59:
            return "freezing_drizzle_or_rain"
        elif 60 <= code <= 69:
            return "rain_snow_combination"
        elif 70 <= code <= 79:
            return "snow_or_sleet"
        elif 80 <= code <= 89:
            return "showers_or_small_hail"
        elif 90 <= code <= 99:
            return "thunderstorm_or_hail"
        else:
            return "unknown"  # Not expected given the spec, but safe fallback


class HolidayFromDatetimeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer adding 'jour_ferie' based on a timezone-aware datetime column.
    """

    def __init__(self,
                 datetime_col: str = "datetime",
                 country_code: str = "metropole"):
        self.datetime_col = datetime_col
        self.country_code = country_code

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_holiday_column_from_datetime(
            df=X,
            datetime_col=self.datetime_col,
            country_code=self.country_code
        )


class WeatherDataEnrichmentTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for enriching a DataFrame with historical weather data
    using Open-Meteo API.

    Args:
        lat_col (str): Column name for latitude.
        lon_col (str): Column name for longitude.
        datetime_col (str): Column name for datetime (tz-aware, UTC).
    """

    def __init__(self, lat_col: str, lon_col: str, datetime_col: str):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.datetime_col = datetime_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df_weather = fetch_weather_data_from_dataframe(
            df,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            datetime_col=self.datetime_col
        )

        if df_weather.empty:
            logger.warning("⚠️ No weather data returned."
                           "Returning original DataFrame.")
            return df

        merged = pd.merge(
            df,
            df_weather,
            how="left",
            on=[self.lat_col, self.lon_col, self.datetime_col]
        )

        return merged


class ColumnNameNormalizerTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to normalize column names:
    - Remove accents
    - Convert to snake_case
    - Replace non-alphanumeric characters
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return normalize_column_names(X.copy())


class SchoolHolidayTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer wrapper for adding school holiday type as a categorical column.
    """

    def __init__(self,
                 datetime_col: str,
                 location: str = "Paris",
                 zone: str = "Zone C"):
        self.datetime_col = datetime_col
        self.location = location
        self.zone = zone

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_school_vacation_column(
            df=X,
            datetime_col=self.datetime_col,
            location=self.location,
            zone=self.zone
        )


class AutoregressiveFeaturesTransformer:
    """
    Adds autoregressive and rolling average features to X.
    Designed to work with (X, X_dates, y) triplets for time series modeling.
    """

    def __init__(
            self,
            nb_ar: int = 1,
            nb_mm: int = 0,
            roll_wind: int = 2):
        self.nb_ar = nb_ar
        self.nb_mm = nb_mm
        self.roll_wind = roll_wind
        self.fitted_ = False

    def fit_transform(self, X: pd.DataFrame, X_dates: pd.DataFrame,
                      y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Applies AR(N) and rolling features to X and aligns X_dates and y.
        Only past data is used: shift(1) avoids data leakage.

        Returns:
            X_transformed, X_dates_transformed, y_transformed
        """
        df = X.copy()
        if self.nb_ar > 0:
            for ar in range(1, self.nb_ar+1):
                df[f"target_ar_{ar}"] = y.shift(ar)
        if self.nb_mm > 0:
            for s in range(1, self.nb_mm+1):
                df[f"target_mm_{self.roll_wind}_{s}"] = y.shift(1).rolling(
                    window=s*self.roll_wind,
                    center=False,
                ).mean()

        # Drop rows with NaN introduced by shift and rolling
        valid_idx = df.dropna().index
        X_transformed = df.loc[valid_idx].reset_index(drop=True)
        y_transformed = y.loc[valid_idx].reset_index(drop=True)
        dates_transformed = X_dates.loc[valid_idx].reset_index(drop=True)

        self.fitted_ = True
        return X_transformed, dates_transformed, y_transformed

    def transform_test(self, X: pd.DataFrame, X_dates: pd.DataFrame,
                       y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Applies same transformation as fit_transform but on test data.
        Requires full y history (at least window+1 rows) to generate features.

        Returns:
            X_transformed, X_dates_transformed, y_transformed
        """
        if not self.fitted_:
            raise RuntimeError("Must call fit_transform before transform_test.")

        df = X.copy()
        if self.nb_ar > 0:
            for ar in range(1, self.nb_ar+1):
                df[f"target_ar_{ar}"] = y.shift(ar)
        if self.nb_mm > 0:
            for s in range(1, self.nb_mm+1):
                df[f"target_mm_{self.roll_wind}_{s}"] = y.shift(1).rolling(
                    window=s*self.roll_wind,
                    center=False,
                ).mean()

        # Drop initial rows with NaN to avoid leakage
        valid_idx = df.dropna().index
        X_transformed = df.loc[valid_idx].reset_index(drop=True)
        y_transformed = y.loc[valid_idx].reset_index(drop=True)
        dates_transformed = X_dates.loc[valid_idx].reset_index(drop=True)

        return X_transformed, dates_transformed, y_transformed
