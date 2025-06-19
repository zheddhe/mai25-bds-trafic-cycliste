import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from smartcheck.dataframe_project_specific import extract_datetime_features

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s]-[%(asctime)s] %(message)s'
)


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

    def __init__(self, timestamp_col: str):
        self.timestamp_col = timestamp_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_t = extract_datetime_features(
            X[[self.timestamp_col]],
            timestamp_col=self.timestamp_col,
        )
        return X_t.drop(columns=self.timestamp_col)

    def get_feature_names_out(self, input_features=None):
        base = self.timestamp_col
        return [
            f"{base}_utc",
            f"{base}_local",
            f"{base}_year",
            f"{base}_month",
            f"{base}_day",
            f"{base}_day_of_year",
            f"{base}_day_of_week",
            f"{base}_hour",
            f"{base}_week",
            f"{base}_dayname",
            f"{base}_monthname"
        ]


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

    def get_feature_names_out(self, input_features=None):
        return self.columns_to_keep


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
        return X[[self._category_col]]

    def get_feature_names_out(self, input_features=None):
        return [self._category_col]

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
