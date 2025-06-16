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
            f"{base}_hour"
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
