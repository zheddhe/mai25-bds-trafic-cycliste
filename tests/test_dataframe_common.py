import pandas as pd
import pytest
from smartcheck.dataframe_common import (
    load_dataset_from_config,
    compare_row_differences,
    display_variable_info,
    detect_and_compare_duplicates
)

# === Tests for load_dataset_from_config ===
# = Test Classes =
class TestLoadDatasetFromConfig:
    def test_load_dataset_from_config(self):
        df = load_dataset_from_config('velib_dispo_data', sep=';')
        assert isinstance(df, pd.DataFrame)


# === Tests for compare_row_differences ===
# = Fixtures =
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Alice"],
        "age": [30, 30, 35],
        "city": ["New York", "Los Angeles", "New York"]
    })
# = Test Classes =
class TestCompareRowDifferences:
    def test_no_differences(self, sample_dataframe: pd.DataFrame):
        result = compare_row_differences(sample_dataframe, 0, 2)
        assert set(result) == {"age"}

    def test_multiple_differences(self, sample_dataframe: pd.DataFrame):
        result = compare_row_differences(sample_dataframe, 0, 1)
        assert set(result) == {"name", "city"}

    def test_all_same(self, sample_dataframe: pd.DataFrame):
        result = compare_row_differences(sample_dataframe, 0, 0)
        assert result == []

    def test_index_out_of_bounds(self, sample_dataframe: pd.DataFrame):
        with pytest.raises(KeyError):
            compare_row_differences(sample_dataframe, 0, 10)


# === Tests pour display_variable_info ===
# = Test Classes =
class TestDisplayVariableInfo:
    def test_series(self, caplog: pytest.LogCaptureFixture):
        series = pd.Series(['apple', 'banana', 'apple', 'orange'], name='fruit')
        with caplog.at_level("INFO"):
            display_variable_info(series)

        assert "Analysis for Series [fruit]:" in caplog.text
        assert "Sorted unique values: ['apple', 'banana', 'orange']" in caplog.text
        assert "Value distribution" in caplog.text

    def test_dataframe(self, caplog: pytest.LogCaptureFixture):
        df = pd.DataFrame({
            'color': ['red', 'blue', 'red', 'green'],
            'size': ['S', 'M', 'M', 'L']
        })
        with caplog.at_level("INFO"):
            display_variable_info(df)

        assert "Analysis for DataFrame:" in caplog.text
        assert "Analysis for column [color]:" in caplog.text
        assert "Sorted unique values: ['blue', 'green', 'red']" in caplog.text
        assert "Analysis for column [size]:" in caplog.text
        assert "Sorted unique values: ['L', 'M', 'S']" in caplog.text

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="Input must be a pandas Series or DataFrame."):
            display_variable_info(['not', 'a', 'pandas', 'object'])


# === Tests pour detect_and_compare_duplicates ===
# = Test Classes =
class TestDetectAndCompareDuplicates:
    def test_detect_and_compare_duplicates(self):
        df = pd.DataFrame({
            'A': [pd.NA, pd.NA, pd.NA, 'x4'],
            'B': ['x1', 'x1', 'x1', 'x4'],
            'C': ['x1', 'x1', 'x1', 'x4'],
        })

        result = detect_and_compare_duplicates(df)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'original_index' in result.columns
        assert list(result.original_index[:2]) == [0, 0]
        assert 'duplicate_index' in result.columns
        assert list(result.duplicate_index[:2]) == [1, 2]
        assert 'nan_columns' in result.columns
        assert result.nan_columns[0] == ['A']
        assert result.nan_columns[1] == ['A']