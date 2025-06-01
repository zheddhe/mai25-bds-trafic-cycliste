import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, Mock
from smartcheck.dataframe_common import (
    _extract_google_drive_file_id,
    _download_google_drive_file,
    _load_data_from_string,
    _load_data_from_local,
    _infer_file_format,
    load_dataset_from_config,
    log_general_info,
    detect_and_log_duplicates_and_missing,
    duplicates_index_map,
    display_variable_info,
    compare_row_differences,
    normalize_column_names,
    analyze_by_reference_variable,
    log_cross_distributions
)


# === Test class for _extract_google_drive_file_id ===
class TestExtractGoogleDriveFileId:

    # === Tests ===

    def test_valid_url(self):
        url = "https://drive.google.com/file/d/1234567890/view?usp=sharing"
        file_id = _extract_google_drive_file_id(url)
        assert file_id == "1234567890"

    def test_invalid_url(self, caplog: pytest.LogCaptureFixture):
        url = "https://drive.google.com/open?id=123456"
        result = _extract_google_drive_file_id(url)
        assert result is None
        assert "Unable to extract file ID" in caplog.text


# === Test class for _download_google_drive_file ===
class TestDownloadGoogleDriveFile:

    # === Data Fixtures ===
    @pytest.fixture
    def html_big_file(self):
        return '''
        <!DOCTYPE html>
        <html>
        <form action="https://drive.google.com/uc">
            <input type="hidden" name="confirm" value="CONFIRM123">
            <input type="hidden" name="uuid" value="UUID456">
        </form>
        </html>
        '''

    @pytest.fixture
    def html_missing_fields(self):
        return '''
        <!DOCTYPE html>
        <html>
        <form action="https://drive.google.com/uc">
            <!-- missing confirm and uuid -->
        </form>
        </html>
        '''

    @pytest.fixture
    def final_file_content(self):
        return "final file content"

    @pytest.fixture
    def error_html(self):
        return "<!DOCTYPE html> error again"

    @pytest.fixture
    def simple_file_content(self):
        return "some file content"

    # === Mock Fixtures ===
    @pytest.fixture
    def mock_response(self):
        return Mock()

    @pytest.fixture
    def mock_session(self, mock_response):
        session = Mock()
        session.get.return_value = mock_response
        return session

    @pytest.fixture
    def patch_session(self, mock_session):
        with patch("smartcheck.dataframe_common.requests.Session",
                   return_value=mock_session):
            yield mock_session

    # === Tests ===
    def test_download_simple_file(self, patch_session,
                                  mock_response, simple_file_content):
        mock_response.text = simple_file_content
        content = _download_google_drive_file("fake_id")
        assert content == simple_file_content
        patch_session.get.assert_called_once()

    def test_download_big_file_with_confirmation(self, patch_session,
                                                 html_big_file, final_file_content):
        patch_session.get.side_effect = [
            Mock(text=html_big_file),
            Mock(status_code=200, text=final_file_content)
        ]
        content = _download_google_drive_file("fake_id")
        assert content == final_file_content
        assert patch_session.get.call_count == 2

    def test_download_big_file_confirmation_failed(self, patch_session,
                                                   html_big_file, error_html, caplog):
        patch_session.get.side_effect = [
            Mock(text=html_big_file),
            Mock(status_code=200, text=error_html)
        ]
        content = _download_google_drive_file("fake_id")
        assert content is None
        assert "Download failed" in caplog.text

    def test_download_missing_confirmation_fields(self, patch_session, mock_response,
                                                  html_missing_fields, caplog):
        mock_response.text = html_missing_fields
        content = _download_google_drive_file("fake_id")
        assert content is None
        assert "Required confirmation fields not found" in caplog.text


# === Test class for _load_data_from_string ===
class TestLoadDataFromString:

    # === Data Fixtures ===
    @pytest.fixture
    def valid_csv(self):
        return "col1,col2\n1,2\n3,4"

    @pytest.fixture
    def valid_json(self):
        return '[{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}]'

    @pytest.fixture
    def malformed_json(self):
        return '{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}'  # liste non encadrée

    # === Tests ===
    def test_load_valid_csv(self, valid_csv):
        df = _load_data_from_string(valid_csv, format_type="csv")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["col1", "col2"]

    def test_load_valid_json(self, valid_json):
        df = _load_data_from_string(valid_json, format_type="json")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["col1", "col2"]

    def test_unsupported_format(self, caplog):
        with caplog.at_level("ERROR"):
            result = _load_data_from_string("dummy", format_type="xml")
            assert result is None
            assert "Unsupported format 'xml'" in caplog.text

    def test_load_malformed_json(self, malformed_json, caplog):
        with caplog.at_level("ERROR"):
            result = _load_data_from_string(malformed_json, format_type="json")
            assert result is None
            assert "Error reading JSON content" in caplog.text


# === Test class for _load_data_from_local ===
class TestLoadDataFromLocal:

    # === Tests ===
    @patch("smartcheck.dataframe_common.os.path.exists",
           return_value=True)
    @patch("smartcheck.dataframe_common.pd.read_csv")
    def test_load_csv(self, mock_read_csv, mock_exists):
        mock_read_csv.return_value = pd.DataFrame({"a": [1]})
        df = _load_data_from_local("some/path.csv", "csv")
        assert isinstance(df, pd.DataFrame)
        mock_read_csv.assert_called_once()

    @patch("smartcheck.dataframe_common.os.path.exists",
           return_value=True)
    @patch("smartcheck.dataframe_common.pd.read_json")
    def test_load_json(self, mock_read_json, mock_exists):
        mock_read_json.return_value = pd.DataFrame({"b": [2]})
        df = _load_data_from_local("some/path.json", "json")
        assert isinstance(df, pd.DataFrame)
        mock_read_json.assert_called_once()

    @patch("smartcheck.dataframe_common.os.path.exists",
           return_value=True)
    @patch("smartcheck.dataframe_common.pd.read_excel")
    def test_load_excel(self, mock_read_excel, mock_exists):
        mock_read_excel.return_value = pd.DataFrame({"c": [3]})
        df = _load_data_from_local("some/path.xlsx", "xlsx")
        assert isinstance(df, pd.DataFrame)
        mock_read_excel.assert_called_once()

    @patch("smartcheck.dataframe_common.os.path.exists",
           return_value=True)
    def test_unsupported_format(self, mock_exists, caplog: pytest.LogCaptureFixture):
        df = _load_data_from_local("some/path.txt", "txt")
        assert df is None
        assert "Unsupported file format" in caplog.text

    @patch("smartcheck.dataframe_common.os.path.exists",
           return_value=False)
    def test_file_not_found(self, mock_exists, caplog: pytest.LogCaptureFixture):
        df = _load_data_from_local("missing/path.csv", "csv")
        assert df is None
        assert "File not found" in caplog.text

    @patch("smartcheck.dataframe_common.os.path.exists",
           return_value=True)
    @patch("smartcheck.dataframe_common.pd.read_csv",
           side_effect=Exception("read error"))
    def test_read_exception(self, mock_read_csv, mock_exists,
                            caplog: pytest.LogCaptureFixture):
        df = _load_data_from_local("some/path.csv", "csv")
        assert df is None
        assert "Error reading local file" in caplog.text


# === Test class for _infer_file_format ===
class TestInferFileFormat:

    # === Tests ===
    def test_format_csv(self):
        assert _infer_file_format("data/test.csv") == "csv"

    def test_format_json_uppercase(self):
        assert _infer_file_format("data/test.JSON") == "json"


# === Test class for load_dataset_from_config ===
class TestLoadDatasetFromConfig:

    # === Tests ===
    @patch("smartcheck.dataframe_common.load_config")
    def test_load_google_drive_wrong_prefix(self, mock_config):
        mock_config.return_value = {
            "data": {
                "input": {
                    "wrong_drive_data": "https://drive.google.com/INCORRECT"
                }
            }
        }
        result = load_dataset_from_config("wrong_drive_data")
        assert result is None

    @patch("smartcheck.dataframe_common.load_config")
    @patch("smartcheck.dataframe_common._load_data_from_local")
    def test_load_local_csv(self, mock_loader, mock_config):
        mock_config.return_value = {
            "data": {
                "input": {
                    "mydata": "path/to/data.csv"
                }
            }
        }
        mock_loader.return_value = pd.DataFrame({"a": [1]})
        df = load_dataset_from_config("mydata")
        assert isinstance(df, pd.DataFrame)

    @patch("smartcheck.dataframe_common.load_config")
    def test_dataset_not_found(self, mock_config, caplog: pytest.LogCaptureFixture):
        mock_config.return_value = {"data": {"input": {}}}
        df = load_dataset_from_config("notfound")
        assert df is None
        assert "not found in configuration" in caplog.text

    @patch("smartcheck.dataframe_common.load_config")
    @patch("smartcheck.dataframe_common._extract_google_drive_file_id",
           return_value="123456")
    @patch("smartcheck.dataframe_common._download_google_drive_file",
           return_value="a,b\n5,6")
    def test_load_google_drive_csv(self, mock_download_google_drive_file,
                                   mock_extract_google_drive_file_id, mock_config):
        mock_config.return_value = {
            "data": {
                "input": {
                    "drive_data": "https://drive.google.com/file/d/123456/view"
                }
            }
        }
        df = load_dataset_from_config("drive_data", sep=",")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 2)


# === Test class for log_general_info ===
class TestLogGeneralInfo:

    # === Data Fixtures ===
    @pytest.fixture
    def valid_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': ['x', 'y', 'y', 'z']
        })

    @pytest.fixture
    def df_with_na_and_duplicates(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 2],
            'B': ['a', 'b', 'b', np.nan]
        })
        return pd.concat([df, df.iloc[[1]]], ignore_index=True)  # Ajoute un duplicata

    # === Tests ===
    def test_logs_general_info(self, caplog: pytest.LogCaptureFixture,
                               valid_df: pd.DataFrame):
        with caplog.at_level("INFO"):
            log_general_info(valid_df)
        assert "Dataset shape" in caplog.text
        assert "For quantitative variable description" in caplog.text
        assert "For quantitative correlation matrix" in caplog.text
        assert "DataFrame Info:" in caplog.text

    def test_logs_with_none_df(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level("ERROR"):
            log_general_info(None)
        assert "Invalid DataFrame provided." in caplog.text

    def test_logs_with_na_and_duplicates(self, caplog: pytest.LogCaptureFixture,
                                         df_with_na_and_duplicates: pd.DataFrame):
        with caplog.at_level("INFO"):
            log_general_info(df_with_na_and_duplicates)
        assert "Dataset shape" in caplog.text
        assert "For quantitative variable description" in caplog.text
        assert "For quantitative correlation matrix" in caplog.text
        assert "DataFrame Info:" in caplog.text


# === Test class pour display_variable_info ===
class TestDisplayVariableInfo:

    # === Tests ===
    def test_series(self, caplog: pytest.LogCaptureFixture):
        series = pd.Series(['apple', 'banana', 'apple', 'orange'], name='fruit')
        with caplog.at_level("INFO"):
            display_variable_info(series)

        assert "Analysis for Series [fruit]:" in caplog.text
        assert "Sorted unique values (first 10): ['apple', 'banana', 'orange']" in caplog.text
        assert "Value distribution (first 10):" in caplog.text

    def test_dataframe(self, caplog: pytest.LogCaptureFixture):
        df = pd.DataFrame({
            'color': ['red', 'blue', 'red', 'green'],
            'size': ['S', 'M', 'M', 'L']
        })
        with caplog.at_level("INFO"):
            display_variable_info(df)

        assert "Analysis for DataFrame:" in caplog.text
        assert "Analysis for column [color]:" in caplog.text
        assert "Sorted unique values (first 10): ['blue', 'green', 'red']" in caplog.text
        assert "Analysis for column [size]:" in caplog.text
        assert "Sorted unique values (first 10): ['L', 'M', 'S']" in caplog.text
        assert "Value distribution (first 10):" in caplog.text

    def test_invalid_type(self):
        with pytest.raises(TypeError,
                           match="Input must be a pandas Series or DataFrame."):
            display_variable_info(['not', 'a', 'pandas', 'object'])


# === Test class for detect_and_log_duplicates_and_missing ===
class TestDetectAndLogDuplicatesAndMissing:

    # === Data Fixtures ===
    @pytest.fixture
    def df_with_missing_and_duplicates(self):
        return pd.DataFrame({
            'A': [1, 2, 2, np.nan, np.nan, 2],
            'B': ['x', 'y', np.nan, 'z', np.nan, np.nan]
        })

    @pytest.fixture
    def df_all_unique(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': ['a', 'b', 'c', 'd']
        })

    @pytest.fixture
    def df_all_missing(self):
        return pd.DataFrame({
            'A': [np.nan, np.nan],
            'B': [np.nan, np.nan]
        })

    # === Tests ===
    def test_detects_missing_and_duplicates(self, caplog: pytest.LogCaptureFixture,
                                            df_with_missing_and_duplicates):
        with caplog.at_level("INFO"):
            unique_dups, total_dups = detect_and_log_duplicates_and_missing(
                df_with_missing_and_duplicates
            )
        assert "Rows with at least one NaN" in caplog.text
        assert "Rows with all values NaN" in caplog.text
        assert "Duplicate rows (NaNs treated as equal)" in caplog.text
        assert "Columns with missing values (normalized total):" in caplog.text
        assert unique_dups == 1
        assert total_dups == 2

    def test_detects_no_issues(self, caplog: pytest.LogCaptureFixture, df_all_unique):
        with caplog.at_level("INFO"):
            unique_dups, total_dups = detect_and_log_duplicates_and_missing(
                df_all_unique
            )
        assert "Rows with at least one NaN: 0" in caplog.text
        assert "Rows with all values NaN: 0" in caplog.text
        assert (
            "Duplicate rows (NaNs treated as equal): 0 unique, 0 total" in caplog.text
        )
        assert unique_dups == 0
        assert total_dups == 0

    def test_all_missing_rows(self, caplog: pytest.LogCaptureFixture, df_all_missing):
        with caplog.at_level("INFO"):
            unique_dups, total_dups = detect_and_log_duplicates_and_missing(
                df_all_missing
            )
        assert "Rows with at least one NaN: 2" in caplog.text
        assert "Rows with all values NaN: 2" in caplog.text
        assert unique_dups == 1
        assert total_dups == 2


# === Test class for duplicates_index_map ===
class TestDuplicatesIndexMap:

    # === Data Fixture ===
    @pytest.fixture
    def df_with_duplicates(self):
        data = {
            'A': [1, 1, 2, 2, 3, np.nan, np.nan, 1],
            'B': ['x', 'x', 'y', 'y', 'z', 'm', 'm', 'x'],
            'C': [np.nan, np.nan, 5, 5, 6, 7, 7, np.nan]
        }
        return pd.DataFrame(data)

    # === Tests ===
    def test_detects_duplicate_groups(self, caplog, df_with_duplicates):
        expected_groups = [
            [0, 1, 7],  # A=1, B='x', C=np.nan
            [2, 3],     # A=2, B='y', C=5
            [5, 6],     # A=np.nan, B='m', C=7
        ]

        with caplog.at_level("DEBUG"):
            result = duplicates_index_map(df_with_duplicates)

        # Normalize order for comparison and verify output
        sorted_result = sorted([sorted(group) for group in result])
        sorted_expected = sorted([sorted(group) for group in expected_groups])
        assert sorted_result == sorted_expected
        # Verify that log messages contain expected groupings
        logged = [
            record.message
            for record in caplog.records
            if "Duplicate group indexes" in record.message
        ]
        assert len(logged) > 0
        # Check log format
        for msg in logged:
            assert msg.startswith("Duplicate group indexes: ")
            group = eval(msg.split(": ", 1)[1])
            assert isinstance(group, list)
            assert all(isinstance(i, int) for i in group)
            assert len(group) > 1  # must be a group (not single entry)


# === Test class for compare_row_differences ===
class TestCompareRowDifferences:

    # === Data Fixture ===
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            "name": ["Alice", "Bob", "Alice"],
            "age": [30, 30, 35],
            "city": ["New York", "Los Angeles", "New York"]
        })

    # === Tests ===
    def test_no_differences(self, sample_dataframe):
        result = compare_row_differences(sample_dataframe, 0, 2)
        assert set(result) == {"age"}

    def test_multiple_differences(self, sample_dataframe):
        result = compare_row_differences(sample_dataframe, 0, 1)
        assert set(result) == {"name", "city"}

    def test_all_same(self, sample_dataframe):
        result = compare_row_differences(sample_dataframe, 0, 0)
        assert result == []

    def test_index_out_of_bounds(self, sample_dataframe):
        with pytest.raises(KeyError):
            compare_row_differences(sample_dataframe, 0, 10)


# === Test class for normalize_column_names ===
class TestNormalizeColumnNames:

    # === Data Fixtures ===
    @pytest.fixture
    def df_original(self):
        return pd.DataFrame(columns=[
            "Borne de paiement disponible",
            "État Général",
            "N° Station",
            "Adresse (complète)",
            "Type_d'utilisation"
        ])

    @pytest.fixture
    def expected_columns(self):
        return [
            "borne_de_paiement_disponible",
            "etat_general",
            "n_station",
            "adresse_complete",
            "type_d_utilisation"
        ]

    # === Tests ===
    def test_columns_are_normalized(self, df_original, expected_columns):
        df_normalized = normalize_column_names(df_original)
        assert list(df_normalized.columns) == expected_columns

    def test_result_is_dataframe_copy(self, df_original):
        df_normalized = normalize_column_names(df_original)
        assert df_normalized is not df_original  # ensure it's a new object
        assert isinstance(df_normalized, pd.DataFrame)


class TestAnalyzeByReferenceVariable:

    # === Fixtures ===
    @pytest.fixture
    def df_mixed(self):
        return pd.DataFrame({
            'Group': ['A', 'A', 'B', 'B', 'B'],
            'Value1': [1, 2, 3, 4, 5],
            'Value2': [10.5, 20.1, 30.3, 40.4, 50.0],
            'Category': ['X', 'Y', 'X', 'X', 'Z']
        })

    @pytest.fixture
    def df_only_categorical(self):
        return pd.DataFrame({
            'Group': ['A', 'A', 'B'],
            'Cat1': ['foo', 'bar', 'foo'],
            'Cat2': ['x', 'x', 'y']
        })

    @pytest.fixture
    def df_only_numeric(self):
        return pd.DataFrame({
            'Group': ['A', 'A', 'B'],
            'Num1': [1, 2, 3],
            'Num2': [4, 5, 6]
        })

    # === Tests ===
    def test_analyze_mixed_data(self, caplog, df_mixed):
        with caplog.at_level("INFO"):
            analyze_by_reference_variable(df_mixed, "Group")

        assert "Distribution of Group:" in caplog.text
        assert "Medians by Group:" in caplog.text
        assert "Modes by Group:" in caplog.text

    def test_only_categorical(self, caplog, df_only_categorical):
        with caplog.at_level("INFO"):
            analyze_by_reference_variable(df_only_categorical, "Group")

        assert "No numeric variables detected" in caplog.text
        assert "Modes by Group:" in caplog.text

    def test_only_numeric(self, caplog, df_only_numeric):
        with caplog.at_level("INFO"):
            analyze_by_reference_variable(df_only_numeric, "Group")

        assert "Medians by Group:" in caplog.text
        assert "No categorical variables detected" in caplog.text

    def test_invalid_column(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        with pytest.raises(ValueError,
                           match="'Missing' must be a column in the DataFrame."):
            analyze_by_reference_variable(df, "Missing")


# === Test class for log_cross_distributions ===
class TestLogCrossDistributions:

    # === Data Fixtures ===
    @pytest.fixture
    def df_example(self):
        return pd.DataFrame({
            'Group': ['A', 'A', 'B', 'B', 'C'],
            'Feature1': ['x', 'y', 'x', 'y', 'z'],
            'Feature2': [1, 2, 1, 2, 1]
        })

    @pytest.fixture
    def df_invalid_reference(self):
        return pd.DataFrame({
            'A': [1, 2, 3]
        })

    @pytest.fixture
    def df_with_unhashable_column(self):
        return pd.DataFrame({
            'Group': ['A', 'B', 'C'],
            'BadColumn': [[1, 2], [3, 4], [5, 6]]  # listes non hashables
        })

    # === Tests ===
    def test_cross_distributions_logging(self, caplog, df_example):
        with caplog.at_level("INFO"):
            log_cross_distributions(df_example, "Group")

        assert "Cross-distribution of Group by Feature1:" in caplog.text
        assert "Cross-distribution of Group by Feature2:" in caplog.text

    def test_invalid_reference_column(self, df_invalid_reference):
        with pytest.raises(ValueError,
                           match="'Group' must be a column in the DataFrame."):
            log_cross_distributions(df_invalid_reference, "Group")

    def test_cross_distribution_raises_exception(self, caplog,
                                                 df_with_unhashable_column):
        with caplog.at_level("WARNING"):
            log_cross_distributions(df_with_unhashable_column, "Group")

        assert "Could not compute cross-distribution for BadColumn" in caplog.text
