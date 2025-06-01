import os
import io
import re
import logging
import pandas as pd
import numpy as np
import requests
import unicodedata
from smartcheck.paths import load_config, get_full_path

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s]-[%(asctime)s] %(message)s')

# Constants
GOOGLE_DRIVE_PREFIX = "https://drive.google.com"
GOOGLE_DRIVE_DOWNLOAD_BASE = "https://drive.google.com/uc?export=download"
NUMERIC_PLACEHOLDER = np.inf
STRING_PLACEHOLDER = "__MISSING__"


def _extract_google_drive_file_id(url):
    """
    Extract the file ID from a Google Drive URL.

    Args:
        url (str): Google Drive file URL.

    Returns:
        str or None: Extracted file ID, or None if extraction fails.
    """
    try:
        return url.split('/d/')[1].split('/')[0]
    except IndexError:
        logger.error("Unable to extract file ID from Google Drive URL.")
        return None


def _download_google_drive_file(file_id):
    """
    Download a file from Google Drive, handling large file confirmation if needed.

    Args:
        file_id (str): ID of the Google Drive file.

    Returns:
        str or None: File content as string if successful, else None.
    """
    session = requests.Session()
    response = session.get(GOOGLE_DRIVE_DOWNLOAD_BASE,
                           params={'id': file_id}, stream=True)
    content = response.text

    if content.lstrip().lower().startswith('<!doctype html'):
        form_url = re.search(r'<form[^>]+action="([^"]+)"', content)
        confirm_token = re.search(r'name="confirm"\s+value="([^"]+)"', content)
        uuid_token = re.search(r'name="uuid"\s+value="([^"]+)"', content)

        if form_url and confirm_token and uuid_token:
            action_url = form_url.group(1)
            params = {
                'id': file_id,
                'export': 'download',
                'confirm': confirm_token.group(1),
                'uuid': uuid_token.group(1)
            }
            logger.info(
                "Big file mechanism : "
                "Attempting download from confirmation URL."
            )
            response = session.get(action_url, params=params, stream=True)
            if (response.status_code == 200 and not
                    response.text.lstrip().lower().startswith('<!doctype html')):
                return response.text
            else:
                logger.error("HTML response received again. Download failed.")
                return None
        else:
            logger.error(
                "Required confirmation fields not found. "
                "File may not be shared publicly."
            )
            return None
    else:
        return content


def _load_data_from_string(content, format_type="csv", *args, **kwargs):
    """
    Load data from a string content in CSV or JSON format.

    Args:
        content (str): The content string of the file.
        format_type (str): Format of the data ('csv' or 'json').

    Returns:
        pd.DataFrame or None: Loaded DataFrame if successful, else None.
    """
    try:
        if format_type == "csv":
            return pd.read_csv(io.StringIO(content), *args, **kwargs)
        elif format_type == "json":
            return pd.read_json(io.StringIO(content), *args, **kwargs)
        else:
            logger.error(
                f"Unsupported format '{format_type}' for string input."
            )
            return None
    except Exception as e:
        logger.error(f"Error reading {format_type.upper()} content: {e}")
        return None


def _load_data_from_local(path, format_type, *args, **kwargs):
    """
    Load a dataset from a local path.

    Args:
        path (str): Relative path to the file.
        format_type (str): File format ('csv', 'json', 'xlsx', 'xls').

    Returns:
        pd.DataFrame or None: Loaded DataFrame if successful, else None.
    """
    full_path = get_full_path(path)
    logger.info(f"Resolved local file path: {full_path}")
    if not os.path.exists(full_path):
        logger.error(f"File not found at: {full_path}")
        logger.debug(f"Current directory contents: {os.listdir()}")
        return None
    try:
        if format_type == "csv":
            return pd.read_csv(str(full_path), *args, **kwargs)
        elif format_type == "json":
            return pd.read_json(str(full_path), *args, **kwargs)
        elif format_type in {"xlsx", "xls"}:
            return pd.read_excel(full_path, *args, **kwargs)
        else:
            logger.error(f"Unsupported file format: {format_type}")
            return None
    except Exception as e:
        logger.error(f"Error reading local file: {e}")
        return None


def _infer_file_format(path):
    """
    Infer file format from file extension.

    Args:
        path (str): File path.

    Returns:
        str: File extension in lowercase.
    """
    return path.split(".")[-1].lower()


def load_dataset_from_config(data_name, *args, **kwargs):
    """
    Load a dataset based on configuration, from either local file or Google Drive.

    Args:
        data_name (str): Dataset name as specified in config['data']['input'].
        *args: Additional positional arguments for pandas loader.
        **kwargs: Additional keyword arguments for pandas loader.

    Returns:
        pd.DataFrame or None: Loaded DataFrame if successful, else None.
    """
    config = load_config()
    file_path = config["data"]["input"].get(data_name)

    if not file_path:
        logger.error(f"Dataset '{data_name}' not found in configuration.")
        return None
    else:
        logger.info(f"File path resolved from configuration : {file_path}.")

    if file_path.startswith(GOOGLE_DRIVE_PREFIX):
        file_id = _extract_google_drive_file_id(file_path)
        if not file_id:
            return None
        else:
            logger.info(f"File ID extracted from URL: {file_id}")
        content = _download_google_drive_file(file_id)
        return _load_data_from_string(content, *args, **kwargs) if content else None
    else:
        format_type = _infer_file_format(file_path)
        return _load_data_from_local(file_path, format_type, *args, **kwargs)


def log_general_info(df) -> None:
    """
    Log general information about a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        None
    """
    if df is None:
        logger.error("Invalid DataFrame provided.")
        return

    logger.info(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    description_str = "df.select_dtypes(include=np.number).describe()"
    logger.info(
        f"For quantitative variable description use:\n{description_str}")
    correlation_str = "df.select_dtypes(include=np.number).corr()"
    logger.info(f"For quantitative correlation matrix use:\n{correlation_str}")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    buffer.close()
    logger.info("DataFrame Info:\n%s", info_str)


def detect_and_log_duplicates_and_missing(df, subset=None):
    """
    Detect and log missing values and duplicate rows in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        subset (list[str] or None): Columns to check for duplicates. If None,
        all columns are used.

    Returns:
        tuple[int, int]: Number of unique duplicates and total duplicates.
    """
    df_sub = df if subset is None else df[subset]

    missing_any = df_sub.isna().any(axis=1).sum()
    missing_all = df_sub.isna().all(axis=1).sum()
    missing_stats = df_sub.isna().sum(axis=0)/len(df_sub)
    missing_stats = (missing_stats[missing_stats > 0].round(5))

    logger.info(f"Rows with at least one NaN: {missing_any}")
    logger.info(f"Rows with all values NaN: {missing_all}")
    logger.info(f"Columns with missing values (normalized total):\n{missing_stats}")

    df_filled = df.copy()
    for col in df_sub.columns:
        if pd.api.types.is_numeric_dtype(df_filled[col]):
            df_filled[col] = df_filled[col].fillna(NUMERIC_PLACEHOLDER)
        else:
            df_filled[col] = df_filled[col].fillna(STRING_PLACEHOLDER)

    dup_keep_first = df_filled.duplicated(subset=subset, keep='first').sum()
    dup_keep_false = df_filled.duplicated(subset=subset, keep=False).sum()

    logger.info(
        "Duplicate rows (NaNs treated as equal): "
        f"{dup_keep_first} unique, {dup_keep_false} total"
    )

    return dup_keep_first, dup_keep_false


def duplicates_index_map(df):
    """
    Identify and group indices of duplicate rows in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        list[list[int]]: List of lists, each containing indices of a duplicate group.
    """
    df_filled = df.copy()

    for col in df_filled.columns:
        if pd.api.types.is_numeric_dtype(df_filled[col]):
            df_filled[col] = df_filled[col].fillna(-9999999)
        else:
            df_filled[col] = df_filled[col].fillna("__MISSING__")

    duplicates_mask = df_filled.duplicated(keep=False)
    duplicates_df = df_filled[duplicates_mask]
    grouped = duplicates_df.groupby(list(duplicates_df.columns))
    duplicate_groups = [list(group.index) for _, group in grouped]

    for group in duplicate_groups:
        logger.debug(f"Duplicate group indexes: {group}")

    return duplicate_groups


def display_variable_info(data, max_values=10):
    """
    Display information about unique values and distribution of a Series or DataFrame
    limiting the number of maximum values displayed for each variable

    Args:
        data (pd.Series or pd.DataFrame): Data to analyze.
        max_values (int) : Number of maximum

    Raises:
        TypeError: If input is not a Series or DataFrame.
    """
    if isinstance(data, pd.Series):
        logger.info(f"Analysis for Series [{data.name}]:")
        unique_values = (
            pd.Series(data.unique())
            .sort_values(na_position='last')
            .tolist()
        )
        logger.info(f"Sorted unique values (first {max_values}): "
                    f"{unique_values}")
        logger.info(f"Value distribution (first {max_values}):\n"
                    f"{data.value_counts(
                        normalize=True,
                        dropna=False
                    ).head(max_values)}")
    elif isinstance(data, pd.DataFrame):
        logger.info("Analysis for DataFrame:")
        for col in data.columns:
            logger.info(f"Analysis for column [{col}]:")
            unique_values = (
                pd.Series(data[col].unique())
                .sort_values(na_position='last')
                .tolist()
            )
            logger.info(f"Sorted unique values (first {max_values}): "
                        f"{unique_values[:max_values]}")
            logger.info(f"Value distribution (first {max_values}):\n"
                        f"{data[col].value_counts(
                            normalize=True,
                            dropna=False
                        ).head(max_values)}")
    else:
        raise TypeError("Input must be a pandas Series or DataFrame.")


def compare_row_differences(df, row_index1, row_index2):
    """
    Compare values of two rows in a DataFrame and log column-wise differences.

    Args:
        df (pd.DataFrame): DataFrame containing the rows.
        row_index1 (int): Index of the first row.
        row_index2 (int): Index of the second row.

    Returns:
        list: Column names where the values differ.
    """
    differences = df.loc[row_index1] != df.loc[row_index2]
    differing_columns = df.columns[differences].tolist()

    if differing_columns:
        logger.info(
            "Differences between rows "
            f"{row_index1} and {row_index2}:"
        )
        logger.info(df.loc[[row_index1, row_index2], differing_columns].to_string())
    else:
        logger.info(
            "No differences found between rows "
            f"{row_index1} and {row_index2}."
        )

    return differing_columns


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame column names to snake_case and remove accents.

    Args:
        df (pd.DataFrame): Input DataFrame with original column names.

    Returns:
        pd.DataFrame: A copy of the DataFrame with normalized column names.
    """

    def normalize(col: str) -> str:
        # Remove accents by normalizing Unicode characters
        col = (
            unicodedata.normalize('NFKD', col)
            .encode('ascii', 'ignore')
            .decode('utf-8')
        )
        # Replace any non-word characters (punctuation, etc.) with spaces
        col = re.sub(r"[^\w\s]", " ", col)
        # Convert spaces to underscores and lowercase everything
        col = re.sub(r"\s+", "_", col).lower()
        # Remove leading/trailing underscores
        return col.strip("_")

    df = df.copy()
    df.columns = [normalize(col) for col in df.columns]
    return df


def analyze_by_reference_variable(df: pd.DataFrame, reference_col: str) -> None:
    """
    Analyze a DataFrame grouped by a reference column.

    Logs:
        - Distribution of the reference column.
        - Median values of numeric variables per group.
        - Mode values of categorical variables per group.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        reference_col (str): Column to group by.

    Raises:
        ValueError: If reference_col is not in df.
    """
    if reference_col not in df.columns:
        raise ValueError(f"'{reference_col}' must be a column in the DataFrame.")

    distribution = df[reference_col].value_counts(
        dropna=False, normalize=True
    ).to_string()
    logger.info("Distribution of %s:\n%s", reference_col, distribution)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != reference_col]
    if numeric_cols:
        medians = df[numeric_cols + [reference_col]] \
            .groupby(reference_col).median().to_string()
        logger.info("Medians by %s:\n%s", reference_col, medians)
    else:
        logger.info("No numeric variables detected")

    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != reference_col]
    if categorical_cols:
        modes_df = {
            col: df.groupby(reference_col)[col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA
            )
            for col in categorical_cols
        }
        modes_df = pd.DataFrame(modes_df)
        logger.info("Modes by %s:\n%s", reference_col, modes_df.to_string())
    else:
        logger.info("No categorical variables detected")


def log_cross_distributions(df: pd.DataFrame, reference_col: str) -> None:
    """
    Log cross-distributions between the reference column and all other columns.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        reference_col (str): Column to compute cross-distributions against.

    Raises:
        ValueError: If reference_col is not in df.
    """
    if reference_col not in df.columns:
        raise ValueError(f"'{reference_col}' must be a column in the DataFrame.")

    other_cols = [col for col in df.columns if col != reference_col]
    for col in other_cols:
        try:
            cross_dist = df.groupby(col)[reference_col].value_counts(
                normalize=True
            ).to_string()
            logger.info("Cross-distribution of %s by %s:\n%s",
                        reference_col, col, cross_dist)
        except Exception as e:
            logger.warning("Could not compute cross-distribution for %s:\n%s", col, e)
