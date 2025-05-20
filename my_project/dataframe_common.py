import os
import yaml
import io
import re
import pandas as pd
import numpy as np
import requests

begin_sep = "--------------------\n"
end_sep =   "\n********************"


def load_dataset_from_config(data_name, *args, **kwargs):
    """
    Loads a dataset based on the configuration file and the provided dataset name.
    Supports both local file paths and Google Drive URLs, including large files
    requiring confirmation for download.

    :param data_name: The name of the dataset as defined in the config file.
    :param args: Additional positional arguments for pandas.read_csv.
    :param kwargs: Additional keyword arguments for pandas.read_csv.
    :return: A pandas DataFrame if successful, otherwise None.
    """
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    print("Project directory:", project_dir)
    config_path = os.path.join(project_dir, 'config', 'config.yaml')

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    file_path = config["data"]["input"][data_name]

    # Handle Google Drive URLs
    if file_path.startswith('https://drive.google.com'):
        try:
            file_id = file_path.split('/d/')[1].split('/')[0]
        except IndexError:
            print("Error: Could not extract Google Drive file ID from configuration.")
            return None

        base_url = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(base_url, params={'id': file_id}, stream=True)
        content = response.text

        # Check if HTML content is returned (i.e., not a raw CSV)
        if content.lstrip().lower().startswith('<!doctype html'):
            # Attempt to parse download confirmation for large files
            action_match = re.search(r'<form[^>]+action="([^"]+)"', content)
            confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', content)
            uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', content)

            if action_match and confirm_match and uuid_match:
                action_url = action_match.group(1)
                confirm_token = confirm_match.group(1)
                uuid_value = uuid_match.group(1)

                params = {
                    'id': file_id,
                    'export': 'download',
                    'confirm': confirm_token,
                    'uuid': uuid_value
                }

                print(f"Downloading from: {action_url} with token confirm={confirm_token}")
                response = session.get(action_url, params=params, stream=True)

                if response.status_code == 200 and not response.text.lstrip().lower().startswith('<!doctype html'):
                    try:
                        df = pd.read_csv(io.StringIO(response.text), *args, **kwargs)
                        print("CSV successfully loaded (large file).")
                        return df
                    except Exception as e:
                        print(f"Error reading CSV: {e}")
                        return None
                else:
                    print("Error: HTML response received again. Download failed.")
                    return None
            else:
                print("Error: Required download form fields not found. The file may not be publicly shared.")
                return None
        else:
            # Direct CSV (small file)
            try:
                df = pd.read_csv(io.StringIO(content), *args, **kwargs)
                print("CSV successfully loaded (small file).")
                return df
            except Exception as e:
                print(f"Error reading CSV: {e}")
                return None

    else:
        # Handle a local file
        full_path = os.path.join(project_dir, file_path)
        print(f"Resolved local file path: {full_path}")
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path, *args, **kwargs)
                print("Local file successfully loaded.")
                return df
            except Exception as e:
                print(f"Error reading local CSV file: {e}")
                return None
        else:
            print(f"File not found at path: {full_path}")
            print("Current directory contents:", os.listdir())
            return None

def print_general_info(df):
    """
    Prints general information about the provided DataFrame.

    This function displays various details about the DataFrame, including its shape
    (rows and columns), a preview of the first and last rows, descriptive statistics
    for quantitative variables, basic correlation among quantitative variables, and
    general information using the DataFrame's info() method. Additionally, it calls for
    supporting functions to detect duplicates and missing values for further analysis.

    :param df: The DataFrame to analyze and display information for.
    :type df: pandas.DataFrame
    :return: None
    """
    if df is None:
        print("Erreur : DataFrame invalide")
        return

    print(begin_sep,"This data set has",df.shape[0],"rows and",df.shape[1], "columns",end_sep)
    print(begin_sep,"Quantitative variables description:",end_sep)
    print(df.select_dtypes(include=np.number).describe())
    print(begin_sep,"Quantitative variables basic correlation:",end_sep)
    print(df.select_dtypes(include=np.number).corr())
    print(begin_sep,"Information:",end_sep)
    df.info()
    detect_duplicates_and_na(df)

def detect_duplicates_and_na(df, subset=None, nan_placeholder="__MISSING__"):
    """
    Detect duplicate rows and NaNs in a dataframe, considering subset columns if provided. Handles NaN
    values by treating them as equal during duplicate checks. Provides summary statistics of NaN and
    duplicate row detections.

    :param df: The dataframe to analyze
    :type df: pandas.DataFrame
    :param subset: Column names to consider for duplicate detection. If None, uses all columns.
    :type subset: list or None
    :param nan_placeholder: Placeholder value for filling non-numeric NaN entries during processing
        to ensure consistent behavior.
    :type nan_placeholder: str
    :return: None
    """
    # work with sub dataset if provided
    df_sub = df.copy() if subset is None else df[subset].copy()
    print(begin_sep,f"Number of Row(s) with at least one NaN detection: {df_sub.isna().any(axis=1).sum()}",end_sep)
    print(begin_sep,f"Number of Row(s) with only NaN detection: {df_sub.isna().all(axis=1).sum()}",end_sep)
    df_sub_filled = df_sub.copy()
    for col in df_sub_filled.columns:
        if df_sub_filled[col].dtype != 'number':
            df_sub_filled[col] = df_sub_filled[col].fillna(nan_placeholder)
        else:
            df_sub_filled[col] = df_sub_filled[col].fillna(-9999999)
    keep_first = df_sub_filled.duplicated(subset=subset, keep='first').sum()
    keep_false = df_sub_filled.duplicated(subset=subset, keep=False).sum()
    print(begin_sep,f"Number of duplicated Row(s) (Nan treated as equal): {keep_first} unique and {keep_false} total",end_sep)
    if keep_false != 0:
        print(detect_and_compare_na_duplicates_verbose(df_sub, nan_placeholder=nan_placeholder))
    return None

def detect_and_compare_na_duplicates_verbose(df, nan_placeholder="__MISSING__"):
    """
    Detect duplicate rows in a DataFrame by treating NaN values as equal, compare duplicates to their corresponding
    original rows, and return detailed information about the differences.

    The function temporarily replaces NaN values in the DataFrame for comparison purposes, computes duplicates
    while treating NaN values as equal, and identifies the columns containing NaN values that differ between
    duplicate and original rows. It provides a verbose output and returns a summary DataFrame of the analysis.

    :param df: The input pandas DataFrame to analyze for duplicates.
    :type df: pandas.DataFrame
    :param nan_placeholder: Placeholder value to use temporarily in place of NaN values for non-numeric
        columns during duplicate detection. Defaults to "__MISSING__".
    :type nan_placeholder: str
    :return: A pandas DataFrame summarizing the duplicate detection and comparison results. Each row contains
        information about an original row index, a duplicate row index, and the columns with differing NaN values.
    :rtype: pandas.DataFrame
    """
    # Copie du DataFrame avec remplacement temporaire des NaN
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].dtype != 'number':
            df_filled[col] = df_filled[col].fillna(nan_placeholder)
        else:
            df_filled[col] = df_filled[col].fillna(-9999999)

    # Détection des doublons (hors première occurrence)
    duplicated_mask = df_filled.duplicated(keep='first')
    # Récupère les indices des doublons (hors premier)
    duplicate_indices = df_filled[duplicated_mask].index

    # Tous les doublons (hors première occurrence)
    duplicated_all_mask = df_filled.duplicated(keep=False)
    # Récupère les indices de tous les doublons
    duplicate_all_indices = df_filled[duplicated_all_mask].index

    results = []

    print(begin_sep,"Duplicated row(s) information when Nan values in columns bypassed (NaN treated as equal):",end_sep)
    for idx in duplicate_indices:
        # Cherche le doublon correspondant précédent
        target_row = df_filled.loc[idx]
        candidates = df_filled[df_filled.index.isin(duplicate_all_indices)]
        candidates = candidates[candidates.eq(target_row).all(axis=1)]

        if not candidates.empty:
            original_idx = candidates.index[0]
            row1 = df.loc[original_idx]
            row2 = df.loc[idx]

            differing_cols = []

            for col in df.columns:
                v1, v2 = row1[col], row2[col]
                if pd.isna(v1) or pd.isna(v2):
                    differing_cols.append(col)

            if len(differing_cols) > 0:
                results.append({
                    'orig_index': original_idx,
                    'dupl_index': idx,
                    'nan_columns': differing_cols,
            })

    return pd.DataFrame(results)

def print_variable_info(obj):
    """
    Analyzes a pandas Series or DataFrame object and prints detailed
    information about its unique values and the distribution of those
    values. The function handles both Series and DataFrame independently
    and processes Series objects one column at a time when working with
    DataFrames. Raises an error for unsupported object types.

    :param obj: The input object to be analyzed. Must be of type
        pandas.Series or pandas.DataFrame.
        - pandas.Series: Will print analysis specific to a single Series.
        - pandas.DataFrame: Each column will be analyzed independently.
    :type obj: Union[pandas.Series, pandas.DataFrame]

    :raises TypeError: If the input object is not of type pandas.Series
        or pandas.DataFrame.
    """
    if isinstance(obj, pd.Series):
        print(f"Series analysis of [{obj.name}]:")
        print("--------------------")
        # Specific Series treatment
        print(f"Unique sorted values for [{obj.name}] :\n",pd.Series(obj.unique()).sort_values(na_position='last').tolist())
        print(f"Value repartition for [{obj.name}] :\n",obj.value_counts())
        print("--------------------")
    elif isinstance(obj, pd.DataFrame):
        print("Dataframe analysis:")
        print("--------------------")
        # Specific DataFrame treatment
        for column in obj.columns:
            print(f"Unique sorted values for [{column}] :\n",pd.Series(obj[column].unique()).sort_values(na_position='last').tolist())
            print(f"Value repartition for [{column}] :\n",obj[column].value_counts())
            print("--------------------")
    else:
        raise TypeError("\nThe object must be a Series or a DataFrame")

def columns_diff_content(df, loc1, loc2):
    diff = df.loc[loc1] != df.loc[loc2]
    columns_diff = df.columns[diff]
    print("Different column content for records ", loc1, " and ", loc2, " :")
    print(df.loc[[loc1, loc2], columns_diff.tolist()])
