"""
preprocessing.py - Data Preprocessing Pipeline

Prepares the raw dataset for clustering by running the following steps in order:

    1. Load        — read 'dataset.csv' from the configured dataset directory
    2. Quality     — report null values and duplicate rows
    3. Outliers    — remove rows with IQR-based outliers across numeric features
    4. Scaling     — apply StandardScaler to selected numeric columns
    5. Params      — persist scaler mean and scale to 'param.json' for later use
    6. Selection   — drop low-signal features ('battery_level', 'brightness')
    7. Save        — write the cleaned dataset to 'cleaned.csv'

Typical usage:
    python preprocessing.py
"""

import os
import json
import pandas as pd
import Clustering.src.common.tools as tools
from sklearn.preprocessing import StandardScaler


def load_data(config):
    """Load the raw dataset from disk.

    Reads 'dataset.csv' from the directory specified in config and returns it
    as a DataFrame. Prints the resolved path and data shape for verification.

    Args:
        config (dict): Configuration dictionary containing:
            - 'dataset_dir' (str): Directory holding 'dataset.csv'.
              Supports '~' home-directory expansion.

    Returns:
        pd.DataFrame: The full raw dataset.
    """
    data = os.path.expanduser(
        os.path.join(config['dataset_dir'], "dataset.csv")
    )

    print("Loading: ", data)
    df = pd.read_csv(data)
    print("Data Shape: ", df.shape)
    return df


def check_null_and_duplicate(df):
    """Print a data-quality report for null values and duplicate rows.

    This function is diagnostic only — it does not modify the DataFrame.
    Review the output before proceeding with further preprocessing.

    Args:
        df (pd.DataFrame): The dataset to inspect.

    Returns:
        None
    """
    null = df.isnull().sum()
    duplicates = df.duplicated().sum()

    print("Null Values:\n", null)
    print("Duplicates: ", duplicates)


def remove_outliers(df):
    """Remove rows that contain outliers in any numeric column using the IQR method.

    A row is flagged as an outlier if any of its numeric values falls outside
    [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]. Only numeric columns are considered;
    non-numeric columns are preserved unchanged.

    Args:
        df (pd.DataFrame): Input dataset, may contain mixed column types.

    Returns:
        pd.DataFrame: Dataset with outlier rows removed.
    """
    nums_features = df.select_dtypes(include=['number'])

    Q1 = nums_features.quantile(0.25)
    Q3 = nums_features.quantile(0.75)
    IQR = Q3 - Q1

    # Build a boolean mask: True where the row has NO outlier in any column
    mask = ~((nums_features < (Q1 - 1.5 * IQR)) | (nums_features > (Q3 + 1.5 * IQR))).any(axis=1)

    df_outlier = df[mask]
    print("Dataset After Outlier Removed: ", df_outlier.shape)
    return df_outlier


def standard_scaler(df):
    """Standardise selected numeric columns to zero mean and unit variance.

    Columns excluded from scaling (retained at their original values):
        timestamp, battery_level, battery_charging, screen_on, brightness

    The fitted scaler's mean and scale parameters are returned so they can be
    saved and reused during inference or inverse transformation.

    Args:
        df (pd.DataFrame): Outlier-free dataset to scale in-place.

    Returns:
        tuple:
            - df (pd.DataFrame): Dataset with scaled numeric columns.
            - mean (list[float]): Per-feature means used by the scaler.
            - scale (list[float]): Per-feature standard deviations used by the scaler.
    """
    scaler = StandardScaler()

    # Select numeric columns, excluding those that should not be normalised
    nums_cols = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
    nums_cols = [col for col in nums_cols if
                 col.lower() not in ['timestamp', 'battery_level', 'battery_charging', 'screen_on', 'brightness']]

    df[nums_cols] = scaler.fit_transform(df[nums_cols])
    print("Dataset Standard Scaled: ", df.shape)

    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()
    return df, mean, scale


def save_scaler_param(mean, scale):
    """Persist the StandardScaler parameters to a JSON file.

    Writes mean and scale lists to 'param.json' inside config['param_dir'].
    This file is later read by inverse.py to reverse the scaling transformation.
    The output directory is created if it does not already exist.

    Args:
        mean (list[float]): Per-feature means from the fitted StandardScaler.
        scale (list[float]): Per-feature standard deviations from the fitted StandardScaler.

    Returns:
        None

    Note:
        This function reads `config` from the module-level global set in __main__.
    """
    scaler_param = {
        "mean": mean,
        "scale": scale
    }

    json_dir = os.path.expanduser(
        os.path.join(config['param_dir'], "param.json")
    )
    os.makedirs(os.path.dirname(json_dir), exist_ok=True)

    with open(json_dir, 'w') as outfile:
        json.dump(scaler_param, outfile)

    print("Saving Scaler Param: ", json_dir)


def feature_selection(df):
    """Drop low-signal features from the dataset.

    Removes 'battery_level' and 'brightness', which are excluded from
    clustering based on domain knowledge. Modification is done in-place.

    Args:
        df (pd.DataFrame): Scaled dataset with all original columns present.

    Returns:
        pd.DataFrame: Dataset with 'battery_level' and 'brightness' removed.
    """
    df.drop(columns=["battery_level", "brightness"], inplace=True)
    print("Feature Selection: ", df.shape)
    return df


def saved_result(df):
    """Save the fully preprocessed dataset to 'cleaned.csv'.

    Writes the final DataFrame to config['dataset_dir']/cleaned.csv without
    the row index. This file is the input consumed by train.py.

    Args:
        df (pd.DataFrame): The cleaned, scaled, and feature-selected dataset.

    Returns:
        None

    Note:
        This function reads `config` from the module-level global set in __main__.
    """
    save_dir = os.path.expanduser(
        os.path.join(config['dataset_dir'], "cleaned.csv")
    )
    df.to_csv(save_dir, index=False)
    print("Saved Result: ", save_dir)


if __name__ == "__main__":
    config = tools.load_config()

    df = load_data(config)
    check_null_and_duplicate(df)           # Diagnostic — does not modify df
    df_outlier = remove_outliers(df)
    df_scaled, mean, scale = standard_scaler(df_outlier)
    save_scaler_param(mean, scale)         # Persist scaler params for inverse.py
    df_selection = feature_selection(df_scaled)
    saved_result(df_selection)