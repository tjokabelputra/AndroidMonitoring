"""
tts.py - Train/Validation/Test Split Utility

This module handles loading, splitting, and saving a classification dataset
into train, validation, and test subsets. The split follows a 60/20/20 ratio
and uses stratified sampling to preserve the class distribution across splits.

Typical usage:
    python tts.py

The script reads configuration (data paths, save directory) from a config
file loaded via `Classification.src.common.tools.load_config()`.
"""

import os
import pandas as pd
import Classification.src.common.tools as tools
from sklearn.model_selection import train_test_split


def load_data(config):
    """Load the raw dataset from disk.

    Reads 'Original.csv' from the directory specified in the config and
    returns it as a DataFrame. Prints the resolved file path and the shape
    of the loaded data for verification.

    Args:
        config (dict): Configuration dictionary containing:
            - 'data_dir' (str): Path to the directory holding 'Original.csv'.
              Supports '~' home-directory expansion.

    Returns:
        pd.DataFrame: The full dataset loaded from 'Original.csv'.
    """
    data = os.path.expanduser(
        os.path.join(config['data_dir'], "Original.csv")
    )

    print("Loading: ", data)
    df = pd.read_csv(data)
    print("Data shape: ", df.shape)

    return df


def split_dataset(df):
    """Split the dataset into train, validation, and test subsets.

    Drops the 'timestamp' column (not a predictive feature) and separates
    features (X) from the label (y = 'Target'). The data is then split with
    stratification to maintain class proportions:

        - Train : 60 % of the full dataset
        - Validation : 20 % of the full dataset
        - Test  : 20 % of the full dataset

    Both split stages use random_state=42 for reproducibility.

    Args:
        df (pd.DataFrame): The full dataset. Must contain 'Target' and
            'timestamp' columns.

    Returns:
        tuple: Six DataFrames/Series in the order
            (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X = df.drop(columns=['Target', 'timestamp'])
    y = df['Target']

    # First split: 60% train, 40% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y)

    # Second split: split the 40% temp equally into 20% val and 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, config):
    """Recombine feature/label splits and save them as CSV files.

    Concatenates each feature matrix with its corresponding label series,
    then writes three CSV files ('train.csv', 'val.csv', 'test.csv') to the
    directory specified in the config. The output directory is created if it
    does not already exist.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_val (pd.DataFrame): Validation feature matrix.
        X_test (pd.DataFrame): Test feature matrix.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
        y_test (pd.Series): Test labels.
        config (dict): Configuration dictionary containing:
            - 'save_dir' (str): Destination directory for the output CSVs.
              Supports '~' home-directory expansion.

    Returns:
        None
    """
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    save_dir = os.path.expanduser(config['save_dir'])
    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(config['save_dir'], 'train.csv')
    val_path = os.path.join(config['save_dir'], 'val.csv')
    test_path = os.path.join(config['save_dir'], 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("=== SPLIT COMPLETE ===")
    print("Train:", train_df.shape)
    print("Val:  ", val_df.shape)
    print("Test: ", test_df.shape)
    print("Saved to:", save_dir)


if __name__ == "__main__":
    # Load configuration (data paths, save directory, etc.)
    config = tools.load_config()

    # Load raw data, split into subsets, and persist to disk
    df = load_data(config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, config)