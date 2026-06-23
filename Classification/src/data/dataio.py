"""
dataio.py - Dataset I/O Utilities

Provides helper functions for loading CSV datasets from disk and splitting
them into feature matrices and label arrays. Used by training and evaluation
scripts throughout the Classification pipeline.
"""

import pandas as pd


def separate_xy(dataframe):
    """Split a DataFrame into a feature matrix and a label array.

    Assumes the last column of the DataFrame is the target label and all
    preceding columns are input features.

    Args:
        dataframe (pd.DataFrame): Dataset where the final column is the label.

    Returns:
        list: [X, y] where
            - X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            - y (np.ndarray): Label array, shape (n_samples,).
    """
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values
    return [X, y]


def load(datapath):
    """Load a CSV dataset from disk and return features and labels.

    Reads the CSV file at `datapath`, prints its shape for verification, then
    splits it into X and y via `separate_xy`.

    Args:
        datapath (str): Path to the CSV file. Must have a header row; the last
            column is treated as the target label.

    Returns:
        list: [X, y] where
            - X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            - y (np.ndarray): Label array, shape (n_samples,).
    """
    dataset = pd.read_csv(datapath, header=0)
    print(dataset.shape)
    [X, y] = separate_xy(dataset)
    return [X, y]