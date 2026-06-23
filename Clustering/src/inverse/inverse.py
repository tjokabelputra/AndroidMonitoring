"""
inverse.py - Inverse Scaling and Label Attachment

Reverses the StandardScaler transformation applied in preprocessing.py to
recover interpretable (original-scale) feature values, then attaches the
cluster labels produced by the PCA clustering pipeline. The result is saved
as 'Original.csv', which feeds into the downstream classification pipeline.

Pipeline:
    1. Load the scaled 'cleaned.csv' dataset
    2. Load cluster labels from 'PCA.csv' (output of train.py PCA runs)
    3. Reverse the scaling using mean/scale from 'param.json'
    4. Attach labels and save 'Original.csv'

Typical usage:
    python inverse.py
"""

import os
import json
import pandas as pd
import numpy as np
import Clustering.src.common.tools as tools


def load_scaled_dataset(config):
    """Load the scaled dataset and drop non-feature indicator columns.

    Reads 'cleaned.csv' from config['dataset_dir'] and removes
    'battery_charging' and 'screen_on', which are binary indicators not
    subject to scaling and not needed for inverse transformation.

    Args:
        config (dict): Configuration dictionary containing:
            - 'dataset_dir' (str): Directory holding 'cleaned.csv'.

    Returns:
        pd.DataFrame: Scaled dataset with binary indicator columns removed.
            Still contains 'timestamp' for re-attachment after inverse scaling.
    """
    data = os.path.expanduser(
        os.path.join(config['dataset_dir'], "cleaned.csv")
    )

    print("Loading: ", data)
    df = pd.read_csv(data)
    df_drop = df.drop(columns=["battery_charging", "screen_on"])
    print("Dataset Shape: ", df_drop.shape)
    return df_drop


def load_pca_label(config):
    """Load the cluster label column from the PCA clustering result.

    Reads 'PCA.csv' from config['result_dir'] and extracts the 'Target'
    column. These labels are produced by train.py and represent the cluster
    assignments used to seed the classification pipeline.

    Args:
        config (dict): Configuration dictionary containing:
            - 'result_dir' (str): Directory holding 'PCA.csv'.

    Returns:
        pd.Series: Integer cluster labels, shape (n_samples,).
    """
    data = os.path.expanduser(
        os.path.join(config['result_dir'], "PCA.csv")
    )

    print("Loading Label")
    df_label = pd.read_csv(data)
    labels = df_label['Target']

    return labels


def inverse_scaler(config, df):
    """Reverse the StandardScaler transformation to restore original feature values.

    Reads the scaler mean and scale from 'param.json', drops the 'timestamp'
    column before inverse-transforming (since it was excluded from scaling),
    then re-attaches it to the result.

    Inverse formula applied element-wise:
        X_original = X_scaled * scale + mean

    Args:
        config (dict): Configuration dictionary containing:
            - 'param_dir' (str): Directory holding 'param.json'.
        df (pd.DataFrame): Scaled dataset including a 'timestamp' column.

    Returns:
        pd.DataFrame: Dataset with all numeric features restored to their
            original scale, with 'timestamp' re-attached as the final column.
    """
    param_dir = os.path.expanduser(
        os.path.join(config['param_dir'], "param.json")
    )

    with open(param_dir) as f:
        params = json.load(f)

    mean = np.array(params['mean'])
    scale = np.array(params['scale'])

    # Preserve timestamp separately — it was not included in scaling
    timestamp = df['timestamp'].values
    X = df.drop(columns=["timestamp"])

    # Apply inverse transformation: X_original = X_scaled * scale + mean
    X_original = X * scale + mean
    df_original = pd.DataFrame(
        X_original,
        columns=X.columns
    )
    df_original['timestamp'] = timestamp

    print("Dataset: ", df_original.columns)
    return df_original


def saved_original(df, label):
    """Attach cluster labels and save the final original-scale dataset.

    Appends the 'Target' label column to the inverse-transformed DataFrame
    and writes it to config['result_dir']/Original.csv. This file is the
    input consumed by the Classification pipeline (tts.py).

    Args:
        df (pd.DataFrame): Inverse-transformed dataset (original feature scale).
        label (pd.Series): Cluster labels to attach as the 'Target' column.

    Returns:
        None

    Note:
        This function reads `config` from the module-level global set in __main__.
    """
    df['Target'] = label

    save_dir = os.path.expanduser(
        os.path.join(config['result_dir'], "Original.csv")
    )

    df.to_csv(save_dir, index=False)
    print("Result Saved")


if __name__ == "__main__":
    config = tools.load_config()
    df_scaled = load_scaled_dataset(config)
    label = load_pca_label(config)
    df_original = inverse_scaler(config, df_scaled)
    saved_original(df_original, label)