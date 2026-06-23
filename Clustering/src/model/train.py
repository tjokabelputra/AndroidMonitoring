"""
train.py - Clustering Model Training Script

Trains and evaluates four clustering configurations on the preprocessed dataset:

    1. KMeans          — direct clustering on the original feature space (k=3)
    2. DBSCAN          — density-based clustering on the original feature space
    3. PCA + KMeans    — KMeans on 2-component PCA projection (k=3)
    4. PCA + DBSCAN    — DBSCAN on the same 2-component PCA projection

Each model is evaluated with three internal metrics (silhouette score,
Davies–Bouldin score, Calinski–Harabasz score) and its labelled dataset and
fitted model object are saved to disk.

PCA component vectors are appended to the shared 'param.json' file so they can
be reused during inference or inverse transformation.

Typical usage:
    python train.py
"""

import os
import json
import joblib
import pandas as pd
import Clustering.src.common.tools as tools
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN
from yellowbrick.cluster import KElbowVisualizer


def load_dataset(config):
    """Load the cleaned dataset and drop non-feature columns.

    Reads 'cleaned.csv' from config['dataset_dir'] and removes the
    'timestamp', 'battery_charging', and 'screen_on' columns, which are
    not used as clustering features.

    Args:
        config (dict): Configuration dictionary containing:
            - 'dataset_dir' (str): Directory holding 'cleaned.csv'.

    Returns:
        pd.DataFrame: Feature-ready DataFrame with timestamp and binary
            indicator columns removed.
    """
    data = os.path.expanduser(
        os.path.join(config['dataset_dir'], "cleaned.csv")
    )

    print("Loading: ", data)
    df = pd.read_csv(data)
    df_drop = df.drop(columns=['timestamp', 'battery_charging', 'screen_on'])
    print("Dataset Shape: ", df_drop.shape)
    return df_drop


def elbow_method(df):
    """Plot the elbow curve to identify the optimal number of KMeans clusters.

    Fits KMeans for k in [1, 9] using the Yellowbrick KElbowVisualizer and
    prints the elbow value (best k). The visualizer also renders a plot.

    Args:
        df (pd.DataFrame): Feature DataFrame; only numeric columns are used.

    Returns:
        None
    """
    X = df.select_dtypes(include=['float64', 'int64']).to_numpy()
    model = KMeans(random_state=42)

    visualizer = KElbowVisualizer(
        model,
        k=(1, 10),
        force_model=True
    )

    visualizer.fit(X)
    best_k = visualizer.elbow_value_
    print("Best K: ", best_k)


def pca_transform(X):
    """Reduce the feature matrix to 2 principal components.

    Fits a PCA on X and returns both the transformed coordinates and the
    component vectors (used to reproduce the projection at inference time).

    Args:
        X (np.ndarray): Numeric feature matrix, shape (n_samples, n_features).

    Returns:
        tuple:
            - PCA_numbers (pd.DataFrame): 2-column DataFrame with columns
              ['PCA1', 'PCA2'], shape (n_samples, 2).
            - X_pca (np.ndarray): Raw PCA-transformed array, shape (n_samples, 2).
            - components (list[list[float]]): PCA component vectors as a
              nested list (shape [2, n_features]), suitable for JSON serialisation.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    PCA_numbers = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    components = pca.components_.tolist()

    return PCA_numbers, X_pca, components


def save_pca_params(components):
    """Append PCA component vectors to the shared param.json file.

    Reads the existing 'param.json' (which already contains scaler mean/scale),
    adds a 'components' key, and writes it back. This keeps all transformation
    parameters in a single file for use during inference or inverse.py.

    Args:
        components (list[list[float]]): PCA component vectors returned by
            pca_transform(), shape [2, n_features].

    Returns:
        None

    Note:
        This function reads `config` from the module-level global set in __main__.
    """
    param_dir = os.path.expanduser(
        os.path.join(config['param_dir'], 'param.json')
    )

    with open(param_dir) as f:
        params = json.load(f)

    params["components"] = components
    with open(param_dir, 'w') as f:
        json.dump(params, f, indent=3)

    print("Successfully Saved PCA Params")


def save_result(df, model, type):
    """Save a labelled DataFrame and its fitted model object to disk.

    Writes the labelled CSV to config['result_dir']/<type>.csv and the
    serialised model to config['model_dir']/<type>.pkl. Both output
    directories are created if they do not already exist.

    Args:
        df (pd.DataFrame): DataFrame with cluster labels in a 'Target' column.
        model: Fitted sklearn clustering model (KMeans or DBSCAN instance).
        type (str): Identifier string used as the filename stem
            (e.g. 'KMeans', 'DBScan', 'PCA_KMeans', 'PCA_DBSCAN').

    Returns:
        None

    Note:
        This function reads `config` from the module-level global set in __main__.
    """
    data_dir = os.path.expanduser(
        os.path.join(config['result_dir'], f"{type}.csv")
    )
    model_dir = os.path.expanduser(
        os.path.join(config['model_dir'], f"{type}.pkl")
    )
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)

    df.to_csv(data_dir, index=False)
    joblib.dump(model, model_dir)

    print(f"Successfully Saved Data and Model {type}")


def evaluate_model(X, labels, type):
    """Print three internal clustering evaluation metrics.

    Reports:
        - Silhouette Score       — higher is better (range -1 to 1)
        - Davies–Bouldin Score   — lower is better (≥ 0)
        - Calinski–Harabasz Score — higher is better (≥ 0)

    Args:
        X (np.ndarray): Feature matrix used for clustering, shape (n_samples, n_features).
        labels (np.ndarray): Cluster label array returned by fit_predict(),
            shape (n_samples,).
        type (str): Model name shown in the printed summary header.

    Returns:
        None
    """
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    print(f"{type} Summary")
    print("Silhouette Score: ", silhouette)
    print("Davies Bouldin Score: ", davies_bouldin)
    print("Calinski Harabasz Score: ", calinski_harabasz)


def train_kmeans(df):
    """Train and evaluate a KMeans model (k=3) on the full feature space.

    Runs the elbow method first for reference, then fits KMeans with k=3,
    appends cluster labels as 'Target', evaluates the model, and saves both
    the labelled data and the fitted model.

    Args:
        df (pd.DataFrame): Cleaned feature DataFrame (no timestamp/indicator cols).

    Returns:
        None
    """
    elbow_method(df)  # Visual aid — does not affect the chosen k
    df_kmeans = df.copy()

    X = df.select_dtypes(include=['float64', 'int64']).to_numpy()
    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(X)
    df_kmeans['Target'] = labels

    evaluate_model(X, labels, "KMeans")
    save_result(df_kmeans, model, 'KMeans')


def train_dbscan(df):
    """Train and evaluate a DBSCAN model on the full feature space.

    Uses fixed hyperparameters (eps=0.5, min_samples=5). Rows assigned label
    -1 by DBSCAN are noise points. Evaluates with internal metrics and saves
    the labelled data and fitted model.

    Args:
        df (pd.DataFrame): Cleaned feature DataFrame (no timestamp/indicator cols).

    Returns:
        None
    """
    df_dbscan = df.copy()
    X = df.select_dtypes(include=['float64', 'int64']).to_numpy()
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(X)
    df_dbscan['Target'] = labels

    evaluate_model(X, labels, "DBScan")
    save_result(df_dbscan, model, 'DBScan')


def train_pca(df):
    """Train KMeans and DBSCAN on a 2-component PCA projection of the data.

    Reduces the feature matrix to 2 dimensions via PCA, saves the component
    vectors to param.json, then trains and evaluates both KMeans (k=3) and
    DBSCAN on the projected coordinates. Results are saved with the type tags
    'PCA_KMeans' and 'PCA_DBSCAN'.

    Args:
        df (pd.DataFrame): Cleaned feature DataFrame (no timestamp/indicator cols).

    Returns:
        None
    """
    X = df.select_dtypes(include=["float64", "int64"]).to_numpy()
    PCA_numbers, X_pca, components = pca_transform(X)
    save_pca_params(components)  # Persist components to param.json

    # ── KMeans on PCA projection ──────────────────────────────────────────────
    df_kmeans = PCA_numbers.copy()
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans_model.fit_predict(X_pca)
    df_kmeans['Target'] = kmeans_labels

    evaluate_model(X_pca, kmeans_labels, "PCA + KMeans")
    save_result(df_kmeans, kmeans_model, 'PCA_KMeans')

    # ── DBSCAN on PCA projection ──────────────────────────────────────────────
    df_dbscan = PCA_numbers.copy()
    dbscan_model = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan_model.fit_predict(X_pca)
    df_dbscan['Target'] = dbscan_labels

    evaluate_model(X_pca, dbscan_labels, "PCA + DBSCAN")
    save_result(df_dbscan, dbscan_model, 'PCA_DBSCAN')


if __name__ == "__main__":
    config = tools.load_config()
    df = load_dataset(config)
    train_kmeans(df)   # KMeans on full feature space
    train_dbscan(df)   # DBSCAN on full feature space
    train_pca(df)      # Both models on 2-component PCA projection