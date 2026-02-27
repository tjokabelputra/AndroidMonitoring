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
    data = os.path.expanduser(
        os.path.join(config['dataset_dir'], "cleaned.csv")
    )

    print("Loading: ", data)
    df = pd.read_csv(data)
    df_drop = df.drop(columns=['timestamp', 'battery_charging', 'screen_on'])
    print("Dataset Shape: ", df_drop.shape)
    return df_drop

def elbow_method(df):
    X = df.select_dtypes(include=['float64','int64']).to_numpy()
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
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    PCA_numbers = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
    components = pca.components_.tolist()

    return PCA_numbers, X_pca, components

def save_pca_params(components):
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
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    print(f"{type} Summary")
    print("Silhouette Score: ", silhouette)
    print("Davies Bouldin Score: ", davies_bouldin)
    print("Calinski Harabasz Score: ", calinski_harabasz)

def train_kmeans(df):
    elbow_method(df)
    df_kmeans = df.copy()

    X = df.select_dtypes(include=['float64','int64']).to_numpy()
    model = KMeans(n_clusters=2, random_state=42)
    labels = model.fit_predict(X)
    df_kmeans['Target'] = labels

    evaluate_model(X, labels, "KMeans")
    save_result(df_kmeans, model, 'KMeans')

def train_dbscan(df):
    df_dbscan = df.copy()
    X = df.select_dtypes(include=['float64','int64']).to_numpy()
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(X)
    df_dbscan['Target'] = labels

    evaluate_model(X, labels, "DBScan")
    save_result(df_dbscan, model, 'DBScan')

def train_pca(df):
    X = df.select_dtypes(include=["float64", "int64"]).to_numpy()
    PCA_numbers, X_pca, components = pca_transform(X)
    save_pca_params(components)

    model = KMeans(n_clusters=2, random_state=42)
    labels = model.fit_predict(X_pca)
    PCA_numbers['Target'] = labels

    evaluate_model(X_pca, labels, "PCA")
    save_result(PCA_numbers, model, 'PCA')

if __name__ == "__main__":
    config = tools.load_config()
    df = load_dataset(config)
    train_kmeans(df)
    train_dbscan(df)
    train_pca(df)