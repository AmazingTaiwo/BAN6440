#   BAN6440 Applied Machine Learning for Analytics
#   Module 4 Assignment - K-Means Python Application
#   Name: Taiwo Babalola
#   Learner ID: 162894
#   Submitted to: Rapheal Wanjiku

#   AppName: ban6440_module_4_assignment_k_means.py
#   Author Taiwo Babalola

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

os.makedirs("kaguya_monoscopic_uncontrolled_observations", exist_ok=True)
with open("kaguya_monoscopic_uncontrolled_observations/sample_1.img", "w") as f:
    f.write("Simulated file content")

def extract_mock_features(directory, feature_dim=5):
    features = []
    file_names = []

    if not os.path.exists(directory):
        raise FileNotFoundError(f"[ERROR] Directory '{directory}' not found.")

    for root, _, files in os.walk(directory):
        for f in files:
            file_path = os.path.join(root, f)
            features.append(np.random.rand(feature_dim))
            file_names.append(file_path)

    if not features:
        raise ValueError("[ERROR] No files found for feature extraction.")

    df = pd.DataFrame(features, columns=[f"feature_{i+1}" for i in range(feature_dim)])
    df["file_name"] = file_names
    print(f"[INFO] Extracted features from {len(df)} files.")
    return df

print(os.listdir("kaguya_monoscopic_uncontrolled_observations"))

# Apply K-Means Clustering
def run_kmeans_clustering(df, n_clusters=3):
    """
    Applies K-Means clustering to numeric features only.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("[ERROR] No numeric features found to cluster.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    if len(scaled_data) < n_clusters:
        raise ValueError(f"[ERROR] Need at least {n_clusters} samples, but got {len(scaled_data)}.")

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_data)
    print(f"[INFO] K-Means clustering completed with {n_clusters} clusters.")
    return labels, model, scaled_data

os.makedirs("kaguya_monoscopic_uncontrolled_observations", exist_ok=True)

for i in range(5):  # Create 5 mock files
    with open(f"kaguya_monoscopic_uncontrolled_observations/sample_{i+1}.img", "w") as f:
        f.write("Dummy lunar image content")


# Visualize with PCA
from sklearn.decomposition import PCA

def plot_clusters(data, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis')
    plt.title("K-Means Clusters of Lunar Data")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def plot_clusters(data, labels, save_path="kmeans_pca_plot.png"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', edgecolor='k', s=60)
    plt.title("K-Means Clusters of Lunar Data")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(save_path)  # Save the plot
    print(f"[INFO] Cluster plot saved as '{save_path}'")
    plt.show()

#   Main flow
if __name__ == "__main__":
    download_dir = "kaguya_monoscopic_uncontrolled_observations"
    df = extract_mock_features(download_dir)
    labels, model, scaled_data = run_kmeans_clustering(df)
    plot_clusters(scaled_data, labels)