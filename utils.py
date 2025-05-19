# utils.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def ensure_dir(directory_path):
    """
    Ensures that a directory exists. If it doesn't, it creates it.

    Args:
        directory_path (str): The path to the directory.
    """
    os.makedirs(directory_path, exist_ok=True)


def set_seeds(seed_value=42):
    """
    Sets random seeds for numpy and tensorflow for reproducibility.

    Args:
        seed_value (int): The seed value to use.
    """
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def plot_feature_correlation_matrix(df, save_path):
    """
    Generates and saves a feature correlation matrix heatmap.

    Args:
        df (pd.DataFrame): DataFrame containing the features.
        save_path (str): Path to save the figure.
    """
    plt.figure(figsize=(15, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Feature correlation matrix saved to {save_path}")


def plot_class_distribution_and_pca(X, y_true, pca_explained_variance_ratio, save_path):
    """
    Generates and saves plots for class distribution and PCA visualization.

    Args:
        X (np.ndarray): Feature data.
        y_true (np.ndarray): True labels.
        pca_explained_variance_ratio (list): Explained variance ratio from PCA.
        save_path (str): Path to save the figure.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(np.bincount(y_true))), np.bincount(y_true))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")

    pca_viz = PCA(n_components=2)
    X_pca_viz = pca_viz.fit_transform(
        X
    )  # Use original X for PCA visualization of true classes

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], c=y_true, cmap="viridis")
    plt.xlabel(f"PC1 ({pca_explained_variance_ratio[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca_explained_variance_ratio[1]:.2%} variance)")
    plt.title("PCA Visualization of True Classes")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Class distribution and PCA plot saved to {save_path}")


def plot_training_history(history, save_path):
    """
    Plots and saves the training and validation loss, and component losses.

    Args:
        history (tf.keras.callbacks.History): Training history object.
        save_path (str): Path to save the figure.
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["decoder_loss"], label="Reconstruction Loss")
    plt.plot(history.history["clustering_loss"], label="Clustering Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Component Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_clustering_comparison(
    X_pca_scaled,
    y_true,
    nn_clusters,
    kmeans_clusters,
    agg_clusters,
    dbscan_clusters,
    latent_pca,
    pca_explained_variance_ratio,
    save_path,
):
    """
    Generates and saves a comprehensive visualization of clustering results.

    Args:
        X_pca_scaled (np.ndarray): PCA transformed scaled data.
        y_true (np.ndarray): True labels.
        nn_clusters (np.ndarray): Clusters from the neural network.
        kmeans_clusters (np.ndarray): Clusters from K-Means.
        agg_clusters (np.ndarray): Clusters from Agglomerative Clustering.
        dbscan_clusters (np.ndarray): Clusters from DBSCAN.
        latent_pca (np.ndarray): PCA transformed latent space data.
        pca_explained_variance_ratio (list): Explained variance ratio from PCA on scaled data.
        save_path (str): Path to save the figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Clustering Results Comparison", fontsize=16)

    # True clusters
    scatter = axes[0, 0].scatter(
        X_pca_scaled[:, 0], X_pca_scaled[:, 1], c=y_true, cmap="viridis"
    )
    axes[0, 0].set_title("True Classes")
    axes[0, 0].set_xlabel(f"PC1 ({pca_explained_variance_ratio[0]:.1%})")
    axes[0, 0].set_ylabel(f"PC2 ({pca_explained_variance_ratio[1]:.1%})")

    # Neural network clusters
    axes[0, 1].scatter(
        X_pca_scaled[:, 0], X_pca_scaled[:, 1], c=nn_clusters, cmap="viridis"
    )
    axes[0, 1].set_title("Neural Network Clustering")
    axes[0, 1].set_xlabel(f"PC1 ({pca_explained_variance_ratio[0]:.1%})")

    # K-means clusters
    axes[0, 2].scatter(
        X_pca_scaled[:, 0], X_pca_scaled[:, 1], c=kmeans_clusters, cmap="viridis"
    )
    axes[0, 2].set_title("K-Means Clustering")
    axes[0, 2].set_xlabel(f"PC1 ({pca_explained_variance_ratio[0]:.1%})")

    # Agglomerative clusters
    axes[1, 0].scatter(
        X_pca_scaled[:, 0], X_pca_scaled[:, 1], c=agg_clusters, cmap="viridis"
    )
    axes[1, 0].set_title("Agglomerative Clustering")
    axes[1, 0].set_xlabel(f"PC1 ({pca_explained_variance_ratio[0]:.1%})")
    axes[1, 0].set_ylabel(f"PC2 ({pca_explained_variance_ratio[1]:.1%})")

    # DBSCAN clusters
    axes[1, 1].scatter(
        X_pca_scaled[:, 0], X_pca_scaled[:, 1], c=dbscan_clusters, cmap="viridis"
    )
    axes[1, 1].set_title("DBSCAN Clustering")
    axes[1, 1].set_xlabel(f"PC1 ({pca_explained_variance_ratio[0]:.1%})")

    # Neural network latent space
    axes[1, 2].scatter(
        latent_pca[:, 0], latent_pca[:, 1], c=y_true, cmap="viridis"
    )  # Color by true classes
    axes[1, 2].set_title("Neural Network Latent Space\n(True Classes)")
    axes[1, 2].set_xlabel("Latent PC1")
    # Assuming latent_pca is 2D, if not, this needs adjustment or a different PCA fit for latent space
    # For simplicity, we assume latent_pca is already the 2D PCA of the latent space.

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Clustering comparison plot saved to {save_path}")


def plot_metrics_comparison(results, save_path):
    """
    Generates and saves bar plots for comparing clustering metrics.

    Args:
        results (dict): Dictionary containing evaluation results for different methods.
        save_path (str): Path to save the figure.
    """
    methods = list(results.keys())
    metrics_to_plot = ["silhouette", "ari", "nmi", "accuracy"]
    metric_names = [
        "Silhouette Score",
        "Adjusted Rand Index",
        "Normalized MI",
        "Alignment Accuracy",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, (metric_key, name) in enumerate(zip(metrics_to_plot, metric_names)):
        values = [
            (
                results[method][metric_key]
                if results[method] and metric_key in results[method]
                else 0
            )
            for method in methods
        ]
        axes[i].bar(methods, values)
        axes[i].set_title(name)
        axes[i].set_ylabel("Score")
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Metrics comparison plot saved to {save_path}")


def plot_davies_bouldin_comparison(results, save_path):
    """
    Generates and saves a bar plot for Davies-Bouldin scores.

    Args:
        results (dict): Dictionary containing evaluation results for different methods.
        save_path (str): Path to save the figure.
    """
    methods = list(results.keys())
    db_values = [
        (
            results[method]["davies_bouldin"]
            if results[method] and "davies_bouldin" in results[method]
            else 0
        )
        for method in methods
    ]
    plt.figure(figsize=(10, 6))
    plt.bar(methods, db_values)
    plt.title("Davies-Bouldin Score (Lower is Better)")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Davies-Bouldin comparison plot saved to {save_path}")
