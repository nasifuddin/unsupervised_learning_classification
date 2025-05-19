# data_loader.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA # Added for PCA explained variance calculation
from utils import plot_feature_correlation_matrix, plot_class_distribution_and_pca

def load_and_analyze_data(figures_dir):
    """
    Loads the Wine dataset, performs basic analysis, and saves visualizations.

    Args:
        figures_dir (str): Directory to save figures.

    Returns:
        tuple: Contains X (features), y_true (true labels), feature_names,
               and pca_explained_variance_ratio for later use in visualization.
    """
    print("="*50)
    print("1. DATASET ANALYSIS")
    print("="*50)

    # Load the Wine dataset
    wine_data = load_wine()
    X = wine_data.data
    y_true = wine_data.target
    feature_names = wine_data.feature_names

    # Create DataFrame for easier analysis
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y_true # Keep target for analysis, but it won't be used for unsupervised learning

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of classes (for reference): {len(np.unique(y_true))}")
    print(f"Class distribution (for reference): {np.bincount(y_true)}")

    # Basic statistics
    print("\nDataset Statistics:")
    print(df.describe())

    # Check for missing values
    print(f"\nMissing values: {df.isnull().sum().sum()}")

    # Feature correlation analysis
    plot_feature_correlation_matrix(df.drop('target', axis=1), f"{figures_dir}/01_feature_correlation_matrix.png")

    # PCA for visualization and getting explained variance ratio
    pca_viz = PCA(n_components=2)
    pca_viz.fit(X) # Fit on original X data
    pca_explained_variance_ratio = pca_viz.explained_variance_ratio_

    # Class distribution and PCA visualization (using original X for PCA of true classes)
    plot_class_distribution_and_pca(X, y_true, pca_explained_variance_ratio, f"{figures_dir}/02_class_distribution_and_pca.png")
    
    return X, y_true, feature_names, pca_explained_variance_ratio
