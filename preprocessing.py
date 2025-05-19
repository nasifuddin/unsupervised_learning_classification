# preprocessing.py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(X):
    """
    Standardizes the features.

    Args:
        X (np.ndarray): Feature data.

    Returns:
        tuple: Scaled feature data (X_scaled) and the scaler object.
    """
    print("\n" + "="*50)
    print("2. DATA PREPROCESSING")
    print("="*50)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Data preprocessing completed:")
    print(f"Original data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Scaled data range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    
    return X_scaled, scaler

def apply_pca_for_visualization(X_scaled, n_components=2):
    """
    Applies PCA to the scaled data, typically for visualization purposes.

    Args:
        X_scaled (np.ndarray): Scaled feature data.
        n_components (int): Number of principal components.

    Returns:
        tuple: PCA transformed data (X_pca) and the PCA object.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"\nApplied PCA for visualization with {n_components} components.")
    print(f"Explained variance ratio by PCA components: {pca.explained_variance_ratio_}")
    return X_pca, pca
