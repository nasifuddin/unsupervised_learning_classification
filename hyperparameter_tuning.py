# hyperparameter_tuning.py
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, davies_bouldin_score
from model import DeepClusteringModel # Assuming model.py is in the same directory

def evaluate_hyperparameters(X_scaled, n_clusters, param_grid, fit_epochs=30):
    """
    Evaluates different hyperparameter combinations for the DeepClusteringModel.

    Args:
        X_scaled (np.ndarray): Scaled input data.
        n_clusters (int): Number of target clusters.
        param_grid (dict): Dictionary with parameters to search.
        fit_epochs (int): Number of epochs to train each model during search.

    Returns:
        tuple: Best hyperparameters (dict) and a list of all results (list of dicts).
    """
    print("\n" + "="*50)
    print("3. HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    
    best_combined_score = -np.inf # Initialize with a very low score
    best_params_found = None # Renamed to avoid conflict
    all_results = [] # Renamed to avoid conflict
    
    for params in ParameterGrid(param_grid):
        print(f"Testing parameters: {params}")
        
        current_model = DeepClusteringModel( # Renamed to avoid conflict
            input_dim=X_scaled.shape[1], 
            n_clusters=n_clusters,
            encoding_dims=params['encoding_dims']
        )
        current_model.build_autoencoder(
            dropout_rate=params['dropout_rate'],
            l2_reg=params['l2_reg']
        )
        current_model.compile_model(
            learning_rate=params['learning_rate'],
            alpha=params['alpha']
        )
        
        # Train with fewer epochs for hyperparameter search
        # Ensure X_scaled is passed correctly.
        # The fit method in DeepClusteringModel initializes K-Means on X_scaled.
        current_model.fit(X_scaled, epochs=fit_epochs, batch_size=params['batch_size'], verbose=0)
        
        # Get clusters and evaluate
        clusters = current_model.predict_clusters(X_scaled)
        n_unique_clusters = len(np.unique(clusters))
        
        current_silhouette = -1.0  # Worst possible silhouette score
        current_davies_bouldin = 10.0  # High DB score indicates poor clustering
        current_combined_score = -10.0  # Very poor combined score

        if n_unique_clusters < 2 or n_unique_clusters > X_scaled.shape[0]-1 : # Check for valid cluster count for metrics
            print(f"Warning: Found {n_unique_clusters} cluster(s). Silhouette and Davies-Bouldin require 2 <= n_labels <= n_samples - 1.")
        else:
            try:
                current_silhouette = silhouette_score(X_scaled, clusters)
                current_davies_bouldin = davies_bouldin_score(X_scaled, clusters)
                # Combined score (higher silhouette, lower davies-bouldin is better)
                current_combined_score = current_silhouette - 0.5 * current_davies_bouldin 
            except ValueError as e:
                print(f"Error in evaluation for params {params}: {e}. Assigning poor score.")
        
        all_results.append({
            'params': params,
            'silhouette': current_silhouette,
            'davies_bouldin': current_davies_bouldin,
            'combined_score': current_combined_score,
            'n_clusters_found': n_unique_clusters
        })
        
        if current_combined_score > best_combined_score:
            best_combined_score = current_combined_score
            best_params_found = params
            
        print(f"Clusters found: {n_unique_clusters}, Silhouette: {current_silhouette:.3f}, Davies-Bouldin: {current_davies_bouldin:.3f}, Combined: {current_combined_score:.3f}")
    
    if best_params_found:
        print(f"\nBest hyperparameters found (Combined Score: {best_combined_score:.3f}):")
        for key, value in best_params_found.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo best hyperparameters found. This might indicate issues during evaluation.")
        # Fallback to a default set of parameters if none are found to be "best"
        # This is crucial if all evaluations failed or resulted in very poor scores.
        # For now, we'll let it return None and handle it in main.py
        # Or, select the first parameter set from the grid as a fallback.
        if all_results:
             best_params_found = all_results[0]['params']
             print(f"Falling back to the first parameter set: {best_params_found}")
        else: # Should not happen if param_grid is not empty
            print("Param grid was empty. Cannot select fallback parameters.")


    return best_params_found, all_results
