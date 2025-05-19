# main.py
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf # For tf.keras.backend.count_params

# Import project modules
from utils import ensure_dir, set_seeds
from data_loader import load_and_analyze_data
from preprocessing import preprocess_data, apply_pca_for_visualization
from model import DeepClusteringModel # Already imported in hyperparameter_tuning and training
from hyperparameter_tuning import evaluate_hyperparameters
from training import train_final_model
from evaluation import perform_traditional_clustering, evaluate_clustering_performance, generate_summary_report
# Visualization functions are now in utils.py and will be called from main where needed
from utils import (
    plot_training_history, # Already called in training.py
    plot_clustering_comparison, 
    plot_metrics_comparison,
    plot_davies_bouldin_comparison
)


# --- Configuration ---
FIGURES_DIR = 'figures'
RANDOM_SEED = 42
# Hyperparameter search space (can be moved to a config file or defined here)
PARAM_GRID = {
    'encoding_dims': [[64, 32, 16], [128, 64, 32], [64, 32]],
    'dropout_rate': [0.1, 0.2], # Reduced for faster example run
    'l2_reg': [0.001, 0.01],    # Reduced
    'learning_rate': [0.001, 0.01],
    'alpha': [0.1, 0.5],
    'batch_size': [16, 32]
}
HP_TUNE_EPOCHS = 10 # Reduced for faster example run, original was 30
FINAL_MODEL_EPOCHS = 20 # Reduced for faster example run, original was 100

def main():
    """
    Main function to run the unsupervised clustering pipeline.
    """
    warnings.filterwarnings('ignore')
    ensure_dir(FIGURES_DIR)
    set_seeds(RANDOM_SEED)

    # 1. DATASET ANALYSIS
    # pca_explained_variance_ratio_orig is from PCA on original X, used for the first PCA plot
    X, y_true, feature_names, pca_explained_variance_ratio_orig = load_and_analyze_data(FIGURES_DIR)
    n_true_clusters = len(np.unique(y_true)) # Number of actual classes in the dataset

    # 2. DATA PREPROCESSING
    X_scaled, scaler = preprocess_data(X)

    # 3. HYPERPARAMETER OPTIMIZATION
    # Note: evaluate_hyperparameters uses X_scaled
    best_params, hp_results_df = evaluate_hyperparameters(
        X_scaled, 
        n_clusters=n_true_clusters, # Use true number of clusters as target for the model
        param_grid=PARAM_GRID,
        fit_epochs=HP_TUNE_EPOCHS 
    )
    
    if best_params is None:
        print("Hyperparameter tuning did not find best parameters. Exiting.")
        # Provide a default set of parameters to proceed if desired, or exit.
        # For this example, we'll try to use the first set from the grid if tuning fails.
        if PARAM_GRID:
            first_params = list(ParameterGrid(PARAM_GRID))[0]
            print(f"Warning: Using default first parameter set due to tuning failure: {first_params}")
            best_params = first_params
        else:
            print("PARAM_GRID is empty. Cannot proceed.")
            return


    # 4. BUILD AND TRAIN FINAL MODEL (using X_scaled)
    # The train_final_model function now also handles printing the model summary.
    final_deep_model, history = train_final_model(
        X_scaled, 
        n_clusters=n_true_clusters, 
        best_params=best_params, 
        figures_dir=FIGURES_DIR,
        epochs=FINAL_MODEL_EPOCHS
    )
    total_model_params = final_deep_model.model.count_params() if final_deep_model.model else 0


    # 6. CLUSTERING COMPARISON
    print("\n" + "="*50)
    print("6. CLUSTERING COMPARISON (Predictions)")
    print("="*50)
    # Get neural network clusters and latent representation
    nn_clusters = final_deep_model.predict_clusters(X_scaled)
    nn_latent_representation = final_deep_model.get_latent_representation(X_scaled)
    print(f"Neural Network predicted clusters: {np.unique(nn_clusters)}")

    # Perform traditional clustering methods
    traditional_clusters = perform_traditional_clustering(
        X_scaled, 
        n_target_clusters=n_true_clusters, # Target same number of clusters
        random_state=RANDOM_SEED
    )

    # 7. EVALUATION METRICS
    print("\n" + "="*50)
    print("7. EVALUATION METRICS")
    print("="*50)
    all_method_results = {}

    # Evaluate Neural Network
    all_method_results['Neural Network'] = evaluate_clustering_performance(
        y_true, nn_clusters, X_scaled, 'Neural Network'
    )
    # Evaluate Traditional Methods
    for method_name, clusters_pred in traditional_clusters.items():
        all_method_results[method_name] = evaluate_clustering_performance(
            y_true, clusters_pred, X_scaled, method_name
        )
    
    # 8. VISUALIZATION of clustering results
    print("\n" + "="*50)
    print("8. CLUSTER VISUALIZATION")
    print("="*50)
    
    # Apply PCA to scaled data for visualization of clusters
    X_pca_scaled, pca_obj_scaled = apply_pca_for_visualization(X_scaled, n_components=2)
    
    # Apply PCA to latent space for visualization
    # Ensure nn_latent_representation is not empty and has enough samples/features for PCA
    if nn_latent_representation.shape[0] > 1 and nn_latent_representation.shape[1] > 1:
        latent_pca_transformed, _ = apply_pca_for_visualization(nn_latent_representation, n_components=2)
    elif nn_latent_representation.shape[0] > 1 and nn_latent_representation.shape[1] == 1: # If latent space is 1D
        # Create a second dimension of zeros for plotting, or handle 1D plotting
        latent_pca_transformed = np.hstack((nn_latent_representation, np.zeros_like(nn_latent_representation)))
        print("Latent space is 1D, padded with zeros for 2D visualization.")
    else:
        print("Latent representation is unsuitable for PCA (e.g., empty or 0 features). Skipping latent space PCA plot.")
        # Create a dummy array for latent_pca_transformed to prevent errors in plot_clustering_comparison
        # It should have 2 columns for the plotting function.
        num_samples = X_pca_scaled.shape[0] # Match number of samples
        latent_pca_transformed = np.zeros((num_samples, 2))


    plot_clustering_comparison(
        X_pca_scaled, 
        y_true, 
        nn_clusters,
        traditional_clusters.get('K-Means', np.zeros_like(y_true)), # Provide default if key missing
        traditional_clusters.get('Agglomerative', np.zeros_like(y_true)),
        traditional_clusters.get('DBSCAN', np.zeros_like(y_true)),
        latent_pca_transformed,
        pca_obj_scaled.explained_variance_ratio_, # Use explained variance from PCA on X_scaled
        f"{FIGURES_DIR}/04_clustering_comparison.png"
    )

    # Metrics comparison bar plots
    plot_metrics_comparison(all_method_results, f"{FIGURES_DIR}/05_metrics_comparison.png")
    plot_davies_bouldin_comparison(all_method_results, f"{FIGURES_DIR}/06_davies_bouldin_comparison.png")

    # 9. SUMMARY REPORT
    generate_summary_report(
        all_method_results,
        X_scaled.shape[1], # input_dim
        best_params,
        n_true_clusters,
        total_model_params
    )

    print("\n" + "="*50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == '__main__':
    main()
