# training.py
from model import DeepClusteringModel # Assuming model.py is in the same directory
from utils import plot_training_history # Assuming utils.py is in the same directory

def train_final_model(X_scaled, n_clusters, best_params, figures_dir, epochs=100):
    """
    Builds, compiles, and trains the final DeepClusteringModel with the best hyperparameters.

    Args:
        X_scaled (np.ndarray): Scaled input data.
        n_clusters (int): Number of target clusters.
        best_params (dict): Dictionary of best hyperparameters.
        figures_dir (str): Directory to save training history plot.
        epochs (int): Number of epochs for final training.

    Returns:
        DeepClusteringModel: The trained model.
        tf.keras.callbacks.History: Training history.
    """
    print("\n" + "="*50)
    print("4. FINAL MODEL TRAINING")
    print("="*50)

    if best_params is None:
        raise ValueError("best_params cannot be None. Hyperparameter tuning might have failed.")

    # Build final model with best hyperparameters
    final_model_instance = DeepClusteringModel( # Renamed to avoid conflict
        input_dim=X_scaled.shape[1], 
        n_clusters=n_clusters,
        encoding_dims=best_params['encoding_dims']
    )

    final_model_instance.build_autoencoder(
        dropout_rate=best_params['dropout_rate'],
        l2_reg=best_params['l2_reg']
    )

    final_model_instance.compile_model(
        learning_rate=best_params['learning_rate'],
        alpha=best_params['alpha']
    )

    # Model Summary
    print("\n" + "="*50)
    print("5. MODEL SUMMARY")
    print("="*50)
    final_model_instance.summary() # Call the summary method of the class

    # Train final model
    print("\nTraining final model...")
    # The fit method of DeepClusteringModel handles KMeans initialization internally
    history = final_model_instance.fit(
        X_scaled, 
        epochs=epochs, 
        batch_size=best_params['batch_size'],
        validation_split=0.2, # As per original script
        verbose=1 # As per original script
    )

    # Plot training history
    plot_training_history(history, f"{figures_dir}/03_training_history.png")
    
    return final_model_instance, history
