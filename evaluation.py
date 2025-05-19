# evaluation.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def _cluster_accuracy(y_true_filtered, y_pred_filtered):
    """
    Calculates clustering accuracy using the Hungarian algorithm for optimal matching.
    Handles cases where the number of predicted clusters might differ from true clusters.
    """
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    
    # If the number of unique true labels is different from unique predicted labels,
    # the confusion matrix might not be square. We need to handle this.
    # The linear_sum_assignment expects a cost matrix where rows are true clusters
    # and columns are predicted clusters.
    
    # Pad the confusion matrix if necessary to make it square or rectangular for assignment
    # This part is tricky if n_true_clusters != n_pred_clusters.
    # A common approach is to map predicted clusters to true clusters.
    # The original code pads to max_size x max_size, which might not be ideal if one dimension is much smaller.

    # Let's use the number of unique true labels and unique predicted labels to form the cost matrix.
    # This assumes we want to find the best mapping from predicted to true.
    
    # If cm is not square, linear_sum_assignment might still work if maximizing overlap.
    # The cost matrix should be constructed such that higher values are better (negative for minimization).
    
    # Optimal assignment using Hungarian algorithm
    # We want to maximize the sum of diagonal elements after optimal permutation.
    # linear_sum_assignment minimizes cost, so we use -cm.
    row_ind, col_ind = linear_sum_assignment(-cm)
    accuracy = cm[row_ind, col_ind].sum() / y_true_filtered.shape[0]
    return accuracy

def evaluate_clustering_performance(y_true, clusters, X_data, method_name):
    """
    Evaluates clustering performance using various internal and external metrics.

    Args:
        y_true (np.ndarray): True class labels.
        clusters (np.ndarray): Predicted cluster labels.
        X_data (np.ndarray): Data used for clustering (e.g., X_scaled).
        method_name (str): Name of the clustering method.

    Returns:
        dict: A dictionary containing the evaluation metrics, or None if evaluation fails.
    """
    print(f"\nEvaluating: {method_name}")
    
    # Handle noise points in DBSCAN (label -1)
    # Only consider points that were assigned to a cluster for metric calculation
    valid_clusters_mask = clusters != -1
    
    if not np.any(valid_clusters_mask): # All points are noise
        print(f"{method_name}: All points classified as noise. Cannot compute metrics.")
        return { # Return a structure with N/A or default bad values
            'silhouette': -1.0, 'davies_bouldin': 10.0, 'ari': -1.0, 
            'nmi': -1.0, 'accuracy': 0.0, 'n_clusters': 0
        }

    y_true_filtered = y_true[valid_clusters_mask]
    clusters_filtered = clusters[valid_clusters_mask]
    X_data_filtered = X_data[valid_clusters_mask]
    
    n_unique_clusters_found = len(np.unique(clusters_filtered))
    
    # Metrics require at least 2 clusters and less than n_samples-1 clusters.
    if not (2 <= n_unique_clusters_found < X_data_filtered.shape[0]):
        print(f"{method_name}: Cannot compute metrics. Found {n_unique_clusters_found} unique clusters "
              f"for {X_data_filtered.shape[0]} samples (after filtering noise). "
              "Silhouette and Davies-Bouldin require 2 <= n_labels <= n_samples - 1.")
        # Return default bad scores if metrics can't be computed
        return {
            'silhouette': -1.0 if n_unique_clusters_found < 2 else silhouette_score(X_data_filtered, clusters_filtered) if X_data_filtered.shape[0] > n_unique_clusters_found else -1.0, # Attempt if possible
            'davies_bouldin': 10.0 if n_unique_clusters_found < 2 else davies_bouldin_score(X_data_filtered, clusters_filtered) if X_data_filtered.shape[0] > n_unique_clusters_found else 10.0, # Attempt if possible
            'ari': adjusted_rand_score(y_true_filtered, clusters_filtered) if len(y_true_filtered) > 1 and len(clusters_filtered) > 1 else -1.0,
            'nmi': normalized_mutual_info_score(y_true_filtered, clusters_filtered) if len(y_true_filtered) > 1 and len(clusters_filtered) > 1 else -1.0,
            'accuracy': _cluster_accuracy(y_true_filtered, clusters_filtered) if len(y_true_filtered) > 0 and len(clusters_filtered) > 0 else 0.0,
            'n_clusters': n_unique_clusters_found
        }

    try:
        silhouette = silhouette_score(X_data_filtered, clusters_filtered)
        davies_bouldin = davies_bouldin_score(X_data_filtered, clusters_filtered)
        ari = adjusted_rand_score(y_true_filtered, clusters_filtered)
        nmi = normalized_mutual_info_score(y_true_filtered, clusters_filtered)
        accuracy = _cluster_accuracy(y_true_filtered, clusters_filtered)
        
        results_dict = {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'ari': ari,
            'nmi': nmi,
            'accuracy': accuracy,
            'n_clusters': n_unique_clusters_found
        }
        
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
        print(f"  Adjusted Rand Index: {ari:.3f}")
        print(f"  Normalized Mutual Information: {nmi:.3f}")
        print(f"  Alignment-based Accuracy: {accuracy:.3f}")
        print(f"  Number of Clusters Found (after filtering): {n_unique_clusters_found}")
        
        return results_dict
        
    except ValueError as e: # Catch specific ValueError from metrics
        print(f"{method_name}: Error computing metrics: {e}. This often happens if too few clusters are found.")
        return { # Return a structure with N/A or default bad values
            'silhouette': -1.0, 'davies_bouldin': 10.0, 'ari': -1.0, 
            'nmi': -1.0, 'accuracy': 0.0, 'n_clusters': n_unique_clusters_found
        }
    except Exception as e: # Catch any other unexpected error
        print(f"{method_name}: Unexpected error computing metrics: {e}")
        return None


def perform_traditional_clustering(X_scaled, n_target_clusters, random_state=42):
    """
    Performs clustering using K-Means, Agglomerative Clustering, and DBSCAN.

    Args:
        X_scaled (np.ndarray): Scaled input data.
        n_target_clusters (int): The target number of clusters (for K-Means and Agglomerative).
        random_state (int): Random state for reproducibility.

    Returns:
        dict: A dictionary where keys are method names and values are predicted cluster labels.
    """
    print("\n" + "="*50)
    print("6. TRADITIONAL CLUSTERING COMPARISON")
    print("="*50)

    cluster_predictions = {}

    # K-Means
    kmeans = KMeans(n_clusters=n_target_clusters, random_state=random_state, n_init='auto')
    cluster_predictions['K-Means'] = kmeans.fit_predict(X_scaled)
    print(f"K-Means completed. Clusters: {np.unique(cluster_predictions['K-Means'])}")

    # Agglomerative Clustering
    # n_clusters cannot be less than 1. If n_target_clusters is 0 or less, this will fail.
    if n_target_clusters < 1:
        print(f"Warning: n_target_clusters is {n_target_clusters}, which is invalid for AgglomerativeClustering. Skipping.")
        cluster_predictions['Agglomerative'] = np.zeros(X_scaled.shape[0], dtype=int) # Placeholder
    else:
        agg_clustering = AgglomerativeClustering(n_clusters=n_target_clusters)
        cluster_predictions['Agglomerative'] = agg_clustering.fit_predict(X_scaled)
        print(f"Agglomerative Clustering completed. Clusters: {np.unique(cluster_predictions['Agglomerative'])}")


    # DBSCAN (estimate eps using k-distance)
    # min_samples for DBSCAN is typically 2 * D (dimensions) or based on domain knowledge.
    # Original script used min_samples=4. Let's make it more robust.
    # A common heuristic for min_samples is ln(N) or a value related to dimensionality.
    # Let's stick to a small constant or D+1. For wine dataset (13 features), D+1 = 14.
    # The original used 4. Let's try min_samples = max(4, X_scaled.shape[1] // 2)
    min_samples_dbscan = max(4, int(X_scaled.shape[1] / 2))
    if X_scaled.shape[0] <= min_samples_dbscan : # Not enough samples for DBSCAN
        print(f"Warning: Not enough samples ({X_scaled.shape[0]}) for DBSCAN with min_samples={min_samples_dbscan}. Skipping DBSCAN.")
        cluster_predictions['DBSCAN'] = np.zeros(X_scaled.shape[0], dtype=int) # Placeholder, all in one cluster (or noise)
    else:
        try:
            # k for k-distance plot is often min_samples - 1
            # The original used n_neighbors=4, which is related to min_samples.
            # If min_samples is k, then we look at the k-th nearest neighbor distance.
            neighbors = NearestNeighbors(n_neighbors=min_samples_dbscan) # k = min_samples
            neighbors_fit = neighbors.fit(X_scaled)
            distances, _ = neighbors_fit.kneighbors(X_scaled)
            
            # Sort distance to k-th nearest neighbor (i.e., distances[:, min_samples_dbscan-1])
            k_distances = np.sort(distances[:, min_samples_dbscan-1], axis=0)
            
            # Heuristic for eps: find the "elbow" in the k-distance plot.
            # Original script used 90th percentile, which is a reasonable heuristic.
            eps = np.percentile(k_distances, 90) 
            print(f"DBSCAN: Estimated eps={eps:.3f} with min_samples={min_samples_dbscan}")

            dbscan = DBSCAN(eps=eps, min_samples=min_samples_dbscan)
            cluster_predictions['DBSCAN'] = dbscan.fit_predict(X_scaled)
            print(f"DBSCAN completed. Clusters: {np.unique(cluster_predictions['DBSCAN'])}")
        except Exception as e:
            print(f"Error during DBSCAN: {e}. Assigning default clusters for DBSCAN.")
            cluster_predictions['DBSCAN'] = np.zeros(X_scaled.shape[0], dtype=int) # Default: all points in one cluster

    return cluster_predictions

def generate_summary_report(all_results, X_scaled_shape_1, best_params_dict, n_true_clusters, total_model_params):
    """
    Prints a summary report of clustering performance and model architecture.

    Args:
        all_results (dict): Dictionary containing evaluation results for all methods.
        X_scaled_shape_1 (int): Input dimension of the scaled data.
        best_params_dict (dict): Dictionary of best hyperparameters for the neural model.
        n_true_clusters (int): Number of true clusters (classes in the dataset).
        total_model_params (int): Total parameters in the trained neural model.
    """
    print("\n" + "="*50)
    print("9. FINAL SUMMARY REPORT")
    print("="*50)

    print("CLUSTERING PERFORMANCE SUMMARY")
    print("-" * 70) # Adjusted width

    summary_data = []
    methods_order = ['Neural Network', 'K-Means', 'Agglomerative', 'DBSCAN'] # Desired order

    for method in methods_order:
        if method in all_results and all_results[method]:
            res = all_results[method]
            summary_data.append([
                method,
                f"{res.get('silhouette', 'N/A'):.3f}" if isinstance(res.get('silhouette'), float) else res.get('silhouette', 'N/A'),
                f"{res.get('davies_bouldin', 'N/A'):.3f}" if isinstance(res.get('davies_bouldin'), float) else res.get('davies_bouldin', 'N/A'),
                f"{res.get('ari', 'N/A'):.3f}" if isinstance(res.get('ari'), float) else res.get('ari', 'N/A'),
                f"{res.get('nmi', 'N/A'):.3f}" if isinstance(res.get('nmi'), float) else res.get('nmi', 'N/A'),
                f"{res.get('accuracy', 'N/A'):.3f}" if isinstance(res.get('accuracy'), float) else res.get('accuracy', 'N/A'),
                res.get('n_clusters', 'N/A')
            ])
        else:
            summary_data.append([method, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

    summary_df = pd.DataFrame(summary_data, columns=[
        'Method', 'Silhouette', 'Davies-Bouldin', 'ARI', 'NMI', 'Accuracy', 'Clusters Found'
    ])

    print(summary_df.to_string(index=False))

    # Find best performing method based on a composite score (e.g., sum of ARI, NMI, Accuracy)
    # Ensure results for 'Neural Network' exist before trying to access them
    if 'Neural Network' in all_results and all_results['Neural Network']:
        best_overall_method = None
        max_composite_score = -np.inf

        for method in methods_order:
            if method in all_results and all_results[method]:
                res = all_results[method]
                # Check if metrics are available and are numbers
                ari = res.get('ari', -1.0) if isinstance(res.get('ari'), (int, float)) else -1.0
                nmi = res.get('nmi', -1.0) if isinstance(res.get('nmi'), (int, float)) else -1.0
                acc = res.get('accuracy', -1.0) if isinstance(res.get('accuracy'), (int, float)) else -1.0
                
                # Ensure Silhouette is positive (or handle negative if that's the scale)
                # Silhouette is [-1, 1]. Davies-Bouldin is [0, inf), lower is better.
                # For simplicity, let's use ARI, NMI, Accuracy for "best" ranking as in original
                composite_score = ari + nmi + acc
                
                if composite_score > max_composite_score:
                    max_composite_score = composite_score
                    best_overall_method = method
        
        if best_overall_method:
            print(f"\nBest overall performance (based on ARI+NMI+Accuracy): {best_overall_method}")
            if best_overall_method == 'Neural Network':
                print("✓ Neural Network clustering achieved/matched best performance!")
            else:
                print(f"Neural Network performance compared to {best_overall_method}.")
        else:
            print("\nCould not determine the best overall performing method from the metrics.")

    else:
        print("\nNeural Network results are not available to determine the best overall method.")

    if best_params_dict:
        print(f"\nDeep Clustering Model Architecture Summary (Best Hyperparameters):")
        print(f"- Input dimension: {X_scaled_shape_1}")
        print(f"- Encoding layers: {best_params_dict.get('encoding_dims', 'N/A')}")
        latent_dim = best_params_dict.get('encoding_dims', [])
        print(f"- Latent dimension: {latent_dim[-1] if latent_dim else 'N/A'}")
        print(f"- Number of clusters targeted: {n_true_clusters}") # This is n_clusters passed to the model
        print(f"- Total parameters in final model: {total_model_params:,}")
        print(f"- Dropout rate: {best_params_dict.get('dropout_rate', 'N/A')}")
        print(f"- L2 regularization: {best_params_dict.get('l2_reg', 'N/A')}")
        print(f"- Learning rate: {best_params_dict.get('learning_rate', 'N/A')}")
        print(f"- Clustering loss weight (alpha): {best_params_dict.get('alpha', 'N/A')}")
    else:
        print("\nBest hyperparameters not available for model architecture summary.")

    print("\nKey Findings (General Observations):")
    print("- Deep clustering aims to learn feature representations tailored for clustering.")
    print("- Performance can be sensitive to architecture, hyperparameters, and initialization.")
    print("- Comparing with traditional methods provides a good baseline.")

