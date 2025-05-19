# Deep Clustering with Neural Networks

## 🎯 Overview

This project implements and evaluates a **deep clustering approach** using neural networks, specifically comparing autoencoder-based clustering with traditional clustering methods. The implementation uses the Wine dataset as a benchmark to demonstrate the effectiveness of learned feature representations for unsupervised clustering tasks.

## 🚀 Key Features

- **Deep Clustering Model**: Autoencoder architecture with integrated clustering layer
- **Comprehensive Evaluation**: Multiple clustering metrics and visualization
- **Traditional Method Comparison**: K-Means, Agglomerative Clustering, and DBSCAN
- **Hyperparameter Optimization**: Automated tuning for optimal performance
- **Rich Visualizations**: PCA plots, correlation matrices, training history, and clustering comparisons

## 📊 Methodology

### Deep Clustering Architecture

The core model combines:

- **Autoencoder**: For learning meaningful feature representations
- **Clustering Layer**: Softmax layer for cluster assignment probabilities
- **Joint Training**: Simultaneous reconstruction and clustering loss optimization

### Pipeline Overview

1. **Dataset Analysis**: Load Wine dataset and perform exploratory analysis
2. **Data Preprocessing**: Standardization and PCA for visualization
3. **Hyperparameter Tuning**: Grid search for optimal model configuration
4. **Model Training**: Train final model with best hyperparameters
5. **Traditional Clustering**: Apply K-Means, Agglomerative, and DBSCAN
6. **Evaluation**: Compare all methods using multiple metrics
7. **Visualization**: Generate comprehensive plots and analysis

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Required packages listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd deep-clustering-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The `requirements.txt` currently lists PyTorch, but the project uses TensorFlow. Install TensorFlow instead:

```bash
pip install tensorflow>=2.8.0 scikit-learn>=1.0.0 matplotlib>=3.3.0 seaborn>=0.11.0 pandas>=1.3.0 numpy>=1.21.0
```

## 🎮 Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
python main.py
```

### Configuration

Modify hyperparameters in `main.py`:

```python
# Hyperparameter search space
PARAM_GRID = {
    'encoding_dims': [[64, 32, 16], [128, 64, 32], [64, 32]],
    'dropout_rate': [0.1, 0.2],
    'l2_reg': [0.001, 0.01],
    'learning_rate': [0.001, 0.01],
    'alpha': [0.1, 0.5],  # Clustering loss weight
    'batch_size': [16, 32]
}

# Training epochs
HP_TUNE_EPOCHS = 10    # For hyperparameter search
FINAL_MODEL_EPOCHS = 20  # For final model training
```

### Output

The pipeline generates:

- **Figures**: Saved in `figures/` directory
  - Feature correlation matrix
  - Class distribution and PCA visualization
  - Training history plots
  - Clustering comparison visualizations
  - Metrics comparison charts
- **Console Output**: Detailed evaluation metrics and summary report

## 📁 Project Structure

```
├── main.py                    # Main pipeline orchestrator
├── data_loader.py            # Dataset loading and analysis
├── preprocessing.py          # Data preprocessing utilities
├── model.py                  # Deep clustering model implementation
├── hyperparameter_tuning.py  # Hyperparameter optimization
├── training.py               # Model training functions
├── evaluation.py             # Clustering evaluation metrics
├── utils.py                  # Visualization and utility functions
├── requirements.txt          # Project dependencies
└── figures/                  # Generated visualizations (created during run)
```

## 🧠 Model Architecture

### Autoencoder Component

```
Input Layer (13 features)
    ↓
Encoder Layers (configurable dimensions)
    ↓ (with BatchNorm + Dropout)
Latent Representation
    ↓
Clustering Layer (softmax)
    ↓
Decoder Layers (reverse of encoder)
    ↓
Output Layer (13 features)
```

### Loss Function

```
Total Loss = Reconstruction Loss + α × Clustering Loss
```

Where:

- **Reconstruction Loss**: Mean Squared Error (MSE)
- **Clustering Loss**: Kullback-Leibler Divergence (KLD)
- **α**: Clustering loss weight (hyperparameter)

## 📈 Evaluation Metrics

The project evaluates clustering performance using:

- **Silhouette Score**: Measures cluster cohesion and separation
- **Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances
- **Adjusted Rand Index (ARI)**: Measures similarity to true clustering
- **Normalized Mutual Information (NMI)**: Information shared between clusters
- **Alignment Accuracy**: Optimal matching between predicted and true clusters

## 🎯 Results

The system generates a comprehensive summary comparing:

1. **Neural Network Clustering**: Deep learning approach
2. **K-Means**: Centroid-based clustering
3. **Agglomerative Clustering**: Hierarchical approach
4. **DBSCAN**: Density-based clustering

### Example Output

```
CLUSTERING PERFORMANCE SUMMARY
----------------------------------------------------------------------
Method           Silhouette  Davies-Bouldin  ARI     NMI     Accuracy  Clusters Found
Neural Network   0.425       1.234          0.678   0.712   0.845     3
K-Means          0.398       1.456          0.645   0.689   0.823     3
Agglomerative    0.412       1.345          0.662   0.698   0.834     3
DBSCAN          0.356       1.678          0.587   0.623   0.756     4
```

## 🔧 Customization

### Using Different Datasets

To use a different dataset, modify `data_loader.py`:

```python
def load_and_analyze_data(figures_dir):
    # Replace with your dataset loading logic
    # Ensure it returns: X, y_true, feature_names, pca_explained_variance_ratio
    pass
```

### Model Architecture

Customize the autoencoder in `model.py`:

```python
# Modify encoding dimensions
encoding_dims = [128, 64, 32, 16]  # Deeper network

# Adjust regularization
dropout_rate = 0.3
l2_reg = 0.005
```

### Clustering Methods

Add new clustering algorithms in `evaluation.py`:

```python
def perform_traditional_clustering(X_scaled, n_target_clusters, random_state=42):
    # Add your clustering method here
    cluster_predictions['NewMethod'] = new_method.fit_predict(X_scaled)
    return cluster_predictions
```

## 📚 Dependencies

- **TensorFlow**: Deep learning framework
- **scikit-learn**: Traditional clustering and metrics
- **NumPy**: Numerical computing
- **pandas**: Data manipulation
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical visualization

## 🤝 Contributing

Feel free to contribute by:

- Adding new clustering algorithms
- Implementing different evaluation metrics
- Enhancing visualizations
- Optimizing model architecture
- Adding support for different datasets

## 📄 License

This project is provided as-is for educational and research purposes.

## 🔍 Technical Notes

1. **Initialization**: K-Means is used to initialize clustering targets
2. **Scalability**: The approach works best with datasets of moderate size
3. **Hyperparameters**: Performance is sensitive to the clustering loss weight (α)
4. **Reproducibility**: Random seeds are set for consistent results

## 📞 Support

For questions or issues, please review the code comments and ensure all dependencies are correctly installed. The project includes extensive error handling and informative console output to help with debugging.

---

_This project demonstrates the power of combining representation learning with clustering in an end-to-end trainable framework._
