# model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans # For initializing cluster targets

class DeepClusteringModel:
    """
    A deep clustering model using an autoencoder architecture with a clustering layer.
    """
    def __init__(self, input_dim, n_clusters, encoding_dims=[64, 32, 16]):
        """
        Initializes the DeepClusteringModel.

        Args:
            input_dim (int): The dimension of the input data.
            n_clusters (int): The number of clusters to find.
            encoding_dims (list): List of integers representing the dimensions of encoder layers.
        """
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.encoding_dims = encoding_dims
        self.model = None
        self.encoder = None
        self.decoder = None
        
    def build_autoencoder(self, dropout_rate=0.2, l2_reg=0.01):
        """
        Builds the autoencoder architecture with a clustering layer.

        Args:
            dropout_rate (float): Dropout rate for regularization.
            l2_reg (float): L2 regularization factor.

        Returns:
            tf.keras.Model: The compiled Keras model.
        """
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for i, dim in enumerate(self.encoding_dims):
            encoded = layers.Dense(dim, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(l2_reg))(encoded)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(dropout_rate)(encoded)
        
        # Latent representation
        latent_dim = self.encoding_dims[-1]
        # Ensure latent_dim is positive, if encoding_dims is empty, this could be an issue.
        if not self.encoding_dims:
            raise ValueError("encoding_dims cannot be empty.")
            
        encoded_output = layers.Dense(latent_dim, activation='relu', name='latent')(encoded) # Renamed to avoid conflict
        
        # Clustering layer
        cluster_layer = layers.Dense(self.n_clusters, activation='softmax', 
                                   name='clustering')(encoded_output) # Use encoded_output
        
        # Decoder
        decoded = encoded_output # Start decoding from the latent representation
        # Iterate in reverse order of encoding_dims, excluding the last one (latent_dim)
        for dim in reversed(self.encoding_dims[:-1]): 
            decoded = layers.Dense(dim, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(l2_reg))(decoded)
            decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dropout(dropout_rate)(decoded)
        
        # Output layer - explicitly name it 'decoder'
        decoded_output = layers.Dense(self.input_dim, activation='linear', name='decoder_output')(decoded) # Renamed to avoid conflict
        
        # Create models
        self.encoder = keras.Model(input_layer, encoded_output, name='encoder')
        # The decoder model should take the latent representation as input
        # Need a new input layer for the standalone decoder model
        latent_input = layers.Input(shape=(latent_dim,), name='latent_input')
        decoded_for_standalone_decoder = latent_input
        for dim in reversed(self.encoding_dims[:-1]):
            decoded_for_standalone_decoder = layers.Dense(dim, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(l2_reg))(decoded_for_standalone_decoder)
            decoded_for_standalone_decoder = layers.BatchNormalization()(decoded_for_standalone_decoder)
            decoded_for_standalone_decoder = layers.Dropout(dropout_rate)(decoded_for_standalone_decoder)
        standalone_decoder_output = layers.Dense(self.input_dim, activation='linear')(decoded_for_standalone_decoder)
        self.decoder = keras.Model(latent_input, standalone_decoder_output, name='decoder')

        self.model = keras.Model(input_layer, [decoded_output, cluster_layer], name='deep_clustering')
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, alpha=0.1):
        """
        Compiles the model with a custom loss function.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            alpha (float): Weight for the clustering loss component.
        """
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'decoder_output': 'mse', # Ensure this matches the output layer name
                'clustering': 'kld'
            },
            loss_weights={'decoder_output': 1.0, 'clustering': alpha} # Ensure this matches
        )
    
    def fit(self, X, epochs=100, batch_size=32, validation_split=0.2, verbose=1, random_state_kmeans=42):
        """
        Trains the model.

        Args:
            X (np.ndarray): Input data.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of data to use for validation.
            verbose (int): Verbosity mode (0, 1, or 2).
            random_state_kmeans (int): Random state for KMeans initialization.

        Returns:
            tf.keras.callbacks.History: Training history object.
        """
        # Prepare targets for training
        y_reconstruction = X.copy()
        y_clustering_target = np.zeros((X.shape[0], self.n_clusters)) # Renamed to avoid confusion
        
        # Initialize clustering targets with K-means
        # Important: Use the latent representation from the *current* encoder for K-Means if pre-training AE first,
        # or use X if training end-to-end directly. The original script uses X.
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state_kmeans, n_init='auto')
        initial_clusters = kmeans.fit_predict(X) # Using X as per original script for initial targets
        for i in range(X.shape[0]):
            y_clustering_target[i, initial_clusters[i]] = 1.0
        
        # Training
        history = self.model.fit(
            X, 
            {'decoder_output': y_reconstruction, 'clustering': y_clustering_target}, # Match output names
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        return history
    
    def predict_clusters(self, X):
        """
        Predicts cluster assignments for the input data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        _, cluster_probs = self.model.predict(X)
        return np.argmax(cluster_probs, axis=1)
    
    def get_latent_representation(self, X):
        """
        Gets the latent representation of the input data from the encoder.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Latent representations.
        """
        if self.encoder is None:
            raise ValueError("Encoder is not built. Call build_autoencoder first.")
        return self.encoder.predict(X)

    def summary(self):
        """Prints the model summaries."""
        if self.model:
            print("Complete Model Architecture:")
            self.model.summary()
        if self.encoder:
            print("\nEncoder Architecture:")
            self.encoder.summary()
        if self.decoder:
            print("\nDecoder Architecture:")
            self.decoder.summary()
        
        if self.model:
            total_params = self.model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
            print(f"\nTotal parameters in deep_clustering model: {total_params:,}")
            print(f"Trainable parameters in deep_clustering model: {trainable_params:,}")

