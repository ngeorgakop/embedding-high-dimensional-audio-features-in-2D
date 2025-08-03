"""Neural network module for predicting t-SNE coordinates from features."""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import json
import logging
from dataclasses import dataclass

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural network functionality disabled.")

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Configuration for neural network training."""
    hidden_layers: List[int] = None  # e.g., [20, 50, 100, 50, 20]
    epochs: int = 100
    batch_size: int = 5
    learning_rate: float = 0.001
    train_size: float = 0.8
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [20, 50, 100, 50, 20]


class TSNERegressor:
    """Neural network for learning t-SNE mappings."""
    
    def __init__(self, config: NetworkConfig):
        """Initialize regressor with configuration.
        
        Args:
            config: Network configuration
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network functionality")
        
        self.config = config
        self.model_2d = None
        self.model_3d = None
        self.feature_dim = None
        
        # Set random seeds for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            tf.random.set_seed(config.random_seed)
    
    def _build_model(self, output_dim: int, input_dim: int) -> keras.Model:
        """Build neural network model.
        
        Args:
            output_dim: Output dimensionality (2 for coordinates, 3 for colors)
            input_dim: Input feature dimensionality
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.config.hidden_layers[0],
            input_dim=input_dim,
            activation='relu',
            kernel_initializer='normal'
        ))
        
        # Hidden layers
        for hidden_size in self.config.hidden_layers[1:]:
            model.add(layers.Dense(
                hidden_size,
                activation='relu',
                kernel_initializer='normal'
            ))
        
        # Output layer
        model.add(layers.Dense(
            output_dim,
            kernel_initializer='normal'
        ))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def prepare_training_data(self, features: np.ndarray, 
                            coordinates_2d: np.ndarray,
                            colors_3d: np.ndarray,
                            filenames: List[str]) -> Dict[str, Any]:
        """Prepare training data from features and t-SNE results.
        
        Args:
            features: High-dimensional feature matrix
            coordinates_2d: 2D t-SNE coordinates
            colors_3d: 3D t-SNE coordinates for colors
            filenames: Corresponding filenames
            
        Returns:
            Dictionary containing prepared datasets
        """
        if len(features) != len(coordinates_2d) or len(features) != len(colors_3d):
            raise ValueError("All input arrays must have the same length")
        
        self.feature_dim = features.shape[1]
        
        # Split data for training and testing
        indices = np.arange(len(features))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=1.0 - self.config.train_size,
            random_state=self.config.random_seed
        )
        
        # Prepare datasets
        train_data = {
            'features': features[train_idx],
            'coordinates_2d': coordinates_2d[train_idx],
            'colors_3d': colors_3d[train_idx],
            'filenames': [filenames[i] for i in train_idx]
        }
        
        test_data = {
            'features': features[test_idx],
            'coordinates_2d': coordinates_2d[test_idx],
            'colors_3d': colors_3d[test_idx],
            'filenames': [filenames[i] for i in test_idx]
        }
        
        logger.info(f"Training data: {len(train_data['features'])} samples")
        logger.info(f"Test data: {len(test_data['features'])} samples")
        
        return {
            'train': train_data,
            'test': test_data,
            'feature_dim': self.feature_dim
        }
    
    def train(self, dataset: Dict[str, Any]) -> Dict[str, float]:
        """Train neural networks for 2D and 3D prediction.
        
        Args:
            dataset: Prepared training dataset
            
        Returns:
            Dictionary containing training metrics
        """
        train_data = dataset['train']
        test_data = dataset['test']
        
        results = {}
        
        # Train 2D coordinate prediction model
        logger.info("Training 2D coordinate prediction model...")
        self.model_2d = self._build_model(2, self.feature_dim)
        
        history_2d = self.model_2d.fit(
            train_data['features'],
            train_data['coordinates_2d'],
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(test_data['features'], test_data['coordinates_2d']),
            verbose=1
        )
        
        # Evaluate 2D model
        pred_2d = self.model_2d.predict(test_data['features'])
        mae_2d = mean_absolute_error(test_data['coordinates_2d'], pred_2d)
        results['mae_2d'] = mae_2d
        
        # Train 3D color prediction model
        logger.info("Training 3D color prediction model...")
        self.model_3d = self._build_model(3, self.feature_dim)
        
        history_3d = self.model_3d.fit(
            train_data['features'],
            train_data['colors_3d'],
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(test_data['features'], test_data['colors_3d']),
            verbose=1
        )
        
        # Evaluate 3D model
        pred_3d = self.model_3d.predict(test_data['features'])
        mae_3d = mean_absolute_error(test_data['colors_3d'], pred_3d)
        results['mae_3d'] = mae_3d
        
        logger.info(f"Training completed")
        logger.info(f"2D MAE: {mae_2d:.6f}")
        logger.info(f"3D MAE: {mae_3d:.6f}")
        
        return results
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict 2D coordinates and 3D colors from features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Tuple of (predicted_2d_coordinates, predicted_3d_colors)
        """
        if self.model_2d is None or self.model_3d is None:
            raise ValueError("Models must be trained before prediction")
        
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.feature_dim}, got {features.shape[1]}")
        
        pred_2d = self.model_2d.predict(features)
        pred_3d = self.model_3d.predict(features)
        
        return pred_2d, pred_3d
    
    def save_models(self, output_directory: Path) -> Dict[str, Path]:
        """Save trained models to disk.
        
        Args:
            output_directory: Directory to save models
            
        Returns:
            Dictionary mapping model types to saved file paths
        """
        if self.model_2d is None or self.model_3d is None:
            raise ValueError("Models must be trained before saving")
        
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save 2D model
        model_2d_path = output_directory / "model_2d.keras"
        self.model_2d.save(model_2d_path)
        saved_files['model_2d'] = model_2d_path
        
        # Save 3D model
        model_3d_path = output_directory / "model_3d.keras"
        self.model_3d.save(model_3d_path)
        saved_files['model_3d'] = model_3d_path
        
        # Save configuration
        config_path = output_directory / "network_config.json"
        config_dict = {
            'hidden_layers': self.config.hidden_layers,
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'train_size': self.config.train_size,
            'random_seed': self.config.random_seed,
            'feature_dim': self.feature_dim
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        saved_files['config'] = config_path
        
        logger.info(f"Models saved to {output_directory}")
        return saved_files
    
    def load_models(self, model_directory: Path) -> None:
        """Load trained models from disk.
        
        Args:
            model_directory: Directory containing saved models
        """
        model_directory = Path(model_directory)
        
        # Load configuration
        config_path = model_directory / "network_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.feature_dim = config_dict.get('feature_dim')
        
        # Load models
        model_2d_path = model_directory / "model_2d.keras"
        model_3d_path = model_directory / "model_3d.keras"
        
        if not model_2d_path.exists() or not model_3d_path.exists():
            raise FileNotFoundError(f"Model files not found in {model_directory}")
        
        self.model_2d = keras.models.load_model(model_2d_path)
        self.model_3d = keras.models.load_model(model_3d_path)
        
        logger.info(f"Models loaded from {model_directory}")
    
    def predict_new_samples(self, features: np.ndarray, 
                           filenames: List[str],
                           existing_embeddings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Predict embeddings for new samples and optionally combine with existing ones.
        
        Args:
            features: Feature matrix for new samples
            filenames: Filenames for new samples
            existing_embeddings: Optional existing embeddings to combine with
            
        Returns:
            Dictionary containing combined embeddings
        """
        # Predict coordinates and colors
        pred_2d, pred_3d = self.predict(features)
        
        # Create new sample data
        new_data = []
        for i, filename in enumerate(filenames):
            sample_data = {
                "path": filename,
                "point": pred_2d[i].tolist(),
                "color": pred_3d[i].tolist()
            }
            new_data.append(sample_data)
        
        # Combine with existing data if provided
        if existing_embeddings:
            combined_data = existing_embeddings.get("data", []) + new_data
        else:
            combined_data = new_data
        
        # Create result dictionary
        result = {
            "data": combined_data,
            "new_samples": new_data,
            "n_new_samples": len(new_data),
            "n_total_samples": len(combined_data)
        }
        
        return result