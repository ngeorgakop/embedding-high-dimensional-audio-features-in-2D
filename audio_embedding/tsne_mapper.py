"""t-SNE mapping module."""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from .config import TSNEConfig
from .utils import normalize_coordinates

logger = logging.getLogger(__name__)


class TSNEMapper:
    """Handles t-SNE dimensionality reduction and coordinate mapping."""
    
    def __init__(self, config: TSNEConfig):
        """Initialize t-SNE mapper with configuration.
        
        Args:
            config: t-SNE configuration
        """
        self.config = config
    
    def fit_transform(self, features: np.ndarray, 
                     n_dims: int = 2) -> np.ndarray:
        """Apply t-SNE to reduce dimensionality of features.
        
        Args:
            features: High-dimensional feature matrix (n_samples, n_features)
            n_dims: Target dimensionality (2 or 3)
            
        Returns:
            Low-dimensional embeddings (n_samples, n_dims)
        """
        if features.shape[0] == 0:
            raise ValueError("Cannot perform t-SNE on empty feature matrix")
        
        if features.shape[1] == 0:
            raise ValueError("Feature matrix has zero dimensions")
        
        logger.info(f"Performing t-SNE reduction from {features.shape[1]}D to {n_dims}D "
                   f"for {features.shape[0]} samples")
        logger.info(f"t-SNE parameters: perplexity={self.config.perplexity}, "
                   f"initial_dims={self.config.initial_dims}")
        
        try:
            # Prepare the data
            data = features.copy()
            
            # Apply PCA preprocessing if requested and beneficial
            if self.config.use_pca and features.shape[1] > self.config.initial_dims:
                logger.info(f"Applying PCA preprocessing to reduce from {features.shape[1]}D "
                           f"to {self.config.initial_dims}D")
                pca = PCA(n_components=self.config.initial_dims, random_state=self.config.random_seed)
                data = pca.fit_transform(features)
                logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
            
            # Create t-SNE instance
            tsne = TSNE(
                n_components=n_dims,
                perplexity=self.config.perplexity,
                n_iter=self.config.max_iterations,
                random_state=self.config.random_seed,
                method='barnes_hut',  # Use Barnes-Hut approximation for speed
                angle=self.config.theta,  # Controls speed/accuracy tradeoff
                learning_rate=self.config.learning_rate,
                early_exaggeration=self.config.early_exaggeration,
                verbose=1 if logger.isEnabledFor(logging.INFO) else 0,
                n_jobs=-1  # Use all available CPU cores
            )
            
            # Fit and transform the data
            embeddings = tsne.fit_transform(data)
            
            logger.info("t-SNE computation completed successfully")
            logger.info(f"Final KL divergence: {tsne.kl_divergence_:.6f}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error during t-SNE computation: {e}")
            raise
    
    def create_embeddings(self, features: np.ndarray, 
                         filenames: List[str]) -> Dict[str, Any]:
        """Create 2D and 3D t-SNE embeddings with normalized coordinates.
        
        Args:
            features: High-dimensional feature matrix
            filenames: List of corresponding filenames
            
        Returns:
            Dictionary containing 2D coordinates, 3D colors, and metadata
        """
        if len(features) != len(filenames):
            raise ValueError("Number of features must match number of filenames")
        
        # Generate 2D embeddings for coordinates
        logger.info("Generating 2D t-SNE embeddings for visualization coordinates...")
        embeddings_2d = self.fit_transform(features, n_dims=2)
        coordinates_2d = normalize_coordinates(embeddings_2d)
        
        # Generate 3D embeddings for colors
        logger.info("Generating 3D t-SNE embeddings for color mapping...")
        embeddings_3d = self.fit_transform(features, n_dims=3)
        colors_3d = normalize_coordinates(embeddings_3d)
        
        # Create data structure for visualization
        visualization_data = []
        for i, filename in enumerate(filenames):
            point_data = {
                "path": filename,
                "point": coordinates_2d[i].tolist(),
                "color": colors_3d[i].tolist()
            }
            visualization_data.append(point_data)
        
        return {
            "data": visualization_data,
            "coordinates_2d": coordinates_2d,
            "colors_3d": colors_3d,
            "metadata": {
                "n_samples": len(filenames),
                "feature_dim": features.shape[1],
                "tsne_config": {
                    "perplexity": self.config.perplexity,
                    "initial_dims": self.config.initial_dims,
                    "max_iterations": self.config.max_iterations,
                    "theta": self.config.theta,
                    "use_pca": self.config.use_pca,
                    "random_seed": self.config.random_seed
                }
            }
        }
    
    def save_embeddings(self, embeddings_dict: Dict[str, Any], 
                       output_directory: Path) -> Dict[str, Path]:
        """Save t-SNE embeddings to various file formats.
        
        Args:
            embeddings_dict: Dictionary containing embeddings and metadata
            output_directory: Directory to save files
            
        Returns:
            Dictionary mapping file types to saved file paths
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save visualization data as JSON
        json_path = output_directory / "points.json"
        with open(json_path, 'w') as f:
            json.dump(embeddings_dict["data"], f, indent=2)
        saved_files["json"] = json_path
        
        # Save coordinates as TSV files (for compatibility with original format)
        tsv_2d_path = output_directory / f"{self.config.initial_dims}.{self.config.perplexity}.2d.tsv"
        np.savetxt(tsv_2d_path, embeddings_dict["coordinates_2d"], 
                  fmt='%.5f', delimiter='\t')
        saved_files["tsv_2d"] = tsv_2d_path
        
        tsv_3d_path = output_directory / f"{self.config.initial_dims}.{self.config.perplexity}.3d.tsv"
        np.savetxt(tsv_3d_path, embeddings_dict["colors_3d"], 
                  fmt='%.5f', delimiter='\t')
        saved_files["tsv_3d"] = tsv_3d_path
        
        # Save filenames
        filenames = [item["path"] for item in embeddings_dict["data"]]
        filenames_path = output_directory / "filenames.txt"
        with open(filenames_path, 'w') as f:
            for filename in filenames:
                f.write(f"{filename}\n")
        saved_files["filenames"] = filenames_path
        
        # Save metadata
        metadata_path = output_directory / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(embeddings_dict["metadata"], f, indent=2)
        saved_files["metadata"] = metadata_path
        
        logger.info(f"Saved embeddings to {output_directory}")
        logger.info(f"Files saved: {list(saved_files.keys())}")
        
        return saved_files
    
    def load_embeddings(self, input_directory: Path) -> Dict[str, Any]:
        """Load previously saved t-SNE embeddings.
        
        Args:
            input_directory: Directory containing saved embeddings
            
        Returns:
            Dictionary containing loaded embeddings and metadata
        """
        input_directory = Path(input_directory)
        
        # Load JSON data
        json_path = input_directory / "points.json"
        if not json_path.exists():
            # Try old format
            json_path = input_directory / "points.txt"
        
        if not json_path.exists():
            raise FileNotFoundError(f"No embeddings file found in {input_directory}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract coordinates and colors
        coordinates_2d = np.array([item["point"] for item in data])
        colors_3d = np.array([item["color"] for item in data])
        
        # Load metadata if available
        metadata_path = input_directory / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return {
            "data": data,
            "coordinates_2d": coordinates_2d,
            "colors_3d": colors_3d,
            "metadata": metadata
        }
    
    def validate_perplexity(self, n_samples: int) -> int:
        """Validate and adjust perplexity based on number of samples.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Validated perplexity value
        """
        # Perplexity should be less than number of samples
        # A good rule of thumb is perplexity < n_samples / 3
        max_perplexity = max(1, n_samples // 3)
        
        if self.config.perplexity > max_perplexity:
            logger.warning(f"Perplexity {self.config.perplexity} is too high for {n_samples} samples. "
                          f"Adjusting to {max_perplexity}")
            return max_perplexity
        
        return self.config.perplexity