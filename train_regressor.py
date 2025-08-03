#!/usr/bin/env python3
"""
Neural network training script for learning t-SNE mappings.

A pythonic, modular approach using proper classes and configuration management.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from audio_embedding import AudioEmbeddingConfig, TSNEMapper
from audio_embedding.neural_network import TSNERegressor, NetworkConfig
from audio_embedding.utils import setup_logging


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train neural network to learn t-SNE mappings from features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--embeddings-dir', '--dir',
        type=Path,
        required=True,
        help='Directory containing t-SNE embeddings and features'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Directory to save trained models (default: same as embeddings-dir)'
    )
    
    # Neural network architecture
    parser.add_argument(
        '--hidden-layers', '--arch',
        nargs='+',
        type=int,
        default=[20, 50, 100, 50, 20],
        help='Hidden layer sizes'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--train-size', '--trainSize',
        type=float,
        default=0.8,
        help='Fraction of data to use for training (rest for testing)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    # Output options
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save prediction results on test set'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Log file path'
    )
    
    return parser


def load_embedding_data(embeddings_dir: Path) -> dict:
    """Load features and embeddings from directory.
    
    Args:
        embeddings_dir: Directory containing embedding files
        
    Returns:
        Dictionary containing loaded data
    """
    logger = logging.getLogger(__name__)
    
    # Load features
    features_path = embeddings_dir / "features.npy"
    if not features_path.exists():
        # Try old format
        features_path = embeddings_dir / "highDfeats.npy"
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found in {embeddings_dir}")
    
    features = np.load(features_path)
    logger.info(f"Loaded features: {features.shape}")
    
    # Load embeddings using TSNEMapper
    tsne_mapper = TSNEMapper(None)  # Config not needed for loading
    embeddings = tsne_mapper.load_embeddings(embeddings_dir)
    
    coordinates_2d = embeddings["coordinates_2d"]
    colors_3d = embeddings["colors_3d"]
    filenames = [item["path"] for item in embeddings["data"]]
    
    logger.info(f"Loaded embeddings: 2D shape {coordinates_2d.shape}, 3D shape {colors_3d.shape}")
    logger.info(f"Number of samples: {len(filenames)}")
    
    # Validate data consistency
    if len(features) != len(coordinates_2d) or len(features) != len(colors_3d):
        raise ValueError("Mismatch in number of samples between features and embeddings")
    
    return {
        'features': features,
        'coordinates_2d': coordinates_2d,
        'colors_3d': colors_3d,
        'filenames': filenames
    }


def main() -> int:
    """Main function."""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Setup logging
        setup_logging(args.log_level, args.log_file)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting neural network training")
        logger.info(f"Embeddings directory: {args.embeddings_dir}")
        
        if not args.embeddings_dir.exists():
            logger.error(f"Embeddings directory not found: {args.embeddings_dir}")
            return 1
        
        # Set output directory
        output_dir = args.output_dir or args.embeddings_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info("Loading embedding data...")
        data = load_embedding_data(args.embeddings_dir)
        
        # Create network configuration
        network_config = NetworkConfig(
            hidden_layers=args.hidden_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_size=args.train_size,
            random_seed=args.random_seed
        )
        
        logger.info(f"Network architecture: {network_config.hidden_layers}")
        logger.info(f"Training parameters: epochs={network_config.epochs}, "
                   f"batch_size={network_config.batch_size}, "
                   f"learning_rate={network_config.learning_rate}")
        
        # Initialize regressor
        regressor = TSNERegressor(network_config)
        
        # Prepare training data
        logger.info("Preparing training data...")
        dataset = regressor.prepare_training_data(
            data['features'],
            data['coordinates_2d'],
            data['colors_3d'],
            data['filenames']
        )
        
        # Train models
        logger.info("Training neural networks...")
        training_results = regressor.train(dataset)
        
        # Save models
        logger.info("Saving trained models...")
        saved_files = regressor.save_models(output_dir)
        
        # Print results
        print("\n=== Training Results ===")
        print(f"2D Coordinate Prediction MAE: {training_results['mae_2d']:.6f}")
        print(f"3D Color Prediction MAE: {training_results['mae_3d']:.6f}")
        print(f"\nModels saved to: {output_dir}")
        for model_type, path in saved_files.items():
            print(f"  {model_type}: {path}")
        
        # Save predictions if requested
        if args.save_predictions:
            logger.info("Generating and saving predictions...")
            
            test_features = dataset['test']['features']
            test_filenames = dataset['test']['filenames']
            test_coords_2d = dataset['test']['coordinates_2d']
            test_colors_3d = dataset['test']['colors_3d']
            
            # Get predictions
            pred_2d, pred_3d = regressor.predict(test_features)
            
            # Create prediction data
            prediction_data = []
            for i, filename in enumerate(test_filenames):
                prediction_data.append({
                    "path": filename,
                    "true_point": test_coords_2d[i].tolist(),
                    "predicted_point": pred_2d[i].tolist(),
                    "true_color": test_colors_3d[i].tolist(),
                    "predicted_color": pred_3d[i].tolist()
                })
            
            # Save predictions
            import json
            predictions_path = output_dir / "test_predictions.json"
            with open(predictions_path, 'w') as f:
                json.dump(prediction_data, f, indent=2)
            
            logger.info(f"Test predictions saved to {predictions_path}")
        
        logger.info("Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())