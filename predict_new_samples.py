#!/usr/bin/env python3
"""
Prediction script for adding new samples to existing t-SNE embeddings.

"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np

from audio_embedding import (
    AudioEmbeddingConfig,
    AudioProcessor,
    FeatureExtractor,
    TSNEMapper
)
from audio_embedding.neural_network import TSNERegressor, NetworkConfig
from audio_embedding.utils import setup_logging


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Predict t-SNE coordinates for new audio samples using trained neural network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--samples-dir', '--samples_dir',
        type=Path,
        required=True,
        help='Directory containing new audio samples'
    )
    parser.add_argument(
        '--models-dir', '--points_dir',
        type=Path,
        required=True,
        help='Directory containing trained models and original embeddings'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        help='Output file for combined embeddings (default: new_embeddings.json)'
    )
    
    # Audio processing parameters
    parser.add_argument(
        '--sample-rate', '--sampleRate',
        type=int,
        default=44100,
        help='Audio sample rate'
    )
    parser.add_argument(
        '--duration-threshold', '--duration_threshold',
        type=float,
        default=10.0,
        help='Duration threshold in seconds'
    )
    
    # Processing options
    parser.add_argument(
        '--config-file',
        type=Path,
        help='JSON configuration file (will try to load from models-dir if not specified)'
    )
    parser.add_argument(
        '--n-processes',
        type=int,
        help='Number of parallel processes for audio loading'
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


def extract_config_from_path(models_dir: Path) -> AudioEmbeddingConfig:
    """Extract configuration parameters from model directory path.
    
    Args:
        models_dir: Directory containing models
        
    Returns:
        Inferred audio embedding configuration
    """
    logger = logging.getLogger(__name__)
    
    # Try to load saved configuration
    config_path = models_dir / "config.json"
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        return AudioEmbeddingConfig.from_json(config_path)
    
    # Fall back to inferring from directory name
    dir_name = models_dir.name
    logger.warning(f"No config file found, inferring parameters from directory name: {dir_name}")
    
    config = AudioEmbeddingConfig()
    
    # Parse directory name (e.g., "mfcc_2048_30_13" or "spectrogram_2048_30_10_10_True")
    parts = dir_name.split('_')
    
    if len(parts) >= 2:
        config.features.feature_type = parts[0]
        
        try:
            if len(parts) >= 3:
                config.audio.n_fft = int(parts[1])
            if len(parts) >= 4:
                config.tsne.perplexity = int(parts[2])
            
            if config.features.feature_type == "mfcc" and len(parts) >= 4:
                config.features.n_mfcc = int(parts[3])
            elif config.features.feature_type == "spectrogram" and len(parts) >= 6:
                config.features.bin_step = int(parts[3])
                config.features.time_step = int(parts[4])
                config.features.use_log_magnitude = parts[5].lower() == 'true'
        
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse all parameters from directory name: {e}")
    
    # Set reasonable defaults for missing parameters
    config.audio.hop_length = config.audio.n_fft // 2
    
    logger.info(f"Inferred configuration: feature_type={config.features.feature_type}, "
               f"n_fft={config.audio.n_fft}, perplexity={config.tsne.perplexity}")
    
    return config


def main() -> int:
    """Main function."""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Setup logging
        setup_logging(args.log_level, args.log_file)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting new sample prediction")
        logger.info(f"Samples directory: {args.samples_dir}")
        logger.info(f"Models directory: {args.models_dir}")
        
        # Validate input directories
        if not args.samples_dir.exists():
            logger.error(f"Samples directory not found: {args.samples_dir}")
            return 1
        
        if not args.models_dir.exists():
            logger.error(f"Models directory not found: {args.models_dir}")
            return 1
        
        # Load or infer configuration
        if args.config_file and args.config_file.exists():
            config = AudioEmbeddingConfig.from_json(args.config_file)
        else:
            config = extract_config_from_path(args.models_dir)
        
        # Override audio parameters from command line
        config.audio.sample_rate = args.sample_rate
        config.audio.duration_threshold = args.duration_threshold
        config.n_processes = args.n_processes
        
        # Initialize components
        audio_processor = AudioProcessor(config.audio)
        feature_extractor = FeatureExtractor(config.audio, config.features)
        
        # Find and load audio files
        logger.info("Finding new audio samples...")
        audio_files = audio_processor.find_audio_files(args.samples_dir)
        
        if not audio_files:
            logger.error(f"No audio files found in {args.samples_dir}")
            return 1
        
        logger.info(f"Loading {len(audio_files)} audio files...")
        audio_data = audio_processor.load_audio_files_parallel(
            audio_files,
            config.n_processes
        )
        
        if not audio_data:
            logger.error("No audio files were successfully loaded")
            return 1
        
        filenames = [item[0] for item in audio_data]
        audio_signals = [item[1] for item in audio_data]
        
        logger.info(f"Successfully loaded {len(audio_signals)} audio files")
        
        # Extract features
        logger.info("Extracting features from new samples...")
        features = []
        
        for i, audio_signal in enumerate(audio_signals):
            try:
                feature_vector = feature_extractor.extract_features(audio_signal)
                features.append(feature_vector)
            except Exception as e:
                logger.error(f"Error extracting features from {filenames[i]}: {e}")
                continue
        
        if not features:
            logger.error("No features were successfully extracted")
            return 1
        
        features = np.array(features)
        logger.info(f"Feature extraction completed. Shape: {features.shape}")
        
        # Load trained models
        logger.info("Loading trained neural network models...")
        regressor = TSNERegressor(NetworkConfig())  # Config will be loaded with models
        regressor.load_models(args.models_dir)
        
        # Load existing embeddings
        logger.info("Loading existing embeddings...")
        tsne_mapper = TSNEMapper(None)
        existing_embeddings = tsne_mapper.load_embeddings(args.models_dir)
        
        # Predict coordinates for new samples
        logger.info("Predicting coordinates for new samples...")
        combined_embeddings = regressor.predict_new_samples(
            features,
            filenames,
            existing_embeddings
        )
        
        # Set output file
        if args.output_file:
            output_file = args.output_file
        else:
            output_file = args.models_dir / "new_embeddings.json"
        
        # Save combined embeddings
        logger.info(f"Saving combined embeddings to {output_file}...")
        
        import json
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(combined_embeddings["data"], f, indent=2)
        
        # Print summary
        print("\n=== Prediction Results ===")
        print(f"New samples processed: {combined_embeddings['n_new_samples']}")
        print(f"Total samples in combined embedding: {combined_embeddings['n_total_samples']}")
        print(f"Combined embeddings saved to: {output_file}")
        
        # Also save just the new samples
        new_samples_file = output_file.parent / f"new_samples_{output_file.name}"
        with open(new_samples_file, 'w') as f:
            json.dump(combined_embeddings["new_samples"], f, indent=2)
        
        print(f"New samples only saved to: {new_samples_file}")
        
        logger.info("Prediction completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Prediction interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())