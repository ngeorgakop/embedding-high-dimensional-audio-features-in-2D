#!/usr/bin/env python3
"""
Main script for audio embedding and t-SNE visualization.

A pythonic, modular approach using proper configuration management and error handling.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from audio_embedding import (
    AudioEmbeddingConfig,
    AudioProcessor,
    FeatureExtractor,
    TSNEMapper
)
from audio_embedding.utils import setup_logging, create_directory_if_not_exists


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract audio features and create t-SNE embeddings for visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--input-dir', '--data_dir',
        type=Path,
        required=True,
        help='Directory containing input audio files'
    )
    parser.add_argument(
        '--output-dir', '--out_dir',
        type=Path,
        required=True,
        help='Directory for output files'
    )
    
    # Audio processing arguments
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
        help='Duration threshold in seconds (crop or pad)'
    )
    parser.add_argument(
        '--n-fft', '--n_fft',
        type=int,
        default=2048,
        help='FFT window size'
    )
    parser.add_argument(
        '--hop-length', '--hop_length',
        type=int,
        default=1024,
        help='STFT hop length'
    )
    
    # Feature extraction arguments
    parser.add_argument(
        '--feature-type', '--featureType',
        choices=['mfcc', 'spectrogram'],
        default='mfcc',
        help='Type of features to extract'
    )
    parser.add_argument(
        '--n-mfcc', '--n_mfcc',
        type=int,
        default=13,
        help='Number of MFCC coefficients'
    )
    parser.add_argument(
        '--n-mels', '--n_mels',
        type=int,
        default=128,
        help='Number of mel bands'
    )
    parser.add_argument(
        '--log-magnitude', '--logMagnitude',
        action='store_true',
        help='Use log magnitude for spectrograms'
    )
    parser.add_argument(
        '--bin-step', '--binStep',
        type=int,
        default=10,
        help='Frequency bin step for spectrograms'
    )
    parser.add_argument(
        '--time-step', '--timeStep',
        type=int,
        default=10,
        help='Time step for spectrograms'
    )
    parser.add_argument(
        '--no-normalize', '--normalization',
        action='store_false',
        dest='normalize',
        help='Disable feature normalization'
    )
    
    # t-SNE arguments
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity parameter'
    )
    parser.add_argument(
        '--initial-dims', '--initial_dims',
        type=int,
        default=30,
        help='Initial dimensionality for t-SNE'
    )
    
    # Processing arguments
    parser.add_argument(
        '--limit', '--lim',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--n-processes',
        type=int,
        default=None,
        help='Number of parallel processes'
    )
    
    # Other arguments
    parser.add_argument(
        '--config-file',
        type=Path,
        help='JSON configuration file'
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


def parse_arguments() -> AudioEmbeddingConfig:
    """Parse command line arguments and create configuration."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging first
    setup_logging(args.log_level, args.log_file)
    
    # Load config from file if specified
    if args.config_file:
        if not args.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {args.config_file}")
        config = AudioEmbeddingConfig.from_json(args.config_file)
        logging.info(f"Loaded configuration from {args.config_file}")
    else:
        config = AudioEmbeddingConfig()
    
    # Override with command line arguments
    config.input_directory = args.input_dir
    config.output_directory = args.output_dir
    config.file_limit = args.limit
    config.n_processes = args.n_processes
    
    # Audio configuration
    config.audio.sample_rate = args.sample_rate
    config.audio.duration_threshold = args.duration_threshold
    config.audio.n_fft = args.n_fft
    config.audio.hop_length = args.hop_length
    
    # Feature configuration
    config.features.feature_type = args.feature_type
    config.features.normalize_features = args.normalize
    config.features.n_mfcc = args.n_mfcc
    config.features.n_mels = args.n_mels
    config.features.use_log_magnitude = args.log_magnitude
    config.features.bin_step = args.bin_step
    config.features.time_step = args.time_step
    
    # t-SNE configuration
    config.tsne.perplexity = args.perplexity
    config.tsne.initial_dims = args.initial_dims
    
    return config


def main() -> int:
    """Main function."""
    try:
        # Parse arguments and create configuration
        config = parse_arguments()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting audio embedding pipeline")
        logger.info(f"Input directory: {config.input_directory}")
        logger.info(f"Output directory: {config.output_directory}")
        logger.info(f"Feature type: {config.features.feature_type}")
        
        # Initialize components
        audio_processor = AudioProcessor(config.audio)
        feature_extractor = FeatureExtractor(config.audio, config.features)
        tsne_mapper = TSNEMapper(config.tsne)
        
        # Find audio files
        logger.info("Finding audio files...")
        audio_files = audio_processor.find_audio_files(config.input_directory)
        
        if not audio_files:
            logger.error(f"No audio files found in {config.input_directory}")
            return 1
        
        # Apply file limit if specified
        if config.file_limit:
            audio_files = audio_files[:config.file_limit]
            logger.info(f"Limited to {len(audio_files)} files")
        
        # Load audio files
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
        logger.info("Extracting features...")
        features = []
        
        for i, audio_signal in enumerate(audio_signals):
            try:
                feature_vector = feature_extractor.extract_features(audio_signal)
                features.append(feature_vector)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(audio_signals)} files")
                    
            except Exception as e:
                logger.error(f"Error extracting features from {filenames[i]}: {e}")
                continue
        
        if not features:
            logger.error("No features were successfully extracted")
            return 1
        
        features = np.array(features)
        logger.info(f"Feature extraction completed. Shape: {features.shape}")
        
        # Create output directory
        output_subdir = config.get_output_subdirectory()
        create_directory_if_not_exists(output_subdir)
        
        # Save features
        features_path = output_subdir / "features.npy"
        np.save(features_path, features)
        logger.info(f"Features saved to {features_path}")
        
        # Validate perplexity
        validated_perplexity = tsne_mapper.validate_perplexity(len(features))
        if validated_perplexity != config.tsne.perplexity:
            config.tsne.perplexity = validated_perplexity
        
        # Create t-SNE embeddings
        logger.info("Creating t-SNE embeddings...")
        embeddings = tsne_mapper.create_embeddings(features, filenames)
        
        # Save embeddings
        saved_files = tsne_mapper.save_embeddings(embeddings, output_subdir)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved in: {output_subdir}")
        logger.info(f"Main visualization file: {saved_files['json']}")
        
        # Save configuration used
        config_path = output_subdir / "config.json"
        config.to_json(config_path)
        logger.info(f"Configuration saved to {config_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())