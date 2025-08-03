#!/usr/bin/env python3
"""
Batch processing script for running multiple t-SNE experiments.

A pythonic approach for running parameter sweeps and batch experiments.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import itertools
import subprocess
import json

from audio_embedding.utils import setup_logging


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run batch t-SNE experiments with parameter sweeps",
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
    
    # Parameter sweep arguments (can specify multiple values)
    parser.add_argument(
        '--feature-types', '--featureType',
        nargs='+',
        choices=['mfcc', 'spectrogram'],
        default=['mfcc'],
        help='Feature types to experiment with'
    )
    parser.add_argument(
        '--n-fft-values', '--n_fft',
        nargs='+',
        type=int,
        default=[2048],
        help='FFT sizes to experiment with'
    )
    parser.add_argument(
        '--hop-length-values', '--hop_length',
        nargs='+',
        type=int,
        default=[1024],
        help='Hop lengths to experiment with'
    )
    parser.add_argument(
        '--perplexity-values', '--perplexity',
        nargs='+',
        type=int,
        default=[30],
        help='Perplexity values to experiment with'
    )
    parser.add_argument(
        '--n-mfcc-values', '--mfcc',
        nargs='+',
        type=int,
        default=[13],
        help='Number of MFCC coefficients to experiment with'
    )
    parser.add_argument(
        '--bin-step-values', '--binStep',
        nargs='+',
        type=int,
        default=[10],
        help='Bin step values for spectrograms'
    )
    parser.add_argument(
        '--time-step-values', '--timeStep',
        nargs='+',
        type=int,
        default=[10],
        help='Time step values for spectrograms'
    )
    parser.add_argument(
        '--log-magnitude-values', '--logMagnitude',
        nargs='+',
        type=str,
        choices=['true', 'false'],
        default=['true'],
        help='Log magnitude options for spectrograms'
    )
    
    # Other parameters
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
    parser.add_argument(
        '--initial-dims', '--initial_dims',
        type=int,
        default=30,
        help='Initial dimensionality for t-SNE'
    )
    parser.add_argument(
        '--n-mels', '--n_mels',
        type=int,
        default=128,
        help='Number of mel bands'
    )
    parser.add_argument(
        '--limit', '--lim',
        type=int,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--normalize/--no-normalize', '--normalization',
        default=True,
        help='Enable/disable feature normalization'
    )
    
    # Processing options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands that would be run without executing them'
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


def generate_parameter_combinations(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Generate all parameter combinations for batch processing.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        List of parameter dictionaries
    """
    combinations = []
    
    for feature_type in args.feature_types:
        if feature_type == 'mfcc':
            # MFCC parameter combinations
            for n_fft, hop_length, perplexity, n_mfcc in itertools.product(
                args.n_fft_values,
                args.hop_length_values,
                args.perplexity_values,
                args.n_mfcc_values
            ):
                params = {
                    'feature_type': feature_type,
                    'n_fft': n_fft,
                    'hop_length': hop_length,
                    'perplexity': perplexity,
                    'n_mfcc': n_mfcc,
                    'n_mels': args.n_mels,
                    'sample_rate': args.sample_rate,
                    'duration_threshold': args.duration_threshold,
                    'initial_dims': args.initial_dims,
                    'normalize': args.normalize
                }
                if args.limit:
                    params['limit'] = args.limit
                combinations.append(params)
        
        else:  # spectrogram
            # Spectrogram parameter combinations
            for n_fft, hop_length, perplexity, bin_step, time_step, log_mag in itertools.product(
                args.n_fft_values,
                args.hop_length_values,
                args.perplexity_values,
                args.bin_step_values,
                args.time_step_values,
                args.log_magnitude_values
            ):
                params = {
                    'feature_type': feature_type,
                    'n_fft': n_fft,
                    'hop_length': hop_length,
                    'perplexity': perplexity,
                    'bin_step': bin_step,
                    'time_step': time_step,
                    'log_magnitude': log_mag.lower() == 'true',
                    'sample_rate': args.sample_rate,
                    'duration_threshold': args.duration_threshold,
                    'initial_dims': args.initial_dims,
                    'normalize': args.normalize
                }
                if args.limit:
                    params['limit'] = args.limit
                combinations.append(params)
    
    return combinations


def build_command(params: Dict[str, Any], input_dir: Path, output_dir: Path) -> List[str]:
    """Build command line for run_audio_embedding.py.
    
    Args:
        params: Parameter dictionary
        input_dir: Input directory path
        output_dir: Output directory path
        
    Returns:
        Command as list of strings
    """
    cmd = [
        'python3', 'run_audio_embedding.py',
        '--input-dir', str(input_dir),
        '--output-dir', str(output_dir),
        '--feature-type', params['feature_type'],
        '--n-fft', str(params['n_fft']),
        '--hop-length', str(params['hop_length']),
        '--perplexity', str(params['perplexity']),
        '--sample-rate', str(params['sample_rate']),
        '--duration-threshold', str(params['duration_threshold']),
        '--initial-dims', str(params['initial_dims'])
    ]
    
    if params.get('normalize', True):
        # normalize is default, only add flag if we want to disable it
        pass
    else:
        cmd.append('--no-normalize')
    
    if 'limit' in params:
        cmd.extend(['--limit', str(params['limit'])])
    
    # Feature-specific parameters
    if params['feature_type'] == 'mfcc':
        cmd.extend(['--n-mfcc', str(params['n_mfcc'])])
        cmd.extend(['--n-mels', str(params['n_mels'])])
    else:  # spectrogram
        cmd.extend(['--bin-step', str(params['bin_step'])])
        cmd.extend(['--time-step', str(params['time_step'])])
        if params['log_magnitude']:
            cmd.append('--log-magnitude')
    
    return cmd


def run_experiment(params: Dict[str, Any], input_dir: Path, output_dir: Path, 
                  dry_run: bool = False) -> bool:
    """Run a single experiment with given parameters.
    
    Args:
        params: Parameter dictionary
        input_dir: Input directory path
        output_dir: Output directory path
        dry_run: If True, only print the command without executing
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Build command
    cmd = build_command(params, input_dir, output_dir)
    
    # Create parameter description
    if params['feature_type'] == 'mfcc':
        param_desc = f"mfcc_nfft{params['n_fft']}_perp{params['perplexity']}_nmfcc{params['n_mfcc']}"
    else:
        param_desc = (f"spec_nfft{params['n_fft']}_perp{params['perplexity']}_"
                     f"bin{params['bin_step']}_time{params['time_step']}_log{params['log_magnitude']}")
    
    logger.info(f"Running experiment: {param_desc}")
    
    if dry_run:
        logger.info(f"Command: {' '.join(cmd)}")
        return True
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Experiment completed successfully: {param_desc}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment failed: {param_desc}")
        logger.error(f"Command: {' '.join(cmd)}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in experiment {param_desc}: {e}")
        return False


def main() -> int:
    """Main function."""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Setup logging
        setup_logging(args.log_level, args.log_file)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting batch t-SNE processing")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        
        if not args.input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            return 1
        
        # Generate parameter combinations
        combinations = generate_parameter_combinations(args)
        logger.info(f"Generated {len(combinations)} parameter combinations")
        
        if args.dry_run:
            logger.info("DRY RUN MODE - Commands will be printed but not executed")
        
        # Run experiments
        successful = 0
        failed = 0
        
        for i, params in enumerate(combinations, 1):
            logger.info(f"Running experiment {i}/{len(combinations)}")
            
            success = run_experiment(params, args.input_dir, args.output_dir, args.dry_run)
            
            if success:
                successful += 1
            else:
                failed += 1
        
        # Summary
        logger.info(f"Batch processing completed")
        logger.info(f"Successful experiments: {successful}")
        logger.info(f"Failed experiments: {failed}")
        logger.info(f"Total experiments: {len(combinations)}")
        
        if not args.dry_run:
            # Save experiment summary
            summary = {
                "total_experiments": len(combinations),
                "successful": successful,
                "failed": failed,
                "parameters_tested": {
                    "feature_types": args.feature_types,
                    "n_fft_values": args.n_fft_values,
                    "hop_length_values": args.hop_length_values,
                    "perplexity_values": args.perplexity_values,
                    "n_mfcc_values": args.n_mfcc_values if 'mfcc' in args.feature_types else None,
                    "bin_step_values": args.bin_step_values if 'spectrogram' in args.feature_types else None,
                    "time_step_values": args.time_step_values if 'spectrogram' in args.feature_types else None,
                    "log_magnitude_values": args.log_magnitude_values if 'spectrogram' in args.feature_types else None,
                }
            }
            
            summary_path = args.output_dir / "batch_summary.json"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Batch summary saved to {summary_path}")
        
        return 0 if failed == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Batch processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Batch processing failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())