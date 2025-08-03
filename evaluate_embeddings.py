#!/usr/bin/env python3
"""
Evaluation script for t-SNE embeddings.

A pythonic, modular approach using proper classes and error handling.
"""

import argparse
import sys
import logging
from pathlib import Path

from audio_embedding import TSNEEvaluator
from audio_embedding.utils import setup_logging


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate t-SNE embeddings based on semantic groupings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--directory', '--dir',
        type=Path,
        required=True,
        help='Directory containing embedding files to evaluate'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=None,
        help='Number of top results to show (default: all)'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        help='Output file for evaluation results (JSON format)'
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


def main() -> int:
    """Main function."""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Setup logging
        setup_logging(args.log_level, args.log_file)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting t-SNE evaluation")
        logger.info(f"Evaluation directory: {args.directory}")
        
        if not args.directory.exists():
            logger.error(f"Directory not found: {args.directory}")
            return 1
        
        # Initialize evaluator
        evaluator = TSNEEvaluator()
        
        # Perform evaluation
        logger.info("Evaluating embeddings...")
        results = evaluator.evaluate_directory(args.directory, args.top)
        
        if not results["results"]:
            logger.warning("No valid evaluation results found")
            return 0
        
        # Print results to console
        evaluator.print_results(results, args.top)
        
        # Save results to file if specified
        if args.output_file:
            evaluator.save_evaluation_results(results, args.output_file)
            
            # Also save best scores in legacy format
            best_scores_path = args.output_file.parent / "best_scores.json"
            best_scores = []
            for result in results["results"]:
                best_scores.append({
                    "total": result["total_mean_distance"],
                    "parameters": result["parameter_info"]
                })
            
            import json
            with open(best_scores_path, 'w') as f:
                json.dump(best_scores, f, indent=2)
            
            logger.info(f"Results saved to {args.output_file}")
            logger.info(f"Best scores saved to {best_scores_path}")
        
        logger.info("Evaluation completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())