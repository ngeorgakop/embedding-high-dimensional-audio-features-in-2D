"""Evaluation module for t-SNE mappings."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from .utils import calculate_distance_matrix

logger = logging.getLogger(__name__)


class TSNEEvaluator:
    """Evaluates t-SNE mappings based on semantic groupings."""
    
    # Keywords to identify different sample types
    SAMPLE_KEYWORDS = {
        'snares': 'snare',
        'kicks': 'kick', 
        'basses': 'bass',
        'hats': 'hat',
        'claps': 'clap',
        'toms': 'tom'
    }
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_directory(self, directory: Path, 
                          top_results: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate all t-SNE mappings in a directory.
        
        Args:
            directory: Directory containing points.json files
            top_results: Number of top results to return (None for all)
            
        Returns:
            Dictionary containing evaluation results
        """
        directory = Path(directory)
        results = []
        
        # Find all points.json files recursively
        points_files = list(directory.rglob("points.json"))
        
        # Also look for old format points.txt files
        if not points_files:
            points_files = list(directory.rglob("points.txt"))
        
        if not points_files:
            logger.warning(f"No embedding files found in {directory}")
            return {"results": [], "summary": {}}
        
        logger.info(f"Found {len(points_files)} embedding files to evaluate")
        
        for points_file in points_files:
            try:
                result = self.evaluate_single_mapping(points_file)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {points_file}: {e}")
        
        # Sort results by total mean distance (lower is better)
        results.sort(key=lambda x: x['total_mean_distance'])
        
        # Limit results if specified
        if top_results:
            results = results[:top_results]
        
        # Create summary
        summary = self._create_summary(results)
        
        return {
            "results": results,
            "summary": summary,
            "evaluation_metadata": {
                "total_mappings_evaluated": len(results),
                "keywords_used": list(self.SAMPLE_KEYWORDS.keys())
            }
        }
    
    def evaluate_single_mapping(self, points_file: Path) -> Optional[Dict[str, Any]]:
        """Evaluate a single t-SNE mapping.
        
        Args:
            points_file: Path to points.json or points.txt file
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Load the points data
            with open(points_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                logger.warning(f"Empty data in {points_file}")
                return None
            
            # Group samples by keywords
            grouped_samples = self._group_samples_by_keywords(data)
            
            # Calculate mean distances for each group
            group_distances = {}
            for group_name, samples in grouped_samples.items():
                if len(samples) < 2:
                    logger.debug(f"Skipping group '{group_name}' with < 2 samples")
                    continue
                
                mean_distance = self._calculate_group_mean_distance(samples)
                group_distances[group_name] = mean_distance
            
            if not group_distances:
                logger.warning(f"No valid groups found in {points_file}")
                return None
            
            # Calculate overall mean distance
            total_mean_distance = np.mean(list(group_distances.values()))
            
            # Get parameter information from file path
            parameter_info = self._extract_parameter_info(points_file)
            
            result = {
                "file_path": str(points_file),
                "parameter_info": parameter_info,
                "group_distances": group_distances,
                "total_mean_distance": total_mean_distance,
                "n_samples": len(data),
                "n_valid_groups": len(group_distances)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating {points_file}: {e}")
            return None
    
    def _group_samples_by_keywords(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group samples by keywords in their filenames.
        
        Args:
            data: List of sample data dictionaries
            
        Returns:
            Dictionary mapping group names to lists of samples
        """
        grouped = {group_name: [] for group_name in self.SAMPLE_KEYWORDS.keys()}
        
        for sample in data:
            file_path = sample.get('path', '').lower()
            
            for group_name, keyword in self.SAMPLE_KEYWORDS.items():
                if keyword in file_path:
                    grouped[group_name].append(sample)
        
        # Remove empty groups
        grouped = {name: samples for name, samples in grouped.items() if samples}
        
        logger.debug(f"Sample grouping: {[(name, len(samples)) for name, samples in grouped.items()]}")
        return grouped
    
    def _calculate_group_mean_distance(self, samples: List[Dict[str, Any]]) -> float:
        """Calculate mean pairwise distance within a group.
        
        Args:
            samples: List of samples in the group
            
        Returns:
            Mean pairwise distance
        """
        if len(samples) < 2:
            return 0.0
        
        # Extract coordinates
        coordinates = np.array([sample['point'] for sample in samples])
        
        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(coordinates)
        
        # Get all non-zero distances (excluding diagonal)
        distances = distance_matrix[distance_matrix > 0]
        
        return np.mean(distances)
    
    def _extract_parameter_info(self, points_file: Path) -> str:
        """Extract parameter information from file path.
        
        Args:
            points_file: Path to points file
            
        Returns:
            String describing the parameters
        """
        # Try to extract from parent directory name
        parent_dir = points_file.parent.name
        
        # Remove common prefixes/suffixes
        if parent_dir.startswith(('mfcc_', 'spectrogram_')):
            return parent_dir
        
        return str(points_file.parent)
    
    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics from evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Summary dictionary
        """
        if not results:
            return {}
        
        total_distances = [r['total_mean_distance'] for r in results]
        
        summary = {
            "best_result": {
                "parameter_info": results[0]['parameter_info'],
                "total_mean_distance": results[0]['total_mean_distance']
            },
            "statistics": {
                "mean_distance": {
                    "min": np.min(total_distances),
                    "max": np.max(total_distances), 
                    "mean": np.mean(total_distances),
                    "std": np.std(total_distances)
                },
                "n_results": len(results)
            }
        }
        
        return summary
    
    def save_evaluation_results(self, results: Dict[str, Any], 
                               output_path: Path) -> None:
        """Save evaluation results to JSON file.
        
        Args:
            results: Dictionary containing evaluation results
            output_path: Path to save results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def print_results(self, results: Dict[str, Any], 
                     top_n: Optional[int] = None) -> None:
        """Print evaluation results in a formatted way.
        
        Args:
            results: Dictionary containing evaluation results
            top_n: Number of top results to print (None for all)
        """
        evaluation_results = results.get("results", [])
        
        if not evaluation_results:
            print("No evaluation results found.")
            return
        
        if top_n:
            evaluation_results = evaluation_results[:top_n]
        
        print(f"\n=== t-SNE Evaluation Results (Top {len(evaluation_results)}) ===")
        print()
        
        for i, result in enumerate(evaluation_results, 1):
            print(f"{i}. Parameters: {result['parameter_info']}")
            print(f"   Total Mean Distance: {result['total_mean_distance']:.6f}")
            print(f"   Samples: {result['n_samples']}, Valid Groups: {result['n_valid_groups']}")
            
            if result.get('group_distances'):
                print("   Group Distances:")
                for group, distance in result['group_distances'].items():
                    print(f"     {group}: {distance:.6f}")
            print()
        
        # Print summary if available
        if "summary" in results and results["summary"]:
            summary = results["summary"]
            print("=== Summary Statistics ===")
            if "best_result" in summary:
                best = summary["best_result"]
                print(f"Best Result: {best['parameter_info']} "
                      f"(distance: {best['total_mean_distance']:.6f})")
            
            if "statistics" in summary:
                stats = summary["statistics"]["mean_distance"]
                print(f"Distance Range: {stats['min']:.6f} - {stats['max']:.6f}")
                print(f"Mean ± Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print()