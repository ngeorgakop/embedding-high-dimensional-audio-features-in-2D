"""Utility functions for audio embedding project."""

import numpy as np
from pathlib import Path
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


def normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Normalize coordinates to [0, 1] range.
    
    Args:
        coordinates: Array of coordinates to normalize
        
    Returns:
        Normalized coordinates
    """
    if coordinates.size == 0:
        return coordinates
    
    # Handle single dimension case
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(-1, 1)
    
    # Normalize each dimension independently
    normalized = np.zeros_like(coordinates)
    
    for dim in range(coordinates.shape[1]):
        col = coordinates[:, dim]
        min_val = np.min(col)
        max_val = np.max(col)
        
        # Avoid division by zero
        if max_val - min_val == 0:
            normalized[:, dim] = 0.5  # Center all points if no variation
        else:
            normalized[:, dim] = (col - min_val) / (max_val - min_val)
    
    return normalized


def create_directory_if_not_exists(directory: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
        
    Returns:
        Path object of the created directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_file_extension(file_path: Union[str, Path], 
                           valid_extensions: List[str]) -> bool:
    """Validate if file has one of the valid extensions.
    
    Args:
        file_path: Path to file
        valid_extensions: List of valid extensions (with or without dots)
        
    Returns:
        True if file has valid extension, False otherwise
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()
    
    # Normalize extensions to include dots
    normalized_extensions = []
    for ext in valid_extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized_extensions.append(ext.lower())
    
    return file_extension in normalized_extensions


def safe_float_conversion(value: str, default: float = 0.0) -> float:
    """Safely convert string to float with default fallback.
    
    Args:
        value: String value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{value}' to float, using default {default}")
        return default


def safe_int_conversion(value: str, default: int = 0) -> int:
    """Safely convert string to int with default fallback.
    
    Args:
        value: String value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted int value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{value}' to int, using default {default}")
        return default


def setup_logging(level: str = "INFO", log_file: Union[str, Path] = None) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging setup complete. Level: {level}")


def calculate_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """Calculate pairwise Euclidean distance matrix.
    
    Args:
        coordinates: Array of coordinates (n_samples, n_dims)
        
    Returns:
        Distance matrix (n_samples, n_samples)
    """
    n_samples = coordinates.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    
    return distance_matrix


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return 0.0
    
    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return size_mb