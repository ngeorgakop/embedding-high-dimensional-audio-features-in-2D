"""
Audio Embedding and t-SNE Visualization Package.

This package provides tools for extracting features from audio files,
creating t-SNE embeddings, and visualizing audio similarity in 2D/3D space.
"""

__version__ = "1.0.0"
__author__ = ""

from .config import AudioEmbeddingConfig
from .feature_extraction import FeatureExtractor
from .audio_processor import AudioProcessor
from .tsne_mapper import TSNEMapper
from .evaluator import TSNEEvaluator

__all__ = [
    "AudioEmbeddingConfig",
    "FeatureExtractor", 
    "AudioProcessor",
    "TSNEMapper",
    "TSNEEvaluator",
]