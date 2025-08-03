"""Configuration management for audio embedding project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union
import json


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 44100
    duration_threshold: float = 10.0
    n_fft: int = 2048
    hop_length: int = 1024
    window_function: str = "hann"


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    feature_type: str = "mfcc"  # "mfcc" or "spectrogram"
    normalize_features: bool = True
    
    # MFCC specific
    n_mfcc: int = 13
    n_mels: int = 128
    
    # Spectrogram specific
    use_log_magnitude: bool = True
    bin_step: int = 10
    time_step: int = 10


@dataclass
class TSNEConfig:
    """t-SNE configuration using scikit-learn implementation."""
    perplexity: int = 30
    initial_dims: int = 30
    n_dims_2d: int = 2
    n_dims_3d: int = 3
    max_iterations: int = 1000
    theta: float = 0.5  # Controls speed/accuracy tradeoff in Barnes-Hut (angle parameter)
    use_pca: bool = True
    random_seed: Optional[int] = None
    learning_rate: Union[str, float] = "auto"  # Can be 'auto' or a float (sklearn parameter)
    early_exaggeration: float = 12.0  # Controls tight clustering in early optimization


@dataclass
class AudioEmbeddingConfig:
    """Main configuration class for the audio embedding project."""
    
    # Data paths
    input_directory: Optional[Path] = None
    output_directory: Optional[Path] = None
    
    # Processing options
    file_limit: Optional[int] = None
    n_processes: Optional[int] = None
    
    # Sub-configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    tsne: TSNEConfig = field(default_factory=TSNEConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.input_directory is not None:
            self.input_directory = Path(self.input_directory)
        if self.output_directory is not None:
            self.output_directory = Path(self.output_directory)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AudioEmbeddingConfig':
        """Create configuration from dictionary."""
        audio_config = AudioConfig(**config_dict.get('audio', {}))
        feature_config = FeatureConfig(**config_dict.get('features', {}))
        tsne_config = TSNEConfig(**config_dict.get('tsne', {}))
        
        # Remove sub-config keys from main dict
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['audio', 'features', 'tsne']}
        
        return cls(
            audio=audio_config,
            features=feature_config,
            tsne=tsne_config,
            **main_config
        )
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'AudioEmbeddingConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'input_directory': str(self.input_directory) if self.input_directory else None,
            'output_directory': str(self.output_directory) if self.output_directory else None,
            'file_limit': self.file_limit,
            'n_processes': self.n_processes,
            'audio': {
                'sample_rate': self.audio.sample_rate,
                'duration_threshold': self.audio.duration_threshold,
                'n_fft': self.audio.n_fft,
                'hop_length': self.audio.hop_length,
                'window_function': self.audio.window_function,
            },
            'features': {
                'feature_type': self.features.feature_type,
                'normalize_features': self.features.normalize_features,
                'n_mfcc': self.features.n_mfcc,
                'n_mels': self.features.n_mels,
                'use_log_magnitude': self.features.use_log_magnitude,
                'bin_step': self.features.bin_step,
                'time_step': self.features.time_step,
            },
            'tsne': {
                'perplexity': self.tsne.perplexity,
                'initial_dims': self.tsne.initial_dims,
                'n_dims_2d': self.tsne.n_dims_2d,
                'n_dims_3d': self.tsne.n_dims_3d,
                'max_iterations': self.tsne.max_iterations,
                'theta': self.tsne.theta,
                'use_pca': self.tsne.use_pca,
                'random_seed': self.tsne.random_seed,
                'learning_rate': self.tsne.learning_rate,
                'early_exaggeration': self.tsne.early_exaggeration,
            }
        }
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_output_subdirectory(self) -> Path:
        """Generate output subdirectory name based on configuration."""
        if not self.output_directory:
            raise ValueError("Output directory not specified")
            
        base_dir = self.output_directory / self.features.feature_type
        
        if self.features.feature_type == "mfcc":
            subdir_name = f"{self.features.feature_type}_{self.audio.n_fft}_{self.tsne.perplexity}_{self.features.n_mfcc}"
        else:  # spectrogram
            subdir_name = (f"{self.features.feature_type}_{self.audio.n_fft}_{self.tsne.perplexity}_"
                          f"{self.features.bin_step}_{self.features.time_step}_{self.features.use_log_magnitude}")
        
        return base_dir / subdir_name