"""Audio processing module."""

import librosa
import numpy as np
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Union
import logging
from multiprocessing import Pool
from functools import partial
from .config import AudioConfig

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio file loading and preprocessing."""
    
    SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.aac'}
    
    def __init__(self, config: AudioConfig):
        """Initialize audio processor with configuration.
        
        Args:
            config: Audio processing configuration
        """
        self.config = config
    
    def find_audio_files(self, directory: Union[str, Path], 
                        extensions: Optional[List[str]] = None) -> List[Path]:
        """Find all audio files in directory recursively.
        
        Args:
            directory: Directory to search for audio files
            extensions: List of file extensions to include (default: all supported)
            
        Returns:
            List of audio file paths
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if extensions is None:
            extensions = self.SUPPORTED_EXTENSIONS
        else:
            extensions = {ext.lower() for ext in extensions}
        
        audio_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                audio_files.append(file_path)
        
        logger.info(f"Found {len(audio_files)} audio files in {directory}")
        return audio_files
    
    def load_audio_file(self, file_path: Union[str, Path]) -> Tuple[str, np.ndarray]:
        """Load and preprocess a single audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (filename, preprocessed_audio_signal)
        """
        file_path = Path(file_path)
        
        try:
            # Load audio file
            audio_signal, sample_rate = librosa.load(
                str(file_path),
                sr=self.config.sample_rate,
                duration=self.config.duration_threshold
            )
            
            # Convert to mono if not already
            audio_signal = librosa.to_mono(audio_signal)
            
            # Get actual duration
            actual_duration = librosa.get_duration(
                y=audio_signal, 
                sr=self.config.sample_rate
            )
            
            # Pad with zeros if audio is shorter than threshold
            if actual_duration < self.config.duration_threshold:
                samples_needed = int(
                    self.config.duration_threshold * self.config.sample_rate - len(audio_signal)
                )
                audio_signal = np.pad(
                    audio_signal, 
                    (0, samples_needed), 
                    mode='constant', 
                    constant_values=0
                )
            
            return str(file_path), audio_signal
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def load_audio_files_parallel(self, file_paths: List[Path], 
                                 n_processes: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """Load multiple audio files in parallel.
        
        Args:
            file_paths: List of audio file paths
            n_processes: Number of processes to use (default: CPU count)
            
        Returns:
            List of tuples (filename, preprocessed_audio_signal)
        """
        if not file_paths:
            return []
        
        logger.info(f"Loading {len(file_paths)} audio files in parallel...")
        
        # Create partial function for multiprocessing
        load_func = partial(self._load_audio_wrapper)
        
        with Pool(processes=n_processes) as pool:
            results = pool.map(load_func, file_paths)
        
        # Filter out None results (failed loads)
        successful_results = [result for result in results if result is not None]
        
        logger.info(f"Successfully loaded {len(successful_results)} out of {len(file_paths)} files")
        return successful_results
    
    def _load_audio_wrapper(self, file_path: Path) -> Optional[Tuple[str, np.ndarray]]:
        """Wrapper for load_audio_file that handles exceptions in multiprocessing.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (filename, audio_signal) or None if loading failed
        """
        try:
            return self.load_audio_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return None
    
    def validate_audio_signal(self, audio_signal: np.ndarray) -> bool:
        """Validate audio signal quality.
        
        Args:
            audio_signal: Audio signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        # Check for empty signal
        if len(audio_signal) == 0:
            return False
        
        # Check for all zeros (silent audio)
        if np.all(audio_signal == 0):
            logger.warning("Audio signal is completely silent")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(audio_signal)) or np.any(np.isinf(audio_signal)):
            logger.warning("Audio signal contains NaN or infinite values")
            return False
        
        # Check for clipping (values outside [-1, 1] range)
        if np.any(np.abs(audio_signal) > 1.0):
            logger.warning("Audio signal appears to be clipped")
            # Don't return False, just warn - clipped audio is still usable
        
        return True
    
    def get_audio_duration(self, file_path: Union[str, Path]) -> float:
        """Get duration of audio file without loading the full signal.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            duration = librosa.get_duration(path=str(file_path))
            return duration
        except Exception as e:
            logger.error(f"Error getting duration for {file_path}: {e}")
            raise
    
    def filter_by_duration(self, file_paths: List[Path], 
                          min_duration: float = 0.1) -> List[Path]:
        """Filter audio files by minimum duration.
        
        Args:
            file_paths: List of audio file paths
            min_duration: Minimum duration in seconds
            
        Returns:
            Filtered list of file paths
        """
        filtered_paths = []
        
        for file_path in file_paths:
            try:
                duration = self.get_audio_duration(file_path)
                if duration >= min_duration:
                    filtered_paths.append(file_path)
                else:
                    logger.debug(f"Skipping {file_path}: duration {duration:.2f}s < {min_duration}s")
            except Exception as e:
                logger.warning(f"Could not get duration for {file_path}, skipping: {e}")
        
        logger.info(f"Filtered {len(filtered_paths)} files out of {len(file_paths)} by duration")
        return filtered_paths