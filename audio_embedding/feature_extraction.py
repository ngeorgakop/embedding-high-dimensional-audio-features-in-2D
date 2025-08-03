"""Audio feature extraction module."""

import numpy as np
import librosa
from skimage.measure import block_reduce
from typing import Optional, Tuple
import logging
from .config import AudioConfig, FeatureConfig

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts audio features for t-SNE embedding."""
    
    def __init__(self, audio_config: AudioConfig, feature_config: FeatureConfig):
        """Initialize feature extractor with configuration.
        
        Args:
            audio_config: Audio processing configuration
            feature_config: Feature extraction configuration
        """
        self.audio_config = audio_config
        self.feature_config = feature_config
        self._window = self._create_window()
    
    def _create_window(self) -> np.ndarray:
        """Create window function for STFT."""
        if self.audio_config.window_function.lower() == "hann":
            return np.hanning(self.audio_config.n_fft)
        elif self.audio_config.window_function.lower() == "hamming":
            return np.hamming(self.audio_config.n_fft)
        else:
            raise ValueError(f"Unsupported window function: {self.audio_config.window_function}")
    
    def extract_features(self, audio_signal: np.ndarray) -> np.ndarray:
        """Extract features from audio signal.
        
        Args:
            audio_signal: Audio signal as numpy array
            
        Returns:
            Feature vector as numpy array
        """
        if self.feature_config.feature_type == "mfcc":
            return self._extract_mfcc_features(audio_signal)
        elif self.feature_config.feature_type == "spectrogram":
            return self._extract_spectrogram_features(audio_signal)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_config.feature_type}")
    
    def _extract_mfcc_features(self, audio_signal: np.ndarray) -> np.ndarray:
        """Extract MFCC features with delta and delta-delta coefficients.
        
        Args:
            audio_signal: Audio signal as numpy array
            
        Returns:
            MFCC feature vector
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_signal,
                sr=self.audio_config.sample_rate,
                n_mfcc=self.feature_config.n_mfcc,
                n_fft=self.audio_config.n_fft,
                hop_length=self.audio_config.hop_length,
                n_mels=self.feature_config.n_mels
            )
            
            # Calculate delta and delta-delta coefficients
            delta_mfcc = librosa.feature.delta(mfcc, order=1)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # Concatenate mean values of MFCC, delta, and delta-delta
            feature_vector = np.concatenate([
                np.mean(mfcc, axis=1),
                np.mean(delta_mfcc, axis=1),
                np.mean(delta2_mfcc, axis=1)
            ])
            
            return self._normalize_features(feature_vector)
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {e}")
            raise
    
    def _extract_spectrogram_features(self, audio_signal: np.ndarray) -> np.ndarray:
        """Extract spectrogram features.
        
        Args:
            audio_signal: Audio signal as numpy array
            
        Returns:
            Spectrogram feature vector
        """
        try:
            # Compute STFT
            stft_matrix = librosa.stft(
                audio_signal,
                n_fft=self.audio_config.n_fft,
                hop_length=self.audio_config.hop_length,
                window=self._window
            )
            
            # Get magnitude
            magnitude = np.abs(stft_matrix)
            
            # Apply log magnitude if specified
            if self.feature_config.use_log_magnitude:
                magnitude = librosa.amplitude_to_db(magnitude)
            
            # Apply block reduction if specified
            if self.feature_config.time_step > 1 or self.feature_config.bin_step > 1:
                magnitude = block_reduce(
                    magnitude,
                    (self.feature_config.bin_step, self.feature_config.time_step),
                    func=np.mean
                )
            
            # Reshape to vector
            feature_vector = magnitude.flatten()
            
            return self._normalize_features(feature_vector)
            
        except Exception as e:
            logger.error(f"Error extracting spectrogram features: {e}")
            raise
    
    def _normalize_features(self, feature_vector: np.ndarray) -> np.ndarray:
        """Normalize feature vector if specified in configuration.
        
        Args:
            feature_vector: Feature vector to normalize
            
        Returns:
            Normalized feature vector
        """
        if not self.feature_config.normalize_features:
            return feature_vector
        
        # Check for zero standard deviation to avoid division by zero
        std = np.std(feature_vector)
        if std == 0:
            logger.warning("Feature vector has zero standard deviation, skipping normalization")
            return feature_vector
        
        # Z-score normalization
        mean = np.mean(feature_vector)
        normalized = (feature_vector - mean) / std
        
        return normalized
    
    def get_feature_dimension(self) -> int:
        """Get the expected dimension of feature vectors.
        
        Returns:
            Expected feature vector dimension
        """
        if self.feature_config.feature_type == "mfcc":
            # MFCC + delta + delta-delta
            return self.feature_config.n_mfcc * 3
        else:
            # For spectrograms, dimension depends on FFT size and reduction factors
            freq_bins = (self.audio_config.n_fft // 2 + 1) // self.feature_config.bin_step
            # Time dimension is variable, so we can't predict exact size
            # Return an estimate based on typical audio length
            time_frames = 100 // self.feature_config.time_step  # rough estimate
            return freq_bins * time_frames