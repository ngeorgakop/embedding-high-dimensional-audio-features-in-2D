# Acknowledgement

The following is mostly based on open source implementations, but heavily inspired by my internship (E. Kokkinis) and undergraduate (EZ Psarakis) supervisors to whom I will always feel grateful for their invaluable support during my first steps in the field.

# Audio Embedding and t-SNE Visualization

A Python package for extracting audio features and creating t-SNE embeddings for similarity visualization. This project maps audio files to 2D/3D coordinates where similar sounds are placed closer together.

## Overview

This package provides tools for:
- Loading and preprocessing audio files
- Extracting audio features (MFCC or spectrograms) 
- Creating t-SNE embeddings for 2D visualization
- Training neural networks to predict embeddings for new samples
- Evaluating embedding quality based on semantic groupings
- Batch processing for parameter sweeps

## Features
- **Feature Extraction**: MFCC and spectrogram features with customizable parameters
- **t-SNE Mapping**: Scikit-learn t-SNE implementation with Barnes-Hut approximation for efficient dimensionality reduction  
- **Neural Network Training**: Learn mappings from features to t-SNE coordinates
- **Evaluation**: Automatic evaluation based on filename keywords
- **Batch Processing**: Parameter sweep capabilities for optimal settings
- **Modular Architecture**: Clean separation of concerns with type hints
- **Configuration Management**: JSON-based configuration system
- **Error Handling**: Robust error handling and validation
- **Parallel Processing**: Efficient multiprocessing support

## Installation

### Prerequisites
```bash
# Install required system packages (example for Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3 python3-pip ffmpeg

# For macOS with Homebrew
brew install python3 ffmpeg
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Additional Setup
The package uses scikit-learn's t-SNE implementation, which should be automatically installed with the requirements. No additional compilation is needed.

## Quick Start

### Basic Usage

1. **Extract features and create t-SNE embeddings:**
```bash
python run_audio_embedding.py \
    --input-dir /path/to/audio/files \
    --output-dir /path/to/output \
    --feature-type mfcc
```

2. **Evaluate embeddings:**
```bash
python evaluate_embeddings.py \
    --directory /path/to/output \
    --top 10
```

3. **Train neural network on embeddings:**
```bash
python train_regressor.py \
    --embeddings-dir /path/to/output/mfcc/mfcc_2048_30_13 \
    --epochs 100
```

4. **Predict coordinates for new samples:**
```bash
python predict_new_samples.py \
    --samples-dir /path/to/new/audio \
    --models-dir /path/to/output/mfcc/mfcc_2048_30_13
```

### Batch Processing

Run experiments with multiple parameter combinations:
```bash
python batch_process.py \
    --input-dir /path/to/audio \
    --output-dir /path/to/experiments \
    --feature-types mfcc spectrogram \
    --perplexity-values 20 30 50 \
    --n-mfcc-values 13 20 30
```

## Configuration

### JSON Configuration Files

You can use JSON configuration files instead of command-line arguments:

```json
{
  "input_directory": "/path/to/audio/files",
  "output_directory": "/path/to/output",
  "file_limit": null,
  "n_processes": null,
  "audio": {
    "sample_rate": 44100,
    "duration_threshold": 10.0,
    "n_fft": 2048,
    "hop_length": 1024,
    "window_function": "hann"
  },
  "features": {
    "feature_type": "mfcc",
    "normalize_features": true,
    "n_mfcc": 13,
    "n_mels": 128,
    "use_log_magnitude": true,
    "bin_step": 10,
    "time_step": 10
  },
  "tsne": {
    "perplexity": 30,
    "initial_dims": 30,
    "n_dims_2d": 2,
    "n_dims_3d": 3,
    "max_iterations": 1000,
    "theta": 0.5,
    "use_pca": true,
    "random_seed": null
  }
}
```

Use with:
```bash
python run_audio_embedding.py --config-file config.json
```

## API Usage

### Programmatic Access

```python
from audio_embedding import (
    AudioEmbeddingConfig,
    AudioProcessor,
    FeatureExtractor,
    TSNEMapper,
    TSNEEvaluator
)

# Create configuration
config = AudioEmbeddingConfig()
config.input_directory = "/path/to/audio"
config.features.feature_type = "mfcc"

# Initialize components
audio_processor = AudioProcessor(config.audio)
feature_extractor = FeatureExtractor(config.audio, config.features)
tsne_mapper = TSNEMapper(config.tsne)

# Process audio files
audio_files = audio_processor.find_audio_files(config.input_directory)
audio_data = audio_processor.load_audio_files_parallel(audio_files)

# Extract features
features = []
for filename, audio_signal in audio_data:
    feature_vector = feature_extractor.extract_features(audio_signal)
    features.append(feature_vector)

# Create embeddings
filenames = [item[0] for item in audio_data]
embeddings = tsne_mapper.create_embeddings(features, filenames)

# Save results
tsne_mapper.save_embeddings(embeddings, config.get_output_subdirectory())
```

## Output Files

The pipeline generates several output files:

- `points.json` - Main visualization data with 2D coordinates and 3D colors (for web viewer)
- `features.npy` - Extracted high-dimensional features
- `*.tsv` - t-SNE coordinates in TSV format (legacy compatibility)
- `filenames.txt` - List of processed filenames
- `metadata.json` - Processing metadata and parameters
- `config.json` - Configuration used for processing

## Web Visualization

The package includes a modern web-based visualizer in `web_viewer/`:

1. **Open the visualizer**: Open `web_viewer/index.html` in your browser
2. **Load data**: Drag and drop a `points.json` file onto the browser
3. **Explore**: Hover over points to see audio filenames and clustering
4. **Customize**: Adjust visualization parameters in the control panel

The web viewer provides an interactive way to explore your audio similarity maps without requiring any software installation.

## Evaluation

The evaluation system groups audio files by keywords in their filenames:
- **snare**, **kick**, **bass**, **hat**, **clap**, **tom**

Files with the same keyword should cluster together in the t-SNE space. The evaluator calculates mean pairwise distances within each group - lower distances indicate better clustering.

Example evaluation output:
```
=== t-SNE Evaluation Results (Top 5) ===

1. Parameters: mfcc_2048_30_13
   Total Mean Distance: 0.234567
   Samples: 1000, Valid Groups: 6
   Group Distances:
     snares: 0.123456
     kicks: 0.234567
     ...
```

## Neural Network Training

The neural network component learns mappings from audio features to t-SNE coordinates, allowing you to:
- Add new samples to existing visualizations without recomputing t-SNE
- Create real-time audio similarity visualization applications
- Transfer learned embeddings to new datasets

### Training Process
1. Load existing t-SNE embeddings and features
2. Split data into training/test sets
3. Train separate networks for 2D coordinates and 3D colors
4. Evaluate on test set using Mean Absolute Error
5. Save trained models for future prediction

## Supported Audio Formats

### Processing Pipeline
- WAV (`.wav`)
- MP3 (`.mp3`) 
- FLAC (`.flac`)
- M4A (`.m4a`)
- AAC (`.aac`)

### Web Visualization
The web viewer works with any `points.json` file generated by the pipeline. For actual audio playback in the browser, additional setup may be required (see `web_viewer/README.md`).

## Performance Tips

1. **Parallel Processing**: Use `--n-processes` to specify the number of CPU cores
2. **File Limiting**: Use `--limit` for testing with smaller datasets
3. **Feature Caching**: Features are saved as `.npy` files for reuse
4. **Memory Management**: Large datasets may require chunked processing

## Troubleshooting

### Common Issues

1. **Scikit-learn Issues**:
   ```bash
   pip install scikit-learn>=1.0.0
   ```

2. **Memory Issues with Large Datasets**:
   - Use `--limit` to process in smaller batches
   - Reduce `--n-fft` or increase `--bin-step`/`--time-step` for spectrograms

3. **TensorFlow/Keras Issues**:
   ```bash
   pip install tensorflow>=2.8.0
   ```

4. **Audio Loading Errors**:
   - Ensure FFmpeg is installed
   - Check audio file formats are supported
   - Verify file permissions



## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Include docstrings for all public methods
4. Add unit tests for new functionality
5. Update documentation for API changes

## License

This project is open source.