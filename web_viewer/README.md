# Audio t-SNE Web Visualizer

A modern, web-based replacement for the C++ openFrameworks viewer. This interactive visualization allows you to explore audio similarity maps created by the t-SNE embedding pipeline.

## Features

### 🎨 **Interactive Visualization**
- Responsive 2D canvas that adapts to any screen size
- Smooth animations and modern UI design
- Real-time mouse interaction with visual feedback
- Color-coded points based on 3D t-SNE embeddings

### 🎵 **Audio Playback Simulation**
- Hover over points to trigger audio playback
- Visual indication of currently playing samples (green circles)
- Configurable playback duration and pause intervals
- Mouse radius control for interaction sensitivity

### ⚙️ **Customizable Controls**
- **Max Duration**: Control how long each sample plays (0.5-10 seconds)
- **Mouse Radius**: Adjust interaction sensitivity (20-200 pixels)
- **Pause Length**: Set delay between consecutive plays (0.1-5 seconds)
- **Point Size**: Change visual size of data points (2-20 pixels)

### 📁 **Easy File Loading**
- Drag and drop points.json files directly onto the browser
- File input control for traditional file selection
- Automatic parsing of t-SNE embedding data
- Real-time statistics and feedback

## Usage

### Quick Start
1. Open `index.html` in any modern web browser
2. Load a `points.json` file created by the audio embedding pipeline
3. Hover over the colored circles to explore the audio space
4. Adjust controls in the right panel to customize the experience

### File Format
The visualizer expects JSON files in this format:
```json
[
  {
    "path": "/path/to/audio/file.wav",
    "point": [0.123, 0.456],
    "color": [0.789, 0.012, 0.345]
  }
]
```

Where:
- `path`: Path to the original audio file
- `point`: 2D t-SNE coordinates (normalized 0-1)
- `color`: 3D t-SNE coordinates for color mapping (normalized 0-1)

## Technical Implementation

### Modern Web Technologies
- **HTML5 Canvas**: High-performance 2D rendering
- **ES6+ JavaScript**: Modern, clean code structure
- **CSS3**: Responsive design with glassmorphism effects
- **Web Audio API Ready**: Framework prepared for audio integration

### Browser Compatibility
- Chrome/Edge 80+
- Firefox 75+
- Safari 13+
- Mobile browsers supported

### Performance
- Efficient canvas rendering for thousands of points
- Smooth 60fps animations
- Minimal memory footprint
- No external dependencies

## Audio Playback Integration

### Current Implementation
The current version simulates audio playback by:
- Displaying the filename of the "playing" audio
- Visual feedback with green circles
- Timing controls for playback duration

### Real Audio Playback Options

To enable actual audio playback, you can extend the visualizer with:

#### Option 1: Local Web Server
```bash
# Serve audio files from a local directory
python -m http.server 8000
```
Then modify the visualizer to load audio files via HTTP requests.

#### Option 2: File Upload Interface
Add a file upload component that allows users to upload both the points.json and corresponding audio files.

#### Option 3: Cloud Integration
Integrate with cloud storage services (AWS S3, Google Cloud Storage) to stream audio files.

#### Example Audio Integration
```javascript
async loadAudioFile(sample) {
    try {
        const response = await fetch(sample.path);
        const arrayBuffer = await response.arrayBuffer();
        const audioContext = new AudioContext();
        sample.audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        sample.isLoaded = true;
    } catch (error) {
        console.error('Error loading audio:', error);
    }
}
```

## Advantages Over C++ Viewer

### ✅ **Cross-Platform**
- Works on any device with a web browser
- No compilation or installation required
- Mobile and tablet support

### ✅ **Modern UI/UX**
- Responsive design that adapts to any screen size
- Smooth animations and visual feedback
- Intuitive drag-and-drop file loading
- Real-time parameter adjustment

### ✅ **Easy Deployment**
- Single HTML file can be shared easily
- No external dependencies
- Can be hosted on any web server
- GitHub Pages compatible

### ✅ **Extensible**
- Easy to modify and enhance
- Can integrate with web APIs
- Supports modern JavaScript features
- Framework ready for additional features

## Customization

### Styling
Modify the CSS in `index.html` to change:
- Color schemes and gradients
- UI component layouts
- Animation effects
- Responsive breakpoints

### Functionality
Extend `visualizer.js` to add:
- Additional visualization modes
- Data export capabilities
- Clustering analysis
- Search and filtering

### Integration
The visualizer can be integrated into:
- Jupyter notebooks
- Web applications
- Research dashboards
- Interactive presentations

## Future Enhancements

- **3D Visualization**: Three.js integration for 3D exploration
- **Audio Streaming**: Direct integration with audio files
- **Clustering Tools**: Interactive cluster analysis
- **Export Features**: Save visualizations as images or videos
- **Collaborative Features**: Share visualizations with others
- **VR/AR Support**: Immersive exploration experiences