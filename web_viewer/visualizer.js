class AudioTSNEVisualizer {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.audioSamples = [];
        this.currentlyPlaying = new Set();
        this.lastPlayTime = new Map();
        
        // Configuration
        this.config = {
            maxDuration: 3.0,
            mouseRadius: 80,
            pauseLength: 1.0,
            pointSize: 6
        };
        
        this.setupCanvas();
        this.setupControls();
        this.setupDragDrop();
        this.setupEventListeners();
        this.animate();
    }
    
    setupCanvas() {
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    
    setupControls() {
        // File input
        const fileInput = document.getElementById('fileInput');
        fileInput.addEventListener('change', (e) => this.loadPointsFile(e.target.files[0]));
        
        // Control sliders
        const controls = ['maxDuration', 'mouseRadius', 'pauseLength', 'pointSize'];
        controls.forEach(control => {
            const slider = document.getElementById(control);
            const valueDisplay = document.getElementById(control + 'Value');
            
            slider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.config[control] = value;
                
                let unit = '';
                if (control.includes('Duration') || control.includes('pause')) unit = 's';
                else if (control.includes('Radius') || control.includes('Size')) unit = 'px';
                
                valueDisplay.textContent = value + unit;
            });
        });
    }
    
    setupDragDrop() {
        const dragDropZone = document.getElementById('dragDropZone');
        
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, this.preventDefaults, false);
        });
        
        // Highlight drop zone when dragging over it
        ['dragenter', 'dragover'].forEach(eventName => {
            document.addEventListener(eventName, () => {
                dragDropZone.classList.add('active');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, () => {
                dragDropZone.classList.remove('active');
            }, false);
        });
        
        // Handle dropped files
        document.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.loadPointsFile(files[0]);
            }
        }, false);
    }
    
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    setupEventListeners() {
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
    }
    
    async loadPointsFile(file) {
        if (!file) return;
        
        const loading = document.getElementById('loading');
        loading.style.display = 'block';
        
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            
            // Reset current state
            this.stopAllAudio();
            this.audioSamples = [];
            
            // Process the points data
            await this.processAudioSamples(data);
            
            this.updateStats();
            console.log(`Loaded ${this.audioSamples.length} audio samples`);
            
        } catch (error) {
            console.error('Error loading points file:', error);
            alert('Error loading points file. Please check the file format.');
        } finally {
            loading.style.display = 'none';
        }
    }
    
    async processAudioSamples(data) {
        for (const item of data) {
            if (item.point && item.color && item.path) {
                const sample = {
                    path: item.path,
                    point: {
                        x: item.point[0],
                        y: item.point[1]
                    },
                    color: {
                        r: Math.floor(item.color[0] * 255),
                        g: Math.floor(item.color[1] * 255),
                        b: Math.floor(item.color[2] * 255)
                    },
                    audio: null,
                    isLoaded: false,
                    isPlaying: false,
                    lastPlayTime: 0
                };
                
                this.audioSamples.push(sample);
            }
        }
    }
    
    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const currentTime = Date.now() / 1000;
        
        for (const sample of this.audioSamples) {
            const canvasX = sample.point.x * this.canvas.width;
            const canvasY = sample.point.y * this.canvas.height;
            
            const distance = Math.sqrt(
                Math.pow(mouseX - canvasX, 2) + Math.pow(mouseY - canvasY, 2)
            );
            
            if (distance < this.config.mouseRadius && 
                !sample.isPlaying && 
                (currentTime - sample.lastPlayTime) > this.config.pauseLength) {
                
                this.playAudioSample(sample);
                break; // Only play one at a time
            }
        }
    }
    
    handleClick(e) {
        // Click can also trigger audio playback for mobile devices
        this.handleMouseMove(e);
    }
    
    async playAudioSample(sample) {
        try {
            // Load audio if not already loaded
            if (!sample.isLoaded) {
                // For web security, we can't directly load files from file paths
                // Instead, we'll show the filename and simulate playback
                console.log(`Would play: ${sample.path}`);
                
                // Create a placeholder audio simulation
                sample.audio = {
                    duration: this.config.maxDuration,
                    currentTime: 0
                };
                sample.isLoaded = true;
            }
            
            // Start "playing"
            sample.isPlaying = true;
            sample.lastPlayTime = Date.now() / 1000;
            this.currentlyPlaying.add(sample);
            
            // Update current track display
            const filename = sample.path.split('/').pop().split('\\').pop();
            document.getElementById('currentTrack').textContent = `♪ ${filename}`;
            
            // Stop after max duration
            setTimeout(() => {
                this.stopAudioSample(sample);
            }, this.config.maxDuration * 1000);
            
        } catch (error) {
            console.error('Error playing audio sample:', error);
        }
    }
    
    stopAudioSample(sample) {
        if (sample.isPlaying) {
            sample.isPlaying = false;
            this.currentlyPlaying.delete(sample);
            
            // Clear current track if this was the only one playing
            if (this.currentlyPlaying.size === 0) {
                document.getElementById('currentTrack').textContent = '';
            }
        }
    }
    
    stopAllAudio() {
        for (const sample of this.audioSamples) {
            this.stopAudioSample(sample);
        }
    }
    
    updateStats() {
        const stats = document.getElementById('stats');
        stats.innerHTML = `
            <div>📊 ${this.audioSamples.length} samples loaded</div>
            <div>🎵 ${this.currentlyPlaying.size} currently playing</div>
        `;
    }
    
    draw() {
        // Clear canvas
        this.ctx.fillStyle = 'rgba(44, 62, 80, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw audio samples
        for (const sample of this.audioSamples) {
            const x = sample.point.x * this.canvas.width;
            const y = sample.point.y * this.canvas.height;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, this.config.pointSize, 0, 2 * Math.PI);
            
            if (sample.isPlaying) {
                // Green for currently playing
                this.ctx.fillStyle = 'rgba(78, 204, 163, 0.8)';
                this.ctx.strokeStyle = 'rgba(78, 204, 163, 1)';
                this.ctx.lineWidth = 2;
                this.ctx.stroke();
            } else {
                // Use the sample's color
                this.ctx.fillStyle = `rgba(${sample.color.r}, ${sample.color.g}, ${sample.color.b}, 0.7)`;
            }
            
            this.ctx.fill();
        }
        
        // Draw mouse radius indicator if hovering
        if (this.mouseX !== undefined && this.mouseY !== undefined) {
            this.ctx.beginPath();
            this.ctx.arc(this.mouseX, this.mouseY, this.config.mouseRadius, 0, 2 * Math.PI);
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
        }
    }
    
    animate() {
        this.draw();
        this.updateStats();
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize the visualizer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.visualizer = new AudioTSNEVisualizer();
    
    // Track mouse position for radius indicator
    document.getElementById('canvas').addEventListener('mousemove', (e) => {
        const rect = e.target.getBoundingClientRect();
        window.visualizer.mouseX = e.clientX - rect.left;
        window.visualizer.mouseY = e.clientY - rect.top;
    });
    
    document.getElementById('canvas').addEventListener('mouseleave', () => {
        window.visualizer.mouseX = undefined;
        window.visualizer.mouseY = undefined;
    });
});

// Note: Actual audio playback would require either:
// 1. A web server to serve audio files
// 2. User uploading audio files along with the points.json
// 3. Integration with cloud storage or audio streaming services
// This demo shows the visualization framework that can be extended with real audio playback.