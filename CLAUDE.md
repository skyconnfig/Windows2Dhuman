# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 2D real-time digital human rendering system (2dHuman) that generates facial animations driven by audio input. The project features ultra-lightweight computation (39 Mflops per frame) and supports both CPU and GPU modes without requiring training.

## Development Environment Setup

### Python Environment
```bash
conda create -n dh_live python=3.11
conda activate dh_live
pip install torch --index-url https://download.pytorch.org/whl/cu124  # Or CPU version: pip install torch
pip install -r requirements.txt
```

### Model Files
Download checkpoint files from [百度网盘](https://pan.baidu.com/s/1jH3WrIAfwI3U5awtnt9KPQ?pwd=ynd7) and extract to `checkpoint/` directory.

## Common Commands

### Video Data Preparation
```bash
# Prepare video data for mini version
python data_preparation_mini.py video_data/000002/video.mp4 video_data/000002

# Prepare web assets  
python data_preparation_web.py video_data/000002
```

### Demo Execution
```bash
# Gradio interface (recommended for first-time users)
python app.py

# Mini demo with audio file (Windows only, requires 16kHz mono WAV)
python demo_mini.py video_data/000002/assets video_data/audio0.wav output.mp4

# Avatar demo
python demo_avatar.py
```

### Web Demo
```bash
# Replace assets folder with new avatar assets (video_data/000002/assets)
# Start web server
python web_demo/server.py
# Access at localhost:8888/static/MiniLive.html
```

## Architecture Overview

### Core Components

1. **Audio Processing (`talkingface/audio_model.py`)**
   - LSTM-based audio-to-blendshape conversion
   - Kaldi native fbank feature extraction
   - Located in `train_audio/` for training components

2. **Face Rendering (`talkingface/render_model_mini.py`)**
   - DINet_mini model for lightweight face generation
   - Input resolution: 128x128 (configurable via `input_height`, `input_width`)
   - GPU/CPU adaptive rendering

3. **3D Rendering Pipeline (`mini_live/`)**
   - OpenGL-based real-time rendering (`opengl_render_interface.py`)
   - Face mesh processing with MediaPipe integration
   - OBJ model utilities for face geometry

4. **Web Interface (`web_source/`)**
   - Browser-based real-time inference
   - JavaScript modules for audio processing and rendering
   - Compressed assets under 3MB total

### Data Flow
1. Audio input → Feature extraction → LSTM model → Blendshape coefficients
2. Video preparation → Face detection → PCA compression → Web assets
3. Real-time: Audio features + Face assets → DINet_mini → Rendered frames

### Key Directories
- `talkingface/`: Core models and utilities
- `mini_live/`: Lightweight rendering engine
- `train_audio/`: Audio model training components
- `web_source/`: Frontend template code
- `checkpoint/`: Pre-trained model weights
- `video_data/`: Processed avatar data storage

## Hardware Requirements

### Minimum (CPU mode)
- Intel i5-8400 / AMD Ryzen 5 2600
- 8GB RAM
- 5GB disk space

### Recommended (GPU mode)  
- Intel i7-10700K / AMD Ryzen 7 3700X
- 16GB RAM
- NVIDIA GTX 1660 Super / RTX 3060
- CUDA 11.0+ support

## Platform Support

| Platform | Video Processing | Offline Synthesis | Web Server | Real-time Chat |
|----------|------------------|-------------------|------------|----------------|
| Windows | ✅ | ✅ | ✅ | ✅ |
| Linux/macOS | ✅ | ❌ | ✅ | ✅ |

## Model Configuration

### Standard Settings
- Standard size: 256x256
- Crop ratio: [0.5, 0.5, 0.5, 0.5] 
- Output resolution: 128x128 (mini version)
- Audio sample rate: 16kHz mono

### Performance Benchmarks
- CPU mode: ~40ms per frame (25 FPS)
- GPU mode: ~15ms per frame (66 FPS)
- Memory usage: 3-6GB RAM, 2-4GB VRAM

## Error Handling

The codebase includes custom exception classes in `data_preparation_mini.py`:
- `VideoProcessingError`: Base video processing exception
- `FFmpegError`: FFmpeg-related errors
- `FaceDetectionError`: Face detection failures
- `FirstFrameFaceDetectionError`: Initial frame processing issues

## Dependencies

Core libraries specified in `requirements.txt`:
- PyTorch (with CUDA support optional)
- MediaPipe for face detection
- OpenGL libraries (glfw, PyOpenGL, pyglm)
- Audio processing (kaldi_native_fbank)
- Web interface (gradio)

## License

MIT License - includes code from DH_live project. See README.txt for attribution details.