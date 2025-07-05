# AI-Powered Fitness Tracker

*Fork of Google's PyCoral library, enhanced with custom fitness tracking application*

## Overview

This project transforms a Raspberry Pi into an intelligent fitness monitoring system that combines computer vision, voice recognition, and environmental sensing. Built on top of Google's PyCoral library, it demonstrates practical edge AI implementation with real-time pose detection and multimodal interaction.

## Key Features

- **Real-time Exercise Counting**: Automatically tracks bicep curls and knee raises using pose detection
- **Voice Control**: Accepts spoken commands for counter reset and environmental queries
- **Performance Optimized**: Achieves 30fps pose detection using Coral Edge TPU (3x improvement over CPU-only)
- **Environmental Monitoring**: Displays temperature and humidity on demand
- **Gesture Controls**: Exit system by bringing wrists together
- **Visual Feedback**: LED matrix displays for exercise counts and environmental data

## Hardware Requirements

- Raspberry Pi 4
- Coral USB Accelerator (Edge TPU)
- Pi Camera Module
- Sense HAT
- Microphone (for voice commands)

## Technical Implementation

### Pose Detection & Exercise Counting
- Uses Google's MoveNet model optimized for Coral Edge TPU
- Tracks 17 keypoints for accurate body pose estimation
- Calculates joint angles to determine exercise completion
- Implements state tracking to prevent double-counting

### Voice Recognition System
- Continuous speech recognition using Google Speech API
- Google Gemini AI for natural language intent classification
- Multiprocessing architecture prevents blocking of main video loop
- Supports commands: reset counter, temperature check, humidity check

### Performance Metrics
- **30fps** real-time pose detection with Coral Edge TPU
- **10fps** baseline performance with CPU-only processing
- **3x performance improvement** through hardware acceleration

## Installation

1. **Install PyCoral Library**
   ```bash
   # On Debian systems (recommended)
   sudo apt-get install python3-pycoral
   
   # Or follow instructions at coral.ai/software/
   ```

2. **Install Additional Dependencies**
   ```bash
   pip3 install opencv-python pillow numpy sense-hat speechrecognition google-generativeai
   ```

3. **Setup Hardware**
   - Connect Coral USB Accelerator
   - Attach Pi Camera Module
   - Install Sense HAT on GPIO pins

4. **Configure API Keys**
   - Add your Google Generative AI API key to the script
   - Ensure microphone permissions are configured

## Usage

```bash
python3 fitness_tracker.py
```

### Voice Commands
- "Start over" / "Restart" → Reset exercise counters
- "How hot is it?" → Display temperature
- "What's the humidity?" → Display humidity levels

### Exercise Detection
- **Bicep Curls**: Monitors right arm elbow angle (red LED display)
- **Knee Raises**: Tracks right ankle above left knee (green LED display)
- **Exit**: Bring both wrists close together

## Project Structure

```
fitness_tracker.py          # Main application
test_data/                  # MoveNet model files
├── movenet_single_pose_lightning_ptq_edgetpu.tflite
```

## Technical Architecture

- **Main Process**: Handles video capture, pose detection, and exercise counting
- **Speech Process**: Runs continuous voice recognition in parallel
- **Queue Communication**: Thread-safe data exchange between processes
- **State Management**: Prevents false positives in exercise counting

## Performance Optimization

The implementation showcases several optimization techniques:
- **Hardware Acceleration**: Coral Edge TPU for 3x faster inference
- **Efficient Processing**: Multiprocessing prevents audio blocking video
- **Memory Management**: Proper tensor handling and cleanup
- **Real-time Constraints**: Maintains 30fps for smooth user experience

## Original PyCoral Library

This project is built upon Google's PyCoral library. For the original documentation and examples, see:
- [PyCoral API Documentation](https://coral.ai/docs/reference/py/)
- [Edge TPU Python Guide](https://coral.ai/docs/edgetpu/tflite-python/)
- [Original Examples](https://github.com/google-coral/pycoral/tree/master/examples)

## License

This project maintains the same license as the original PyCoral library. See LICENSE file for details.

## Acknowledgments

- Google Coral team for the PyCoral library and Edge TPU optimization
- Google for MoveNet pose detection model
- Raspberry Pi Foundation for accessible edge computing platform
