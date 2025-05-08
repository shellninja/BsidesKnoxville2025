# Deepfake Mask Detector

A real-time tool for detecting potential mask artifacts in deepfake videos by analyzing edge consistency, color patterns, facial landmarks, and blending quality.

## Features

- Real-time video analysis from webcam or video files
- Multi-metric detection approach:
  - Edge consistency analysis
  - Color consistency analysis
  - Facial landmark analysis
  - Blending artifact detection
- Visual feedback with confidence scores
- FPS counter to monitor performance

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- dlib
- SciPy
- shape_predictor_68_face_landmarks.dat (facial landmark predictor model)

## Installation

1. Clone this repository or download the source code.

2. Install required Python packages:
   ```bash
   pip install opencv-python numpy dlib scipy
   ```

3. Download the facial landmark predictor model:
   ```bash
   # Option 1: Using wget
   wget https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   
   # Option 2: Manual download
   # Download from https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   # and extract it to the project directory
   ```

## Usage

### Analyzing webcam feed:
```bash
python deepfake_detector.py
```

### Analyzing a video file:
```bash
python deepfake_detector.py --video_path path/to/video.mp4
```

### Controls:
- Press 'q' to exit the program

## How It Works

The detector uses a multi-metric approach to identify potential deepfake masks:

1. **Edge Consistency Analysis**: Examines edge patterns around facial boundaries to detect unnatural transitions that might indicate mask edges.

2. **Color Consistency Analysis**: Compares color histograms between face and surrounding areas to identify unnatural color transitions.

3. **Facial Landmark Analysis**: Checks facial proportions and symmetry for unnatural distortions.

4. **Blending Artifact Detection**: Identifies artifacts in color transitions that might indicate poor blending of a face mask.

For each face in the frame, the detector calculates individual scores for each metric and combines them into an overall confidence score. Results are displayed in real-time with color-coded indicators.

## Limitations

- No deepfake detection method is 100% accurate
- Performance depends on video quality and lighting conditions
- May produce false positives or false negatives
- Requires clear, well-lit facial images for best results
- Computational requirements may impact performance on older hardware

## Disclaimer

This tool is provided for research and educational purposes only. It should not be used as the sole means to authenticate or verify content. The techniques used are not foolproof and may not detect sophisticated deepfakes or may incorrectly flag authentic content.

## License

[Specify license information here]