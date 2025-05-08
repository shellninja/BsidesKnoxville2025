# Deepfake Mask Detector

A Python tool for real-time detection of potential deepfake masks in videos or webcam feeds using computer vision techniques.

## Description

This application analyzes video frames to identify potential deepfake manipulations by examining various visual artifacts that commonly appear in AI-generated or manipulated faces. The detector uses multiple analysis methods to generate probability scores for each detected face, indicating how likely it is to be a deepfake.

## Features

- Real-time deepfake detection for video files or webcam feeds
- Multi-faceted analysis using six different detection techniques:
  - Edge consistency analysis
  - Color consistency analysis
  - Facial landmark analysis
  - Blending artifact detection
  - Jawline consistency analysis
  - Texture consistency analysis
- Visual feedback with face highlighting and detailed metrics panel
- FPS counter to monitor performance
- Adjustable detection parameters for fine-tuning

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- dlib
- SciPy
- Pre-trained facial landmark model (`shape_predictor_68_face_landmarks.dat`)

## Installation

1. Clone this repository or download the source code
2. Install required packages:
```bash
pip install opencv-python numpy dlib scipy
```
3. Download the facial landmark predictor file:
   - The `shape_predictor_68_face_landmarks.dat` file can be downloaded from the dlib model repository
   - Place this file in the same directory as the script

## Usage

### Using the webcam:
```bash
python deepfake_detector.py
```

### Analyzing a video file:
```bash
python deepfake_detector.py --video_path path/to/your/video.mp4
```

### Controls:
- Press 'q' to quit the application

## How It Works

The detector analyzes each frame through multiple detection methods:

1. **Edge Detection Analysis**: Identifies unnatural edges that might indicate mask boundaries
2. **Color Consistency Analysis**: Examines color transitions between face and surrounding areas
3. **Facial Landmark Analysis**: Checks for unnatural positioning of facial features
4. **Blending Artifact Detection**: Identifies artifacts from poor blending in manipulated videos
5. **Jawline Consistency Analysis**: Specifically examines the jawline area for mask artifacts
6. **Texture Consistency Analysis**: Detects unnatural smoothness or repetitive patterns

These analyses are combined with optimized weights to calculate an overall probability score. The detector shows a visualization with face boxes (green for likely real, red for likely fake) and a detailed metrics panel.

## Limitations

As stated in the application's disclaimer, no deepfake detection method is 100% accurate. This tool should be used as an aid for initial assessment only, not as definitive proof of manipulation. Factors such as lighting conditions, camera quality, and video compression can affect detection accuracy.

## Notes

- The detector requires good lighting conditions for optimal performance
- Detection accuracy may vary depending on the quality and resolution of the input video
- The system is designed to identify common deepfake artifacts but may not detect all types of manipulations

## License

[Specify your license here or remove this section if not applicable]