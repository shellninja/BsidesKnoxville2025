# Face Mesh Visualizer for Deepfakes

A Python tool for visualizing and extracting 3D facial meshes using MediaPipe's Face Mesh model. This tool demonstrates the type of facial landmark detection techniques commonly used in deepfake creation and analysis.

## Description

This application processes videos or webcam feeds to detect and visualize facial meshes in both 2D and 3D. It extracts 468 facial landmarks and displays their connections in real-time, providing insight into how facial tracking works for deepfake technology. The tool can also extract facial mesh data from images and save it in OBJ format for use in 3D applications.

## Features

- Real-time facial mesh detection and visualization
- 2D overlay of facial mesh on video frames
- Interactive 3D visualization of the facial mesh
- Extraction of facial mesh data to OBJ file format
- Support for video files or webcam input
- Detailed visual feedback with status messages

## Requirements

- Python 3.6+
- OpenCV (cv2)
- MediaPipe
- NumPy
- Matplotlib (with 3D plotting support)

## Installation

1. Clone this repository or download the source code
2. Install the required packages:
```bash
pip install opencv-python mediapipe numpy matplotlib
```

## Usage

### Processing Video

```python
from face_mesh_visualizer import FaceMeshVisualizer

# Initialize the visualizer
visualizer = FaceMeshVisualizer()

# Process a video file
visualizer.process_video(
    input_video_path="input.mp4",
    output_video_path="output.mp4",
    show_3d=True,  # Enable 3D visualization
    show_2d=True   # Enable 2D visualization
)

# Or use webcam (pass 0 as the input_video_path)
visualizer.process_video(
    input_video_path=0,
    output_video_path="webcam_output.mp4",
    show_3d=True,
    show_2d=True
)
```

### Extracting Mesh from Image

```python
# Extract facial mesh from an image and save as OBJ file
visualizer.extract_and_save_mesh(
    input_image_path="face.jpg",
    output_obj_path="face_mesh.obj"
)
```

## Controls

- Press `ESC` key to exit the application while processing video

## Output

- **2D Visualization**: A video file showing the facial mesh overlay on each frame
- **3D Visualization**: A real-time 3D plot showing the facial landmarks and their connections
- **OBJ File**: When using the mesh extraction feature, an OBJ file containing the 3D mesh data

## How It Works

The visualizer uses MediaPipe's Face Mesh model to detect 468 facial landmarks in 3D space. These landmarks track specific facial features such as the eyes, nose, mouth, and overall face shape. The connections between these landmarks form a mesh that can be used to understand how deepfake systems track and manipulate facial features.

The 2D visualization draws this mesh directly on the video frames, while the 3D visualization creates an interactive plot showing the spatial relationships between the landmarks.

## Applications

- Educational demonstrations of facial tracking technology
- Research into deepfake creation and detection methods
- Preliminary step for creating 3D face models
- Understanding the underlying mechanics of face swapping algorithms

## Limitations

As noted in the disclaimer shown during processing, no deepfake detection or creation technology is 100% accurate. The mesh extraction quality depends on lighting conditions, face orientation, and camera quality.

## License

[Specify your license here or remove this section if not applicable]