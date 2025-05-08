# Eye Aspect Ratio (EAR) Deepfake Detection

This tool analyzes videos to detect potential deepfakes by examining eye blinking patterns, which are often inaccurately reproduced in synthetic media. The code provides both visual and numerical metrics to assess the likelihood that a video has been manipulated.

## How It Works

The tool uses the Eye Aspect Ratio (EAR) technique, which measures the ratio between the height and width of the eye. When we blink, this ratio decreases significantly. By tracking this metric over time, we can analyze blinking patterns and identify unnatural behavior that often appears in deepfakes.

### The Science Behind It

1. **Normal Human Blinking**: The average person blinks 15-20 times per minute, with each blink lasting 100-400 milliseconds
2. **Deepfake Limitations**: Many deepfake algorithms struggle to reproduce natural blinking patterns, typically resulting in:
   - Less frequent blinking
   - Unnatural blink duration
   - Inconsistent blink timing
3. **Detection Method**: By measuring EAR, we can identify these anomalies without requiring the original video

## Features

- Real-time EAR calculation and visualization
- Moving graph display of EAR values
- Visual highlighting of eye landmarks
- Deepfake likelihood assessment based on blink frequency analysis
- Progress tracking for long video processing
- Comprehensive summary statistics

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- dlib
- NumPy
- SciPy
- Matplotlib

## Installation

1. Install required Python packages:
   ```
   pip install opencv-python dlib scipy matplotlib numpy
   ```

2. Download the facial landmark predictor file:
   - Download from: https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract it and place it in the same directory as this script

## Usage

1. Configure the input and output paths in the script:
   ```python
   input_video = "path/to/your/video.mp4"
   output_video = "path/to/output.mp4"
   ```

2. Run the script:
   ```
   python ear_detection.py
   ```

3. Review the output video and console statistics

## Interpreting Results

The tool provides several metrics to help assess if a video might be a deepfake:

- **EAR Value**: The current eye aspect ratio (lower values indicate more closed eyes)
- **Blink Ratio**: Percentage of frames where blinking is detected
- **Deepfake Likelihood**: Assessment based on blink frequency compared to natural human blinking patterns

### Warning Signs

- Unusually low blink rates (<5% of frames showing blinks)
- Very consistent EAR values with minimal variation
- Sudden, unnatural jumps in EAR values

## Limitations

- Requires visible eyes and good lighting conditions
- May produce false positives with videos of people intentionally not blinking
- Works best with frontal face views
- Not foolproof - should be used as one tool in a broader detection strategy

## For Security Professionals

This tool demonstrates just one technique in deepfake detection. For comprehensive security, consider:

1. Combining multiple detection methods
2. Implementing media authentication protocols
3. Training staff on verification procedures
4. Establishing clear incident response plans for suspected deepfakes

## References

- Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. 21st Computer Vision Winter Workshop.
- Li, Y., Chang, M. C., & Lyu, S. (2018). In Ictu Oculi: Exposing AI Generated Fake Face Videos by Detecting Eye Blinking.
