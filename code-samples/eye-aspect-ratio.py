import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def eye_aspect_ratio(eye):
    """
    Calculate the eye aspect ratio (EAR)
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    where p1, ..., p6 are 2D facial landmark points
    """
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

# Helper function to draw text with outline
def draw_text_with_outline(img, text, position, font, font_scale, text_color, outline_color, thickness=1, outline_thickness=3):
    # Draw the outline (stroke)
    cv2.putText(img, text, position, font, font_scale, outline_color, outline_thickness)
    # Draw the inner text
    cv2.putText(img, text, position, font, font_scale, text_color, thickness)

def process_video(input_video_path, output_video_path, show_landmarks=True):
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Define indices for the facial landmarks for the left and right eyes
    # These indices are based on the 68-point facial landmark detector
    LEFT_EYE_INDICES = list(range(36, 42))
    RIGHT_EYE_INDICES = list(range(42, 48))

    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Variables for EAR tracking
    ear_values = []
    frame_numbers = []
    frame_count = 0
    ear_window_size = 30  # Number of frames to display in the moving window
    
    # EAR thresholds
    EAR_THRESHOLD = 0.2  # Threshold for detecting a blink
    SUSPICIOUS_BLINK_RATIO = 0.05  # Expected ratio of frames with blinks (5%)
    
    # Frame array for storing data
    frames_below_threshold = 0

    print(f"Processing video with {total_frames} frames...")
    start_time = time.time()

    # Create a small plot area for the EAR graph
    plot_height = 100
    plot_width = 200
    
    # Disclaimer text
    disclaimer = "DISCLAIMER: No deepfake detection technology can be 100% accurate."
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            frames_per_second = frame_count / elapsed
            estimated_time = (total_frames - frame_count) / frames_per_second
            print(f"Processed {frame_count}/{total_frames} frames. Est. time remaining: {estimated_time:.2f}s")

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        # Initialize EAR for this frame
        current_ear = 0

        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)

            # Extract coordinates for left and right eyes
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES])

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Average the EAR
            current_ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below blink threshold
            if current_ear < EAR_THRESHOLD:
                frames_below_threshold += 1
            
            # Draw eye landmarks if requested
            if show_landmarks:
                # Draw left eye
                for i in range(0, len(left_eye)-1):
                    cv2.line(frame, tuple(left_eye[i]), tuple(left_eye[i+1]), (0, 255, 0), 1)
                cv2.line(frame, tuple(left_eye[-1]), tuple(left_eye[0]), (0, 255, 0), 1)
                
                # Draw right eye
                for i in range(0, len(right_eye)-1):
                    cv2.line(frame, tuple(right_eye[i]), tuple(right_eye[i+1]), (0, 255, 0), 1)
                cv2.line(frame, tuple(right_eye[-1]), tuple(right_eye[0]), (0, 255, 0), 1)

        # Store EAR value and frame number
        ear_values.append(current_ear)
        frame_numbers.append(frame_count)
        
        # Keep only the most recent values for the moving window
        if len(ear_values) > ear_window_size:
            ear_values.pop(0)
            frame_numbers.pop(0)

        # Create small plot of EAR values
        if len(ear_values) > 1:
            # Create a white background for the plot
            plot_bg = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            
            # Normalize the EAR values to fit in the plot
            min_ear = min(0.1, min(ear_values)) if ear_values else 0.1
            max_ear = max(0.4, max(ear_values)) if ear_values else 0.4
            normalized_ears = [(ear - min_ear) / (max_ear - min_ear) * (plot_height - 20) for ear in ear_values]
            
            # Draw the EAR threshold line
            threshold_y = int((1 - (EAR_THRESHOLD - min_ear) / (max_ear - min_ear)) * (plot_height - 20))
            cv2.line(plot_bg, (0, threshold_y), (plot_width, threshold_y), (255, 0, 0), 1)
            
            # Draw the EAR values
            for i in range(len(normalized_ears) - 1):
                pt1 = (int(i * plot_width / len(normalized_ears)), plot_height - 10 - int(normalized_ears[i]))
                pt2 = (int((i + 1) * plot_width / len(normalized_ears)), plot_height - 10 - int(normalized_ears[i + 1]))
                cv2.line(plot_bg, pt1, pt2, (0, 0, 255), 2)
            
            # Place the plot in the top right corner of the frame
            frame[10:10+plot_height, width-10-plot_width:width-10] = plot_bg
        
        # Add metrics text to the frame with black outline
        blink_ratio = frames_below_threshold / frame_count if frame_count > 0 else 0
        
        # Draw text with black outline
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 0, 255)  # Red
        outline_color = (0, 0, 0)  # Black
        
        # EAR value text
        draw_text_with_outline(
            frame, f"EAR: {current_ear:.3f}", (10, 30), 
            font, 0.7, text_color, outline_color, 2, 4
        )
        
        # Blink ratio text
        draw_text_with_outline(
            frame, f"Blink ratio: {blink_ratio:.3f}", (10, 60),
            font, 0.7, text_color, outline_color, 2, 4
        )
        
        # Add deepfake likelihood indicator
        deepfake_likelihood = "LOW" if blink_ratio >= SUSPICIOUS_BLINK_RATIO else "HIGH"
        indicator_color = (0, 255, 0) if deepfake_likelihood == "LOW" else (0, 0, 255)  # Green for LOW, Red for HIGH
        
        draw_text_with_outline(
            frame, f"Deepfake likelihood: {deepfake_likelihood}", (10, 90),
            font, 0.7, indicator_color, outline_color, 2, 4
        )
        
        # Add disclaimer at the bottom of the frame
        # Calculate position to center the text
        text_size = cv2.getTextSize(disclaimer, font, 0.6, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 20  # 20 pixels from the bottom
        
        draw_text_with_outline(
            frame, disclaimer, (text_x, text_y),
            font, 0.6, (255, 255, 255), outline_color, 2, 4  # White text with black outline
        )
        
        # Write frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    
    # Calculate final statistics
    final_blink_ratio = frames_below_threshold / frame_count if frame_count > 0 else 0
    deepfake_likelihood = "LOW" if final_blink_ratio >= SUSPICIOUS_BLINK_RATIO else "HIGH"
    
    print(f"\nProcessing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Frames with blinks: {frames_below_threshold}")
    print(f"Blink ratio: {final_blink_ratio:.3f}")
    print(f"Deepfake likelihood: {deepfake_likelihood}")
    print(f"Output saved to: {output_video_path}")

if __name__ == "__main__":
    # Replace with your input and output paths
    #input_video = "/Users/davidhawthorne/Downloads/Kennedy.mp4"
    input_video = "/Users/davidhawthorne/Downloads/hacker_dude_short.mp4"
    output_video = "output_with_ear.mp4"
    
    # NOTE: You must download the facial landmark predictor from:
    # https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    # Extract it and place it in the same directory as this script
    
    process_video(input_video, output_video)