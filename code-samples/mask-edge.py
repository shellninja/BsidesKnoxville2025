import cv2
import numpy as np
import dlib
from scipy.spatial import distance
import time
import argparse

def detect_mask_issues_in_frame(frame):
    """
    Detect potential mask issues in a single video frame
    
    Args:
        frame: Video frame to analyze
        
    Returns:
        Processed frame with detection results and confidence scores
    """
    # Convert to RGB for processing (OpenCV loads as BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Detect faces
    faces = detector(frame_rgb)
    
    results = {}
    for i, face in enumerate(faces):
        # Get facial landmarks
        landmarks = predictor(frame_rgb, face)
        landmarks_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        
        # 1. Edge detection analysis
        edge_score = analyze_edge_consistency(frame_rgb, face, landmarks_points)
        
        # 2. Color consistency analysis
        color_score = analyze_color_consistency(frame_rgb, face, landmarks_points)
        
        # 3. Facial landmark analysis
        landmark_score = analyze_facial_landmarks(landmarks_points)
        
        # 4. Blending artifact detection
        blending_score = detect_blending_artifacts(frame_rgb, face)
        
        # Calculate overall detection confidence
        overall_score = (edge_score + color_score + landmark_score + blending_score) / 4.0
        
        results[f"face_{i}"] = {
            "edge_consistency_score": edge_score,
            "color_consistency_score": color_score,
            "landmark_naturalness_score": landmark_score,
            "blending_quality_score": blending_score,
            "overall_deepfake_probability": overall_score,
            "is_likely_deepfake": overall_score > 0.65,
            "face_coords": (face.left(), face.top(), face.width(), face.height())
        }
    
    return results

def analyze_edge_consistency(image, face, landmarks_points):
    """Detect unnatural edges that might indicate mask boundaries"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Create face mask using landmarks
    mask = np.zeros_like(gray)
    face_polygon = np.array(landmarks_points, dtype=np.int32)
    cv2.fillConvexPoly(mask, face_polygon, 255)
    
    # Get edge pixels around the face boundary (dilate mask slightly)
    boundary = cv2.dilate(mask, np.ones((5,5), np.uint8)) - cv2.erode(mask, np.ones((5,5), np.uint8))
    boundary_edges = cv2.bitwise_and(edges, boundary)
    
    # Calculate edge density along boundary
    edge_density = np.sum(boundary_edges > 0) / np.sum(boundary > 0) if np.sum(boundary > 0) > 0 else 0
    
    # Normalize score (higher means more likely to be a deepfake)
    return min(1.0, edge_density * 5.0)

def analyze_color_consistency(image, face, landmarks_points):
    """Analyze color consistency between face and surrounding areas"""
    # Create face mask
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    face_polygon = np.array(landmarks_points, dtype=np.int32)
    cv2.fillConvexPoly(mask, face_polygon, 255)
    
    # Dilate to get surrounding area
    surrounding_mask = cv2.dilate(mask, np.ones((20,20), np.uint8)) - mask
    
    # Get face and surrounding areas
    face_area = cv2.bitwise_and(image, image, mask=mask)
    surrounding_area = cv2.bitwise_and(image, image, mask=surrounding_mask)
    
    # Calculate color histograms
    face_hist = cv2.calcHist([face_area], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    surrounding_hist = cv2.calcHist([surrounding_area], [0, 1, 2], surrounding_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    cv2.normalize(face_hist, face_hist)
    cv2.normalize(surrounding_hist, surrounding_hist)
    
    # Compare histograms (higher difference = more likely deepfake)
    hist_diff = cv2.compareHist(face_hist, surrounding_hist, cv2.HISTCMP_CHISQR)
    
    # Normalize score
    return min(1.0, hist_diff / 10.0)

def analyze_facial_landmarks(landmarks_points):
    """Analyze facial landmarks for unnatural positioning"""
    # Convert to numpy array for easier math
    landmarks = np.array(landmarks_points)
    
    # Calculate facial symmetry
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    nose_tip = landmarks[30]
    
    # Check if eyes are level
    eye_level_diff = abs(left_eye[1] - right_eye[1])
    
    # Check facial proportions
    eye_distance = distance.euclidean(left_eye, right_eye)
    face_height = distance.euclidean(landmarks[8], np.mean(landmarks[27:28], axis=0))
    proportion = eye_distance / face_height if face_height > 0 else 0
    
    # Typical human face proportions
    ideal_proportion = 0.4
    proportion_diff = abs(proportion - ideal_proportion)
    
    # Combine metrics into score
    score = (eye_level_diff / 10.0) + (proportion_diff * 2.0)
    return min(1.0, score)

def detect_blending_artifacts(image, face):
    """Detect blending artifacts that might indicate mask edges"""
    # Extract the face region
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_region = image[y:y+h, x:x+w]
    
    # Convert to different color spaces to detect inconsistencies
    hsv = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)
    
    # Apply Laplacian filter to detect sharp transitions
    laplacian = cv2.Laplacian(lab[:,:,0], cv2.CV_64F)
    
    # Calculate statistics
    mean_lap = np.mean(np.abs(laplacian))
    std_hsv = np.std(hsv[:,:,1])  # Saturation channel
    
    # Combine metrics
    score = (mean_lap / 10.0) + (std_hsv / 50.0)
    return min(1.0, score)

def draw_results_on_frame(frame, results):
    """Draw detection results on the frame in the bottom right corner with black outline for text"""
    h, w = frame.shape[:2]
    
    # Create panel in bottom right corner
    panel_width = w // 3
    panel_height = h // 3
    panel_x = w - panel_width - 20
    panel_y = h - panel_height - 20
    
    # Create semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                 (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add disclaimer at the top of the panel
    disclaimer = "DISCLAIMER: No deepfake detection"
    disclaimer2 = "method is 100% accurate"
    
    # Function to draw text with black outline
    def draw_text_with_outline(img, text, position, font_scale=0.6, thickness=1):
        # Draw black outline
        cv2.putText(img, text, (position[0]-1, position[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 0, 0), thickness*3, cv2.LINE_AA)
        cv2.putText(img, text, (position[0]+1, position[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 0, 0), thickness*3, cv2.LINE_AA)
        cv2.putText(img, text, (position[0], position[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 0, 0), thickness*3, cv2.LINE_AA)
        cv2.putText(img, text, (position[0], position[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 0, 0), thickness*3, cv2.LINE_AA)
        # Draw white text
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    draw_text_with_outline(frame, disclaimer, (panel_x + 10, panel_y + 20), 0.5, 1)
    draw_text_with_outline(frame, disclaimer2, (panel_x + 10, panel_y + 40), 0.5, 1)
    
    y_offset = 70
    
    if not results:
        draw_text_with_outline(frame, "No faces detected", (panel_x + 10, panel_y + y_offset))
        return frame
    
    for face_id, metrics in results.items():
        if "error" in metrics:
            draw_text_with_outline(frame, f"Error: {metrics['error']}", (panel_x + 10, panel_y + y_offset))
            y_offset += 20
            continue
        
        # Draw rectangle around detected face
        x, y, w, h = metrics["face_coords"]
        color = (0, 0, 255) if metrics["is_likely_deepfake"] else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add face number label near the face
        probability = metrics["overall_deepfake_probability"] * 100
        face_label = f"{face_id}: {probability:.1f}%"
        cv2.putText(frame, face_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add metrics to panel
        draw_text_with_outline(frame, f"Face {face_id}:", (panel_x + 10, panel_y + y_offset))
        y_offset += 20
        
        draw_text_with_outline(frame, f"Edge: {metrics['edge_consistency_score']:.2f}", 
                         (panel_x + 20, panel_y + y_offset), 0.5)
        y_offset += 20
        
        draw_text_with_outline(frame, f"Color: {metrics['color_consistency_score']:.2f}", 
                         (panel_x + 20, panel_y + y_offset), 0.5)
        y_offset += 20
        
        draw_text_with_outline(frame, f"Landmark: {metrics['landmark_naturalness_score']:.2f}", 
                         (panel_x + 20, panel_y + y_offset), 0.5)
        y_offset += 20
        
        draw_text_with_outline(frame, f"Blending: {metrics['blending_quality_score']:.2f}", 
                         (panel_x + 20, panel_y + y_offset), 0.5)
        y_offset += 20
        
        draw_text_with_outline(frame, f"Overall: {metrics['overall_deepfake_probability']:.2f}", 
                         (panel_x + 20, panel_y + y_offset), 0.5)
        y_offset += 20
        
        likelihood = "LIKELY FAKE" if metrics["is_likely_deepfake"] else "LIKELY REAL"
        color = (0, 0, 255) if metrics["is_likely_deepfake"] else (0, 255, 0)
        
        # Draw black outline
        cv2.putText(frame, likelihood, (panel_x + 20, panel_y + y_offset-1), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, likelihood, (panel_x + 20, panel_y + y_offset+1), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, likelihood, (panel_x + 20-1, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, likelihood, (panel_x + 20+1, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        
        # Draw colored text
        cv2.putText(frame, likelihood, (panel_x + 20, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        y_offset += 30
    
    # Add FPS indicator
    draw_text_with_outline(frame, f"FPS: {fps:.1f}", (panel_x + 10, panel_y + panel_height - 10), 0.5)
    
    return frame

def process_video(video_path=None):
    """Process video file or webcam feed and show results in real-time"""
    global fps
    
    # Initialize video capture from webcam if no path provided
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Use default webcam
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_count = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps_count}")
    
    # Create window
    cv2.namedWindow("Deepfake Mask Detector", cv2.WINDOW_NORMAL)
    
    # For FPS calculation
    prev_time = 0
    fps = 0
    
    while True:
        # Calculate FPS
        current_time = time.time()
        if current_time - prev_time > 0:
            fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
        
        # Detect mask issues
        results = detect_mask_issues_in_frame(frame)
        
        # Draw results on frame
        frame_with_results = draw_results_on_frame(frame, results)
        
        # Show frame
        cv2.imshow("Deepfake Mask Detector", frame_with_results)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Global variable for FPS
    global fps
    fps = 0

    
    
    parser = argparse.ArgumentParser(description='Detect mask issues in deepfake videos')
    parser.add_argument('--video_path', type=str, help='Path to the video file (optional, uses webcam if not provided)', default=None)
    args = parser.parse_args()
    
    # Run video processor
    process_video(args.video_path)