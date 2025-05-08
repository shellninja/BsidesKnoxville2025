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
    # Handle grayscale images by converting to RGB if needed
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Preprocess the frame
    frame = preprocess_frame(frame)
    
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
        
        # 5. Jawline consistency analysis
        jawline_score = analyze_jawline_consistency(frame_rgb, landmarks_points)
        
        # 6. Texture consistency analysis (new)
        texture_score = analyze_texture_consistency(frame_rgb, face, landmarks_points)
        
        # Calculate overall detection confidence with balanced approach
        # Using weighted average with optimized weights
        raw_score = (
            edge_score * 0.1 +            # Edge consistency (low weight)
            color_score * 0.2 +           # Color consistency (medium weight)
            landmark_score * 0.1 +       # Facial landmarks (higher weight)
            blending_score * 0.15 +       # Blending artifacts (medium weight)
            jawline_score * 0.15 +        # Jawline consistency (medium weight)
            texture_score * 0.15          # Texture consistency (medium weight)
        )
        
        # Apply a mild bias adjustment for balance
        bias_adjustment = 0.1  # Moderate bias to reduce false positives
        overall_score = max(0.0, raw_score - bias_adjustment)
        
        results[f"face_{i}"] = {
            "edge_consistency_score": edge_score,
            "color_consistency_score": color_score,
            "landmark_naturalness_score": landmark_score,
            "blending_quality_score": blending_score,
            "jawline_consistency_score": jawline_score,
            "texture_consistency_score": texture_score,
            "overall_deepfake_probability": overall_score,
            "is_likely_deepfake": overall_score > 0.5,  # Balanced threshold
            "face_coords": (face.left(), face.top(), face.width(), face.height())
        }
    
    return results

def preprocess_frame(frame):
    """Preprocessing to normalize the image before analysis"""
    # Normalize brightness and contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return normalized

def analyze_edge_consistency(image, face, landmarks_points):
    """Detect unnatural edges that might indicate mask boundaries"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection with higher thresholds to detect only stronger edges
    # Original was (50, 150), higher values mean fewer edges detected
    edges = cv2.Canny(gray, 80, 180)
    
    # Create face mask using landmarks
    mask = np.zeros_like(gray)
    face_polygon = np.array(landmarks_points, dtype=np.int32)
    cv2.fillConvexPoly(mask, face_polygon, 255)
    
    # Get edge pixels around the face boundary (dilate mask slightly)
    boundary = cv2.dilate(mask, np.ones((5,5), np.uint8)) - cv2.erode(mask, np.ones((5,5), np.uint8))
    boundary_edges = cv2.bitwise_and(edges, boundary)
    
    # Calculate edge density along boundary
    edge_density = np.sum(boundary_edges > 0) / np.sum(boundary > 0) if np.sum(boundary > 0) > 0 else 0
    
    # Normalize score (higher means more likely to be a deepfake) with greatly reduced sensitivity
    return min(1.0, edge_density * 3.0)  # Significantly reduced from 6.0 to 3.0

def analyze_color_consistency(image, face, landmarks_points):
    """Analyze color consistency between face and surrounding areas"""
    # Create face mask
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    face_polygon = np.array(landmarks_points, dtype=np.int32)
    cv2.fillConvexPoly(mask, face_polygon, 255)
    
    # Create a smaller inner face mask to avoid edge effects
    inner_mask = cv2.erode(mask, np.ones((5,5), np.uint8))
    
    # Create band around face (not too far from face)
    outer_mask = cv2.dilate(mask, np.ones((15,15), np.uint8)) - mask
    
    # Get face and surrounding areas
    face_area = cv2.bitwise_and(image, image, mask=inner_mask)
    surrounding_area = cv2.bitwise_and(image, image, mask=outer_mask)
    
    # Convert to LAB color space which better represents human color perception
    face_lab = cv2.cvtColor(face_area, cv2.COLOR_RGB2LAB)
    surrounding_lab = cv2.cvtColor(surrounding_area, cv2.COLOR_RGB2LAB)
    
    # Calculate mean color values for inner face and surrounding area
    face_mean = cv2.mean(face_lab, mask=inner_mask)[:3]  # Ignore alpha
    surrounding_mean = cv2.mean(surrounding_lab, mask=outer_mask)[:3]  # Ignore alpha
    
    # Calculate color histograms (RGB is better for histogram comparison)
    face_hist = cv2.calcHist([face_area], [0, 1, 2], inner_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    surrounding_hist = cv2.calcHist([surrounding_area], [0, 1, 2], outer_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    cv2.normalize(face_hist, face_hist)
    cv2.normalize(surrounding_hist, surrounding_hist)
    
    # Calculate histogram difference (higher = more likely fake)
    hist_diff = cv2.compareHist(face_hist, surrounding_hist, cv2.HISTCMP_CHISQR)
    hist_score = min(1.0, hist_diff / 15.0)
    
    # Calculate Euclidean distance between mean LAB values
    # L*a*b* was designed so that equal distances in the space correspond to equal perceived color differences
    lab_diff = np.sqrt(sum((face_mean[i] - surrounding_mean[i])**2 for i in range(3)))
    
    # Natural faces have somewhat smooth color transitions to surrounding areas
    # But not completely uniform (people have different skin tones than background)
    # Unrealistically large or small differences are suspicious
    lab_score = 0.0
    if lab_diff < 5.0:  # Suspiciously similar colors
        lab_score = (5.0 - lab_diff) / 5.0 * 0.5  # Less suspicious than too different
    elif lab_diff > 30.0:  # Suspiciously different colors
        lab_score = min(1.0, (lab_diff - 30.0) / 20.0)
    else:
        lab_score = 0.0  # Natural color difference
    
    # Final score combining both metrics
    # Camera quality and lighting can affect color metrics significantly
    # so we reduce the overall weight to avoid false positives
    combined_score = (hist_score * 0.4) + (lab_score * 0.6)
    return min(0.5, combined_score * 0.7)

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
    
    # Deepfake-specific check: unnatural eye shapes/positions
    # Get bounding box of each eye
    left_eye_x = [landmarks[i][0] for i in range(36, 42)]
    left_eye_y = [landmarks[i][1] for i in range(36, 42)]
    right_eye_x = [landmarks[i][0] for i in range(42, 48)]
    right_eye_y = [landmarks[i][1] for i in range(42, 48)]
    
    # Calculate aspect ratios of eyes (width/height)
    left_eye_width = max(left_eye_x) - min(left_eye_x)
    left_eye_height = max(left_eye_y) - min(left_eye_y)
    right_eye_width = max(right_eye_x) - min(right_eye_x)
    right_eye_height = max(right_eye_y) - min(right_eye_y)
    
    left_eye_ratio = left_eye_width / left_eye_height if left_eye_height > 0 else 0
    right_eye_ratio = right_eye_width / right_eye_height if right_eye_height > 0 else 0
    
    # Check for unnaturally consistent eye shapes (too similar to each other)
    eye_ratio_diff = abs(left_eye_ratio - right_eye_ratio)
    
    # Most people have slightly different eye shapes - too identical is suspicious
    eye_ratio_score = 0.0
    if eye_ratio_diff < 0.1:  # Very similar eye shapes
        eye_ratio_score = 0.7 * (1.0 - (eye_ratio_diff / 0.1))
    else:
        eye_ratio_score = 0.0
    
    # Check for unnatural symmetry in facial features
    # Calculate distances from midline to landmark pairs
    midline_x = nose_tip[0]
    
    # Check left/right facial feature distances (eyebrows, mouth corners)
    left_brow = landmarks[19]  # Left eyebrow
    right_brow = landmarks[24]  # Right eyebrow
    left_mouth = landmarks[48]  # Left mouth corner
    right_mouth = landmarks[54]  # Right mouth corner
    
    left_brow_dist = abs(midline_x - left_brow[0])
    right_brow_dist = abs(midline_x - right_brow[0])
    left_mouth_dist = abs(midline_x - left_mouth[0])
    right_mouth_dist = abs(midline_x - right_mouth[0])
    
    # Calculate relative asymmetry ratios
    brow_asymmetry = abs(left_brow_dist - right_brow_dist) / max(left_brow_dist, right_brow_dist) if max(left_brow_dist, right_brow_dist) > 0 else 0
    mouth_asymmetry = abs(left_mouth_dist - right_mouth_dist) / max(left_mouth_dist, right_mouth_dist) if max(left_mouth_dist, right_mouth_dist) > 0 else 0
    
    # Real faces have some natural asymmetry - too symmetrical is suspicious
    symmetry_score = 0.0
    if brow_asymmetry < 0.05 and mouth_asymmetry < 0.05:  # Unnaturally symmetrical
        symmetry_score = 0.8 * (1.0 - ((brow_asymmetry + mouth_asymmetry) / 0.1))
    else:
        symmetry_score = 0.0
    
    # Combine all metrics with balanced weights
    combined_score = (
        (eye_level_diff / 30.0) * 0.1 +           # 10% weight
        (proportion_diff * 1.2) * 0.2 +           # 20% weight
        eye_ratio_score * 0.4 +                   # 40% weight (deepfake specific)
        symmetry_score * 0.3                      # 30% weight (deepfake specific)
    )
    
    # Cap but with a higher maximum to better detect deepfakes
    return min(0.8, combined_score)

def detect_blending_artifacts(image, face):
    """Detect blending artifacts that might indicate mask edges"""
    # Extract the face region
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_region = image[y:y+h, x:x+w]
    
    # Apply Gaussian blur to reduce noise before analysis
    face_region_blurred = cv2.GaussianBlur(face_region, (5, 5), 0)
    
    # Convert to different color spaces to detect inconsistencies
    hsv = cv2.cvtColor(face_region_blurred, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(face_region_blurred, cv2.COLOR_RGB2LAB)
    
    # Apply Laplacian filter to detect sharp transitions using 2nd channel of LAB
    # This focuses more on color transitions than brightness
    laplacian = cv2.Laplacian(lab[:,:,1], cv2.CV_64F)
    
    # Calculate statistics
    mean_lap = np.mean(np.abs(laplacian))
    std_hsv = np.std(hsv[:,:,1])  # Saturation channel
    
    # Combine metrics with greatly reduced sensitivity
    score = (mean_lap / 10.0) + (std_hsv / 60.0)  # Significantly reduced sensitivity
    return min(1.0, score * 0.8)  # Further reduce with a scaling factor

def analyze_jawline_consistency(image, landmarks_points):
    """Specifically analyze the jawline area for mask artifacts"""
    # Extract jawline landmarks (points 0-16 in dlib's 68 point model)
    jawline = np.array(landmarks_points[0:17], dtype=np.int32)
    
    # Create a mask just for the jawline area with some padding
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Draw slightly thicker line along jawline
    for i in range(len(jawline)-1):
        cv2.line(mask, tuple(jawline[i]), tuple(jawline[i+1]), 255, 5)
    
    # Dilate to get surrounding area
    jawline_area = cv2.dilate(mask, np.ones((10,10), np.uint8))
    
    # Convert to grayscale and get gradients
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Use a larger kernel size (5 instead of 3) to smooth out gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Only look at the jawline area
    jawline_gradients = cv2.bitwise_and(sobel_mag.astype(np.uint8), jawline_area)
    
    # Calculate statistics
    mean_gradient = np.mean(jawline_gradients[jawline_area > 0]) if np.sum(jawline_area > 0) > 0 else 0
    
    # Check for unnatural gradient patterns
    # Apply Gaussian blur to smooth the gradient image for analysis
    jawline_gradients_blurred = cv2.GaussianBlur(jawline_gradients, (5, 5), 0)
    
    # Calculate variance of gradients - too consistent is suspicious
    gradient_variance = np.var(jawline_gradients_blurred[jawline_area > 0]) if np.sum(jawline_area > 0) > 0 else 0
    
    # Calculate score based on both mean and variance
    # Too low variance is suspicious (too smooth transitions)
    variance_score = 0.0
    if gradient_variance < 20.0:  # Suspiciously uniform gradient
        variance_score = (20.0 - gradient_variance) / 20.0
    else:
        variance_score = 0.0  # Natural variation
    
    # Higher mean gradient can indicate unnatural transitions
    mean_score = min(1.0, mean_gradient / 100.0)
    
    # Combine scores with more emphasis on variance (unnatural smoothness)
    combined_score = (mean_score * 0.3) + (variance_score * 0.7)
    
    # Cap the final score at 0.4 to reduce false positives
    return min(0.4, combined_score)

def analyze_texture_consistency(image, face, landmarks_points):
    """Analyze skin texture for unnatural smoothness or repetitive patterns common in deepfakes"""
    # Extract the face region
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_region = image[y:y+h, x:x+w]
    
    # Create a mask for just the skin areas (excluding eyes, eyebrows, mouth)
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    face_polygon = np.array(landmarks_points, dtype=np.int32)
    cv2.fillConvexPoly(mask, face_polygon, 255)
    
    # Create masks for features to exclude
    features_mask = np.zeros_like(mask)
    
    # Exclude eyes
    left_eye = np.array(landmarks_points[36:42], dtype=np.int32)
    right_eye = np.array(landmarks_points[42:48], dtype=np.int32)
    cv2.fillConvexPoly(features_mask, left_eye, 255)
    cv2.fillConvexPoly(features_mask, right_eye, 255)
    
    # Exclude mouth
    mouth = np.array(landmarks_points[48:60], dtype=np.int32)
    cv2.fillConvexPoly(features_mask, mouth, 255)
    
    # Create skin-only mask
    skin_mask = cv2.bitwise_and(mask, cv2.bitwise_not(features_mask))
    skin_mask_roi = skin_mask[y:y+h, x:x+w]  # Region of interest for the face area
    
    # Convert face to grayscale for texture analysis
    if face_region.size == 0:  # Check if face_region is empty
        return 0.0
    
    gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
    # 1. Check for artificial smoothness using local binary patterns (simplified)
    # Apply Gaussian blur and compare with original
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blurred)
    
    # Calculate texture variation statistics
    texture_mean = np.mean(diff[skin_mask_roi > 0]) if np.sum(skin_mask_roi > 0) > 0 else 0
    texture_std = np.std(diff[skin_mask_roi > 0]) if np.sum(skin_mask_roi > 0) > 0 else 0
    
    # Deepfakes often have unnaturally smooth skin or repetitive texture patterns
    smoothness_score = 0.0
    if texture_std < 3.0:  # Unnaturally consistent texture
        smoothness_score = 0.7 * (1.0 - (texture_std / 3.0))
    elif texture_mean < 2.0:  # Unnaturally smooth (too little texture)
        smoothness_score = 0.6 * (1.0 - (texture_mean / 2.0))
    else:
        smoothness_score = 0.0
    
    # 2. Check for repeating patterns using auto-correlation
    # Apply Laplacian for edge enhancement to highlight texture
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Compute gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(sobelx, sobely)
    
    # Calculate gradient statistics
    gradient_std = np.std(gradient_mag[skin_mask_roi > 0]) if np.sum(skin_mask_roi > 0) > 0 else 0
    
    # Many deepfakes have artificially low variance in skin texture
    pattern_score = 0.0
    if gradient_std < 5.0:  # Suspicious lack of natural texture variation
        pattern_score = 0.6 * (1.0 - (gradient_std / 5.0))
    else:
        pattern_score = 0.0
    
    # Combine scores with emphasis on the stronger indicator
    final_score = max(smoothness_score, pattern_score) * 0.7 + min(smoothness_score, pattern_score) * 0.3
    
    return min(0.8, final_score)

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
        
        draw_text_with_outline(frame, f"Jawline: {metrics['jawline_consistency_score']:.2f}", 
                         (panel_x + 20, panel_y + y_offset), 0.5)
        y_offset += 20
        
        draw_text_with_outline(frame, f"Texture: {metrics['texture_consistency_score']:.2f}", 
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