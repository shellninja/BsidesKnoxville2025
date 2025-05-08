import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

class FaceMeshVisualizer:
    """
    A class to visualize facial meshes used in deepfake creation
    using MediaPipe's Face Mesh model.
    """
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # MediaPipe Face Mesh uses 468 landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define specific face regions for visualization
        self.FACE_CONNECTIONS = self.mp_face_mesh.FACEMESH_TESSELATION
        self.CONTOURS = self.mp_face_mesh.FACEMESH_CONTOURS
        self.LIPS = self.mp_face_mesh.FACEMESH_LIPS
        self.LEFT_EYE = self.mp_face_mesh.FACEMESH_LEFT_EYE
        self.RIGHT_EYE = self.mp_face_mesh.FACEMESH_RIGHT_EYE
        self.LEFT_EYEBROW = self.mp_face_mesh.FACEMESH_LEFT_EYEBROW
        self.RIGHT_EYEBROW = self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW
        
        # Drawing specifications
        self.mesh_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1)
        self.contour_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=1)
    
    def process_video(self, input_video_path, output_video_path, show_3d=True, show_2d=True):
        """
        Process a video file to extract and visualize facial mesh
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save the output video
            show_3d: Whether to show a 3D visualization
            show_2d: Whether to show a 2D visualization
        """
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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Set up 3D plot if needed
        if show_3d:
            plt.ion()
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
        # Process each frame
        frame_count = 0
        start_time = time.time()
        print(f"Processing video with {total_frames} frames...")
        
        # Add a disclaimer
        disclaimer = "DISCLAIMER: No deepfake detection/creation technology can be 100% accurate."
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                frames_per_second = frame_count / elapsed
                estimated_time = (total_frames - frame_count) / frames_per_second
                print(f"Processed {frame_count}/{total_frames} frames. Est. time remaining: {estimated_time:.2f}s")
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_mesh.process(rgb_frame)
            
            # Draw annotations
            annotated_frame = frame.copy()
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get the 3D coordinates
                    landmarks_3d = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                    
                    # Draw 2D mesh on the frame
                    if show_2d:
                        # Draw face mesh
                        self.mp_drawing.draw_landmarks(
                            image=annotated_frame,
                            landmark_list=face_landmarks,
                            connections=self.FACE_CONNECTIONS,
                            landmark_drawing_spec=self.mesh_drawing_spec,
                            connection_drawing_spec=self.mesh_drawing_spec)
                        
                        # Draw contours for emphasis
                        self.mp_drawing.draw_landmarks(
                            image=annotated_frame,
                            landmark_list=face_landmarks,
                            connections=self.CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.contour_drawing_spec)
                    
                    # Draw 3D mesh in a separate window
                    if show_3d and frame_count % 5 == 0:  # Update 3D plot every 5 frames for performance
                        self._visualize_3d_mesh(ax, landmarks_3d)
            
            # Add text explaining what's happening
            # Draw text with black outline for visibility
            self._draw_text_with_outline(
                annotated_frame, 
                "Facial Mesh Extraction for Deepfakes", 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                (0, 0, 0), 
                2, 
                3
            )
            
            self._draw_text_with_outline(
                annotated_frame, 
                "Detecting 468 facial landmarks in 3D space", 
                (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                (0, 0, 0), 
                2, 
                3
            )
            
            if results.multi_face_landmarks:
                self._draw_text_with_outline(
                    annotated_frame, 
                    "Face detected and mesh created", 
                    (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    (0, 0, 0), 
                    2, 
                    3
                )
            else:
                self._draw_text_with_outline(
                    annotated_frame, 
                    "No face detected", 
                    (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 255), 
                    (0, 0, 0), 
                    2, 
                    3
                )
                
            # Add disclaimer at the bottom
            text_size = cv2.getTextSize(disclaimer, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = (annotated_frame.shape[1] - text_size[0]) // 2
            text_y = annotated_frame.shape[0] - 20
            
            self._draw_text_with_outline(
                annotated_frame, 
                disclaimer, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                (0, 0, 0), 
                2, 
                3
            )
            
            # Write frame to output video
            out.write(annotated_frame)
            
            # Show frame
            cv2.imshow("Facial Mesh Visualization", annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        if show_3d:
            plt.close(fig)
            plt.ioff()
            
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Output saved to: {output_video_path}")
    
    def _draw_text_with_outline(self, img, text, position, font, font_scale, text_color, outline_color, thickness=1, outline_thickness=2):
        """Helper function to draw text with an outline for better visibility"""
        # Draw the outline (stroke)
        cv2.putText(img, text, position, font, font_scale, outline_color, outline_thickness)
        # Draw the inner text
        cv2.putText(img, text, position, font, font_scale, text_color, thickness)
    
    def _visualize_3d_mesh(self, ax, landmarks_3d):
        """Visualize the 3D mesh in a matplotlib 3D plot"""
        ax.clear()
        
        # Extract x, y, z coordinates
        x = landmarks_3d[:, 0]
        y = landmarks_3d[:, 1]
        z = landmarks_3d[:, 2]
        
        # Adjust the coordinates for better visualization
        x = -x  # Flip x for a better view
        
        # Plot the points
        ax.scatter(x, y, z, c='g', marker='o', s=10)
        
        # Draw connections for better visualization
        for connection in self.CONTOURS:
            idx1, idx2 = connection
            ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]], 'b-', linewidth=1)
        
        # Set equal aspect ratio
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Facial Landmarks')
        
        # Set the same scale for all axes
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.draw()
        plt.pause(0.001)  # Pause to update the plot

    def extract_and_save_mesh(self, input_image_path, output_obj_path):
        """
        Extract face mesh from an image and save as OBJ file
        Args:
            input_image_path: Path to input image
            output_obj_path: Path to save the OBJ file
        """
        # Read image
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error: Could not read image {input_image_path}")
            return
            
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            print("No face detected in the image")
            return
            
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract 3D landmarks
        landmarks_3d = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
        
        # Save as OBJ file
        with open(output_obj_path, 'w') as f:
            # Write vertices
            for i, (x, y, z) in enumerate(landmarks_3d):
                # Scale to typical OBJ size and adjust coordinates for OBJ format
                x = x * 100
                y = -y * 100  # Flip y for OBJ
                z = -z * 100  # Flip z for OBJ
                f.write(f"v {x} {y} {z}\n")
            
            # Write faces (triangles)
            # MediaPipe uses a specific topology - we'll use the FACE_CONNECTIONS
            for connection in self.FACE_CONNECTIONS:
                # Convert to 1-indexed for OBJ format
                idx1, idx2 = connection[0] + 1, connection[1] + 1
                # Since we only have edges defined, we'll create small triangles
                if idx1 < idx2:  # Avoid duplicates
                    # Find a third point to form a triangle
                    # This is a simplification and won't create a perfect mesh
                    for idx3 in range(1, len(landmarks_3d) + 1):
                        if idx3 != idx1 and idx3 != idx2:
                            # Check if idx3 has connections to both idx1 and idx2
                            # This is a simple heuristic and can be improved
                            if (idx1-1, idx3-1) in self.FACE_CONNECTIONS or (idx3-1, idx1-1) in self.FACE_CONNECTIONS:
                                if (idx2-1, idx3-1) in self.FACE_CONNECTIONS or (idx3-1, idx2-1) in self.FACE_CONNECTIONS:
                                    f.write(f"f {idx1} {idx2} {idx3}\n")
                                    break
            
        print(f"Mesh saved to {output_obj_path}")


if __name__ == "__main__":
    # Initialize the visualizer
    visualizer = FaceMeshVisualizer()
    
    # Define paths
    video_path = 0  # Use 0 for webcam or replace with your video file path
    output_path = "output_mesh_visualization.mp4"
    
    # Check if the input video exists
    if not os.path.exists(video_path):
        print(f"Error: Input video not found at {video_path}")
        print("Please update the video_path variable to point to your video file.")
    else:
        # Process the video
        visualizer.process_video(video_path, output_path, show_3d=True, show_2d=True)
        
    # You can also extract mesh from a single image
    # image_path = "input_image.jpg"  # Replace with your image path
    # output_obj = "face_mesh.obj"
    # visualizer.extract_and_save_mesh(image_path, output_obj)