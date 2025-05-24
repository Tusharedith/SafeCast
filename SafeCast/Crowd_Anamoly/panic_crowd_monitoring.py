import cv2
import numpy as np
import time
import datetime
import threading
import requests
import json
from collections import deque


class PanicCrowdMonitoring:
    def __init__(self, camera_id=0, motion_threshold=2500, density_threshold=40, 
                 panic_detection_frames=15, alert_cooldown=60):
        """
        Initialize the Panic Crowd Monitoring system.
        
        Args:
            camera_id: Camera identifier (0 for default webcam)
            motion_threshold: Threshold for motion detection sensitivity
            density_threshold: Percentage of frame that must contain movement to indicate high density
            panic_detection_frames: Number of consecutive frames needed to confirm panic
            alert_cooldown: Time in seconds between alerts to prevent spam
        """
        self.camera_id = camera_id
        self.motion_threshold = motion_threshold
        self.density_threshold = density_threshold
        self.panic_detection_frames = panic_detection_frames
        self.alert_cooldown = alert_cooldown
          
        # State variables
        self.is_running = False
        self.last_alert_time = 0
        self.consecutive_panic_frames = 0
        self.motion_history = deque(maxlen=30)  # Store recent motion values
        self.baseline_motion = None
        
        # Video capture
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False)
        
        # Optical flow parameters for directional movement
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # For optical flow analysis
        self.prev_gray = None
        self.prev_points = None
        
    def initialize_camera(self):
        """Initialize the camera connection."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera with ID {self.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read frame from camera")
        
        # Convert to grayscale for optical flow
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect features to track
        self.prev_points = cv2.goodFeaturesToTrack(
            self.prev_gray, mask=None, **self.feature_params)
        
    def detect_motion(self, frame):
        """
        Detect motion in the frame using background subtraction.
        
        Returns:
            tuple: (motion_mask, motion_level, density_percentage)
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate motion level (sum of white pixels)
        motion_level = np.sum(fg_mask == 255)
        
        # Calculate density percentage
        total_pixels = self.frame_width * self.frame_height
        density_percentage = (motion_level / total_pixels) * 100
        
        return fg_mask, motion_level, density_percentage
    
    def analyze_flow_direction(self, frame):
        """
        Analyze optical flow to detect directional movement patterns.
        
        Returns:
            dict: Flow analysis results with direction counts and average speed
        """
        if self.prev_points is None or len(self.prev_points) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_gray = gray
            return {"direction_counts": {}, "avg_speed": 0, "is_divergent": False}
        
        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params)
        
        # Filter only valid points
        if next_points is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return {"direction_counts": {}, "avg_speed": 0, "is_divergent": False}
            
        good_old = self.prev_points[status == 1]
        good_new = next_points[status == 1]
        
        # Calculate flow vectors
        flow_vectors = []
        direction_counts = {"up": 0, "down": 0, "left": 0, "right": 0}
        speeds = []
        center_x = self.frame_width / 2
        center_y = self.frame_height / 2
        
        # Check if people are moving away from a central point (divergence)
        away_from_center = 0
        toward_center = 0
        
        for i, (old, new) in enumerate(zip(good_old, good_new)):
            x1, y1 = old.ravel()
            x2, y2 = new.ravel()
            
            # Calculate vector
            dx = x2 - x1
            dy = y2 - y1
            
            # Calculate speed (magnitude of the vector)
            speed = np.sqrt(dx*dx + dy*dy)
            speeds.append(speed)
            
            # Count directions
            if abs(dx) > abs(dy):  # Horizontal movement is stronger
                direction_counts["right" if dx > 0 else "left"] += 1
            else:  # Vertical movement is stronger
                direction_counts["down" if dy > 0 else "up"] += 1
                
            # Check if moving away from or toward center
            vec_to_center_x = center_x - x1
            vec_to_center_y = center_y - y1
            dot_product = dx * vec_to_center_x + dy * vec_to_center_y
            if dot_product < 0:  # Moving away from center
                away_from_center += 1
            else:
                toward_center += 1
        
        # Calculate if movement is divergent (people moving away from center)
        is_divergent = away_from_center > 1.5 * toward_center and away_from_center > 10
        
        # Update previous points and gray image
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        
        return {
            "direction_counts": direction_counts,
            "avg_speed": np.mean(speeds) if speeds else 0,
            "is_divergent": is_divergent,
            "away_from_center": away_from_center,
            "toward_center": toward_center
        }
    
    def is_panic_situation(self, motion_level, density_percentage, flow_analysis):
        """
        Determine if current conditions indicate a panic situation.
        
        Returns:
            bool: True if panic is detected, False otherwise
        """
        # If motion baseline isn't established yet, update it
        if self.baseline_motion is None and len(self.motion_history) >= 20:
            self.baseline_motion = np.mean(self.motion_history)
        
        # Store current motion level for baseline calculation
        self.motion_history.append(motion_level)
        
        # Can't determine panic until we have a baseline
        if self.baseline_motion is None:
            return False
        
        # Check if motion is significantly above baseline
        motion_spike = motion_level > (self.baseline_motion * 2)
        
        # Check if density is above threshold
        high_density = density_percentage > self.density_threshold
        
        # Check if motion direction indicates panic (divergent or mainly in one direction)
        direction_counts = flow_analysis["direction_counts"]
        max_direction_count = max(direction_counts.values()) if direction_counts else 0
        total_tracked_points = sum(direction_counts.values()) if direction_counts else 0
        
        # Directional flow indicates panic if:
        # - Movement is divergent (people fleeing from a central point)
        # - OR movement is predominantly in one direction (>70% of tracked points)
        direction_panic = (
            flow_analysis["is_divergent"] or 
            (total_tracked_points > 10 and max_direction_count / total_tracked_points > 0.7)
        )
        
        # High speed can also indicate panic
        high_speed = flow_analysis["avg_speed"] > 5.0  # Threshold based on calibration
        
        # Consider panic when multiple conditions are met
        is_panic = (motion_spike and high_density) or (direction_panic and high_speed)
        
        return is_panic
    
    def send_alert(self, frame, metadata):
        """
        Send an alert to security operators with evidence.
        
        Args:
            frame: Current frame to include as evidence
            metadata: Dictionary with additional information about the event
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        alert_id = f"panic_alert_{timestamp}"
        
        # Save evidence
        evidence_path = f"evidence/{alert_id}.jpg"
        cv2.imwrite(evidence_path, frame)
        
        # Create alert payload
        alert = {
            "alert_id": alert_id,
            "alert_type": "panic_crowd",
            "timestamp": timestamp,
            "camera_id": self.camera_id,
            "evidence_path": evidence_path,
            "metadata": metadata
        }
        
        # In a real system, this would send to a central server or notification system
        # For demonstration, we'll print the alert to console
        print(f"🚨 ALERT: Panic crowd behavior detected!")
        print(f"Time: {timestamp}")
        print(f"Camera ID: {self.camera_id}")
        print(f"Evidence saved: {evidence_path}")
        print(f"Metadata: {metadata}")
        
        # In a real implementation, you would send this to your security team:
        # self.send_to_security_team(alert)
        
        # Update last alert time to prevent alert spam
        self.last_alert_time = time.time()
    
    def send_to_security_team(self, alert):
        """
        Send alert to security team's endpoint.
        In production, this would communicate with your security system API.
        """
        try:
            # Example API endpoint (not functional in this demo)
            response = requests.post(
                "https://api.safecast.internal/alerts",
                json=alert,
                headers={"Authorization": "Bearer YOUR_API_KEY"}
            )
            if response.status_code == 200:
                print("Alert sent successfully to security team")
            else:
                print(f"Failed to send alert: {response.status_code}")
        except Exception as e:
            print(f"Error sending alert: {str(e)}")
    
    def process_frame(self, frame):
        """
        Process a single frame to detect panic crowd behavior.
        
        Returns:
            tuple: (processed_frame, is_panic_detected)
        """
        # Make a copy for visualization
        display_frame = frame.copy()
        
        # Detect motion
        motion_mask, motion_level, density_percentage = self.detect_motion(frame)
        
        # Analyze flow direction
        flow_analysis = self.analyze_flow_direction(frame)
        
        # Check if current conditions indicate panic
        is_panic = self.is_panic_situation(motion_level, density_percentage, flow_analysis)
        
        # Update panic detection counter
        if is_panic:
            self.consecutive_panic_frames += 1
        else:
            self.consecutive_panic_frames = max(0, self.consecutive_panic_frames - 1)
        
        # Trigger alert if panic persists for enough frames and cooldown period passed
        alert_triggered = False
        current_time = time.time()
        if (self.consecutive_panic_frames >= self.panic_detection_frames and 
                current_time - self.last_alert_time > self.alert_cooldown):
            
            metadata = {
                "motion_level": float(motion_level),
                "density_percentage": float(density_percentage),
                "avg_speed": float(flow_analysis["avg_speed"]),
                "is_divergent": flow_analysis["is_divergent"],
                "direction_counts": flow_analysis["direction_counts"]
            }
            
            self.send_alert(frame, metadata)
            alert_triggered = True
            self.consecutive_panic_frames = 0  # Reset counter after alert
        
        # Visualize results on display frame
        self.visualize_results(
            display_frame, 
            motion_mask, 
            density_percentage, 
            flow_analysis, 
            is_panic,
            alert_triggered
        )
        
        return display_frame, is_panic or alert_triggered
    
    def visualize_results(self, frame, motion_mask, density, flow_analysis, is_panic, alert_triggered):
        """Add visualization elements to the frame."""
        # Add motion mask as overlay
        motion_overlay = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        alpha = 0.3
        cv2.addWeighted(motion_overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Display stats
        cv2.putText(frame, f"Density: {density:.1f}%", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Speed: {flow_analysis['avg_speed']:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display panic status
        status_color = (0, 0, 255) if is_panic else (0, 255, 0)
        status_text = "PANIC DETECTED" if is_panic else "Normal"
        cv2.putText(frame, status_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # If alert was triggered, show prominent warning
        if alert_triggered:
            cv2.putText(frame, "ALERT SENT", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            # Draw red border
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 10)
    
    def run(self):
        """Main processing loop."""
        if not self.is_running:
            self.is_running = True
            self.initialize_camera()
            
            try:
                while self.is_running:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read frame, exiting...")
                        break
                    
                    processed_frame, _ = self.process_frame(frame)
                    
                    # Display the resulting frame
                    cv2.imshow('SafeCasts - Panic Crowd Monitoring', processed_frame)
                    
                    # Break the loop with 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            finally:
                self.stop()
    
    def stop(self):
        """Stop processing and release resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
    def run_threaded(self):
        """Run the monitoring in a separate thread."""
        thread = threading.Thread(target=self.run)
        thread.daemon = True
        thread.start()
        return thread


class DatasetTrainer:
    """
    Class to train the panic detection model using the multi-view high-density anomalous crowd dataset.
    """
    def __init__(self, dataset_path):
        """
        Initialize the dataset trainer.
        
        Args:
            dataset_path: Path to the multi-view high-density anomalous crowd dataset
        """
        self.dataset_path = dataset_path
        self.model_params = {
            'motion_threshold': 2500,
            'density_threshold': 40,
            'panic_detection_frames': 15,
            # 'flow_threshold': 5.0
        }
        
    def load_dataset(self):
        """
        Load the dataset videos and annotations.
        Returns a list of (video_path, annotations) pairs.
        """
        # In a real implementation, this would parse the dataset structure
        # For this example, we'll return a placeholder
        print(f"Loading dataset from {self.dataset_path}...")
        
        # This is a placeholder - in real implementation, you would:
        # 1. Scan the directory for video files
        # 2. Load corresponding annotation files
        # 3. Return paired data
        
        # Placeholder for dataset structure
        dataset = [
            {
                "video_path": f"{self.dataset_path}/video1.mp4",
                "annotations": {
                    "panic_segments": [(10, 20), (45, 60)],  # (start_sec, end_sec)
                    "type": "stampede"
                }
            },
            {
                "video_path": f"{self.dataset_path}/video2.mp4",
                "annotations": {
                    "panic_segments": [(5, 15), (30, 40)],
                    "type": "overcrowding"
                }
            }
            # More videos would be included here
        ]
        
        return dataset
        
    def extract_features(self, video_path):
        """
        Extract motion and density features from a video file.
        """
        features = []
        labels = []
        
        # Create a monitoring instance for feature extraction
        monitor = PanicCrowdMonitoring(camera_id=video_path)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        # Initialize background subtractor within this context
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False)
        
        prev_gray = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for optical flow
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Initialize optical flow on first frame
                if prev_gray is None:
                    prev_gray = gray
                    prev_points = cv2.goodFeaturesToTrack(
                        gray, mask=None, **monitor.feature_params)
                    continue
                
                # Apply background subtraction
                fg_mask = bg_subtractor.apply(frame)
                
                # Clean up noise
                kernel = np.ones((5, 5), np.uint8)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                
                # Calculate motion level
                motion_level = np.sum(fg_mask == 255)
                
                # Calculate density percentage
                h, w = frame.shape[:2]
                total_pixels = h * w
                density_percentage = (motion_level / total_pixels) * 100
                
                # Calculate optical flow
                if prev_points is not None and len(prev_points) > 0:
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, prev_points, None, **monitor.lk_params)
                    
                    # Filter only valid points
                    if next_points is not None and len(next_points) > 0:
                        good_old = prev_points[status == 1]
                        good_new = next_points[status == 1]
                        
                        # Calculate flow vectors
                        speeds = []
                        for old, new in zip(good_old, good_new):
                            x1, y1 = old.ravel()
                            x2, y2 = new.ravel()
                            
                            # Calculate speed (magnitude of the vector)
                            dx = x2 - x1
                            dy = y2 - y1
                            speed = np.sqrt(dx*dx + dy*dy)
                            speeds.append(speed)
                        
                        avg_speed = np.mean(speeds) if speeds else 0
                    else:
                        avg_speed = 0
                        
                    # Update previous points
                    prev_gray = gray
                    prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **monitor.feature_params)
                else:
                    avg_speed = 0
                    prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **monitor.feature_params)
                
                # Create feature vector
                feature = {
                    'motion_level': motion_level,
                    'density_percentage': density_percentage,
                    'avg_speed': avg_speed,
                    'frame_number': frame_count,
                    'time': frame_count / fps  # Time in seconds
                }
                
                features.append(feature)
                frame_count += 1
                
        finally:
            cap.release()
        
        return features, fps
    
    def add_labels(self, features, fps, annotations):
        """
        Add labels to features based on annotations.
        """
        labels = [0] * len(features)  # Initialize all as non-panic
        
        # Mark panic segments based on annotations
        for start_sec, end_sec in annotations['panic_segments']:
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            
            for i, feature in enumerate(features):
                if start_frame <= feature['frame_number'] <= end_frame:
                    labels[i] = 1  # Mark as panic
        
        return labels
    
    def train(self):
        """
        Train the model using the dataset.
        """
        dataset = self.load_dataset()
        all_features = []
        all_labels = []
        
        for item in dataset:
            print(f"Processing {item['video_path']}...")
            
            # Extract features
            features, fps = self.extract_features(item['video_path'])
            if features is None:
                continue
                
            # Add labels
            labels = self.add_labels(features, fps, item['annotations'])
            
            all_features.extend(features)
            all_labels.extend(labels)
        
        # Now we can use the features and labels to find optimal parameters
        self.optimize_parameters(all_features, all_labels)
        
        return self.model_params
    
    def optimize_parameters(self, features, labels):
        """
        Optimize the detection parameters based on labeled data.
        """
        print("Optimizing detection parameters...")
        
        # For simplicity, we'll use a basic grid search approach
        # In a real implementation, you might use more sophisticated methods
        
        best_params = self.model_params.copy()
        best_f1 = 0
        
        # Define parameter search grid
        param_grid = {
            'motion_threshold': [1000, 2000, 3000, 4000],
            'density_threshold': [20, 30, 40, 50],
            'panic_detection_frames': [5, 10, 15, 20],
            'flow_threshold': [3.0, 5.0, 7.0, 9.0]
        }
        
        # Extract feature arrays for easier processing
        motion_levels = [f['motion_level'] for f in features]
        densities = [f['density_percentage'] for f in features]
        speeds = [f['avg_speed'] for f in features]
        
        # Simple parameter optimization (would be more sophisticated in real implementation)
        print("Finding optimal motion threshold...")
        best_params['motion_threshold'] = np.percentile(motion_levels, 85)
        
        print("Finding optimal density threshold...")
        best_params['density_threshold'] = np.percentile(densities, 85)
        
        print("Finding optimal flow threshold...")
        best_params['flow_threshold'] = np.percentile(speeds, 85)
        
        print(f"Optimized parameters: {best_params}")
        self.model_params = best_params
    
    # def save_model_params(self, output_path):
    #     """
    #     Save the optimized model parameters to a file.
    #     """
    #     import json
    #     with open(output_path, 'w') as f:
    #         json.dump(self.model_params, f, indent=4)
    #     print(f"Model parameters saved to {output_path}")
    def save_model_params(self, output_path):
        """Save model parameters to a JSON file with NumPy type conversion."""
        serializable_params = {}
        for key, value in self.model_params.items():
            if isinstance(value, (np.float32, np.float64)):
                serializable_params[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                serializable_params[key] = int(value)
            else:
                serializable_params[key] = value
    
    # Save to file        
        with open(output_path, 'w') as f:
            json.dump(serializable_params, f, indent=4)
    
        print(f"Model parameters saved to {output_path}")

def main():
    """
    Main function to run the panic crowd monitoring system.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='SafeCasts Panic Crowd Monitoring')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--train', action='store_true', help='Train using dataset')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--model', type=str, default='model_params.json', 
                       help='Path to model parameters (for loading or saving)')
    
    args = parser.parse_args()
    
    if args.train:
        if not args.dataset:
            print("Error: --dataset argument is required with --train")
            return
            
        # Train the model
        trainer = DatasetTrainer(args.dataset)
        trainer.train()
        trainer.save_model_params(args.model)
    else:
        # Load model parameters if available
        model_params = {}
        try:
            import json
            with open(args.model, 'r') as f:
                model_params = json.load(f)
            print(f"Loaded model parameters from {args.model}")
        except FileNotFoundError:
            print(f"Model parameter file {args.model} not found, using defaults")
        
        # Run panic detection on camera feed
        monitor = PanicCrowdMonitoring(
            camera_id=args.camera,
            motion_threshold=model_params.get('motion_threshold', 2500),
            density_threshold=model_params.get('density_threshold', 40),
            panic_detection_frames=model_params.get('panic_detection_frames', 15)
        )
        
        print("Starting panic crowd monitoring...")
        print("Press 'q' to quit")
        monitor.run()


if __name__ == "__main__":
    # Create directory for evidence if it doesn't exist
    import os
    os.makedirs("evidence", exist_ok=True)
    
    main()