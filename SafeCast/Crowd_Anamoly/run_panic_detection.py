import cv2
import numpy as np
import time
import datetime
import threading
import json
import os
import argparse

# Import our panic detection system
from panic_crowd_monitoring import PanicCrowdMonitoring

def run_panic_detection(model_path, camera_id=0, video_path=None, output_dir=None):
    """
    Run the panic detection system with trained model parameters.
    
    Args:
        model_path: Path to the trained model parameters JSON file
        camera_id: Camera ID to use (default: 0, for webcam)
        video_path: Optional path to video file instead of camera
        output_dir: Optional directory to save alert evidence
    """
    # Load model parameters if available
    model_params = {}
    try:
        with open(model_path, 'r') as f:
            model_params = json.load(f)
        print(f"Loaded model parameters from {model_path}")
    except FileNotFoundError:
        print(f"Model parameter file {model_path} not found, using defaults")
    
    # Create evidence directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Alert evidence will be saved to {output_dir}")
    else:
        output_dir = "evidence"
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the monitor with our trained parameters
    monitor = PanicCrowdMonitoring(
        camera_id=video_path if video_path else camera_id,
        motion_threshold=model_params.get('motion_threshold', 2500),
        density_threshold=model_params.get('density_threshold', 40),
        panic_detection_frames=model_params.get('panic_detection_frames', 15),
        # flow_threshold=model_params.get('flow_threshold', 5.0)
    )
    
    # Override the send_alert method to save evidence to our specified directory
    original_send_alert = monitor.send_alert
    
    def custom_send_alert(frame, metadata):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        alert_id = f"panic_alert_{timestamp}"
        
        # Save evidence
        evidence_path = os.path.join(output_dir, f"{alert_id}.jpg")
        cv2.imwrite(evidence_path, frame)
        
        # Create alert payload
        alert = {
            "alert_id": alert_id,
            "alert_type": "panic_crowd",
            "timestamp": timestamp,
            "camera_id": monitor.camera_id,
            "evidence_path": evidence_path,
            "metadata": metadata
        }
        
        # Save metadata alongside the image
        metadata_path = os.path.join(output_dir, f"{alert_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(alert, f, indent=4)
        
        print(f"\n🚨 ALERT: Panic crowd behavior detected!")
        print(f"Time: {timestamp}")
        print(f"Camera/Video: {monitor.camera_id}")
        print(f"Evidence saved: {evidence_path}")
        print(f"Metadata saved: {metadata_path}")
        print(f"Density: {metadata['density_percentage']:.1f}%")
        print(f"Motion level: {metadata['motion_level']}")
        print(f"Average speed: {metadata['avg_speed']:.1f}")
        print(f"Is divergent: {metadata['is_divergent']}")
        print(f"Direction counts: {metadata['direction_counts']}")
        
        # Update last alert time to prevent alert spam
        monitor.last_alert_time = time.time()
    
    # Replace the send_alert method
    monitor.send_alert = custom_send_alert
    
    print("\n=== SafeCasts Panic Crowd Monitoring ===")
    print(f"Using {'video file: ' + video_path if video_path else 'camera ID: ' + str(camera_id)}")
    print("Press 'q' to quit")
    
    # Run the monitor
    monitor.run()

def main():
    """Parse command line arguments and run the system."""
    parser = argparse.ArgumentParser(description='SafeCasts Panic Crowd Monitoring')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--video', type=str, help='Path to video file (instead of camera)')
    parser.add_argument('--model', type=str, default='panic_model_params.json', 
                       help='Path to model parameters')
    parser.add_argument('--output', type=str, help='Directory to save alert evidence')
    
    args = parser.parse_args()
    
    run_panic_detection(
        model_path=args.model,
        camera_id=args.camera,
        video_path=args.video,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()