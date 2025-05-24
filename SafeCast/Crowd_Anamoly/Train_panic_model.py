import kagglehub
import os
import json
import cv2
import numpy as np
import sys
import glob
from collections import deque

# Import our panic detection system
# Make sure panic_crowd_monitoring.py is in the same directory
from panic_crowd_monitoring import PanicCrowdMonitoring, DatasetTrainer

def download_dataset():
    """Download the Kaggle dataset if not already downloaded."""
    print("Downloading multiview-highdensity-anomalous-crowd dataset...")
    path = kagglehub.dataset_download("angelchi56/multiview-highdensity-anomalous-crowd")
    print(f"Dataset downloaded to: {path}")
    return path

def prepare_dataset(dataset_path):
    """
    Prepare the dataset for training.
    This function will scan the dataset directory and create annotations if needed.
    """
    print(f"Preparing dataset from {dataset_path}...")
    
    # Check dataset structure
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
    
    print(f"Found {len(video_files)} video files in dataset")
    
    # Check if annotations exist, if not, create a simple annotation file
    annotations_file = os.path.join(dataset_path, "annotations.json")
    
    if not os.path.exists(annotations_file):
        print("No annotations found. Creating a simple annotation file based on directory structure...")
        
        annotations = {}
        
        # Detect videos that contain panic based on folder names or filenames
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            
            # Simplified logic - assume videos in folders/with names containing certain keywords are panic videos
            panic_keywords = ['panic', 'stampede', 'anomaly', 'abnormal', 'crowd', 'emergency']
            
            is_panic = any(keyword in video_path.lower() for keyword in panic_keywords)
            
            if is_panic:
                # For panic videos, assume the entire video contains panic behavior
                # In a real-world scenario, you'd need more accurate annotations
                video_duration = get_video_duration(video_path)
                
                annotations[video_name] = {
                    "panic_segments": [[0, video_duration]],
                    "type": "panic"
                }
            else:
                annotations[video_name] = {
                    "panic_segments": [],
                    "type": "normal"
                }
        
        # Save annotations
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=4)
        
        print(f"Created annotations file: {annotations_file}")
    else:
        print(f"Found existing annotations file: {annotations_file}")
    
    dataset_info = {
        "path": dataset_path,
        "videos": video_files,
        "annotations_file": annotations_file
    }
    
    return dataset_info

def get_video_duration(video_path):
    """Get the duration of a video in seconds."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return 60  # Default to 60 seconds if can't open
        
        # Get frame count and FPS
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 60
        
        cap.release()
        return duration
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 60  # Default to 60 seconds on error

def load_annotations(annotations_file):
    """Load annotations from file."""
    with open(annotations_file, 'r') as f:
        return json.load(f)

def train_model(dataset_info, output_model_path="model_params.json"):
    """
    Train the panic detection model using the dataset.
    """
    print("Starting model training...")
    
    # Load annotations
    annotations = load_annotations(dataset_info["annotations_file"])
    
    # Create a custom dataset structure for our trainer
    # The trainer expects a list of items with video_path and annotations
    dataset = []
    
    for video_path in dataset_info["videos"]:
        video_name = os.path.basename(video_path)
        
        if video_name in annotations:
            dataset.append({
                "video_path": video_path,
                "annotations": annotations[video_name]
            })
        else:
            print(f"Warning: No annotations found for {video_name}, skipping")
    
    print(f"Prepared {len(dataset)} videos with annotations for training")
    
    # Create trainer instance
    trainer = DatasetTrainer(dataset_path=dataset_info["path"])
    
    # Override the load_dataset method to use our prepared dataset
    original_load_dataset = trainer.load_dataset
    trainer.load_dataset = lambda: dataset
    
    # Train the model
    model_params = trainer.train()
    
    # Save model parameters
    trainer.save_model_params(output_model_path)
    
    print(f"Training complete. Model parameters saved to {output_model_path}")
    return model_params

def main():
    """Main function to download dataset and train model."""
    # Download dataset
    dataset_path = download_dataset()
    
    # Prepare dataset
    dataset_info = prepare_dataset(dataset_path)
    
    # Train model
    output_model_path = "panic_model_params.json"
    model_params = train_model(dataset_info, output_model_path)
    
    print("\nTraining completed successfully!")
    print(f"Optimized model parameters: {model_params}")
    print(f"To use this model with the panic detection system:")
    print(f"python panic_crowd_monitoring.py --camera 0 --model {output_model_path}")

if __name__ == "__main__":
    main()