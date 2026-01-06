"""
Visualize tracking results by drawing bounding boxes on video
"""

import cv2
import numpy as np
from pathlib import Path
import os

def visualize_tracking(seq_name, img_dir, result_file, output_video):
    """
    Draw tracking results on images and save as video
    """
    
    # Load tracking results
    results = {}
    try:
        with open(result_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = int(float(parts[2])), int(float(parts[3])), \
                             int(float(parts[4])), int(float(parts[5]))
                
                if frame_id not in results:
                    results[frame_id] = []
                results[frame_id].append((track_id, x, y, w, h))
    except Exception as e:
        print(f"Error reading {result_file}: {e}")
        return
    
    # Get image files
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    
    if len(img_files) == 0:
        print(f"No images found in {img_dir}")
        return
    
    print(f"  Found {len(img_files)} images, {len(results)} frames with tracks")
    
    # Read first image to get video dimensions
    first_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    if first_img is None:
        print(f"Could not read first image")
        return
        
    height, width = first_img.shape[:2]
    
    # Create output directory
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 25.0, (width, height))
    
    # Process each frame
    for frame_idx, img_file in enumerate(img_files, start=1):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Draw tracking results
        if frame_idx in results:
            for track_id, x, y, w, h in results[frame_idx]:
                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw track ID
                text = f"ID: {track_id}"
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
        
        # Add frame number
        frame_text = f"Frame: {frame_idx}"
        cv2.putText(img, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2)
        
        out.write(img)
        
        if frame_idx % 100 == 0:
            print(f"    Frame {frame_idx}")
    
    out.release()
    print(f"✓ Video saved: {output_video}")

def main():
    DATASET_ROOT = 'DanceTrack/test1'
    RESULTS_DIR = 'results/yolov10n_diffmot'
    VIDEO_OUT_DIR = 'videos/yolov10n_diffmot'
    
    Path(VIDEO_OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get all sequences from results directory (more reliable)
    result_files = sorted(Path(RESULTS_DIR).glob('*.txt'))
    
    if len(result_files) == 0:
        print(f"No result files found in {RESULTS_DIR}")
        return
    
    print(f"Found {len(result_files)} tracking result files\n")
    
    for result_file in result_files:
        seq_name = result_file.stem  # Remove .txt extension
        img_dir = Path(DATASET_ROOT) / seq_name / 'img1'
        output_video = Path(VIDEO_OUT_DIR) / f"{seq_name}.mp4"
        
        if not img_dir.exists():
            print(f"⚠ Skipping {seq_name}: no images at {img_dir}")
            continue
        
        print(f"Processing {seq_name}...")
        visualize_tracking(seq_name, str(img_dir), str(result_file), str(output_video))

if __name__ == '__main__':
    main()
