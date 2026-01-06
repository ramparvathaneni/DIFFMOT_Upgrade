"""
Batch YOLOv10n Detection Generation for all DanceTrack sequences
Uses yolov10detections.py for each sequence
"""

import os
import subprocess
from pathlib import Path
import sys

def main():
    # Configuration
    DATASET_ROOT = 'DanceTrack/test1'
    DET_OUT_DIR = 'detections'
    MODEL_PATH = 'yolov10n.pt'  # Path to YOLOv10n model weights
    DEVICE = 'cuda'
    CONF_THRESHOLD = 0.01
    
    # Create output directory
    Path(DET_OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get all sequences
    dataset_path = Path(DATASET_ROOT)
    sequences = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(sequences)} sequences to process")
    print(f"Model: {MODEL_PATH}")
    print(f"Sequences: {sequences}\n")
    
    successful = 0
    failed = 0
    
    for seq_name in sequences:
        img_dir = dataset_path / seq_name / 'img1'
        output_txt = Path(DET_OUT_DIR) / seq_name / 'det.txt'
        
        # Check if required files exist
        if not img_dir.exists():
            print(f"⚠ Skipping {seq_name}: {img_dir} not found")
            failed += 1
            continue
        
        # Check if already processed
        if output_txt.exists():
            print(f"✓ {seq_name}: Already processed (skipping)")
            successful += 1
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {seq_name}")
        print(f"{'='*70}")
        print(f"Images: {img_dir}")
        print(f"Output: {output_txt}")
        
        cmd = [
            'python', 'yolov10detections.py',
            '--img-dir', str(img_dir),
            '--output-txt', str(output_txt),
            '--model-path', MODEL_PATH,
            '--conf', str(CONF_THRESHOLD),
            '--device', DEVICE
        ]
        
        try:
            result = subprocess.run(cmd, check=True)
            print(f"✓ {seq_name}: Success")
            successful += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ {seq_name}: Failed with error code {e.returncode}")
            failed += 1
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Successful: {successful}")
    print(f"Failed/Skipped: {failed}")
    print(f"Total: {len(sequences)}")
    
    if failed == 0:
        print(f"\n✓ All detections generated successfully!")
        print(f"Detections saved to: {DET_OUT_DIR}/")
    else:
        print(f"\n⚠ Some sequences failed. Check errors above.")

if __name__ == '__main__':
    main()
