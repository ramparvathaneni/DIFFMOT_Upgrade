"""
Batch ReID Feature Extraction for all DanceTrack sequences
Uses extract_reid_features.py for each sequence
"""

import os
import subprocess
from pathlib import Path
import sys

def main():
    # Configuration
    DATASET_ROOT = 'DanceTrack/test1'
    DET_DIR = 'detections'
    REID_OUT_DIR = 'reid_embeddings'
    MODEL = 'osnet_x1_0'
    DEVICE = 'cuda'
    
    # Get all sequences
    dataset_path = Path(DATASET_ROOT)
    sequences = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(sequences)} sequences to process")
    print(f"Sequences: {sequences}\n")
    
    successful = 0
    failed = 0
    
    for seq_name in sequences:
        img_dir = dataset_path / seq_name / 'img1'
        det_file = Path(DET_DIR) / seq_name / 'det.txt'
        output_npy = Path(REID_OUT_DIR) / seq_name / 'features.npy'
        
        # Check if required files exist
        if not img_dir.exists():
            print(f"⚠ Skipping {seq_name}: {img_dir} not found")
            failed += 1
            continue
        
        if not det_file.exists():
            print(f"⚠ Skipping {seq_name}: {det_file} not found")
            failed += 1
            continue
        
        # Check if already processed
        if output_npy.exists():
            print(f"✓ {seq_name}: Already processed (skipping)")
            successful += 1
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {seq_name}")
        print(f"{'='*70}")
        print(f"Images: {img_dir}")
        print(f"Detections: {det_file}")
        print(f"Output: {output_npy}")
        
        cmd = [
            'python', 'extract_reid_features.py',
            '--img-dir', str(img_dir),
            '--det-file', str(det_file),
            '--output', str(output_npy),
            '--model', MODEL,
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
        print(f"\n✓ All ReID features extracted successfully!")
    else:
        print(f"\n⚠ Some sequences failed. Check errors above.")

if __name__ == '__main__':
    main()
