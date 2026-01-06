"""
Extract ReID features from YOLOv10n detections using torchreid
Compatible with DiffMOT pipeline
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys

# Add torchreid to path
sys.path.insert(0, 'external/deep-person-reid')

from torchreid.utils import FeatureExtractor

def load_detections(det_file):
    """Load detections from MOT format det.txt"""
    detections = defaultdict(list)
    with open(det_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6])
            
            # Filter low confidence detections
            if conf > 0.3:
                detections[frame_id].append([x, y, w, h])
    
    return detections

def extract_reid_features_simple(img_dir, det_file, output_npy, model_name='osnet_x1_0', device='cuda'):
    """
    Extract ReID features using torchreid models
    
    Args:
        img_dir: Path to image directory (img1 folder)
        det_file: Path to detection file (det.txt)
        output_npy: Output path for .npy file
        model_name: torchreid model name (e.g., 'osnet_x1_0', 'resnet50')
        device: 'cuda' or 'cpu'
    """
    
    print(f"Loading model: {model_name}")
    
    # Initialize FeatureExtractor from torchreid
    try:
        extractor = FeatureExtractor(
            model_name=model_name,
            device=device
        )
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Available models may be: osnet_x1_0, resnet50, resnet101, etc.")
        return
    
    # Load detections
    print(f"Loading detections from {det_file}")
    detections = load_detections(det_file)
    print(f"Loaded {len(detections)} frames with detections")
    
    # Get image files
    img_files = sorted(list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png')))
    print(f"Found {len(img_files)} image files")
    
    if len(img_files) == 0:
        print(f"ERROR: No images found in {img_dir}")
        return
    
    # Extract features frame by frame
    reid_features = {}
    
    for frame_idx, img_file in enumerate(tqdm(img_files, desc="Extracting ReID features"), start=1):
        
        if frame_idx not in detections:
            continue
        
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not read {img_file}")
            continue
        
        # Extract crops for all detections in this frame
        crops = []
        for x, y, w, h in detections[frame_idx]:
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            # Ensure valid crop bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                crops.append(crop)
        
        # Extract features for this frame's crops
        if len(crops) > 0:
            try:
                features = extractor(crops)  # Returns tensor of shape (N, feature_dim)
                reid_features[frame_idx] = features.cpu().numpy()
            except Exception as e:
                print(f"Error extracting features for frame {frame_idx}: {e}")
                continue
    
    # Save features
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    np.save(output_npy, reid_features, allow_pickle=True)
    print(f"\n✓ ReID features saved to {output_npy}")
    print(f"  Total frames with features: {len(reid_features)}")
    
    # Print sample feature shape
    if len(reid_features) > 0:
        sample_frame = list(reid_features.keys())[0]
        sample_features = reid_features[sample_frame]
        print(f"  Sample: Frame {sample_frame} has {len(sample_features)} detections")
        print(f"  Feature shape per detection: {sample_features[0].shape}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract ReID features from YOLOv10n detections')
    parser.add_argument('--img-dir', type=str, required=True, help="Path to img1 folder")
    parser.add_argument('--det-file', type=str, required=True, help="Path to det.txt")
    parser.add_argument('--output', type=str, required=True, help="Output .npy file path")
    parser.add_argument('--model', type=str, default='osnet_x1_0', help="ReID model name")
    parser.add_argument('--device', type=str, default='cuda', help="Device: cuda or cpu")
    
    args = parser.parse_args()
    
    extract_reid_features_simple(
        img_dir=args.img_dir,
        det_file=args.det_file,
        output_npy=args.output,
        model_name=args.model,
        device=args.device
    )

