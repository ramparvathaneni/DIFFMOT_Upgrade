"""
Convert extracted ReID features from .npy to pickle cache format
Compatible with DiffMOT's EmbeddingComputer
"""

import os
import numpy as np
import pickle
from pathlib import Path

def main():
    REID_DIR = 'reid_embeddings'
    DATASET = 'dancetrack'
    
    # Get all sequence directories
    reid_path = Path(REID_DIR)
    sequences = sorted([d.name for d in reid_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(sequences)} sequences with ReID features")
    print(f"Converting to pickle cache format...\n")
    
    for seq_name in sequences:
        npy_file = reid_path / seq_name / 'features.npy'
        pickle_file = reid_path / f"{seq_name}_embedding.pkl"
        
        if not npy_file.exists():
            print(f"⚠ Skipping {seq_name}: {npy_file} not found")
            continue
        
        try:
            # Load numpy features (dictionary: frame_id -> embeddings)
            features_dict = np.load(npy_file, allow_pickle=True).item()
            
            # Convert to pickle cache format
            # The cache format expects: tag (seq:frame) -> embeddings
            cache = {}
            for frame_id, embs in features_dict.items():
                tag = f"{seq_name}:{frame_id}"
                cache[tag] = embs
            
            # Save as pickle
            with open(pickle_file, 'wb') as f:
                pickle.dump(cache, f)
            
            num_frames = len(cache)
            print(f"✓ {seq_name}: {num_frames} frames cached")
            print(f"  Saved to: {pickle_file}")
            
        except Exception as e:
            print(f"✗ {seq_name}: Failed to convert - {e}")
            continue
    
    print(f"\n✓ All embeddings converted to pickle cache!")

if __name__ == '__main__':
    main()
