# PIPELINE.md - Technical Architecture Documentation

## Complete Technical Architecture Guide

This document provides detailed technical explanation of the DiffMOT + YOLOv10n tracking pipeline architecture.

---

## 📋 Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Detection Pipeline](#detection-pipeline)
3. [Feature Extraction Pipeline](#feature-extraction-pipeline)
4. [Tracking Pipeline](#tracking-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Data Formats](#data-formats)
7. [Configuration Parameters](#configuration-parameters)

---

## Pipeline Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Video Input                          │
│              (DanceTrack Sequences)                     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│         STAGE 1: Detection (YOLOv10n)                   │
│  - Load video frames from img1/ directory               │
│  - Run YOLOv10n person detector                         │
│  - Generate bounding boxes with confidence scores       │
│  - Output: detections/dancetrack/<seq>/det.txt          │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│         STAGE 2: Feature Extraction (FastReID)          │
│  - Crop detected persons from frames                    │
│  - Extract 512-dim appearance embeddings                │
│  - Use OSNet backbone                                   │
│  - Output: cache/embeddings/<seq>/*.pkl                 │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│         STAGE 3: Multi-Object Tracking (DiffMOT)        │
│  - Frame-by-frame association                           │
│  - D2MP motion prediction                               │
│  - Track management (birth, death, update)              │
│  - Output: results/<seq>.txt                            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│         STAGE 4: Visualization (Optional)               │
│  - Draw bounding boxes on frames                        │
│  - Add track IDs                                        │
│  - Generate MP4 videos                                  │
│  - Output: videos/<seq>.mp4                             │
└─────────────────────────────────────────────────────────┘
```

---

## Detection Pipeline

### YOLOv10n Architecture

**Model**: YOLOv10n (nano variant)
- Parameters: ~2.8M
- Input: 640×640 RGB images
- Output: Person bounding boxes with confidence scores

### Detection Process

```python
# Pseudo-code for detection pipeline
for each_sequence in test_sequences:
    for each_frame in sequence:
        # 1. Load frame
        image = load_frame(frame_path)
        
        # 2. Preprocess
        input_tensor = preprocess(image)  # Resize to 640×640, normalize
        
        # 3. Run detection
        detections = yolov10n_model(input_tensor)
        
        # 4. Post-process
        boxes = non_max_suppression(detections, conf_threshold=0.1)
        
        # 5. Filter person class (class_id = 0)
        person_boxes = filter_class(boxes, class_id=0)
        
        # 6. Save to det.txt
        save_detections(person_boxes, output_file)
```

### Detection Output Format

File: `detections/dancetrack/<sequence_name>/det.txt`

Format: MOT standard (CSV)
```
frame_id, track_id, x, y, w, h, confidence, -1, -1, -1
```

Example:
```
1,-1,245.2,156.8,89.3,234.5,0.95,-1,-1,-1
1,-1,567.1,189.2,92.1,241.3,0.87,-1,-1,-1
2,-1,247.5,158.2,90.1,235.8,0.94,-1,-1,-1
```

Fields:
- `frame_id`: 1-indexed frame number
- `track_id`: -1 (not assigned yet)
- `x, y`: Top-left corner coordinates
- `w, h`: Width and height
- `confidence`: Detection confidence (0-1)
- Last 3 fields: Not used (-1)

---

## Feature Extraction Pipeline

### FastReID with OSNet Backbone

**Architecture**: OSNet (Omni-Scale Network)
- Parameters: 2.2M
- Input: Cropped person images (256×128)
- Output: 512-dimensional embedding vectors

### Feature Extraction Process

```python
# Pseudo-code for feature extraction
for each_sequence in test_sequences:
    # Load all detections
    detections = load_detections(det_file)
    
    # Group by frame
    frames_data = group_by_frame(detections)
    
    for frame_id, frame_detections in frames_data:
        # 1. Load frame image
        image = load_frame(frame_path)
        
        # 2. Crop detected persons
        crops = []
        for detection in frame_detections:
            crop = extract_crop(image, detection.bbox)
            crops.append(crop)
        
        # 3. Batch process crops
        crop_batch = preprocess_crops(crops)  # Resize to 256×128
        
        # 4. Extract features
        embeddings = fastreid_model(crop_batch)  # Shape: [N, 512]
        
        # 5. L2 normalize
        embeddings = l2_normalize(embeddings)
        
        # 6. Save embeddings
        save_embeddings(embeddings, cache_file)
```

### Feature Cache Format

File: `cache/embeddings/<sequence_name>/<frame_id>.pkl`

Structure (Python pickle):
```python
{
    'frame_id': 1,
    'embeddings': numpy.array([N, 512]),  # N detections
    'boxes': numpy.array([N, 4]),         # Bounding boxes
    'scores': numpy.array([N])            # Confidence scores
}
```

---

## Tracking Pipeline

### DiffMOT Architecture

**Components**:
1. **Data Association**: Match detections to existing tracks
2. **D2MP Motion Predictor**: Diffusion-based motion prediction
3. **Track Management**: Create, update, and terminate tracks

### Tracking Process

```python
# Pseudo-code for tracking
tracker = DiffMOT(config)
tracks = []  # Active tracks

for frame_id in range(1, num_frames + 1):
    # 1. Load detections and embeddings
    detections = load_frame_detections(frame_id)
    embeddings = load_frame_embeddings(frame_id)
    
    # 2. Predict track positions (D2MP)
    if len(tracks) > 0:
        predicted_boxes = d2mp_predict(tracks, frame_id)
    
    # 3. Data association
    matches, unmatched_dets, unmatched_tracks = associate(
        detections, tracks, embeddings, predicted_boxes
    )
    
    # 4. Update matched tracks
    for track_id, det_id in matches:
        tracks[track_id].update(detections[det_id], embeddings[det_id])
    
    # 5. Create new tracks for unmatched detections
    for det_id in unmatched_dets:
        if detections[det_id].score > high_threshold:
            new_track = Track(detections[det_id], embeddings[det_id])
            tracks.append(new_track)
    
    # 6. Handle unmatched tracks
    for track_id in unmatched_tracks:
        tracks[track_id].mark_missed()
        if tracks[track_id].time_since_update > max_age:
            tracks.remove(track_id)
    
    # 7. Output results
    save_tracking_results(frame_id, tracks)
```

### Data Association Details

**Cost Matrix Computation**:
```python
# Appearance cost (embedding similarity)
appearance_cost = 1 - cosine_similarity(track_embeddings, det_embeddings)

# Motion cost (IoU-based)
motion_cost = 1 - iou(predicted_boxes, detection_boxes)

# Combined cost
total_cost = w_appearance * appearance_cost + w_motion * motion_cost
```

**Matching Algorithm**: Hungarian algorithm (linear assignment)

### D2MP Motion Prediction

**Diffusion Model for Motion Prediction**:
- Input: Historical track trajectory (last 5-10 frames)
- Output: Predicted bounding box for next frame
- Training: Learns to denoise corrupted motion trajectories

```python
# D2MP prediction
def d2mp_predict(track_history, target_frame):
    # 1. Extract historical trajectory
    trajectory = track_history[-10:]  # Last 10 frames
    
    # 2. Add noise (diffusion forward process)
    noisy_trajectory = add_gaussian_noise(trajectory)
    
    # 3. Denoise (diffusion reverse process)
    for t in reversed(range(diffusion_steps)):
        noisy_trajectory = denoise_step(noisy_trajectory, t)
    
    # 4. Extract predicted box
    predicted_box = noisy_trajectory[-1]
    return predicted_box
```

---

## Training Pipeline

### Training D2MP Model

**Training Data**: Prepared from ground truth tracks

```python
# Training process
d2mp_model = D2MP(config)
optimizer = Adam(lr=0.0001)

for epoch in range(800):
    for batch in train_dataloader:
        # 1. Get trajectory sequences
        trajectories = batch['trajectories']  # Shape: [B, T, 4]
        
        # 2. Add noise (diffusion forward)
        t = random.randint(0, diffusion_steps)
        noisy_trajectories = add_noise(trajectories, t)
        
        # 3. Predict noise
        predicted_noise = d2mp_model(noisy_trajectories, t)
        
        # 4. Compute loss
        loss = mse_loss(predicted_noise, actual_noise)
        
        # 5. Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 6. Validation
    if epoch % 20 == 0:
        validate(d2mp_model, val_dataloader)
        save_checkpoint(d2mp_model, epoch)
```

### Training Data Preparation

Script: `prepare_trackers_gt.py`

**Process**:
1. Load ground truth annotations
2. Extract track trajectories
3. Convert to YOLO format (normalized coordinates)
4. Save per-track trajectory files

**Output Format**: `DanceTrack/trackers_gt/<seq>/img1/<track_id>.txt`

```
0 frame_id norm_cx norm_cy norm_w norm_h visibility
```

Example:
```
0 1 0.456 0.512 0.089 0.234 1.0
0 2 0.459 0.515 0.090 0.236 1.0
0 3 0.462 0.518 0.091 0.237 1.0
```

---

## Data Formats

### Input: DanceTrack Dataset Structure

```
DanceTrack/
├── train1/
│   ├── dancetrack0001/
│   │   ├── img1/
│   │   │   ├── 00000001.jpg
│   │   │   ├── 00000002.jpg
│   │   │   └── ...
│   │   ├── gt/
│   │   │   └── gt.txt
│   │   └── seqinfo.ini
│   ├── dancetrack0002/
│   └── ...
├── val/
│   └── (same structure)
└── test1/
    └── (same structure, no gt/)
```

### Intermediate: Detection Files

`detections/dancetrack/<seq>/det.txt`
```
frame_id,track_id,x,y,w,h,conf,-1,-1,-1
```

### Intermediate: Feature Cache

`cache/embeddings/<seq>/<frame_id>.pkl`
```python
{
    'embeddings': np.array([N, 512]),
    'boxes': np.array([N, 4]),
    'scores': np.array([N])
}
```

### Output: Tracking Results

`results/<seq>.txt`
```
frame_id,track_id,x,y,w,h,1.0,-1,-1,-1
```

---

## Configuration Parameters

### Testing Configuration

`configs/yolov10n_dancetrack.yaml`

```yaml
# Mode
eval_mode: true

# Dataset
dataset: dancetrack
data_root: DanceTrack/test1

# Detection
det_thresh: 0.1          # Minimum detection confidence

# Tracking
high_thres: 0.6          # High confidence threshold for new tracks
low_thres: 0.4           # Low confidence threshold
new_track_thresh: 0.6    # Threshold to create new track
track_buffer: 30         # Max frames to keep lost track
match_thresh: 0.8        # Matching threshold

# Association weights
w_assoc_emb: 2.2         # Weight for appearance similarity
w_assoc_mot: 0.8         # Weight for motion similarity

# Feature extraction
batch_size: 1024         # Batch size for ReID
```

### Training Configuration

`configs/yolov10n_dancetrack_train.yaml`

```yaml
# Mode
eval_mode: false

# Dataset
dataset: dancetrack
data_root: DanceTrack/train1
val_root: DanceTrack/val

# Training
epochs: 800
batch_size: 2048
lr: 0.0001
weight_decay: 0.0001
eval_every: 20

# D2MP parameters
diffusion_steps: 100
trajectory_length: 10
noise_schedule: linear
```

---

## Performance Characteristics

### Speed Benchmarks

| Component | FPS | Time (test set) | Hardware |
|-----------|-----|----------------|----------|
| YOLOv10n Detection | 40-50 | 30-60 min | RTX 3090 |
| FastReID Extraction | 100+ | 30-60 min | RTX 3090 |
| DiffMOT Tracking | 50-100 | 10-20 min | RTX 3090 |
| D2MP Training | - | 2-4 hours | RTX 3090 |

### Memory Requirements

| Stage | GPU Memory | RAM |
|-------|------------|-----|
| Detection | 4-6 GB | 8 GB |
| Feature Extraction | 6-8 GB | 16 GB |
| Tracking | 4-6 GB | 16 GB |
| Training | 10-12 GB | 32 GB |

---

## Implementation Notes

### Key Design Decisions

1. **Two-stage processing**: Separate detection and tracking for flexibility
2. **Feature caching**: Precompute embeddings to speed up tracking
3. **Relative paths**: All paths relative to project root for portability
4. **Batch processing**: Batch ReID extraction for efficiency

### Optimization Tips

1. Increase `batch_size` for faster ReID extraction (if GPU memory allows)
2. Use lower `det_thresh` to get more detections (trade-off with false positives)
3. Adjust `w_assoc_emb` to balance appearance vs motion importance
4. Tune `track_buffer` based on occlusion duration in your dataset

---

## References

- **DiffMOT**: https://github.com/Kroery/DiffMOT
- **YOLOv10**: https://github.com/THU-MIG/yolov10
- **FastReID**: https://github.com/Megvii-BaseDetection/FastReID
- **DanceTrack**: https://dancetrack.github.io/

---

**Last Updated**: January 7, 2026

For more information, see [README.md](README.md) and [REFERENCES.md](REFERENCES.md)
