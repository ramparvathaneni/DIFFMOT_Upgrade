# tools/yolov10_to_mot_det.py
"""
Run YOLOv10 inference on per-sequence image folders and write MOTChallenge-style det files.
Usage:
 python tools/yolov10_to_mot_det.py --weights /path/to/yolov10n.pt \
    --seq_root /path/to/Dataset/train --out_dir ./detections/yolov10n --imgsz 640 --conf 0.35
Assumes seq_root has subfolders per sequence. Each sequence folder contains images (000001.jpg ...).
"""

import os, argparse
from pathlib import Path
import tqdm

# Try to import ultralytics or yolov10 API
try:
    # ultralytics style
    from ultralytics import YOLOv10
    has_ultralytics = True
except Exception:
    has_ultralytics = False

def run_inference_ultralytics(weights, img_path, imgsz, conf):
    model = YOLO(weights)  # loads weights / pretrained
    # run single-image predict (returns a results list)
    res = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, verbose=False)
    # res[0].boxes.xyxy, res[0].boxes.conf
    boxes = []
    try:
        b = res[0].boxes.xyxy.cpu().numpy()
        c = res[0].boxes.conf.cpu().numpy()
        for i in range(len(b)):
            boxes.append((float(b[i,0]), float(b[i,1]), float(b[i,2]), float(b[i,3]), float(c[i])))
    except Exception:
        # fallback if structure differs
        for r in res:
            for box in getattr(r.boxes, 'xyxy', []):
                # attempt to parse
                arr = box.cpu().numpy()
                boxes.append((float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]), 1.0))
    return boxes

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--seq_root', required=True, help='dir with subfolders per sequence containing images')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.3)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not has_ultralytics:
        print("Warning: ultralytics not detected; ensure your YOLOv10 import works (THU repo may expose a different API).")

    seqs = sorted([d for d in Path(args.seq_root).iterdir() if d.is_dir()])
    for seq in seqs:
        img_files = sorted([p for p in seq.iterdir() if p.suffix.lower() in ('.jpg','.png')])
        out_lines = []
        # frame indices must start at 1
        for f_idx, imgp in enumerate(img_files, start=1):
            if has_ultralytics:
                boxes = run_inference_ultralytics(args.weights, imgp, args.imgsz, args.conf)
            else:
                # If using THU yolov10 API, the user must adapt this call to their API.
                raise RuntimeError("No detection API available. Install ultralytics or adapt script for THU yolov10 API.")
            for (x1,y1,x2,y2,score) in boxes:
                w = x2 - x1
                h = y2 - y1
                # MOT format: frame, id, left, top, width, height, conf, -1, -1, -1
                out_lines.append(f"{f_idx},0,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n")
        out_path = os.path.join(args.out_dir, f"{seq.name}.txt")
        with open(out_path, 'w') as f:
            f.writelines(out_lines)
        print(f"Saved {len(out_lines)} dets -> {out_path}")

if __name__ == '__main__':
    main()

