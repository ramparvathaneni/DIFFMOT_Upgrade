from ultralytics import YOLOv10
import cv2
import os
from tqdm import tqdm
from pathlib import Path

def run_yolov10n_on_sequence(img_dir, output_txt, model_path="yolov10n.pt", conf=0.01, device="cuda"):
    model = YOLOv10(model_path)
    model.to(device)

    img_files = sorted(list(Path(img_dir).glob("*.jpg")))
    with open(output_txt, "w") as f:
        for idx, img_file in enumerate(tqdm(img_files, desc="Detecting")):
            img = cv2.imread(str(img_file))
            results = model.predict(img, conf=conf, device=device, verbose=False)
            for result in results:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_score = float(box.conf[0])
                    width = x2 - x1
                    height = y2 - y1
                    # MOT format: frame,id,x,y,w,h,conf,-1,-1,-1
                    line = f"{idx+1},-1,{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf_score:.6f},-1,-1,-1\n"
                    f.write(line)

if __name__ == "__main__":
    img_dir = "./DanceTrack/test1/dancetrack0003/img1"
    output_txt = "detections/sequence_name/det.txt"
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    run_yolov10n_on_sequence(img_dir, output_txt)

