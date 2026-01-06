import cv2
from ultralytics import YOLOv10
from pathlib import Path
from tqdm import tqdm
import argparse
import os

def run_yolov10n_on_sequence(
    img_dir, 
    output_txt, 
    model_path="yolov10n.pt", 
    conf=0.01, 
    device="cuda", 
    save_video=False, 
    video_out_path=None
):
    model = YOLOv10(model_path)
    model.to(device)

    img_files = sorted(list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png")))
    if not img_files:
        raise FileNotFoundError(f"No images found in {img_dir}. Check the path and file extensions.")
    if save_video:
        example_img = cv2.imread(str(img_files[0]))
        height, width = example_img.shape[:2]
        video_out_path = video_out_path or os.path.join(os.path.dirname(output_txt), "vis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_out_path, fourcc, 30, (width, height))

    with open(output_txt, "w") as f:
        for idx, img_file in enumerate(tqdm(img_files, desc="Detecting")):
            img = cv2.imread(str(img_file))
            results = model.predict(img, conf=conf, device=device, verbose=False)
            # Draw boxes if visualizing
            for result in results:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_score = float(box.conf[0])
                    width_box = x2 - x1
                    height_box = y2 - y1
                    # MOT format
                    line = f"{idx+1},-1,{x1:.2f},{y1:.2f},{width_box:.2f},{height_box:.2f},{conf_score:.6f},-1,-1,-1\n"
                    f.write(line)
                    if save_video:
                        x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(img, (x1_, y1_), (x2_, y2_), (0,255,0), 2)
                        cv2.putText(
                            img, f"{conf_score:.2f}", 
                            (x1_, max(0, y1_-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2
                        )
            if save_video:
                video_writer.write(img)
    if save_video:
        video_writer.release()
        print(f"✓ Visualization video saved to {video_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--output-txt", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="yolov10n.pt")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-out-path", type=str, default=None)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    run_yolov10n_on_sequence(
        args.img_dir, 
        args.output_txt,
        model_path=args.model_path,
        conf=args.conf,
        device=args.device,
        save_video=args.save_video,
        video_out_path=args.video_out_path
    )

