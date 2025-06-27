import cv2
import yaml
import os
from tqdm import tqdm


def load_video_path(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['match']['video_path_1']


def crop_video(video_path, crop_region=(800, 580, 580, 40)):
    x, y, w, h = crop_region

    # Parse directory and filename
    original_dir = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)

    # Create output path like: original_dir/name_cropped.ext
    output_path = os.path.join(original_dir, f"{name}_cropped{ext}")

    # Abort if output already exists
    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}. Aborting.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    with tqdm(total=total_frames, desc="Cropping video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cropped_frame = frame[y:y+h, x:x+w]
            out.write(cropped_frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Cropped video saved to: {output_path}")


if __name__ == "__main__":
    config_file = "config.yaml"  # Change this if needed
    video_path = load_video_path(config_file)
    crop_video(video_path)
