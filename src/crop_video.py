import cv2
import yaml
import os
from tqdm import tqdm


def load_video_path(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['video_path_0']


def crop_video(video_path, crop_region=(1050, 560, 220, 60)):
    x, y, w, h = crop_region

    original_dir = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)

    renamed_original_path = os.path.join(original_dir, f"{name}_ori{ext}")
    cropped_output_path = os.path.join(original_dir, f"{name}{ext}")

    # If renamed file already exists, assume cropping was done
    # if os.path.exists(cropped_output_path):
    #     print(f"Cropped file already exists: {cropped_output_path}. Aborting.")
    #     return

    os.rename(video_path, renamed_original_path)
    print(f"Original file renamed to: {renamed_original_path}")

    cap = cv2.VideoCapture(renamed_original_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {renamed_original_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cropped_output_path, fourcc, fps, (w, h))

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
    print(f"Cropped video saved to: {cropped_output_path}")


if __name__ == "__main__":
    config_file = "config.yaml"  # Change this if needed
    video_path = load_video_path(config_file)
    crop_video(video_path)
