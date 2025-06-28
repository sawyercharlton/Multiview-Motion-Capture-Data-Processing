import cv2
import os


video_path = r"C:\Users\yuboh\GitHub\data\RecSync0\Standalone\20250626_190845.mp4"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Failed to open video: {video_path}")

# Get video properties
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = frame_count / fps if fps > 0 else 0

print(f"Video path     : {video_path}")
print(f"Frame count    : {frame_count}")
print(f"Frame rate (FPS): {fps:.2f}")
print(f"Duration (sec) : {duration:.2f}")

cap.release()
