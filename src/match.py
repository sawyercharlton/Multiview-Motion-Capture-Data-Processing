import os
import yaml
import cv2
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import re
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import shutil
import os
import re
from pathlib import Path


ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.npy', '.png', '.pcd']


def extract_frames(video_path, output_dir):
    # Remove the output directory if it already exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    video_name = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    # Video length info
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        duration_sec = frame_count / fps
    else:
        duration_sec = 0
        logging.warning(f"[{video_name}] FPS is zero or unavailable. Frame count: {frame_count}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame-{frame_idx + 1}.png")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    logging.info(
        f"[{video_name}] Video length: {duration_sec:.2f} seconds "
        f"({frame_count} frames @ {fps:.2f} FPS); extracted {frame_idx} frames to {output_dir}"
    )


def format_threshold_filename(threshold_ns):
    """Convert nanoseconds to a compact filename-safe string like 30ms or 250us."""
    if threshold_ns % 1e9 == 0:
        return f"{int(threshold_ns / 1e9)}s"
    elif threshold_ns % 1e6 == 0:
        return f"{int(threshold_ns / 1e6)}ms"
    elif threshold_ns % 1e3 == 0:
        return f"{int(threshold_ns / 1e3)}us"
    else:
        return f"{threshold_ns}ns"


def parse_duration_ns(duration_str):
    """Parse human-friendly duration strings to nanoseconds."""
    units = {
        "ns": 1,
        "us": int(1e3),
        "ms": int(1e6),
        "s":  int(1e9)
    }

    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ns|us|ms|s)", duration_str.strip())
    if not match:
        raise ValueError(f"Invalid duration format: '{duration_str}'")

    value, unit = match.groups()
    return int(float(value) * units[unit])


def setup_logger(log_file_path):
    """
    Configures logging to write messages to both a file and the terminal.
    If the file exists, new messages will be appended.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def match_frames_from_csv(csv_path_left, csv_path_right, output_csv_path, threshold_ns):
    """
    Match frame timestamps from two CSV files instead of image filenames.
    Each CSV should have one timestamp per line (as integer nanoseconds).
    """
    # Load CSVs
    df_left = pd.read_csv(csv_path_left, header=None, names=["t"])
    df_right = pd.read_csv(csv_path_right, header=None, names=["t"])

    # Prepare dataframes
    left = pd.DataFrame({
        't': df_left["t"].astype(np.int64),
        'left': df_left["t"].astype(np.int64)
    })

    right = pd.DataFrame({
        't': df_right["t"].astype(np.int64),
        'right_int': df_right["t"].astype(np.int64),
        'right': df_right["t"].astype(str)
    })

    # Perform matching
    df = pd.merge_asof(
        left.sort_values('t'),
        right.sort_values('t'),
        on='t',
        tolerance=threshold_ns,
        allow_exact_matches=True,
        direction='nearest'
    )
    df = df.dropna()
    df = df.drop(['t', 'right_int'], axis=1).reset_index(drop=True)

    # Save result
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Matched {df.shape[0]} frame pairs from CSVs (threshold = {threshold_ns / 1e6:.1f} ms) → {output_csv_path}")


def match_frames_full_from_csv(csv_path_left, csv_path_right, output_csv_path, threshold_ns):
    """
    Match all timestamps from left CSV with right CSV (full match), including unmatched.
    Each CSV should have one timestamp per line (as integer nanoseconds).
    """
    # Load CSVs
    df_left = pd.read_csv(csv_path_left, header=None, names=["t"])
    df_right = pd.read_csv(csv_path_right, header=None, names=["t"])

    # Prepare dataframes
    left_df = pd.DataFrame({
        't': df_left["t"].astype(np.int64),
        'left': df_left["t"].astype(np.int64)
    })

    right_df = pd.DataFrame({
        't': df_right["t"].astype(np.int64),
        'right_int': df_right["t"].astype(np.int64),
        'right': df_right["t"].astype(str)
    })

    # Match frames with merge_asof
    matched_df = pd.merge_asof(
        left_df.sort_values('t'),
        right_df.sort_values('t'),
        on='t',
        tolerance=threshold_ns,
        allow_exact_matches=True,
        direction='nearest'
    )

    matched_df = matched_df.reset_index(drop=True)
    full_output_df = matched_df[['left', 'right']].copy()

    # Fill unmatched with empty string
    full_output_df['left'] = full_output_df['left'].astype("Int64")
    full_output_df['right'] = full_output_df['right'].fillna('').astype(str)

    # Add match status
    full_output_df["matched"] = full_output_df["right"].apply(lambda x: x != "")
    full_output_df.to_csv(output_csv_path, index=False)

    logging.info(f"Full match completed: {len(full_output_df)} entries → {output_csv_path}")

def match_frames_from_csv_multi(csv_path_left, csv_path_right_list, output_csv_path, threshold_ns):
    df_left = pd.read_csv(csv_path_left, header=None, names=["t"])
    base_df = pd.DataFrame({'t': df_left["t"].astype(np.int64), 'left': df_left["t"].astype(str)})

    for idx, csv_path_right in enumerate(csv_path_right_list):
        df_right = pd.read_csv(csv_path_right, header=None, names=["t"])
        shifted_idx = idx + 1
        right_df = pd.DataFrame({
            't': df_right["t"].astype(np.int64),
            f'right{shifted_idx}_int': df_right["t"].astype(np.int64),
            f'right{shifted_idx}': df_right["t"].astype(str)
        })

        base_df = pd.merge_asof(
            base_df.sort_values('t'),
            right_df.sort_values('t'),
            on='t',
            tolerance=threshold_ns,
            allow_exact_matches=True,
            direction='nearest'
        )

        base_df.drop(columns=[f'right{shifted_idx}_int'], inplace=True)

    # Drop all rows with any missing values
    base_df.dropna(inplace=True)

    # Drop the 't' column used for merge key
    base_df.drop(columns=['t'], inplace=True)

    base_df.to_csv(output_csv_path, index=False, header=False)
    logging.info(f"Multi-match → {output_csv_path}")

def match_frames_full_from_csv_multi(csv_path_left, csv_path_right_list, output_csv_path, threshold_ns):
    df_left = pd.read_csv(csv_path_left, header=None, names=["t"])
    base_df = pd.DataFrame({'t': df_left["t"].astype(np.int64), 'left': df_left["t"].astype(np.int64)})

    for idx, csv_path_right in enumerate(csv_path_right_list, start=1):  # start at 1 for labeling
        df_right = pd.read_csv(csv_path_right, header=None, names=["t"])
        right_df = pd.DataFrame({
            't': df_right["t"].astype(np.int64),
            f'right{idx}_int': df_right["t"].astype(np.int64),
            f'right{idx}': df_right["t"].astype(str)
        })
        base_df = pd.merge_asof(
            base_df.sort_values('t'),
            right_df.sort_values('t'),
            on='t',
            tolerance=threshold_ns,
            allow_exact_matches=True,
            direction='nearest'
        )
        base_df.drop(columns=[f'right{idx}_int'], inplace=True)

    base_df = base_df.reset_index(drop=True)
    for idx in range(1, len(csv_path_right_list) + 1):
        base_df[f'right{idx}'] = base_df[f'right{idx}'].fillna('').astype(str)
        base_df[f'matched_{idx}'] = base_df[f'right{idx}'].apply(lambda x: x != '')

    base_df.drop(columns=['t'], inplace=True)

    total_frames = len(base_df)
    matched_1_only = ((base_df["matched_1"]) & (~base_df["matched_2"])).sum()
    matched_2_only = ((~base_df["matched_1"]) & (base_df["matched_2"])).sum()
    matched_1 = (base_df["matched_1"]).sum()
    matched_2 = (base_df["matched_2"]).sum()
    matched_both = ((base_df["matched_1"]) & (base_df["matched_2"])).sum()

    logging.info(f"Total frames in device_0: {total_frames}")
    logging.info(f"matched_1: {matched_1}")
    logging.info(f"matched_2: {matched_2}")
    logging.info(f"matched_both: {matched_both}")
    logging.info(f"matched_1_only: {matched_1_only}")
    logging.info(f"matched_2_only: {matched_2_only}")

    base_df.to_csv(output_csv_path, index=False)
    logging.info(f"Multi-match (full) → {output_csv_path}")


def parse_duration_seconds(duration_str):
    units = {
        'ns': 1e-9,
        'us': 1e-6,
        'ms': 1e-3,
        's': 1.0
    }
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ns|us|ms|s)", duration_str.strip())
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}")
    value, unit = match.groups()
    return float(value) * units[unit]


def get_csv_path_from_video(video_path):
    """
    Given a video file path, returns the corresponding timestamp CSV file path
    by parsing the video name and assuming the CSV is located in the parent directory.
    """
    video_name = Path(video_path).stem
    match = re.search(r"VID_((\d|_)+)", video_name)
    if not match:
        raise ValueError(f"[ERROR] Video name format is incorrect: {video_name}")
    video_date = match.group(1)
    csv_path = Path(video_path).parent.parent / f"{video_date}.csv"
    return csv_path


def extract_frame_data(target_dir, video_path):
    """
    Renames extracted frames in `target_dir` based on timestamps in the CSV file
    associated with the given `video_path`.
    """
    csv_path = get_csv_path_from_video(video_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"[ERROR] Timestamp CSV not found: {csv_path}")

    with csv_path.open() as f:
        filename_timestamps = [(line.strip(), int(line.strip())) for line in f if line.strip()]

    target_dir = Path(target_dir)
    frame_files = sorted([f for f in target_dir.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS])

    if len(frame_files) != len(filename_timestamps):
        raise ValueError(
            f"[ERROR] Frame count ({len(frame_files)}) does not match timestamp count ({len(filename_timestamps)})."
        )

    extension = frame_files[0].suffix

    for i, (timestamp_name, _) in enumerate(filename_timestamps):
        old_name = target_dir / f"frame-{i + 1}{extension}"
        new_name = target_dir / f"{timestamp_name}{extension}"
        if old_name.exists():
            old_name.rename(new_name)
        else:
            raise FileNotFoundError(f"[ERROR] Expected frame not found: {old_name}")

def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    video_path_0 = config["video_path_0"]
    video_path_1 = config["video_path_1"]
    video_path_2 = config["video_path_2"]
    threshold_str = config.get("threshold")
    threshold_ns = parse_duration_ns(threshold_str)

    video_name_0 = Path(video_path_0).stem
    base_output = os.path.join("output", "exp", video_name_0)
    os.makedirs(base_output, exist_ok=True)

    output_path_0 = os.path.join(base_output, "0")
    output_path_1 = os.path.join(base_output, "1")
    output_path_2 = os.path.join(base_output, "2")

    log_file_path = os.path.join(base_output, "match.log")
    setup_logger(log_file_path)
    logging.info("==== New matching session started ====")

    extract_flag = config.get("extract", False)
    if extract_flag:
        logging.info("Extracting frames for all videos as instructed by config.yaml...")
        extract_frames(video_path_0, output_path_0)
        extract_frame_data(output_path_0, video_path_0)

        extract_frames(video_path_1, output_path_1)
        extract_frame_data(output_path_1, video_path_1)

        extract_frames(video_path_2, output_path_2)
        extract_frame_data(output_path_2, video_path_2)
    else:
        logging.info("Frame extraction is disabled via config.yaml.")

    csv_0 = get_csv_path_from_video(video_path_0)
    csv_1 = get_csv_path_from_video(video_path_1)
    csv_2 = get_csv_path_from_video(video_path_2)

    threshold_str_for_filename = format_threshold_filename(threshold_ns)

    matched_csv_path = os.path.join(base_output, f"matched_multi_{threshold_str_for_filename}.csv")
    matched_csv_full_path = os.path.join(base_output, f"matched_multi_{threshold_str_for_filename}_full.csv")

    match_frames_from_csv_multi(csv_0, [csv_1, csv_2], matched_csv_path, threshold_ns)
    match_frames_full_from_csv_multi(csv_0, [csv_1, csv_2], matched_csv_full_path, threshold_ns)


if __name__ == "__main__":
    main()

