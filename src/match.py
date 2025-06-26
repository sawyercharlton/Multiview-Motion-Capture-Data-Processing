import os
import re
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


ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.npy', '.png', '.pcd']


def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
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


def extract_frame_data(target_dir, video_path):
    video_root, video_filename = os.path.split(video_path)
    video_name, _ = os.path.splitext(video_filename)
    video_date = re.sub(r"VID_((\d|_)*)", r"\1", video_name)

    video_parent_dir = os.path.abspath(os.path.join(video_root, os.pardir))
    csv_path = os.path.join(video_parent_dir, f"{video_date}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Timestamp CSV not found: {csv_path}")

    with open(csv_path) as f:
        filename_timestamps = [
            (line.strip(), int(line)) for line in f.readlines()
        ]

    frame_files = sorted([
        f for f in os.listdir(target_dir)
        if os.path.splitext(f)[1] in ALLOWED_EXTENSIONS
    ])

    if len(frame_files) != len(filename_timestamps):
        print(f"[WARNING] Frame count ({len(frame_files)}) does not match timestamp count ({len(filename_timestamps)}). Proceeding anyway.")

    _, extension = os.path.splitext(frame_files[0])

    for i, (timestamp_name, _) in enumerate(filename_timestamps[:len(frame_files)]):
        old_name = os.path.join(target_dir, f"frame-{i + 1}.png")
        new_name = os.path.join(target_dir, f"{timestamp_name}{extension}")
        os.rename(old_name, new_name)


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


def match_frames(vid_left, vid_right, output_csv_path, threshold_ns):
    out_images_left = sorted(glob.glob(os.path.join(vid_left, "*")))
    out_images_right = sorted(glob.glob(os.path.join(vid_right, "*")))

    image_timestamps_left = [
        int(os.path.splitext(os.path.basename(x))[0]) for x in out_images_left
    ]
    image_timestamps_right = [
        int(os.path.splitext(os.path.basename(x))[0]) for x in out_images_right
    ]

    left = pd.DataFrame({
        't': np.array(image_timestamps_left, dtype=np.int64),
        'left': np.array(image_timestamps_left, dtype=np.int64)
    })

    right = pd.DataFrame({
        't': np.array(image_timestamps_right, dtype=np.int64),
        'right_int': np.array(image_timestamps_right, dtype=np.int64),
        'right': list(map(str, image_timestamps_right))
    })

    df = pd.merge_asof(
        left, right, on='t',
        tolerance=threshold_ns,
        allow_exact_matches=True,
        direction='nearest'
    )
    df = df.dropna()
    df = df.drop(['t', 'right_int'], axis=1).reset_index(drop=True)
    df.to_csv(output_csv_path, index=False)
    logging.info(f" Matched {df.shape[0]} frame pairs (threshold = {threshold_ns / 1e6:.1f} ms) → {output_csv_path}")


def match_frames_full(vid_left, vid_right, output_csv_path, threshold_ns):
    out_images_left = sorted(glob.glob(os.path.join(vid_left, "*")))
    out_images_right = sorted(glob.glob(os.path.join(vid_right, "*")))

    image_timestamps_left = [
        int(os.path.splitext(os.path.basename(x))[0]) for x in out_images_left
    ]
    image_timestamps_right = [
        int(os.path.splitext(os.path.basename(x))[0]) for x in out_images_right
    ]

    left_df = pd.DataFrame({
        't': np.array(image_timestamps_left, dtype=np.int64),
        'left': np.array(image_timestamps_left, dtype=np.int64)
    })

    right_df = pd.DataFrame({
        't': np.array(image_timestamps_right, dtype=np.int64),
        'right_int': np.array(image_timestamps_right, dtype=np.int64),
        'right': list(map(str, image_timestamps_right))
    })

    # Match frames with merge_asof
    matched_df = pd.merge_asof(
        left_df, right_df, on='t',
        tolerance=threshold_ns,
        allow_exact_matches=True,
        direction='nearest'
    )

    matched_df = matched_df.reset_index(drop=True)
    full_output_df = matched_df[['left', 'right']].copy()

    # Fill NaN with empty string for unmatched
    full_output_df['left'] = full_output_df['left'].astype("Int64")  # nullable int
    full_output_df['right'] = full_output_df['right'].fillna('').astype(str)

    # Add match status column
    full_output_df["matched"] = full_output_df["right"].apply(lambda x: x != "")
    full_output_df.to_csv(output_csv_path, index=False)


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


def load_and_process_pair(path_a, path_b, threshold_s):
    df_a = pd.read_csv(path_a, header=None, names=["timestamp"])
    df_b = pd.read_csv(path_b, header=None, names=["timestamp"])
    df_a["timestamp"] = df_a["timestamp"].astype(np.int64) / 1e9
    df_b["timestamp"] = df_b["timestamp"].astype(np.int64) / 1e9
    df_a_sorted = df_a.sort_values("timestamp").reset_index(drop=True)
    df_b_sorted = df_b.sort_values("timestamp").reset_index(drop=True)
    df_a_sorted["key"] = df_a_sorted["timestamp"]
    df_b_sorted["key"] = df_b_sorted["timestamp"]
    matched = pd.merge_asof(
        df_a_sorted, df_b_sorted, on="key", direction="nearest",
        tolerance=threshold_s, suffixes=("_left", "_right")
    )
    matched["matched"] = ~matched["timestamp_right"].isna()
    return df_a_sorted, df_b_sorted, matched


def plot_matching(ax, df_a, df_b, matched, label_a, label_b,
                  color_a='orange', color_b='blue', match_color='green'):
    ax.scatter(df_a["timestamp"], [1] * len(df_a), color=color_a, label=label_a, s=0.5)
    ax.scatter(df_b["timestamp"], [0] * len(df_b), color=color_b, label=label_b, s=0.5)
    for _, row in matched[matched["matched"]].iterrows():
        ax.plot([row["timestamp_left"], row["timestamp_right"]], [1, 0],
                color=match_color, linewidth=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([label_b, label_a])
    ax.grid(True, axis='x', linestyle='--', linewidth=0.3)
    ax.legend(loc='upper right')


def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    video_path_0 = config["match"]["video_path_0"]
    video_path_1 = config["match"]["video_path_1"]
    threshold_str = config["match"].get("threshold")
    threshold_ns = parse_duration_ns(threshold_str)

    video_name_0 = Path(video_path_0).stem
    base_output = os.path.join("output", "exp", video_name_0)

    output_path_0 = os.path.join(base_output, "0")
    output_path_1 = os.path.join(base_output, "1")

    log_file_path = os.path.join(base_output, "match.log")
    setup_logger(log_file_path)
    logging.info("==== New matching session started ====")

    if not os.path.exists(output_path_0):
        logging.info(f"Output path {output_path_0} does not exist. Extracting video 0...")
        extract_frames(video_path_0, output_path_0)
        extract_frame_data(output_path_0, video_path_0)
    else:
        logging.info(f"Output path {output_path_0} already exists. Skipping extraction for video 0.")

    if not os.path.exists(output_path_1):
        logging.info(f"Output path {output_path_1} does not exist. Extracting video 1...")
        extract_frames(video_path_1, output_path_1)
        extract_frame_data(output_path_1, video_path_1)
    else:
        logging.info(f"Output path {output_path_1} already exists. Skipping extraction for video 1.")

    threshold_str_for_filename = format_threshold_filename(threshold_ns)
    matched_csv_name = f"matched_{threshold_str_for_filename}.csv"
    matched_csv_path = os.path.join(base_output, matched_csv_name)
    matched_csv_full_name = f"matched_{threshold_str_for_filename}_full.csv"
    matched_csv_full_path = os.path.join(base_output, matched_csv_full_name)
    match_frames(output_path_0, output_path_1, matched_csv_path, threshold_ns)
    match_frames_full(output_path_0, output_path_1, matched_csv_full_path, threshold_ns)


    if config["match"].get("visualize", False):
        try:
            video_root_0, video_filename_0 = os.path.split(video_path_0)
            video_name_0, _ = os.path.splitext(video_filename_0)
            video_date_0 = re.sub(r"VID_((\d|_)*)", r"\1", video_name_0)
            video_parent_dir_0 = os.path.abspath(os.path.join(video_root_0, os.pardir))
            csv_path_0 = os.path.join(video_parent_dir_0, f"{video_date_0}.csv")

            video_root_1, video_filename_1 = os.path.split(video_path_1)
            video_name_1, _ = os.path.splitext(video_filename_1)
            video_date_1 = re.sub(r"VID_((\d|_)*)", r"\1", video_name_1)
            video_parent_dir_1 = os.path.abspath(os.path.join(video_root_1, os.pardir))
            csv_path_1 = os.path.join(video_parent_dir_1, f"{video_date_1}.csv")

            pairs = [
                (csv_path_0, csv_path_0),
                (csv_path_1, csv_path_1),
                (csv_path_0, csv_path_1),
            ]

            color_schemes = [
                ('orange', 'orange', 'orange'),
                ('blue', 'blue', 'blue'),
                ('orange', 'blue', 'green')
            ]

            output_paths = [
                ("output_path_0", "output_path_0"),  # First figure
                ("output_path_1", "output_path_1"),  # Second figure
                ("output_path_0", "output_path_1"),  # Third figure
            ]

            titles = [
                "Timestamp: output_path_0",
                "Timestamp: output_path_1",
                "Timestamp Matching: output_path_0 vs output_path_1"
            ]

            threshold_s = parse_duration_seconds(threshold_str)
            fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

            for i, (path_a, path_b) in enumerate(pairs):
                df_a, df_b, matched_df = load_and_process_pair(path_a, path_b, threshold_s)
                color_a, color_b, match_color = color_schemes[i]
                out_a, out_b = output_paths[i]

                plot_matching(
                    axes[i], df_a, df_b, matched_df,
                    out_a, out_b,
                    color_a=color_a, color_b=color_b, match_color=match_color
                )
                axes[i].set_title(titles[i])

            min_ts = min([pd.read_csv(a, header=None).min()[0] for a, _ in pairs]) / 1e9
            max_ts = max([pd.read_csv(b, header=None).max()[0] for _, b in pairs]) / 1e9
            xticks = np.arange(np.floor(min_ts), np.ceil(max_ts)+1, 1)
            axes[-1].set_xticks(xticks)
            axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
            plt.xticks(rotation=45)
            axes[-1].set_xlabel("Timestamp (seconds)")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.warning(f"[WARNING] Visualization failed: {e}")


if __name__ == "__main__":
    main()

