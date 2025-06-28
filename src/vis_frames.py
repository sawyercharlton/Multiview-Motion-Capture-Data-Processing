import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import FuncFormatter
import os
from pathlib import Path


def parse_duration_seconds(duration_str):
    units = {'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1.0}
    import re
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
                  color_a='orange', color_b='blue', match_color='green', unmatched_len=0.4):
    ax.scatter(df_a["timestamp"], [1] * len(df_a), color=color_a, label=label_a, s=0.5)
    ax.scatter(df_b["timestamp"], [0] * len(df_b), color=color_b, label=label_b, s=0.5)

    for _, row in matched[matched["matched"]].iterrows():
        ax.plot([row["timestamp_left"], row["timestamp_right"]], [1, 0],
                color=match_color, linewidth=0.5)

    unmatched_left = matched[~matched["matched"]]["timestamp_left"]
    for ts in unmatched_left:
        ax.plot([ts, ts], [1, 1 - unmatched_len], color=color_a, linewidth=0.5)

    matched_right_set = set(matched[matched["matched"]]["timestamp_right"].dropna())
    unmatched_right = df_b[~df_b["timestamp"].isin(matched_right_set)]["timestamp"]
    for ts in unmatched_right:
        ax.plot([ts, ts], [0, 0 + unmatched_len], color=color_b, linewidth=0.5)

    ax.set_yticks([0, 1])
    # ax.set_yticklabels([label_b, label_a])
    ax.yaxis.set_visible(False)


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


def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    csv_0 = get_csv_path_from_video(config["video_path_0"])
    csv_1 = get_csv_path_from_video(config["video_path_1"])
    csv_2 = get_csv_path_from_video(config["video_path_2"])
    threshold_s = parse_duration_seconds(config["threshold"])

    df_01_a, df_01_b, matched_01 = load_and_process_pair(csv_0, csv_1, threshold_s)
    df_02_a, df_02_b, matched_02 = load_and_process_pair(csv_0, csv_2, threshold_s)
    # mark the start-time (we’ll display everything relative to this)
    origin = df_01_a["timestamp"].min()



    fig, ax = plt.subplots(figsize=(15, 3))
    # ymin, ymax = ax.get_ylim()
    # y_margin = 0.3 * (ymax - ymin)
    # ax.set_ylim(ymin - y_margin, ymax + y_margin)
    # fig.subplots_adjust(top=0.70, bottom=0.30)


    # Reusable function for custom y-coordinates
    def plot_custom_matching(ax, df_a, df_b, matched, y_a, y_b,
                             label_a, label_b,
                             color_a='orange', color_b='blue', match_color='green',
                             unmatched_len=0):

        ax.scatter(df_a["timestamp"], [y_a] * len(df_a), color=color_a, label=label_a, s=0.6)
        ax.scatter(df_b["timestamp"], [y_b] * len(df_b), color=color_b, label=label_b, s=0.6)

        for _, row in matched[matched["matched"]].iterrows():
            ax.plot([row["timestamp_left"], row["timestamp_right"]], [y_a, y_b],
                    color=match_color, linewidth=0.4)

        unmatched_left = matched[~matched["matched"]]["timestamp_left"]
        for ts in unmatched_left:
            ax.plot([ts, ts], [y_a, y_a + unmatched_len * np.sign(y_b - y_a)],
                    color=color_a, linewidth=0.5)

        matched_right_set = set(matched[matched["matched"]]["timestamp_right"].dropna())
        unmatched_right = df_b[~df_b["timestamp"].isin(matched_right_set)]["timestamp"]
        for ts in unmatched_right:
            ax.plot([ts, ts], [y_b, y_b - unmatched_len * np.sign(y_b - y_a)],
                    color=color_b, linewidth=0.5)

    # Device 0 vs Device 1: Device 1 at y=1, Device 0 at y=0
    plot_custom_matching(ax, df_01_a, df_01_b, matched_01,
                         y_a=0, y_b=1,
                         label_a="Device 0", label_b="Device 1",
                         color_a='orange', color_b='blue', match_color='green')

    # Device 0 vs Device 2: Device 2 at y=-1, Device 0 at y=0 again
    plot_custom_matching(ax, df_02_a, df_02_b, matched_02,
                         y_a=0, y_b=-1,
                         label_a="Device 0", label_b="Device 2",
                         color_a='orange', color_b='purple', match_color='green')

    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Device 2", "Device 0", "Device 1"])
    ax.set_title("Frame Matching Visualization (Device 0 as anchor)")
    origin = df_01_a["timestamp"].min()

    # at the end, instead of your current formatter:
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x - origin:.1f}")
    )
    ax.set_xlabel("Elapsed time (seconds)")

    # ax.set_xlabel("Timestamp (seconds)")
    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.margins(y=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
