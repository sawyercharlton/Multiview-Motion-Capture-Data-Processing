import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# ====== CONFIGURATION SECTION ======
CSV_A_PATH =  r"C:\Users\yuboh\GitHub\data\RecSync0\Standalone\20250626_190213.csv"
CSV_B_PATH =  r"C:\Users\yuboh\GitHub\data\RecSync0\Standalone\20250626_190213.csv"
THRESHOLD = "30ms"                         # Threshold as string (e.g., "30ms", "1s")
LABEL_A = "Device 0"                       # Label for Device A
LABEL_B = "Device 1"                       # Label for Device B
FIGSIZE = (15, 4)                          # Size of the output figure
UNMATCHED_LEN = 0.4                        # Length of the unmatched vertical bars
# ====================================

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


def main():
    threshold_s = parse_duration_seconds(THRESHOLD)
    df_a, df_b, matched = load_and_process_pair(CSV_A_PATH, CSV_B_PATH, threshold_s)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_matching(ax, df_a, df_b, matched, LABEL_A, LABEL_B, unmatched_len=UNMATCHED_LEN)
    ax.set_title("Frame Matching Visualization")
    ax.set_xlabel("Timestamp (seconds)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
