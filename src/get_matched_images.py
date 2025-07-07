import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_matched_images(config):
    csv_path = config.get("get_matched_images")
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Matched CSV not found or not defined: {csv_path}")

    root = os.path.dirname(csv_path)

    dirs = {
        "0": os.path.join(root, "0"),
        "1": os.path.join(root, "1"),
        "2": os.path.join(root, "2"),
        "0_matched": os.path.join(root, "0_matched"),
        "1_matched": os.path.join(root, "1_matched"),
        "2_matched": os.path.join(root, "2_matched"),
    }

    os.makedirs(dirs["0_matched"], exist_ok=True)
    os.makedirs(dirs["1_matched"], exist_ok=True)
    os.makedirs(dirs["2_matched"], exist_ok=True)

    match_df = pd.read_csv(csv_path, header=None)

    for _, row in tqdm(match_df.iterrows(), total=len(match_df), desc="Copying matched images"):
        try:
            top_id, middle_id, bottom_id = row[0], row[1], row[2]

            for i, img_id in enumerate([top_id, middle_id, bottom_id]):
                src_dir = dirs[str(i)]
                dst_dir = dirs[f"{i}_matched"]
                src_path = os.path.join(src_dir, f"{img_id}.png")
                dst_path = os.path.join(dst_dir, f"{img_id}.png")

                if os.path.exists(src_path):
                    img = Image.open(src_path)
                    img.save(dst_path)
                else:
                    print(f"Warning: Missing image {src_path}")

        except Exception as e:
            print(f"Failed to process row {row.tolist()}: {e}")


if __name__ == "__main__":
    config = load_config()
    save_matched_images(config)
