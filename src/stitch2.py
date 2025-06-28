import os
import yaml
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def stitch_images(config):
    stitch_config = config["stitch"]
    threshold_str = stitch_config["threshold"]
    root = stitch_config["root"]
    output_dir = os.path.join(root, f"stitched_{threshold_str}.csv")
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir)


    matched_csv_name = f"matched_{threshold_str}.csv"
    matched_csv_path = os.path.join(root, matched_csv_name)

    if not os.path.exists(matched_csv_path):
        raise FileNotFoundError(f"Matched CSV not found: {matched_csv_path}")

    match_df = pd.read_csv(matched_csv_path)

    left_dir = os.path.join(root, stitch_config["left_subdir"])
    right_dir = os.path.join(root, stitch_config["right_subdir"])

    for idx, row in tqdm(match_df.iterrows(), total=len(match_df), desc="Stitching"):
        left_path = os.path.join(left_dir, f"{row['left']}.png")
        right_path = os.path.join(right_dir, f"{row['right']}.png")

        try:
            left_img = Image.open(left_path)
            right_img = Image.open(right_path)

            # Match height
            if left_img.height != right_img.height:
                new_height = min(left_img.height, right_img.height)
                left_img = left_img.resize((left_img.width, new_height))
                right_img = right_img.resize((right_img.width, new_height))

            # Stitch side by side
            total_width = left_img.width + right_img.width
            stitched_img = Image.new('RGB', (total_width, left_img.height))
            stitched_img.paste(left_img, (0, 0))
            stitched_img.paste(right_img, (left_img.width, 0))

            # output_filename = f"{idx:04d}.png"
            output_filename = f"{row['left']}+{row['right']}.png"

            stitched_img.save(os.path.join(output_dir, output_filename))
            # print(f"Stitched {output_filename}: {row['left']} + {row['right']}")

        except Exception as e:
            print(f"Failed to stitch {idx}: {e}")


if __name__ == "__main__":
    config = load_config()
    stitch_images(config)
