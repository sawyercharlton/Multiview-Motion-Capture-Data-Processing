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


def stitch_images(config):
    csv_path = config.get("stitch_ref")
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Matched CSV not found or not defined: {csv_path}")

    root = os.path.dirname(csv_path)
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join(root, stem)
    os.makedirs(output_dir, exist_ok=True)

    match_df = pd.read_csv(csv_path, header=None)  # Read without header

    top_dir = os.path.join(root, "0")
    middle_dir = os.path.join(root, "1")
    bottom_dir = os.path.join(root, "2")

    for _, row in tqdm(match_df.iterrows(), total=len(match_df), desc="Stitching"):
        try:
            top_id, middle_id, bottom_id = row[0], row[1], row[2]

            top_path = os.path.join(top_dir, f"{top_id}.png")
            middle_path = os.path.join(middle_dir, f"{middle_id}.png")
            bottom_path = os.path.join(bottom_dir, f"{bottom_id}.png")

            top_img = Image.open(top_path)
            middle_img = Image.open(middle_path)
            bottom_img = Image.open(bottom_path)

            # Resize to match smallest width
            new_width = min(top_img.width, middle_img.width, bottom_img.width)
            resize = lambda img: img.resize((new_width, int(img.height * new_width / img.width)))
            top_img = resize(top_img)
            middle_img = resize(middle_img)
            bottom_img = resize(bottom_img)

            # Stack images vertically
            total_height = top_img.height + middle_img.height + bottom_img.height
            stitched_img = Image.new('RGB', (new_width, total_height))
            stitched_img.paste(top_img, (0, 0))
            stitched_img.paste(middle_img, (0, top_img.height))
            stitched_img.paste(bottom_img, (0, top_img.height + middle_img.height))

            # Save
            output_filename = f"{top_id}+{middle_id}+{bottom_id}.png"
            stitched_img.save(os.path.join(output_dir, output_filename))

        except Exception as e:
            print(f"Failed to stitch row {row.tolist()}: {e}")


if __name__ == "__main__":
    config = load_config()
    stitch_images(config)
