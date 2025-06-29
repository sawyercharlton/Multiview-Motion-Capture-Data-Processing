import os
import yaml
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_config(path="config.yaml"):
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

    match_df = pd.read_csv(csv_path)

    left_dir = os.path.join(root, "0")
    right_dir = os.path.join(root, "1")

    for _, row in tqdm(match_df.iterrows(), total=len(match_df), desc="Stitching"):
        try:
            left_id, right_id = row['left'], row['right']

            left_path = os.path.join(left_dir, f"{left_id}.png")
            right_path = os.path.join(right_dir, f"{right_id}.png")

            left_img = Image.open(left_path)
            right_img = Image.open(right_path)

            # Match to smallest height
            new_height = min(left_img.height, right_img.height)
            resize = lambda img: img.resize((int(img.width * new_height / img.height), new_height))
            left_img = resize(left_img)
            right_img = resize(right_img)

            # Stitch side by side
            total_width = left_img.width + right_img.width
            stitched_img = Image.new('RGB', (total_width, new_height))
            stitched_img.paste(left_img, (0, 0))
            stitched_img.paste(right_img, (left_img.width, 0))

            # Save
            output_filename = f"{left_id}+{right_id}.png"
            stitched_img.save(os.path.join(output_dir, output_filename))

        except Exception as e:
            print(f"Error stitching {left_id} and {right_id}: {e}")


if __name__ == "__main__":
    config = load_config()
    stitch_images(config)
