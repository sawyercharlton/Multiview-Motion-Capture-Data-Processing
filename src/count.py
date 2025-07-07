import os

# Path to the folder
folder_path = r'C:\Users\yuboh\GitHub\Multiview-Motion-Capture-Data-Processing\src\output\exp\VID_20250702_125200\0_matched'
target_filename = '1751482324934908414.png'

# Get all PNG files in the folder and sort them
all_images = sorted(f for f in os.listdir(folder_path) if f.endswith('.png'))

# Find the index of the target image
if target_filename in all_images:
    index = all_images.index(target_filename)
    print(f"Number of images before '{target_filename}':", index)
else:
    print(f"'{target_filename}' not found in the folder.")
