import cv2
import yaml


def load_video_path_0(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['video_path_0']


def draw_grid(image, step=100, color=(0, 255, 0), thickness=1, label=True):
    height, width = image.shape[:2]

    for x in range(0, width, step):
        cv2.line(image, (x, 0), (x, height), color, thickness)
        if label:
            cv2.putText(image, f'{x}', (x + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for y in range(0, height, step):
        cv2.line(image, (0, y), (width, y), color, thickness)
        if label:
            cv2.putText(image, f'{y}', (5,
                                        y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to read the first frame of the video: {video_path}")
    return frame


def resize_to_fit_screen(image, max_width=1920, max_height=1080):
    height, width = image.shape[:2]
    if width <= max_width and height <= max_height:
        return image  # No need to resize

    scale = min(max_width / width, max_height / height)
    new_size = (int(width * scale), int(height * scale))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized

def main(config_path):
    path0 = load_video_path_0(config_path)

    print(f"Processing {path0}")
    frame = extract_first_frame(path0)
    grid_frame = draw_grid(frame.copy(), step=100)

    resized_grid = resize_to_fit_screen(grid_frame)
    window_name = "Grid on Frame from video_path_0"
    cv2.imshow(window_name, resized_grid)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    config_file = "config.yaml"  # Path to your uploaded config
    main(config_file)
