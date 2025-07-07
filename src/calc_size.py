import os

def get_file_size(path):
    """Return the size of a file in bytes."""
    return os.path.getsize(path)

def get_dir_size(path):
    """Recursively return the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except FileNotFoundError:
                pass  # skip broken symlinks or deleted files
    return total_size

def human_readable_size(size_bytes):
    """Convert bytes to a human-readable format (KB, MB, GB, etc.)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def calculate_size(path):
    if os.path.isfile(path):
        size = get_file_size(path)
        print(f"File size: {human_readable_size(size)}")
    elif os.path.isdir(path):
        size = get_dir_size(path)
        print(f"Directory size: {human_readable_size(size)}")
    else:
        print("Invalid path.")

if __name__ == "__main__":
    user_input = input("Enter the full path to a file or directory: ").strip()
    calculate_size(user_input)
