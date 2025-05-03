import os
import random

def generate_split_lists(dataset_root, output_dir, train_ratio=0.8, val_ratio=0.1):
    image_paths = []

    # Recursively walk the dataset and collect image paths
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.jpg'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, dataset_root)
                image_paths.append(relative_path.replace("\\", "/"))  # Windows-safe

    print(f"Total images found: {len(image_paths)}")

    # Shuffle for randomness
    random.shuffle(image_paths)

    # Split
    total = len(image_paths)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_paths = image_paths[:train_end]
    val_paths = image_paths[train_end:val_end]
    test_paths = image_paths[val_end:]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write to files
    with open(os.path.join(output_dir, "train_gt.txt"), "w") as f:
        f.write("\n".join(train_paths))
    with open(os.path.join(output_dir, "val_gt.txt"), "w") as f:
        f.write("\n".join(val_paths))
    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_paths))

    print("List files generated:")
    print(f"- train_gt.txt: {len(train_paths)} samples")
    print(f"- val_gt.txt: {len(val_paths)} samples")
    print(f"- test.txt: {len(test_paths)} samples")

# Customize this with your actual path
if __name__ == "__main__":
    generate_split_lists(
        dataset_root="/home/kciri/scnn-lane-detection/data/CULane",      # Adjust to match your folder
        output_dir="/home/kciri/scnn-lane-detection/data/CULane/list",   # Folder to save txt files
    )
