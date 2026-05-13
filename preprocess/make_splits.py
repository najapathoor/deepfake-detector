import os
import shutil
import random

# Paths to actual videos
categories = {
    r"C:\deepcheck\original_sequences\youtube\c23\videos": "real",
    r"C:\deepcheck\manipulated_sequences\Deepfakes\c23\videos": "fake"
}

split_path = r"C:\deepcheck\dataset_split"
splits = ["train", "val", "test"]
split_ratio = [0.7, 0.15, 0.15]

# Create split folders
for split in splits:
    for cat_name in categories.values():
        os.makedirs(os.path.join(split_path, split, cat_name), exist_ok=True)

# Split videos
for src_path, new_name in categories.items():
    videos = [v for v in os.listdir(src_path) if v.endswith(".mp4")]
    random.shuffle(videos)

    total = len(videos)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    splits_files = {
        "train": videos[:train_end],
        "val": videos[train_end:val_end],
        "test": videos[val_end:]
    }

    # Copy videos
    for split in splits:
        for video in splits_files[split]:
            src = os.path.join(src_path, video)
            dst = os.path.join(split_path, split, new_name, video)
            shutil.copy2(src, dst)

print("Dataset split complete with real/fake folders!")
